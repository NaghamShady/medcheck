"""Medicine details lookup helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rapidfuzz import fuzz, process

from .preprocessing import normalize_drug_name, safe_str, join_non_empty

# Common dosage-form suffixes for near-exact generic product labels
_FORM_SUFFIXES = (
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "syrup",
    "injection",
    "cream",
    "gel",
    "ointment",
    "drops",
    "suspension",
    "solution",
    "powder",
)


def _collect_row_values(row, columns: List[str]) -> List[str]:
    values = []
    for col in columns:
        if col in row.index:
            values.append(safe_str(row[col]))
    return values


def _compact_name(col: str) -> str:
    return "".join(ch for ch in str(col).lower() if ch.isalnum())


def _is_combo_product(norm_name: str) -> bool:
    """Heuristic: combination packs / multi-ingredient brand labels."""
    if "/" in norm_name or "+" in norm_name or " plus " in f" {norm_name} ":
        return True
    if re.search(r"\d+\s*mg\s*/\s*\d+", norm_name):
        return True
    return False


def _product_relevance_score(norm_name: str, target: str) -> int:
    """
    Higher is better. Prefer exact / near-generic labels over branded combos.
    """
    if not norm_name or not target:
        return -1

    if norm_name == target:
        return 1000

    for form in _FORM_SUFFIXES:
        if norm_name == f"{target} {form}":
            return 950
        if norm_name.startswith(f"{target} {form}"):
            score = 850 - len(norm_name)
            if _is_combo_product(norm_name):
                score -= 250
            return score

    # Whole-word start: "ibuprofen 400mg tablet"
    if re.match(rf"^{re.escape(target)}(\s|$)", norm_name):
        score = 700 - min(len(norm_name), 200)
        if _is_combo_product(norm_name):
            score -= 300
        return score

    # Contains as whole word
    if re.search(rf"(^|\s){re.escape(target)}(\s|$)", norm_name):
        score = 400 - min(len(norm_name), 200)
        if _is_combo_product(norm_name):
            score -= 250
        return score

    if target in norm_name:
        score = 150 - min(len(norm_name), 120)
        if _is_combo_product(norm_name):
            score -= 200
        return score

    return -1


def _best_product_hits(
    medicines_df: pd.DataFrame, target: str, limit: int = 25
) -> pd.DataFrame:
    """Rank catalog rows for a drug token and return the best matches."""
    scored: List[Tuple[int, int]] = []
    norms = medicines_df["_name_norm"].tolist()
    for idx, norm in enumerate(norms):
        score = _product_relevance_score(norm, target)
        if score >= 0:
            scored.append((score, idx))

    if not scored:
        return medicines_df.iloc[0:0]

    scored.sort(key=lambda x: (-x[0], x[1]))
    top = [i for _, i in scored[:limit]]
    out = medicines_df.iloc[top].copy()
    out["_relevance"] = [s for s, _ in scored[:limit]]
    return out


def get_medicine_details(
    medicine_name: str,
    medicines_df: Optional[pd.DataFrame],
    medicine_map: Optional[Dict[str, Any]] = None,
    fuzzy_threshold: float = 0.70,
    lookup_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Return medicine details for a medication.

    Preference order:
    1. Exact normalized product/generic name match
    2. Near-exact labels (e.g. "ibuprofen tablet")
    3. Best single-ingredient-like products containing the drug name
    4. RapidFuzz fallback on a narrowed candidate set

    The displayed medicine name stays as the requested drug (e.g. Ibuprofen),
    not a random brand combo pack.
    """
    empty = {
        "name": medicine_name,
        "generic": "",
        "uses": "",
        "side_effects": "",
        "substitutes": "",
        "found": False,
        "match_note": "",
        "source": "Medicine Details Dataset",
    }

    if medicines_df is None or medicines_df.empty or not medicine_map:
        return empty

    name_col = medicine_map.get("name")
    if not name_col:
        return empty

    candidates = [medicine_name] + list(lookup_names or [])
    # De-dupe while preserving order
    seen = set()
    search_terms = []
    for c in candidates:
        n = normalize_drug_name(c)
        if n and n not in seen:
            seen.add(n)
            search_terms.append(n)

    if not search_terms:
        return empty

    if "_name_norm" not in medicines_df.columns:
        medicines_df = medicines_df.copy()
        medicines_df["_name_norm"] = (
            medicines_df[name_col].map(safe_str).map(normalize_drug_name)
        )

    hits = medicines_df.iloc[0:0]
    match_note = ""

    for target in search_terms:
        exact = medicines_df[medicines_df["_name_norm"] == target]
        if not exact.empty:
            hits = exact
            match_note = "exact name match"
            break

        ranked = _best_product_hits(medicines_df, target)
        if not ranked.empty and int(ranked.iloc[0]["_relevance"]) >= 700:
            hits = ranked
            match_note = "near-exact / generic-preferring product match"
            break

        if not ranked.empty and int(ranked.iloc[0]["_relevance"]) >= 0:
            # Keep as fallback but continue searching other terms first
            if hits.empty:
                hits = ranked
                match_note = "best product match containing the drug name"

    # Fuzzy fallback if still nothing strong
    if hits.empty or (
        "_relevance" in hits.columns and int(hits.iloc[0].get("_relevance", 0)) < 150
    ):
        target = search_terms[0]
        prefix = target[:4] if len(target) >= 4 else target[:3]
        candidates_df = medicines_df[
            medicines_df["_name_norm"].str.contains(prefix, na=False, regex=False)
        ]
        if not candidates_df.empty:
            if len(candidates_df) > 8000:
                candidates_df = candidates_df.head(8000)
            choices = candidates_df["_name_norm"].tolist()
            results = process.extract(
                target,
                choices,
                scorer=fuzz.WRatio,
                score_cutoff=int(fuzzy_threshold * 100),
                limit=25,
            )
            if results:
                # Prefer non-combination packs among fuzzy hits
                ranked_fuzzy = []
                for match_norm, score, idx in results:
                    penalty = 40 if _is_combo_product(match_norm) else 0
                    ranked_fuzzy.append((score - penalty, -len(match_norm), idx))
                ranked_fuzzy.sort(reverse=True)
                best_idx = ranked_fuzzy[0][2]
                fuzzy_hit = candidates_df.iloc[[best_idx]]
                if hits.empty:
                    hits = fuzzy_hit
                    match_note = "related catalog products (brand-name match)"


    if hits.empty:
        return empty

    # Aggregate details from the top few relevant rows
    top_rows = hits.head(8)
    generic_col = medicine_map.get("generic")
    uses_col = medicine_map.get("uses")
    sides_col = medicine_map.get("side_effects")
    subs_col = medicine_map.get("substitutes")
    sub_list = medicine_map.get("substitute_list") or []
    use_list = medicine_map.get("use_list") or []
    side_list = medicine_map.get("side_effect_list") or []

    uses_parts: List[str] = []
    side_parts: List[str] = []
    substitutes_parts: List[str] = []
    generics: List[str] = []
    example_products: List[str] = []

    for _, row in top_rows.iterrows():
        product = safe_str(row[name_col])
        if product:
            example_products.append(product)

        if generic_col and generic_col in medicines_df.columns:
            generics.append(safe_str(row[generic_col]))
        else:
            for col in medicines_df.columns:
                if _compact_name(col) == "chemicalclass":
                    generics.append(safe_str(row[col]))
                    break

        if uses_col and uses_col in medicines_df.columns:
            uses_parts.append(safe_str(row[uses_col]))
        uses_parts.extend(_collect_row_values(row, use_list))

        if sides_col and sides_col in medicines_df.columns:
            side_parts.append(safe_str(row[sides_col]))
        side_parts.extend(_collect_row_values(row, side_list))

        if subs_col and subs_col in medicines_df.columns:
            substitutes_parts.append(safe_str(row[subs_col]))
        substitutes_parts.extend(_collect_row_values(row, sub_list))
        # Brand products themselves can be listed as substitutes / examples
        if product and normalize_drug_name(product) not in {
            normalize_drug_name(medicine_name),
            *search_terms,
        }:
            substitutes_parts.append(product)

    if not any(uses_parts):
        for col in medicines_df.columns:
            if _compact_name(col) == "therapeuticclass":
                uses_parts.append(safe_str(top_rows.iloc[0][col]))
                break

    # Prefer a clean generic label
    generic = ""
    for g in generics:
        if g:
            generic = g
            break
    if not generic:
        # If chemical class missing, use the requested medicine name as generic label
        generic = medicine_name

    # Deduplicate while preserving order
    def _uniq(items: List[str], limit: int = 12) -> str:
        out = []
        seen_local = set()
        for item in items:
            key = item.strip().lower()
            if not item or key in seen_local:
                continue
            seen_local.add(key)
            out.append(item)
            if len(out) >= limit:
                break
        return join_non_empty(out)

    display_name = medicine_name
    note = ""
    if match_note:
        note = (
            "Product-level rows in this catalog use brand names; "
            f"details below are summarized for {medicine_name}."
        )

    return {
        "name": display_name,
        "generic": generic,
        "uses": _uniq(uses_parts, limit=8),
        "side_effects": _uniq(side_parts, limit=12),
        "substitutes": _uniq(substitutes_parts, limit=10),
        "found": True,
        "match_note": note,
        "source": "Medicine Details Dataset",
    }
