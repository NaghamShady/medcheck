"""Medication name matching with exact, RapidFuzz, and MiniLM embeddings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .brand_aliases import BRAND_ALIASES
from .preprocessing import normalize_drug_name, format_display_name, safe_str

# Common layman / brand names → dataset preferred names (normalized keys)
DRUG_ALIASES = BRAND_ALIASES

_FORM_WORDS = re.compile(
    r"\b(tablet|tablets|capsule|capsules|injection|syrup|cream|gel|"
    r"ointment|drops|suspension|solution|powder|mg|mcg|ml|g)\b"
)
_DOSE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|%)\b")


@dataclass
class MatchResult:
    entered: str
    matched: str  # Canonical dataset name used for interaction lookup
    method: str
    confidence: float
    status: str  # accepted | needs_review | rejected | not_found
    display_name: str = ""  # User-facing label (e.g. Aspirin / Vitamin C)
    generic_name: str = ""  # Shown under display when different (e.g. Ascorbic acid)
    candidates: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.matched or self.entered
        if not self.generic_name and self.matched and self.display_name:
            if normalize_drug_name(self.matched) != normalize_drug_name(self.display_name):
                self.generic_name = self.matched


def _build_norm_lookup(drug_names: List[str]) -> Dict[str, str]:
    norm_to_display: Dict[str, str] = {}
    for name in drug_names:
        n = normalize_drug_name(name)
        if n and n not in norm_to_display:
            norm_to_display[n] = name
    return norm_to_display


def _clean_product_text(text: str) -> str:
    t = normalize_drug_name(text)
    t = _DOSE.sub(" ", t)
    t = _FORM_WORDS.sub(" ", t)
    t = re.sub(r"[+/]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def resolve_brand_from_catalog(
    entered: str,
    drug_names: List[str],
    medicines_df: Optional[pd.DataFrame],
) -> Optional[Tuple[str, str]]:
    """
    If the entered name matches a medicine product brand stem, try to recover
    the interaction generic from the product / substitute text.

    Returns (canonical_display_name, method_label) or None.
    """
    if medicines_df is None or medicines_df.empty or not drug_names:
        return None

    norm = normalize_drug_name(entered)
    if not norm or len(norm) < 3:
        return None

    norm_to_display = _build_norm_lookup(drug_names)

    if "_name_norm" in medicines_df.columns:
        mask = (medicines_df["_name_norm"] == norm) | medicines_df["_name_norm"].str.startswith(
            norm + " ", na=False
        )
    elif "name" in medicines_df.columns:
        name_series = medicines_df["name"].map(safe_str).map(normalize_drug_name)
        mask = (name_series == norm) | name_series.str.startswith(norm + " ", na=False)
    else:
        return None

    hits = medicines_df.loc[mask]
    if hits.empty:
        return None

    sub_cols = [c for c in medicines_df.columns if str(c).lower().startswith("substitute")]
    texts: List[str] = []
    for _, row in hits.head(40).iterrows():
        texts.append(safe_str(row.get("name", "")))
        for c in sub_cols:
            val = safe_str(row.get(c, ""))
            if val:
                texts.append(val)

    # Strong: an interaction drug appears as a whole token in product text
    for text in texts:
        cleaned = _clean_product_text(text)
        tokens = cleaned.split()
        for drug_norm, display in norm_to_display.items():
            drug_tokens = drug_norm.split()
            if len(drug_tokens) == 1 and drug_tokens[0] in tokens:
                if drug_tokens[0] == norm:
                    continue
                return display, "Brand catalog match"
            if len(drug_tokens) > 1 and drug_norm in cleaned:
                return display, "Brand catalog match"

    return None


def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load the pretrained SentenceTransformer model.

    Returns (model, error_message). model is None if loading fails.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        return model, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def create_drug_embeddings(model, drug_names: List[str]) -> Optional[np.ndarray]:
    """Precompute embeddings for all unique medication names."""
    if model is None or not drug_names:
        return None
    try:
        embeddings = model.encode(
            drug_names,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings)
    except Exception:
        return None


def _cosine_similarity_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity assuming vectors may or may not be L2-normalized."""
    from sklearn.metrics.pairwise import cosine_similarity

    q = query_vec.reshape(1, -1)
    return cosine_similarity(q, matrix).flatten()


def _token_overlap_bonus(query: str, candidate: str) -> float:
    """Reward candidates that keep distinctive tokens (e.g. vitamin c vs vitamin a)."""
    q_tokens = set(normalize_drug_name(query).split())
    c_tokens = set(normalize_drug_name(candidate).split())
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
    q_vit = re.match(r"^vitamin\s+([a-z0-9]+)$", normalize_drug_name(query))
    c_vit = re.match(r"^vitamin\s+([a-z0-9]+)$", normalize_drug_name(candidate))
    if q_vit and c_vit and q_vit.group(1) != c_vit.group(1):
        return -0.35
    return 0.08 * overlap


def _shared_char_bigrams(a: str, b: str) -> float:
    """Cheap similarity guard — brands vs wrong generics often share almost nothing."""
    a = normalize_drug_name(a).replace(" ", "")
    b = normalize_drug_name(b).replace(" ", "")
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ba = {a[i : i + 2] for i in range(len(a) - 1)}
    bb = {b[i : i + 2] for i in range(len(b) - 1)}
    if not ba:
        return 0.0
    return len(ba & bb) / len(ba)


def match_medication_name(
    entered: str,
    drug_names: List[str],
    drug_embeddings: Optional[np.ndarray] = None,
    model=None,
    threshold: float = 0.75,
    medicines_df: Optional[pd.DataFrame] = None,
) -> MatchResult:
    """
    Match a user-entered medication name to the closest known drug.

    Priority:
    1. Exact normalized match
    2. Alias / synonym match (brands)
    3. Medicine-catalog brand → generic
    4. RapidFuzz (+ optional MiniLM), with safeguards against near-miss vitamins
    """
    display_entered = format_display_name(entered)
    norm_entered = normalize_drug_name(entered)

    if not norm_entered or not drug_names:
        return MatchResult(
            entered=display_entered,
            matched="",
            method="None",
            confidence=0.0,
            status="not_found",
        )

    norm_to_display = _build_norm_lookup(drug_names)

    # 1) Exact match
    if norm_entered in norm_to_display:
        return MatchResult(
            entered=display_entered,
            matched=norm_to_display[norm_entered],
            method="Exact match",
            confidence=1.0,
            status="accepted",
            display_name=norm_to_display[norm_entered],
        )

    # 1b) Common alias / synonym match
    alias_norm = DRUG_ALIASES.get(norm_entered)
    if alias_norm:
        friendly = display_entered
        if norm_entered.startswith("vitamin") or norm_entered.startswith("vit "):
            friendly = " ".join(w.capitalize() for w in norm_entered.split())
        if alias_norm in norm_to_display:
            canonical = norm_to_display[alias_norm]
            return MatchResult(
                entered=display_entered,
                matched=canonical,
                method="Alias / synonym match",
                confidence=1.0,
                status="accepted",
                display_name=friendly,
                generic_name=canonical,
            )
        # Known brand, but its generic is not in the interaction CSV —
        # do NOT fall through to fuzzy (Lantus → Lansoprazole).
        return MatchResult(
            entered=display_entered,
            matched=alias_norm,
            method="Brand known — generic not in interaction dataset",
            confidence=1.0,
            status="not_in_dataset",
            display_name=friendly,
            generic_name=alias_norm.title() if alias_norm else "",
        )

    # 1c) Exact vitamin letter form present in catalog
    vit = re.match(r"^vitamin\s+([a-z0-9]+)$", norm_entered)
    if vit:
        target = f"vitamin {vit.group(1)}"
        if target in norm_to_display:
            return MatchResult(
                entered=display_entered,
                matched=norm_to_display[target],
                method="Exact match",
                confidence=1.0,
                status="accepted",
                display_name=norm_to_display[target],
            )

    # 1d) Brand product found in medicine catalog → generic in interaction list
    catalog_hit = resolve_brand_from_catalog(entered, drug_names, medicines_df)
    if catalog_hit:
        canonical, method = catalog_hit
        return MatchResult(
            entered=display_entered,
            matched=canonical,
            method=method,
            confidence=0.95,
            status="accepted",
            display_name=display_entered,
            generic_name=canonical,
        )

    fuzz_score = 0.0
    fuzz_match = ""
    choices = list(norm_to_display.keys())
    fuzz_results = process.extract(
        norm_entered,
        choices,
        scorer=fuzz.WRatio,
        limit=8,
    )
    best_fuzz = None
    best_fuzz_adj = -1.0
    for cand_norm, raw, _ in fuzz_results or []:
        base = float(raw) / 100.0
        # Reject lookalike mismatches (Norvasc → Atorvastatin style)
        if _shared_char_bigrams(norm_entered, cand_norm) < 0.18 and base < 0.92:
            continue
        adj = base + _token_overlap_bonus(norm_entered, cand_norm)
        if adj > best_fuzz_adj:
            best_fuzz_adj = adj
            best_fuzz = (cand_norm, min(base, 1.0))
    if best_fuzz:
        fuzz_match_norm, fuzz_score = best_fuzz
        fuzz_match = norm_to_display.get(fuzz_match_norm, fuzz_match_norm)

    mini_score = 0.0
    mini_match = ""
    if model is not None and drug_embeddings is not None and len(drug_embeddings) == len(drug_names):
        try:
            query_emb = model.encode(
                [display_entered],
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            sims = _cosine_similarity_matrix(query_emb[0], drug_embeddings)
            top_idx = np.argsort(sims)[::-1][:8]
            best_mini_adj = -1.0
            for idx in top_idx:
                name = drug_names[int(idx)]
                base = float(sims[int(idx)])
                if _shared_char_bigrams(norm_entered, name) < 0.18 and base < 0.92:
                    continue
                adj = base + _token_overlap_bonus(norm_entered, name)
                if adj > best_mini_adj:
                    best_mini_adj = adj
                    mini_score = base
                    mini_match = name
        except Exception:
            mini_score = 0.0
            mini_match = ""

    method = "RapidFuzz"
    best_match = fuzz_match
    best_score = fuzz_score

    if mini_match and mini_score > 0:
        same = normalize_drug_name(fuzz_match) == normalize_drug_name(mini_match)
        fuzz_adj = fuzz_score + _token_overlap_bonus(norm_entered, fuzz_match)
        mini_adj = mini_score + _token_overlap_bonus(norm_entered, mini_match)

        if same:
            best_score = 0.55 * fuzz_score + 0.45 * mini_score
            best_match = fuzz_match or mini_match
            method = "Combined match"
        elif fuzz_adj >= mini_adj:
            best_score = fuzz_score
            best_match = fuzz_match
            method = "RapidFuzz"
        else:
            best_score = mini_score
            best_match = mini_match
            method = "MiniLM semantic match"
    elif not fuzz_match and mini_match:
        best_score = mini_score
        best_match = mini_match
        method = "MiniLM semantic match"

    q_vit = re.match(r"^vitamin\s+([a-z0-9]+)$", norm_entered)
    if q_vit and best_match:
        b_vit = re.match(r"^vitamin\s+([a-z0-9]+)$", normalize_drug_name(best_match))
        if b_vit and b_vit.group(1) != q_vit.group(1):
            if q_vit.group(1) == "c" and "ascorbic acid" in norm_to_display:
                return MatchResult(
                    entered=display_entered,
                    matched=norm_to_display["ascorbic acid"],
                    method="Alias / synonym match",
                    confidence=1.0,
                    status="accepted",
                    display_name="Vitamin C",
                    generic_name=norm_to_display["ascorbic acid"],
                )
            best_score = min(best_score, threshold - 0.01)

    candidates = []
    if fuzz_match:
        candidates.append({"name": fuzz_match, "score": fuzz_score, "source": "RapidFuzz"})
    if mini_match and normalize_drug_name(mini_match) != normalize_drug_name(fuzz_match or ""):
        candidates.append({"name": mini_match, "score": mini_score, "source": "MiniLM"})

    if not best_match or best_score < threshold:
        return MatchResult(
            entered=display_entered,
            matched=best_match or "",
            method=method if best_match else "None",
            confidence=round(best_score, 4),
            status="needs_review" if best_match else "not_found",
            display_name=best_match or display_entered,
            candidates=candidates or None,
        )

    return MatchResult(
        entered=display_entered,
        matched=best_match,
        method=method,
        confidence=round(best_score, 4),
        status="accepted",
        display_name=best_match,
        candidates=candidates or None,
    )


def match_all_medications(
    entered_names: List[str],
    drug_names: List[str],
    drug_embeddings: Optional[np.ndarray] = None,
    model=None,
    threshold: float = 0.75,
    medicines_df: Optional[pd.DataFrame] = None,
) -> List[MatchResult]:
    """Match a list of entered medication names."""
    return [
        match_medication_name(
            name,
            drug_names,
            drug_embeddings=drug_embeddings,
            model=model,
            threshold=threshold,
            medicines_df=medicines_df,
        )
        for name in entered_names
    ]
