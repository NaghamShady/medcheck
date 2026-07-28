"""Drug interaction lookup and severity inference."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .preprocessing import normalize_drug_name, safe_str


HIGH_KEYWORDS = [
    "contraindicated",
    "serious",
    "severe",
    "life-threatening",
    "life threatening",
    "fatal",
    "major",
    "significantly increase",
    "bleeding risk",
    "toxicity",
    "avoid",
    "do not combine",
    "hemorrhage",
    "haemorrhage",
    "gastrointestinal bleeding",
    "increase the anticoagulant",
    "anticoagulant activities",
]

MODERATE_KEYWORDS = [
    "moderate",
    "may increase",
    "monitor",
    "caution",
    "adjustment",
    "reduced effectiveness",
    "may decrease",
    "potential interaction",
    "risk or severity",
    "adverse effects",
    "can be increased",
    "serum concentration",
]

MINOR_KEYWORDS = [
    "minor",
    "mild",
    "low clinical significance",
    "limited clinical",
]


def generate_medication_pairs(medications: List[str]) -> List[Tuple[str, str]]:
    """Generate every unique unordered medication pair."""
    unique = []
    seen = set()
    for med in medications:
        key = normalize_drug_name(med)
        if key and key not in seen:
            seen.add(key)
            unique.append(med)
    return list(combinations(unique, 2))


def infer_severity(description: str) -> Tuple[str, str]:
    """
    Infer severity from interaction description keywords.

    Returns (severity_label, source) where source is 'keyword-inferred'.
    Used only when the dataset has no severity column / value.
    """
    text = (description or "").lower()
    if not text.strip():
        return "Unknown / Review Required", "keyword-inferred"

    # Strong clinical patterns first (before broad "may increase")
    if "anticoagulant" in text and any(
        k in text for k in ["increase", "potentiat", "enhance", "elevat"]
    ):
        return "Major", "keyword-inferred"
    if "bleeding" in text or "haemorrhag" in text or "hemorrhag" in text:
        return "Major", "keyword-inferred"

    for kw in HIGH_KEYWORDS:
        if kw in text:
            return "Major", "keyword-inferred"

    for kw in MODERATE_KEYWORDS:
        if kw in text:
            return "Moderate", "keyword-inferred"

    for kw in MINOR_KEYWORDS:
        if kw in text:
            return "Minor", "keyword-inferred"

    return "Unknown / Review Required", "keyword-inferred"


def normalize_severity_label(raw: str) -> str:
    """Map dataset severity values to standard labels."""
    text = safe_str(raw).lower()
    if not text:
        return ""

    if any(k in text for k in ["contraindic", "major", "severe", "high", "serious"]):
        return "Major"
    if any(k in text for k in ["moderate", "medium"]):
        return "Moderate"
    if any(k in text for k in ["minor", "mild", "low"]):
        return "Minor"
    if "unknown" in text:
        return "Unknown / Review Required"
    return safe_str(raw).title() or "Unknown / Review Required"


def build_interaction_index(interactions_df: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    """Map frozenset-like sorted drug pairs to the first matching row position."""
    index: Dict[Tuple[str, str], int] = {}
    if interactions_df is None or interactions_df.empty:
        return index
    d1 = interactions_df["_drug_1_norm"].tolist()
    d2 = interactions_df["_drug_2_norm"].tolist()
    for i, (a, b) in enumerate(zip(d1, d2)):
        if not a or not b:
            continue
        key = (a, b) if a <= b else (b, a)
        if key not in index:
            index[key] = i
    return index


def find_drug_interaction(
    drug_a: str,
    drug_b: str,
    interactions_df: Optional[pd.DataFrame],
    interaction_map: Optional[Dict[str, Any]] = None,
    pair_index: Optional[Dict[Tuple[str, str], int]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Search the interaction dataset for a drug pair (order-independent).

    Returns a result dict or None if no interaction found.
    """
    if interactions_df is None or interactions_df.empty:
        return None

    a = normalize_drug_name(drug_a)
    b = normalize_drug_name(drug_b)
    if not a or not b or a == b:
        return None

    if "_drug_1_norm" not in interactions_df.columns or "_drug_2_norm" not in interactions_df.columns:
        return None

    row = None
    if pair_index is not None:
        key = (a, b) if a <= b else (b, a)
        pos = pair_index.get(key)
        if pos is not None:
            row = interactions_df.iloc[pos]
    else:
        mask = (
            ((interactions_df["_drug_1_norm"] == a) & (interactions_df["_drug_2_norm"] == b))
            | ((interactions_df["_drug_1_norm"] == b) & (interactions_df["_drug_2_norm"] == a))
        )
        hits = interactions_df.loc[mask]
        if not hits.empty:
            row = hits.iloc[0]

    if row is None:
        return None
    imap = interaction_map or {}

    desc_col = imap.get("interaction")
    sev_col = imap.get("severity")
    d1_col = imap.get("drug_1")
    d2_col = imap.get("drug_2")

    description = safe_str(row[desc_col]) if desc_col else ""
    severity_source = "dataset"
    if sev_col and sev_col in interactions_df.columns:
        severity = normalize_severity_label(row[sev_col])
        if not severity:
            severity, severity_source = infer_severity(description)
    else:
        severity, severity_source = infer_severity(description)

    # Prefer user-facing names from the pair query
    return {
        "drug_a": drug_a,
        "drug_b": drug_b,
        "severity": severity,
        "severity_source": severity_source,
        "description": description or "No description available in the dataset.",
        "source": "Drug Interaction Dataset",
        "dataset_drug_1": safe_str(row[d1_col]) if d1_col else drug_a,
        "dataset_drug_2": safe_str(row[d2_col]) if d2_col else drug_b,
    }


def check_all_pairs(
    medications: List[str],
    interactions_df: Optional[pd.DataFrame],
    interaction_map: Optional[Dict[str, Any]] = None,
    pair_index: Optional[Dict[Tuple[str, str], int]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Check all medication pairs.

    Returns (found_interactions, all_pair_results).
    """
    pairs = generate_medication_pairs(medications)
    found: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []

    if pair_index is None and interactions_df is not None and not interactions_df.empty:
        pair_index = build_interaction_index(interactions_df)

    for a, b in pairs:
        result = find_drug_interaction(
            a, b, interactions_df, interaction_map, pair_index=pair_index
        )
        if result:
            found.append(result)
            all_results.append({**result, "found": True})
        else:
            all_results.append(
                {
                    "drug_a": a,
                    "drug_b": b,
                    "severity": None,
                    "description": (
                        "We don't have enough information to check this combination. "
                        "To stay safe, please ask your doctor or pharmacist."
                    ),
                    "source": "Drug Interaction Dataset",
                    "found": False,
                }
            )

    return found, all_results
