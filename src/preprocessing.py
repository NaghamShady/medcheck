"""Text cleaning and medication name normalization helpers."""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple


def expand_compact_name(name: str) -> str:
    """
    Expand compact names like VitaminC / ibuprofen400mg into spaced forms.
    Helps matching when users omit spaces.
    """
    if name is None:
        return ""
    text = str(name).strip()
    # CamelCase: VitaminC -> Vitamin C
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Letter-digit boundaries: vitaminC2 -> vitamin C 2 (after lower we still help here)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_drug_name(name: str) -> str:
    """Normalize a drug name for comparison (lowercase, trimmed, compact spaces)."""
    if name is None:
        return ""
    text = expand_compact_name(str(name))
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-+/]", "", text)
    # Common compact forms after lowercasing: vitaminc already split by CamelCase;
    # also handle all-lowercase compact vitamins: vitaminc, vitamine
    text = re.sub(r"\bvitamin([a-ek])\b", r"vitamin \1", text)
    text = re.sub(r"\bvit([a-ek])\b", r"vitamin \1", text)
    return text.strip()


def format_display_name(name: str) -> str:
    """Return a clean display version of a medication name."""
    if name is None:
        return ""
    text = expand_compact_name(str(name))
    text = re.sub(r"\s+", " ", text)
    return text


def parse_medication_input(raw_text: str) -> Tuple[List[str], List[str]]:
    """
    Parse user input into unique medications.

    Accepts one medication per line or comma-separated values.
    Returns (display_names, normalized_names) with duplicates removed.
    """
    if not raw_text or not str(raw_text).strip():
        return [], []

    # Split on newlines and commas
    parts = re.split(r"[\n,;]+", str(raw_text))

    display_names: List[str] = []
    seen_normalized: set = set()

    for part in parts:
        display = format_display_name(part)
        if not display:
            continue
        normalized = normalize_drug_name(display)
        if not normalized or normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)
        display_names.append(display)

    normalized_names = [normalize_drug_name(n) for n in display_names]
    return display_names, normalized_names


def safe_str(value) -> str:
    """Convert a value to a display string, treating missing values as empty."""
    if value is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "nat", ""}:
        return ""
    return text


def join_non_empty(values: Iterable, separator: str = ", ") -> str:
    """Join non-empty string values."""
    items = [safe_str(v) for v in values]
    items = [v for v in items if v]
    return separator.join(items)
