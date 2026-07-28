"""CSV loading, column detection, and dataset validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .preprocessing import normalize_drug_name, safe_str

# Candidate column name aliases (lowercase keys)
INTERACTION_ALIASES: Dict[str, List[str]] = {
    "drug_1": [
        "drug 1",
        "drug_1",
        "drug1",
        "drug a",
        "drug_a",
        "medication 1",
        "medication_1",
        "medicine 1",
        "medicine_1",
    ],
    "drug_2": [
        "drug 2",
        "drug_2",
        "drug2",
        "drug b",
        "drug_b",
        "medication 2",
        "medication_2",
        "medicine 2",
        "medicine_2",
    ],
    "interaction": [
        "interaction",
        "description",
        "interaction description",
        "interaction_description",
        "interaction desc",
        "effect",
        "details",
        "interaction_effect",
    ],
    "severity": [
        "severity",
        "level",
        "risk",
        "severity_level",
        "interaction severity",
        "interaction_severity",
    ],
}

MEDICINE_ALIASES: Dict[str, List[str]] = {
    "name": [
        "medicine name",
        "medicine_name",
        "name",
        "drug name",
        "drug_name",
        "medication name",
        "medication_name",
        "product name",
        "product_name",
    ],
    "generic": [
        "generic name",
        "generic_name",
        "generic",
        "salt composition",
        "salt_composition",
        "composition",
        "active ingredient",
        "active_ingredient",
        "salt",
        "chemical class",
        "chemical_class",
    ],
    "uses": [
        "uses",
        "use",
        "indication",
        "indications",
        "therapeutic uses",
        "therapeutic_uses",
        "purpose",
    ],
    "side_effects": [
        "side effects",
        "side_effects",
        "side effect",
        "sideeffect",
        "adverse effects",
        "adverse_effects",
        "adverse reactions",
    ],
    "substitutes": [
        "substitutes",
        "substitute",
        "substitute medicines",
        "substitute_medicines",
        "alternatives",
        "alternative",
    ],
}


def _normalize_column_key(col: str) -> str:
    return re.sub(r"\s+", " ", str(col).strip().lower()).strip()


def _compact_col(col: str) -> str:
    return re.sub(r"[\s_]+", "", _normalize_column_key(col))


def _collect_numbered_columns(df: pd.DataFrame, prefixes: List[str]) -> List[str]:
    """Collect columns like use0, sideEffect12, substitute3."""
    found: List[Tuple[int, str]] = []
    compact_prefixes = [p.replace(" ", "").replace("_", "").lower() for p in prefixes]

    for col in df.columns:
        key = _compact_col(col)
        for prefix in compact_prefixes:
            if key.startswith(prefix):
                suffix = key[len(prefix) :]
                if suffix.isdigit():
                    found.append((int(suffix), col))
                    break

    found.sort(key=lambda x: x[0])
    # Preserve order, unique columns
    seen = set()
    ordered = []
    for _, col in found:
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    return ordered


def detect_columns(
    df: pd.DataFrame,
    aliases: Dict[str, List[str]],
    required: Optional[List[str]] = None,
) -> Tuple[Dict[str, Optional[str]], List[str]]:
    """
    Map logical field names to actual DataFrame columns.

    Returns (mapping, missing_required_fields).
    """
    col_lookup = {_normalize_column_key(c): c for c in df.columns}
    mapping: Dict[str, Optional[str]] = {}

    for field, candidates in aliases.items():
        found = None
        for candidate in candidates:
            key = _normalize_column_key(candidate)
            if key in col_lookup:
                found = col_lookup[key]
                break
        if found is None and field in col_lookup:
            found = col_lookup[field]
        mapping[field] = found

    # Numbered multi-columns used by common public medicine datasets
    mapping["substitute_list"] = _collect_numbered_columns(  # type: ignore
        df, ["substitute"]
    )
    mapping["use_list"] = _collect_numbered_columns(df, ["use", "uses"])  # type: ignore
    mapping["side_effect_list"] = _collect_numbered_columns(  # type: ignore
        df, ["sideeffect", "side effect", "side_effect"]
    )

    # If a single "uses"/"side effects" column was incorrectly picked as use0, clear it
    # when numbered lists exist and the mapped column is itself numbered.
    for field, list_key in (("uses", "use_list"), ("side_effects", "side_effect_list")):
        mapped = mapping.get(field)
        numbered = mapping.get(list_key) or []
        if mapped and numbered and mapped in numbered:
            mapping[field] = None

    missing: List[str] = []
    if required:
        for field in required:
            if not mapping.get(field):
                missing.append(field)

    return mapping, missing


def read_csv_flexible(path_or_buffer) -> pd.DataFrame:
    """Read a CSV trying common encodings."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
    last_error: Optional[Exception] = None

    for encoding in encodings:
        try:
            return pd.read_csv(path_or_buffer, encoding=encoding, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if hasattr(path_or_buffer, "seek"):
                try:
                    path_or_buffer.seek(0)
                except Exception:
                    pass

    raise ValueError(f"Could not read CSV. Last error: {last_error}")


def load_datasets(
    interaction_path: Optional[str | Path] = None,
    medicine_path: Optional[str | Path] = None,
    interaction_buffer=None,
    medicine_buffer=None,
) -> Dict[str, Any]:
    """
    Load and validate both datasets.

    Returns a dict with dataframes, column maps, and status messages.
    """
    result: Dict[str, Any] = {
        "interactions": None,
        "medicines": None,
        "interaction_map": {},
        "medicine_map": {},
        "errors": [],
        "warnings": [],
        "interaction_source": None,
        "medicine_source": None,
    }

    # --- Interactions ---
    try:
        if interaction_buffer is not None:
            interactions = read_csv_flexible(interaction_buffer)
            result["interaction_source"] = "uploaded"
        elif interaction_path and Path(interaction_path).exists():
            interactions = read_csv_flexible(interaction_path)
            result["interaction_source"] = str(interaction_path)
        else:
            interactions = None
            result["errors"].append(
                "Drug interaction CSV not found. Please upload drug_interactions.csv."
            )

        if interactions is not None:
            imap, missing = detect_columns(
                interactions,
                INTERACTION_ALIASES,
                required=["drug_1", "drug_2", "interaction"],
            )
            if missing:
                result["errors"].append(
                    _format_column_error(
                        "Drug Interaction Dataset",
                        list(interactions.columns),
                        missing,
                        INTERACTION_ALIASES,
                    )
                )
            else:
                interactions = interactions.copy()
                interactions["_drug_1_norm"] = (
                    interactions[imap["drug_1"]].map(safe_str).map(normalize_drug_name)
                )
                interactions["_drug_2_norm"] = (
                    interactions[imap["drug_2"]].map(safe_str).map(normalize_drug_name)
                )
                interactions = interactions[
                    (interactions["_drug_1_norm"] != "")
                    & (interactions["_drug_2_norm"] != "")
                ].drop_duplicates(subset=["_drug_1_norm", "_drug_2_norm"], keep="first")
                result["interactions"] = interactions
                result["interaction_map"] = imap
                # Built lazily in check_all_pairs; placeholder for callers
                result["interaction_index"] = None
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"Failed to load interaction dataset: {exc}")

    # --- Medicines ---
    try:
        if medicine_buffer is not None:
            medicines = read_csv_flexible(medicine_buffer)
            result["medicine_source"] = "uploaded"
        elif medicine_path and Path(medicine_path).exists():
            medicines = read_csv_flexible(medicine_path)
            result["medicine_source"] = str(medicine_path)
        else:
            medicines = None
            result["errors"].append(
                "Medicine details CSV not found. Please upload medicine_details.csv."
            )

        if medicines is not None:
            mmap, missing = detect_columns(
                medicines,
                MEDICINE_ALIASES,
                required=["name"],
            )
            if missing:
                result["errors"].append(
                    _format_column_error(
                        "Medicine Details Dataset",
                        list(medicines.columns),
                        missing,
                        MEDICINE_ALIASES,
                    )
                )
            else:
                medicines = medicines.copy()
                medicines["_name_norm"] = (
                    medicines[mmap["name"]].map(safe_str).map(normalize_drug_name)
                )
                medicines = medicines[medicines["_name_norm"] != ""].drop_duplicates(
                    subset=["_name_norm"], keep="first"
                )
                result["medicines"] = medicines
                result["medicine_map"] = mmap

                # Helpful note when product catalog is huge
                if len(medicines) > 20000:
                    result["warnings"].append(
                        "Large medicine catalog detected. Semantic embeddings use "
                        "interaction-dataset drug names for speed; product details "
                        "are matched with exact / fuzzy search."
                    )
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"Failed to load medicine details dataset: {exc}")

    return result


def build_unique_drug_list(
    interactions: Optional[pd.DataFrame],
    medicines: Optional[pd.DataFrame],
    interaction_map: Optional[Dict] = None,
    medicine_map: Optional[Dict] = None,
    include_medicine_products: bool = False,
) -> List[str]:
    """
    Build a sorted unique list of medication display names.

    By default uses interaction-dataset drug names (best for pairing + embeddings).
    Optionally includes medicine product names (can be very large).
    """
    names: Dict[str, str] = {}

    if interactions is not None and interaction_map:
        for col_key in ("drug_1", "drug_2"):
            col = interaction_map.get(col_key)
            if not col or col not in interactions.columns:
                continue
            for raw in interactions[col].tolist():
                display = safe_str(raw)
                norm = normalize_drug_name(display)
                if norm and norm not in names:
                    names[norm] = display

    if include_medicine_products and medicines is not None and medicine_map:
        col = medicine_map.get("name")
        if col and col in medicines.columns:
            for raw in medicines[col].tolist():
                display = safe_str(raw)
                norm = normalize_drug_name(display)
                if norm and norm not in names:
                    names[norm] = display

    return sorted(names.values(), key=lambda x: normalize_drug_name(x))


def _format_column_error(
    dataset_name: str,
    found_columns: List[str],
    missing_fields: List[str],
    aliases: Dict[str, List[str]],
) -> str:
    expected_examples = []
    for field in missing_fields:
        examples = aliases.get(field, [field])[:4]
        expected_examples.append(f"  - {field}: e.g. {', '.join(examples)}")

    return (
        f"{dataset_name}: could not identify required columns.\n"
        f"Columns found: {found_columns}\n"
        f"Missing fields:\n" + "\n".join(expected_examples) + "\n"
        "Please rename your CSV columns to match one of the expected names."
    )
