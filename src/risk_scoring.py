"""Overall risk scoring for medication combinations."""

from __future__ import annotations

from typing import Any, Dict, List


SEVERITY_SCORE = {
    "none": 0,
    "minor": 1,
    "moderate": 2,
    "major": 3,
    "severe": 4,
    "contraindicated": 4,
    "high": 4,
    "unknown / review required": 1,
    "unknown": 1,
}


def _severity_bucket(label: str) -> str:
    text = (label or "").strip().lower()
    if not text:
        return "unknown"
    if any(k in text for k in ["contraindic", "severe", "major", "high", "serious"]):
        return "major"
    if "moderate" in text or "medium" in text:
        return "moderate"
    if "minor" in text or "mild" in text or "low" in text:
        return "minor"
    return "unknown"


def calculate_overall_risk(
    interactions: List[Dict[str, Any]],
    num_medications: int,
    num_pairs: int,
) -> Dict[str, Any]:
    """
    Calculate overall risk classification.

    Priority: highest individual severity overrides total score.
    """
    counts = {"minor": 0, "moderate": 0, "major": 0, "unknown": 0}
    total_score = 0
    max_level = 0  # 0 none, 1 minor, 2 moderate, 3+ major

    for item in interactions:
        bucket = _severity_bucket(item.get("severity", ""))
        counts[bucket if bucket in counts else "unknown"] = (
            counts.get(bucket if bucket in counts else "unknown", 0) + 1
        )
        score = SEVERITY_SCORE.get(bucket, 1)
        if "contraindic" in (item.get("severity") or "").lower():
            score = 4
        total_score += score
        if bucket == "major":
            max_level = max(max_level, 3)
        elif bucket == "moderate":
            max_level = max(max_level, 2)
        elif bucket == "minor":
            max_level = max(max_level, 1)
        else:
            max_level = max(max_level, 1)

    if not interactions:
        level = "grey"
        title = "Not enough information"
        message = (
            "We don't have enough information to check this combination. "
            "To stay safe, please ask your doctor or pharmacist."
        )
    elif max_level >= 3 or counts["major"] > 0:
        level = "red"
        title = "High-risk combination"
        message = (
            "At least one major interaction was detected. "
            "Consult a healthcare professional before combining these medications."
        )
    elif max_level >= 2 or counts["moderate"] > 0:
        level = "yellow"
        title = "Moderate attention needed"
        message = (
            "At least one moderate interaction was detected, with no major interactions found. "
            "Discuss this combination with a doctor or pharmacist."
        )
    else:
        level = "yellow"
        title = "Moderate attention needed"
        message = (
            "Minor or unclassified interactions were detected. "
            "Review the details below and consult a healthcare professional if unsure."
        )

    return {
        "level": level,
        "title": title,
        "message": message,
        "num_medications": num_medications,
        "num_pairs": num_pairs,
        "num_interactions": len(interactions),
        "num_major": counts["major"],
        "num_moderate": counts["moderate"],
        "num_minor": counts["minor"],
        "num_unknown": counts["unknown"],
        "total_score": total_score,
        "severity_counts": counts,
    }
