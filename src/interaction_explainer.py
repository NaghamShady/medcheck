"""LLM-powered rewriting for drug interaction descriptions."""

from __future__ import annotations

import os
from typing import Optional, Tuple


DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


def get_gemini_api_key(streamlit_secrets=None) -> str:
    """Read Gemini API key from Streamlit secrets or environment variables."""
    if streamlit_secrets is not None:
        try:
            key = streamlit_secrets.get("GEMINI_API_KEY", "")
            if key:
                return str(key).strip()
        except Exception:
            pass
        try:
            key = streamlit_secrets.get("gemini", {}).get("api_key", "")
            if key:
                return str(key).strip()
        except Exception:
            pass

    return os.getenv("GEMINI_API_KEY", "").strip()


def get_gemini_model_name(streamlit_secrets=None) -> str:
    """Read an optional Gemini model override."""
    if streamlit_secrets is not None:
        try:
            model = streamlit_secrets.get("GEMINI_MODEL", "")
            if model:
                return str(model).strip()
        except Exception:
            pass
        try:
            model = streamlit_secrets.get("gemini", {}).get("model", "")
            if model:
                return str(model).strip()
        except Exception:
            pass

    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def simplify_interaction_description(
    drug_a: str,
    drug_b: str,
    severity: str,
    description: str,
    api_key: str,
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Rewrite a dataset interaction sentence in plain language.

    Returns (plain_text, error). The caller should fall back to the original
    dataset description when plain_text is None.
    """
    if not api_key:
        return None, "Gemini API key is not configured."
    if not description:
        return None, "No interaction description was provided."

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:  # noqa: BLE001
        return None, f"Gemini SDK is not installed: {exc}"

    prompt = f"""
Rewrite this medicine interaction for a general patient.

Rules:
- Use one short sentence, maximum 24 words.
- Use simple everyday language.
- Mention both medicine names.
- Do not add facts that are not in the source sentence.
- Do not give treatment advice beyond asking a doctor/pharmacist if risk is unclear.
- Do not say the medicines are safe.

Medicine A: {drug_a}
Medicine B: {drug_b}
Severity: {severity or "Unknown"}
Source sentence: {description}
""".strip()

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=80,
                temperature=0.2,
                candidate_count=1,
                system_instruction=(
                    "You translate drug interaction wording into plain English. "
                    "You are careful, concise, and never invent medical details."
                ),
            ),
        )
        text = (getattr(response, "text", "") or "").strip()
        if not text:
            return None, "Gemini returned an empty response."
        return " ".join(text.split()), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Gemini request failed: {exc}"
