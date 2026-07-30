"""LLM-powered rewriting for drug interaction descriptions."""

from __future__ import annotations

import base64
import json
import os
import re
import ssl
from typing import Optional, Tuple
from urllib import error, request

try:
    import certifi

    # Force Python HTTPS calls to use certifi's CA bundle (fixes macOS CERTIFICATE_VERIFY_FAILED).
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    certifi = None  # type: ignore


OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "openrouter/auto"
DEFAULT_OPENROUTER_VISION_MODEL = "openai/gpt-4o-mini"
DEFAULT_OPENROUTER_APP_NAME = "MedCheck"
SUPPORTED_IMAGE_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
MIN_EXPLANATION_WORDS = 8
MAX_EXPLANATION_WORDS = 32
INCOMPLETE_ENDINGS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "or",
    "the",
    "to",
    "taking",
    "with",
}


def _get_secret_value(streamlit_secrets, key: str) -> str:
    if streamlit_secrets is None:
        return ""
    try:
        value = streamlit_secrets.get(key, "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return ""


def _get_nested_secret_value(streamlit_secrets, section: str, key: str) -> str:
    if streamlit_secrets is None:
        return ""
    try:
        value = streamlit_secrets.get(section, {}).get(key, "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return ""


def get_openrouter_api_key(streamlit_secrets=None) -> str:
    """Read OpenRouter API key from Streamlit secrets or environment variables."""
    if streamlit_secrets is not None:
        key = _get_secret_value(streamlit_secrets, "OPENROUTER_API_KEY")
        if key:
            return key
        key = _get_nested_secret_value(streamlit_secrets, "openrouter", "api_key")
        if key:
            return key

    return os.getenv("OPENROUTER_API_KEY", "").strip()


def get_openrouter_model_name(streamlit_secrets=None) -> str:
    """Read an optional OpenRouter model override."""
    if streamlit_secrets is not None:
        model = _get_secret_value(streamlit_secrets, "OPENROUTER_MODEL")
        if model:
            return model
        model = _get_nested_secret_value(streamlit_secrets, "openrouter", "model")
        if model:
            return model

    return os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip() or DEFAULT_OPENROUTER_MODEL


def get_openrouter_vision_model_name(streamlit_secrets=None) -> str:
    """Read an optional OpenRouter vision model override."""
    if streamlit_secrets is not None:
        model = _get_secret_value(streamlit_secrets, "OPENROUTER_VISION_MODEL")
        if model:
            return model
        model = _get_nested_secret_value(streamlit_secrets, "openrouter", "vision_model")
        if model:
            return model

    return (
        os.getenv("OPENROUTER_VISION_MODEL", DEFAULT_OPENROUTER_VISION_MODEL).strip()
        or DEFAULT_OPENROUTER_VISION_MODEL
    )


def get_openrouter_site_url(streamlit_secrets=None) -> str:
    """Read an optional OpenRouter referer URL for app attribution."""
    if streamlit_secrets is not None:
        site_url = _get_secret_value(streamlit_secrets, "OPENROUTER_SITE_URL")
        if site_url:
            return site_url
        site_url = _get_nested_secret_value(streamlit_secrets, "openrouter", "site_url")
        if site_url:
            return site_url

    return os.getenv("OPENROUTER_SITE_URL", "").strip()


def get_openrouter_app_name(streamlit_secrets=None) -> str:
    """Read an optional OpenRouter app title for app attribution."""
    if streamlit_secrets is not None:
        app_name = _get_secret_value(streamlit_secrets, "OPENROUTER_APP_NAME")
        if app_name:
            return app_name
        app_name = _get_nested_secret_value(streamlit_secrets, "openrouter", "app_name")
        if app_name:
            return app_name

    return os.getenv("OPENROUTER_APP_NAME", DEFAULT_OPENROUTER_APP_NAME).strip() or DEFAULT_OPENROUTER_APP_NAME


def _build_openrouter_headers(api_key: str, site_url: str, app_name: str) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": app_name or DEFAULT_OPENROUTER_APP_NAME,
    }
    if site_url:
        headers["HTTP-Referer"] = site_url
    return headers


def _extract_response_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        content = " ".join(
            str(part.get("text") or part.get("content") or "")
            for part in content
            if isinstance(part, dict)
        )
    return " ".join(str(content).strip().strip("\"'`").split())


def _is_complete_explanation(text: str) -> bool:
    words = text.split()
    if len(words) < MIN_EXPLANATION_WORDS:
        return False
    if len(words) > MAX_EXPLANATION_WORDS:
        return False
    if text[-1:] not in ".!?":
        return False

    last_word = words[-1].strip(".,!?;:").lower()
    return last_word not in INCOMPLETE_ENDINGS


def _ssl_context():
    """Prefer certifi CA bundle; fall back to default system context."""
    if certifi is not None:
        try:
            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass
    return ssl.create_default_context()


def _post_openrouter(payload: dict, headers: dict) -> dict:
    encoded_payload = json.dumps(payload).encode("utf-8")
    req = request.Request(
        OPENROUTER_CHAT_COMPLETIONS_URL,
        data=encoded_payload,
        headers=headers,
        method="POST",
    )
    with request.urlopen(req, timeout=30, context=_ssl_context()) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _parse_medication_name_response(text: str) -> list[str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    cleaned = re.sub(r"^\s*json\s*", "", cleaned, flags=re.IGNORECASE)

    json_start = min(
        [idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx >= 0],
        default=-1,
    )
    if json_start > 0:
        json_end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if json_end > json_start:
            cleaned = cleaned[json_start : json_end + 1]

    names = []
    try:
        data = json.loads(cleaned)
        raw_names = data.get("medications", []) if isinstance(data, dict) else data
        if isinstance(raw_names, list):
            names = [str(name) for name in raw_names]
    except json.JSONDecodeError:
        for line in cleaned.splitlines():
            line = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
            if line:
                names.append(line)

    unique = []
    seen = set()
    for name in names:
        normalized = " ".join(name.strip().strip("\"'`,;").split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


def _format_extracted_medication_name(name: str) -> str:
    def format_word(match: re.Match) -> str:
        word = match.group(0)
        if word.isupper() and len(word) > 4:
            return word.capitalize()
        return word

    return re.sub(r"[A-Z]{2,}", format_word, name)


def extract_medication_names_from_images(
    images: list[tuple[str, bytes]],
    api_key: str,
    model_name: str = DEFAULT_OPENROUTER_VISION_MODEL,
    site_url: str = "",
    app_name: str = DEFAULT_OPENROUTER_APP_NAME,
) -> Tuple[list[str], Optional[str]]:
    """
    Extract visible medication names from up to ten uploaded images.

    Returns (names, error). The caller should leave manual input unchanged when
    names is empty.
    """
    if not api_key:
        return [], "OpenRouter API key is not configured."
    if not images:
        return [], "Upload at least one photo first."
    if len(images) > 10:
        return [], "Please upload no more than 10 photos."

    content = [
        {
            "type": "text",
            "text": (
                "Extract medication names visible in these photos. Look for names on "
                "boxes, bottles, blister packs, labels, and prescriptions. Return JSON "
                'only in this exact shape: {"medications":["name"]}. Include only names '
                "you can read; do not guess. Do not include doses, counts, directions, "
                "or warnings unless they are part of the medicine name."
            ),
        }
    ]
    for content_type, image_bytes in images:
        if content_type not in SUPPORTED_IMAGE_CONTENT_TYPES:
            return [], f"Unsupported image type: {content_type or 'unknown'}."

    for content_type, image_bytes in images:
        base64_image = base64.b64encode(image_bytes).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{content_type};base64,{base64_image}"},
            }
        )

    payload = {
        "model": model_name or DEFAULT_OPENROUTER_VISION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You read medication packaging and labels. You return only valid JSON "
                    "with medication names that are visibly present."
                ),
            },
            {"role": "user", "content": content},
        ],
        "max_tokens": 220,
        "temperature": 0,
    }

    try:
        data = _post_openrouter(
            payload,
            _build_openrouter_headers(api_key, site_url, app_name),
        )
        text = _extract_response_text(data)
        names = [
            _format_extracted_medication_name(name)
            for name in _parse_medication_name_response(text)
        ]
        if not names:
            return [], "OpenRouter could not find readable medication names in those photos."
        return names, None
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        detail = " ".join(detail.split())[:300]
        return [], f"OpenRouter image extraction failed ({exc.code}): {detail or exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return [], f"OpenRouter image extraction failed: {exc}"


def simplify_interaction_description(
    drug_a: str,
    drug_b: str,
    severity: str,
    description: str,
    api_key: str,
    model_name: str = DEFAULT_OPENROUTER_MODEL,
    site_url: str = "",
    app_name: str = DEFAULT_OPENROUTER_APP_NAME,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Rewrite a dataset interaction sentence in plain language.

    Returns (plain_text, error). The caller should fall back to the original
    dataset description when plain_text is None.
    """
    if not api_key:
        return None, "OpenRouter API key is not configured."
    if not description:
        return None, "No interaction description was provided."

    prompt = f"""
Rewrite this medicine interaction for a general patient.

Rules:
- Use one complete sentence, 8 to 32 words.
- Use simple everyday language.
- Mention both medicine names.
- Do not add facts that are not in the source sentence.
- Do not give treatment advice beyond asking a doctor/pharmacist if risk is unclear.
- Do not say the medicines are safe.
- Return only the sentence; no bullets, labels, or preface.
- Make sure the sentence is complete and ends with punctuation.

Medicine A: {drug_a}
Medicine B: {drug_b}
Severity: {severity or "Unknown"}
Source sentence: {description}
""".strip()

    headers = _build_openrouter_headers(api_key, site_url, app_name)

    payload = {
        "model": model_name or DEFAULT_OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You translate drug interaction wording into plain English. "
                    "You are careful, concise, and never invent medical details."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 140,
        "temperature": 0.2,
    }

    try:
        data = _post_openrouter(payload, headers)
        text = _extract_response_text(data)
        if not text:
            return None, "OpenRouter returned an empty response."
        if _is_complete_explanation(text):
            return text, None

        retry_payload = {
            **payload,
            "messages": [
                *payload["messages"],
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        "That answer was too short or incomplete. Rewrite it as one complete "
                        "plain-English sentence, 8 to 32 words, ending with punctuation."
                    ),
                },
            ],
        }
        retry_data = _post_openrouter(retry_payload, headers)
        retry_text = _extract_response_text(retry_data)
        if _is_complete_explanation(retry_text):
            return retry_text, None

        return None, f"OpenRouter returned an incomplete summary: {text[:80]}"
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        detail = " ".join(detail.split())[:300]
        return None, f"OpenRouter request failed ({exc.code}): {detail or exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return None, f"OpenRouter request failed: {exc}"
