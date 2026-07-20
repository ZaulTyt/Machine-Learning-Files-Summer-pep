import json
import logging
import re
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"

# Groq safe limit: ~6 000 chars leaves plenty of room for the system prompt
# and the model's JSON response within Groq's request-body size limit.
GROQ_MAX_CHARS  = 8_000
# OpenRouter models generally accept larger payloads
OR_MAX_CHARS    = 30_000

GROQ_NATIVE_MODELS = {
    "meta-llama/llama-4-scout-17b-16e-instruct",
}

SYSTEM_PROMPT = (
    "You are an expert fact-checker and claim extraction assistant.\n"
    "Analyze the input text and extract specific, high-priority factual claims.\n\n"
    "CONSTRAINTS:\n"
    "1. Extract ONLY claims that contain concrete facts: statistics, percentages, dates, "
    "revenue figures, growth rates, or technical metrics.\n"
    "2. Skip subjective statements, opinions, or qualitative assertions.\n"
    "3. Output ONLY a valid JSON array. No markdown, no explanation.\n"
    "4. Each object must have exactly one key: 'claim'.\n\n"
    "Example:\n"
    "[{\"claim\": \"OpenAI was founded in December 2015.\"}, "
    "{\"claim\": \"The global economy grew by 3.2% in 2023.\"}]"
)


def _truncate(text: str, max_chars: int) -> str:
    """Keep the first 75 % and last 25 % of the budget to preserve structure."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.75)
    tail = max_chars - head
    logger.warning(
        f"Text ({len(text):,} chars) truncated to {max_chars:,} chars "
        f"(head={head}, tail={tail})."
    )
    return text[:head] + "\n\n[...text truncated...]\n\n" + text[-tail:]


def _parse_claims(content: str) -> list:
    """Parse the LLM JSON response into a validated list of claim dicts."""
    # Strip markdown fences
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content
        if content.endswith("```"):
            content = content[:-3].strip()
        elif "```" in content:
            content = content.split("```")[0].strip()

    claims_list = json.loads(content)

    if isinstance(claims_list, dict):
        for k in ("claims", "data", "results"):
            if k in claims_list and isinstance(claims_list[k], list):
                claims_list = claims_list[k]
                break

    if not isinstance(claims_list, list):
        raise ValueError("Parsed content is not a list.")

    validated = []
    for item in claims_list:
        if isinstance(item, dict) and "claim" in item:
            validated.append({"claim": str(item["claim"]).strip()})
        elif isinstance(item, str) and item.strip():
            validated.append({"claim": item.strip()})

    return validated


def _call_api(payload: dict, headers: dict, url: str, timeout: int = 45) -> str:
    """Make the API call and return the raw content string."""
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    if "choices" not in result or not result["choices"]:
        raise ValueError(f"Unexpected API response: {result}")
    return result["choices"][0]["message"]["content"].strip()


def extract_claims(text: str, api_key: str,
                   model: str = "qwen/qwen-2.5-coder-32b-instruct:free") -> list:
    """
    Extracts factual claims from document text via LLM.

    Automatically selects the right endpoint (Groq / OpenRouter) and
    applies safe character limits to avoid HTTP 413 errors.
    """
    if not api_key:
        raise ValueError("API key is required to extract claims.")

    is_groq = api_key.startswith("gsk_")
    url     = GROQ_URL if is_groq else OPENROUTER_URL

    # Resolve model ID
    if is_groq:
        if ":" in model or ("/" in model and model not in GROQ_NATIVE_MODELS):
            model = "llama-3.3-70b-versatile"

    # Build headers
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if not is_groq:
        headers["HTTP-Referer"] = "https://jetro.ai"
        headers["X-Title"]      = "Fact-Check Agent"

    # Attempt with decreasing text sizes if we hit a 413
    max_chars_steps = (
        [GROQ_MAX_CHARS, GROQ_MAX_CHARS // 2, GROQ_MAX_CHARS // 4]
        if is_groq else
        [OR_MAX_CHARS, OR_MAX_CHARS // 2]
    )

    last_error: Exception = RuntimeError("Unknown error")

    for max_chars in max_chars_steps:
        truncated_text = _truncate(text, max_chars)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Document Text:\n\n{truncated_text}"},
            ],
            "temperature": 0.1,
        }

        try:
            content = _call_api(payload, headers, url)
            claims  = _parse_claims(content)
            logger.info(f"Extracted {len(claims)} claims (input={len(truncated_text):,} chars).")
            return claims

        except requests.exceptions.HTTPError as he:
            if he.response.status_code == 413:
                logger.warning(
                    f"HTTP 413 with {len(truncated_text):,} chars — retrying with smaller input…"
                )
                last_error = he
                continue  # try next smaller size

            # Any other HTTP error → surface immediately
            try:
                err_msg = he.response.json().get("error", {}).get("message", he.response.text)
            except Exception:
                err_msg = he.response.text
            logger.error(f"HTTP {he.response.status_code} error: {err_msg}")
            raise ValueError(f"API Error {he.response.status_code}: {err_msg}")

        except json.JSONDecodeError as je:
            logger.error(f"JSON parse error: {je}")
            # Regex fallback
            fallback = re.findall(r'"claim"\s*:\s*"([^"]+)"', content if 'content' in dir() else "")
            if fallback:
                return [{"claim": c} for c in fallback]
            raise ValueError("Failed to parse the model's response as JSON. Try again.")

        except Exception as e:
            logger.error(f"Claim extraction error: {e}")
            raise

    # All size retries exhausted
    raise ValueError(
        f"Request too large even after reducing input size. "
        f"Try uploading a shorter document. (Last error: {last_error})"
    )
