import json
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"

# ── Groq-native models (pass full ID without fallback) ──────────────────────
GROQ_NATIVE_MODELS = {
    "meta-llama/llama-4-scout-17b-16e-instruct",
}


def _safe_groq_model(model: str) -> str:
    """Return a Groq-compatible model ID (strips OpenRouter-style suffixes)."""
    if ":" in model:
        return "llama-3.3-70b-versatile"
    if "/" in model and model not in GROQ_NATIVE_MODELS:
        return "llama-3.3-70b-versatile"
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Groq compound-beta  →  web search is built-in, no DuckDuckGo needed
# ─────────────────────────────────────────────────────────────────────────────
def verify_claim_compound(claim: str, api_key: str) -> dict:
    """
    Uses Groq compound-beta to search the web AND verify the claim in one call.
    Returns the same dict shape as verify_claim(), plus an 'evidence' key.
    """
    system_prompt = (
        "You are a fact-checking engine with real-time web search access.\n"
        "Search the web to find current evidence about the claim, then classify it.\n\n"
        "CLASSIFICATION RULES:\n"
        "- 'Verified'   : claim is factually correct and supported by evidence.\n"
        "- 'Inaccurate' : claim is partially correct but contains a clear error.\n"
        "- 'False'      : claim directly contradicts reliable evidence.\n\n"
        "OUTPUT — respond ONLY with valid JSON, no markdown, no prose:\n"
        "{\n"
        "  \"status\": \"Verified\" | \"Inaccurate\" | \"False\",\n"
        "  \"confidence\": <integer 0-100>,\n"
        "  \"reason\": \"<1-2 sentences citing your findings>\",\n"
        "  \"correct_fact\": \"<corrected fact if Inaccurate/False, else empty string>\",\n"
        "  \"sources\": [\"<url1>\", \"<url2>\"]\n"
        "}"
    )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "compound-beta",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f'Fact-check this claim using web search: "{claim}"'},
        ],
        "temperature": 0.1,
    }

    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"].strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content
            content = content.rstrip("`").strip()
            if "```" in content:
                content = content.split("```")[0].strip()

        verification = json.loads(content)

        status = verification.get("status", "Needs Review")
        if status not in ("Verified", "Inaccurate", "False"):
            status = "Needs Review"

        confidence = 50
        try:
            confidence = int(verification.get("confidence", 50))
        except (TypeError, ValueError):
            pass

        sources = verification.get("sources", [])
        evidence = [
            {"title": f"Source {i+1}", "snippet": "", "url": src}
            for i, src in enumerate(sources) if isinstance(src, str) and src.startswith("http")
        ]

        return {
            "status":       status,
            "confidence":   confidence,
            "reason":       verification.get("reason", ""),
            "correct_fact": verification.get("correct_fact", ""),
            "evidence":     evidence,
        }

    except requests.exceptions.HTTPError as he:
        try:
            err_msg = he.response.json().get("error", {}).get("message", he.response.text)
        except Exception:
            err_msg = he.response.text
        logger.error(f"compound-beta HTTP error: {err_msg}")
        return {
            "status": "Needs Review", "confidence": 0,
            "reason": f"API Error {he.response.status_code}: {err_msg}",
            "correct_fact": "", "evidence": [],
        }
    except json.JSONDecodeError as je:
        logger.error(f"JSON parse error from compound-beta: {je}")
        return {
            "status": "Needs Review", "confidence": 0,
            "reason": "Failed to parse compound-beta response as JSON.",
            "correct_fact": "", "evidence": [],
        }
    except Exception as e:
        logger.error(f"compound-beta error: {e}")
        return {
            "status": "Needs Review", "confidence": 0,
            "reason": f"Verification error: {e}",
            "correct_fact": "", "evidence": [],
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Standard path  →  evidence already retrieved by web_search.py
# ─────────────────────────────────────────────────────────────────────────────
def verify_claim(claim: str, evidence: list, api_key: str,
                 model: str = "qwen/qwen-2.5-coder-32b-instruct:free") -> dict:
    """
    Compares the claim against the collected web evidence and determines factual accuracy.
    Used for OpenRouter keys (evidence pre-fetched via DuckDuckGo).
    """
    if not evidence:
        return {
            "status": "Needs Review", "confidence": 0,
            "reason": "No search results found on the live web to verify this claim.",
            "correct_fact": "",
        }

    if not api_key:
        raise ValueError("API key is required to verify claims.")

    # Build evidence block
    evidence_block = "\n".join(
        f"Source [{i+1}]:\nTitle: {e.get('title','')}\n"
        f"URL: {e.get('url','')}\nSnippet: {e.get('snippet','')}\n"
        for i, e in enumerate(evidence)
    )

    system_prompt = (
        "You are an expert fact-checking engine.\n"
        "Verify a factual claim against the provided search evidence.\n\n"
        "GUIDELINES:\n"
        "1. Do NOT mark a claim as False if it is a valid generalization of the evidence.\n"
        "2. Base classification strictly on the provided evidence, not prior knowledge.\n"
        "3. Rounded numbers are acceptable — treat as Verified.\n\n"
        "STATUS:\n"
        "- 'Verified'  : factually correct per evidence.\n"
        "- 'Inaccurate': partially correct but contains a clear error.\n"
        "- 'False'     : directly contradicts evidence.\n\n"
        "OUTPUT — ONLY valid JSON, no markdown:\n"
        "{\n"
        "  \"status\": \"Verified\" | \"Inaccurate\" | \"False\",\n"
        "  \"confidence\": <integer 0-100>,\n"
        "  \"reason\": \"<1-2 sentence explanation citing sources>\",\n"
        "  \"correct_fact\": \"<correction if needed, else empty string>\"\n"
        "}"
    )

    is_groq = api_key.startswith("gsk_")
    url     = GROQ_URL if is_groq else OPENROUTER_URL
    if is_groq:
        model = _safe_groq_model(model)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if not is_groq:
        headers["HTTP-Referer"] = "https://jetro.ai"
        headers["X-Title"]      = "Fact-Check Agent"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f'Claim: "{claim}"\n\nSearch Evidence:\n{evidence_block}'},
        ],
        "temperature": 0.1,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"].strip()

        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content
            if content.endswith("```"):
                content = content[:-3].strip()
            elif "```" in content:
                content = content.split("```")[0].strip()

        verification = json.loads(content)

        status = verification.get("status", "Needs Review")
        if status not in ("Verified", "Inaccurate", "False"):
            status = "Needs Review"

        confidence = 50
        try:
            confidence = int(verification.get("confidence", 50))
        except (TypeError, ValueError):
            pass

        return {
            "status":       status,
            "confidence":   confidence,
            "reason":       verification.get("reason", ""),
            "correct_fact": verification.get("correct_fact", ""),
        }

    except requests.exceptions.HTTPError as he:
        try:
            err_msg = he.response.json().get("error", {}).get("message", he.response.text)
        except Exception:
            err_msg = he.response.text
        logger.error(f"HTTP error during verification: {err_msg}")
        return {
            "status": "Needs Review", "confidence": 0,
            "reason": f"API Error {he.response.status_code}: {err_msg}",
            "correct_fact": "",
        }
    except json.JSONDecodeError as je:
        logger.error(f"JSON decode error: {je}")
        return {
            "status": "Needs Review", "confidence": 0,
            "reason": "Failed to parse the fact-checking engine's response.",
            "correct_fact": "",
        }
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return {
            "status": "Needs Review", "confidence": 0,
            "reason": f"System error: {e}",
            "correct_fact": "",
        }
