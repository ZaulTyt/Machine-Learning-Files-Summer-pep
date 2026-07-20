import time
import random
import logging
from ddgs import DDGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_last_search_time: float = 0.0


def search_claim(claim: str, max_results: int = 5) -> list:
    """
    Searches DuckDuckGo (via the ddgs package) for the given claim.

    Applies a global rate-limit guard and exponential back-off on retries.

    Args:
        claim: The factual claim to search for.
        max_results: Maximum number of search results to retrieve.

    Returns:
        List of dicts with keys: 'title', 'snippet', 'url'.
    """
    global _last_search_time

    query = claim.strip().strip('."\'?!,;:-')
    if len(query) > 200:
        query = query[:200]

    logger.info(f"Searching: '{query[:80]}'")

    results = []

    for attempt in range(3):
        # ── Minimum gap between searches (avoid rate-limiting) ───────────
        now = time.time()
        min_gap = random.uniform(2.0, 3.5)
        elapsed = now - _last_search_time
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)

        if attempt > 0:
            backoff = (2 ** attempt) + random.uniform(0.5, 1.5)
            logger.info(f"Retry {attempt + 1}/3 — back-off {backoff:.1f}s")
            time.sleep(backoff)

        try:
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(query, max_results=max_results))

            _last_search_time = time.time()

            for r in ddg_results:
                results.append({
                    "title":   r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url":     r.get("href", ""),
                })

            if results:
                logger.info(f"Got {len(results)} results.")
                return results

            logger.warning(f"Empty results on attempt {attempt + 1}.")
            _last_search_time = time.time()

        except Exception as e:
            _last_search_time = time.time()
            logger.warning(f"Search error (attempt {attempt + 1}/3): {type(e).__name__}: {e}")

    logger.error(f"All search attempts failed for: '{query[:80]}'")
    return []
