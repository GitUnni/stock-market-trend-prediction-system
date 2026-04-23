"""
Market News Router
──────────────────
Data source : Marketaux API  (https://www.marketaux.com — 100 req/day free)
AI layer    : LangChain + Groq (trend detection, market-mood summary)

Add to your .env:
    MARKETAUX_API_KEY=<your_key>
    GROQ_API_KEY=<your_key>        # already present from research.py

Debug endpoint (does NOT count toward your Marketaux quota):
    GET /news/debug  →  shows raw API response + any errors
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests as req
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv("app/.env")
MARKETAUX_KEY = os.getenv("MARKETAUX_API_KEY")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")

router = APIRouter(prefix="/news", tags=["News"])
logger = logging.getLogger(__name__)

MARKETAUX_URL = "https://api.marketaux.com/v1/news/all"


# ── Schemas ───────────────────────────────────────────────────────────────────

class Article(BaseModel):
    title:           str
    url:             str
    published_at:    str
    source:          str
    description:     str
    sentiment:       str          # positive | negative | neutral
    sentiment_score: float
    image_url:       str | None
    is_global:       bool


class NewsResponse(BaseModel):
    articles:      list[Article]
    trend_summary: str
    market_mood:   str            # bullish | bearish | cautious
    mood_emoji:    str
    fetched_at:    str


# ── Marketaux helpers ─────────────────────────────────────────────────────────

def _fetch_marketaux(params: dict, label: str = "") -> list[dict]:
    """
    Raw Marketaux call. Raises RuntimeError with a clear message on failure
    so callers can surface the problem rather than silently returning [].
    """
    if not MARKETAUX_KEY:
        raise RuntimeError(
            "MARKETAUX_API_KEY is not set in your .env file. "
            "Get a free key at https://www.marketaux.com and add "
            "MARKETAUX_API_KEY=your_key to app/.env"
        )

    try:
        resp = req.get(
            MARKETAUX_URL,
            params={"api_token": MARKETAUX_KEY, "language": "en", **params},
            timeout=15,
        )
    except req.exceptions.Timeout:
        raise RuntimeError(f"Marketaux request timed out ({label})")
    except req.exceptions.ConnectionError:
        raise RuntimeError(f"Could not connect to Marketaux API ({label}). Check your internet connection.")

    # Surface HTTP errors with the actual Marketaux error message
    if not resp.ok:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:300]
        raise RuntimeError(
            f"Marketaux API error {resp.status_code} ({label}): {detail}"
        )

    data = resp.json()
    articles = data.get("data", [])
    logger.info(f"Marketaux [{label}]: fetched {len(articles)} articles")
    return articles


def _parse_article(raw: dict, is_global: bool) -> Article | None:
    """Convert a Marketaux article dict into our Article model."""
    title = (raw.get("title") or "").strip()
    url   = (raw.get("url")   or "").strip()
    if not title or not url:
        return None

    # Sentiment: average entity-level scores provided by Marketaux
    entities = raw.get("entities") or []
    scores = [
        e["sentiment_score"]
        for e in entities
        if isinstance(e, dict) and e.get("sentiment_score") is not None
    ]
    sentiment_score = round(sum(scores) / len(scores), 3) if scores else 0.0

    if sentiment_score > 0.1:
        sentiment = "positive"
    elif sentiment_score < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Marketaux returns source as a plain string (domain), not a dict
    raw_source = raw.get("source") or ""
    source_name = (
        raw_source.get("name", "") if isinstance(raw_source, dict)
        else str(raw_source)
    )

    description = (
        raw.get("description") or raw.get("snippet") or raw.get("summary") or ""
    )[:300]

    return Article(
        title=title,
        url=url,
        published_at=raw.get("published_at") or "",
        source=source_name,
        description=description,
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        image_url=raw.get("image_url") or None,
        is_global=is_global,
    )


def fetch_all_news(limit_per_batch: int = 10) -> list[Article]:
    """
    Two Marketaux calls:
      1. Indian market news  (countries=in)
      2. Global macro news that moves Indian markets
    Deduplicates by URL, sorts newest-first.
    Both batches are hard-limited to the last 7 days via published_after.
    """
    from datetime import timedelta
    # Hard cutoff — no article older than 7 days will be returned
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")

    errors = []

    # Batch 1 — Indian news
    try:
        indian_raw = _fetch_marketaux(
            {
                "countries":       "in",
                "filter_entities": "true",
                "limit":           limit_per_batch,
                "sort":            "published_desc",
                "published_after": cutoff,
            },
            label="India",
        )
    except RuntimeError as e:
        logger.warning(f"India batch failed: {e}")
        errors.append(str(e))
        indian_raw = []

    # Batch 2 — Global macro
    try:
        global_raw = _fetch_marketaux(
            {
                "search": (
                    "oil price OR US Federal Reserve OR China economy "
                    "OR geopolitical OR war sanctions"
                ),
                "limit":           limit_per_batch,
                "sort":            "published_desc",
                "published_after": cutoff,
            },
            label="Global",
        )
    except RuntimeError as e:
        logger.warning(f"Global batch failed: {e}")
        errors.append(str(e))
        global_raw = []

    # If BOTH batches failed, surface the error
    if not indian_raw and not global_raw and errors:
        raise RuntimeError(errors[0])

    seen_urls: set[str] = set()
    articles:  list[Article] = []

    for raw, is_global in (
        [(r, False) for r in indian_raw] + [(r, True) for r in global_raw]
    ):
        art = _parse_article(raw, is_global)
        if art and art.url not in seen_urls:
            seen_urls.add(art.url)
            articles.append(art)

    def _pub_key(a: Article) -> datetime:
        try:
            return datetime.fromisoformat(a.published_at.replace("Z", "+00:00"))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    articles.sort(key=_pub_key, reverse=True)
    logger.info(f"Total deduplicated articles: {len(articles)}")
    return articles


# ── LangChain trend analysis ──────────────────────────────────────────────────

def analyse_trends(articles: list[Article]) -> tuple[str, str, str]:
    """
    LangChain + Groq → (trend_summary, market_mood, mood_emoji)
    Falls back to sentiment-counting heuristic if LLM call fails.
    """
    headlines = "\n".join(
        f"- [{'GLOBAL' if a.is_global else 'INDIA'}] [{a.sentiment.upper()}] {a.title}"
        for a in articles[:15]
    )
    today = datetime.now().strftime("%B %d, %Y")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a senior equity strategist specialising in Indian financial markets. "
            "Synthesise recent headlines into a crisp market narrative for institutional investors. "
            "Be factual, concise, and avoid speculation.",
        ),
        (
            "human",
            f"Today is {today}. Headlines below (INDIA = direct, GLOBAL = macro affecting India):\n\n"
            f"{headlines}\n\n"
            f"Respond in EXACTLY this format (no extra text):\n"
            f"MOOD: <bullish|bearish|cautious>\n"
            f"SUMMARY: <3-5 sentences: dominant trend, key global risk/tailwind, "
            f"sectors to watch, investor stance>",
        ),
    ])

    mood    = "cautious"
    summary = ""

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=400,
            groq_api_key=GROQ_API_KEY,
        )
        result = (prompt | llm).invoke({}).content.strip()
        logger.info(f"LLM raw response: {result[:200]}")

        for line in result.splitlines():
            upper = line.upper()
            if upper.startswith("MOOD:"):
                raw_mood = line.split(":", 1)[1].strip().lower()
                if raw_mood in ("bullish", "bearish", "cautious"):
                    mood = raw_mood
            elif upper.startswith("SUMMARY:"):
                summary = line.split(":", 1)[1].strip()

        if not summary:
            summary = result  # use full response if parsing failed

    except Exception as e:
        logger.warning(f"LLM trend analysis failed, using heuristic: {e}")
        pos = sum(1 for a in articles if a.sentiment == "positive")
        neg = sum(1 for a in articles if a.sentiment == "negative")
        if pos > neg + 2:
            mood    = "bullish"
            summary = "Market sentiment is broadly positive based on recent headlines."
        elif neg > pos + 2:
            mood    = "bearish"
            summary = "Market sentiment is broadly negative based on recent headlines."
        else:
            mood    = "cautious"
            summary = "Mixed signals in the market. Investors should tread carefully."

    emoji_map = {"bullish": "📈", "bearish": "📉", "cautious": "⚠️"}
    return summary, mood, emoji_map.get(mood, "📊")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/feed", response_model=NewsResponse)
async def get_news_feed():
    """Main news feed — consumes 2 Marketaux API calls per request."""
    try:
        articles = fetch_all_news(limit_per_batch=10)

        if not articles:
            return NewsResponse(
                articles=[],
                trend_summary="No articles were returned by Marketaux. Your API key may be invalid or the quota may be exhausted.",
                market_mood="cautious",
                mood_emoji="⚠️",
                fetched_at=datetime.now().isoformat(),
            )

        trend_summary, market_mood, mood_emoji = analyse_trends(articles)

        return NewsResponse(
            articles=articles,
            trend_summary=trend_summary,
            market_mood=market_mood,
            mood_emoji=mood_emoji,
            fetched_at=datetime.now().isoformat(),
        )

    except RuntimeError as e:
        # Surface Marketaux config/auth errors clearly
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in news feed")
        raise HTTPException(status_code=500, detail=f"News feed error: {str(e)}")


@router.get("/debug")
async def debug_news():
    """
    Diagnostic endpoint — does NOT trigger LLM analysis.
    Use this to check your API key and see raw Marketaux output.
    Visit: GET /news/debug
    """
    result = {
        "marketaux_key_set": bool(MARKETAUX_KEY),
        "groq_key_set":      bool(GROQ_API_KEY),
        "india_batch":       {"ok": False, "count": 0, "error": None, "sample": None},
        "global_batch":      {"ok": False, "count": 0, "error": None, "sample": None},
    }

    # Test India batch
    try:
        india = _fetch_marketaux(
            {"countries": "in", "filter_entities": "true", "limit": 3, "sort": "published_desc"},
            label="debug-india",
        )
        result["india_batch"] = {
            "ok":     True,
            "count":  len(india),
            "error":  None,
            "sample": india[0] if india else None,   # show first raw article
        }
    except RuntimeError as e:
        result["india_batch"]["error"] = str(e)

    # Test Global batch
    try:
        glob = _fetch_marketaux(
            {"search": "oil price OR US Federal Reserve", "limit": 3, "sort": "published_desc"},
            label="debug-global",
        )
        result["global_batch"] = {
            "ok":     True,
            "count":  len(glob),
            "error":  None,
            "sample": glob[0] if glob else None,
        }
    except RuntimeError as e:
        result["global_batch"]["error"] = str(e)

    return result