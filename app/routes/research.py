"""
Equity Research Agent
─────────────────────
Search backend  : Tavily (primary) → DuckDuckGo (fallback)
Query strategy  : LLM reformulates the user question into focused queries
Recency         : Tavily `days` filter + date-aware article ranking
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
import requests as req
import os
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv("app/.env")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

router = APIRouter(prefix="/research", tags=["Research"])

TODAY        = datetime.now()
CURRENT_YEAR = TODAY.year
TODAY_STR    = TODAY.strftime("%B %d, %Y")


# Schemas

class ResearchQuery(BaseModel):
    question: str

class ResearchResponse(BaseModel):
    answer: str
    sources: list[dict]


# LLM helpers

def call_groq(messages: list[dict], max_tokens: int = 1024) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    resp = req.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def reformulate_queries(question: str) -> list[str]:
    """
    Use Groq to turn the user's natural-language question into 3 tight,
    search-engine-friendly queries optimised for recent Indian financial data.
    """
    prompt = f"""Today is {TODAY_STR}.

Convert the following user question into exactly 3 short search queries
(4–8 words each) that will find precise, recent results on a search engine.

Rules:
- Include "{CURRENT_YEAR}" in at least 2 of the queries.
- Use financial keywords: "market cap", "market capitalisation", "NSE", "BSE".
- For "most valuable company" questions, search for market capitalisation rankings.
- Output one query per line. No numbering, no explanation, nothing else.

User question: {question}"""

    raw = call_groq([{"role": "user", "content": prompt}], max_tokens=120)
    queries = [q.strip() for q in raw.strip().splitlines() if q.strip()]
    if not queries:
        queries = [f"{question} India {CURRENT_YEAR}"]
    return queries[:3]


# Search: Tavily (primary)

def tavily_search(query: str, max_results: int = 5, days: int = 30) -> list[dict]:
    """
    Tavily Search API — purpose-built for AI agents, reliable recency filter.
    Get a free key at https://tavily.com (1,000 searches/month free).
    """
    if not TAVILY_API_KEY:
        return []
    try:
        payload = {
            "api_key":        TAVILY_API_KEY,
            "query":          query,
            "search_depth":   "advanced",
            "max_results":    max_results,
            "days":           days,
            "include_answer": False,
        }
        resp = req.post("https://api.tavily.com/search", json=payload, timeout=15)
        resp.raise_for_status()
        return [
            {
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "snippet": r.get("content", r.get("snippet", "")),
            }
            for r in resp.json().get("results", [])
        ]
    except Exception:
        return []


# Search: DuckDuckGo (fallback)

def duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results, timelimit="m"))
        except Exception:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))

        return [
            {
                "title":   r.get("title", ""),
                "url":     r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]
    except Exception:
        return []


# Unified search orchestrator

def search_financial_news(queries: list[str]) -> tuple[str, list[dict]]:
    """
    Run all reformulated queries. Try Tavily first, fall back to DDG.
    Deduplicate by URL across all queries.
    """
    all_results: list[dict] = []
    seen_urls:   set[str]   = set()

    for query in queries:
        # Pass 1: Tavily, last 30 days
        results = tavily_search(query, max_results=5, days=30)

        # Pass 2: Tavily, last 90 days
        if len(results) < 2:
            results = tavily_search(query, max_results=5, days=90)

        # Pass 3: DDG fallback
        if len(results) < 2:
            results = duckduckgo_search(f"{query} India", max_results=5)

        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                domain = re.sub(r"https?://(www\.)?", "", url).split("/")[0]
                all_results.append({**r, "source": domain})

    if not all_results:
        return "No search results found.", []

    lines = [
        f"[{i}] {r['title']}\n"
        f"    Source : {r['source']}\n"
        f"    URL    : {r['url']}\n"
        f"    Snippet: {r['snippet']}\n"
        for i, r in enumerate(all_results, 1)
    ]
    return "\n".join(lines), all_results


# Article scraping

def _parse_pub_date(soup: BeautifulSoup) -> datetime | None:
    candidates = []
    for meta in soup.find_all("meta"):
        prop = (meta.get("property", "") + meta.get("name", "")).lower()
        if any(k in prop for k in ["published", "date", "created", "modified"]):
            content = meta.get("content", "")
            if content:
                candidates.append(content)
    for tag in soup.find_all("time"):
        dt = tag.get("datetime", "")
        if dt:
            candidates.append(dt)

    for raw in candidates:
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d",
            "%B %d, %Y", "%d %B %Y",
        ):
            try:
                cleaned = re.sub(r"[+-]\d{2}:\d{2}$", "", raw.strip())
                return datetime.strptime(cleaned, fmt)
            except ValueError:
                continue
    return None


def scrape_article(url: str) -> tuple[str, float]:
    """Returns (article_text, age_in_days). age=inf on failure."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    try:
        resp = req.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return f"Could not fetch article: {e}", float("inf")

    soup = BeautifulSoup(resp.text, "lxml")
    pub  = _parse_pub_date(soup)
    age  = (TODAY - pub.replace(tzinfo=None)).days if pub else float("inf")
    pub_str = pub.strftime("%B %d, %Y") if pub else "Unknown date"

    for tag in soup(["script", "style", "nav", "footer",
                     "header", "aside", "form", "iframe"]):
        tag.decompose()

    article_tag = (
        soup.find("article")
        or soup.find(class_=lambda c: c and "article" in c.lower())
        or soup.find("main")
        or soup.body
    )
    paragraphs = article_tag.find_all("p") if article_tag else soup.find_all("p")
    text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
    text = " ".join(text.split())

    if not text:
        return "Could not extract article text.", float("inf")
    if len(text) > 3000:
        text = text[:3000] + "... [truncated]"

    return f"Publication date: {pub_str}\n\nArticle content:\n{text}", age


# Main agent

def run_equity_research_agent(question: str) -> tuple[str, list[dict]]:

    system_prompt = f"""You are an expert equity research analyst specialising \
in Indian financial markets. Today is {TODAY_STR}.

You will be given:
  • A user question about Indian stocks or financials.
  • A list of search-result snippets from trusted sources.
  • (Optionally) the full text of one scraped article.

YOUR TASK:
1. Read ALL the provided content carefully.
2. Extract the specific figure, metric, or fact the user asked for.
3. Give a direct, factual answer (2–4 sentences) that includes:
   - The exact number/value with units (e.g. ₹19.88 lakh crore).
   - The date or time period it covers.
4. End with a "Sources:" line listing the URLs you used.

RECENCY RULES (CRITICAL):
- Today is {TODAY_STR}. ALWAYS use the most recent data point available.
- If multiple figures exist across different dates, choose the LATEST one and
  state its date explicitly.
- If the most recent data is older than 6 months, flag it clearly:
  "Note: most recent data found is from [date]; more current figures may exist."
- NEVER present a 2023 or 2024 figure as current for a question about 2026.

STRICT RULES:
- Only use figures that appear verbatim in the provided content.
- Never fabricate numbers.
- If no relevant figures appear anywhere, say so and suggest a rephrased query."""

    # Step 1: Reformulate question into focused search queries
    queries = reformulate_queries(question)

    # Step 2: Search
    search_text, raw_results = search_financial_news(queries)

    # Step 3: Scrape the freshest useful article
    candidates: list[tuple[float, str, str]] = []
    for r in raw_results[:6]:
        url = r.get("url", "")
        if not url.startswith("http"):
            continue
        text, age = scrape_article(url)
        if "Could not" not in text and len(text) > 300:
            candidates.append((age, text, url))

    article_text = ""
    used_url     = ""
    if candidates:
        candidates.sort(key=lambda x: x[0])   # youngest first
        _, article_text, used_url = candidates[0]

    # Step 4: Build answer prompt
    user_content = (
        f"Question: {question}\n"
        f"Today's date: {TODAY_STR}\n"
        f"Search queries used: {' | '.join(queries)}\n\n"
        f"=== Search Results ===\n{search_text}\n\n"
    )
    if article_text:
        user_content += (
            f"=== Full Article ({used_url}) ===\n{article_text}\n\n"
        )
    user_content += (
        f"Using ONLY the content above, give the most recent and specific answer. "
        f"Today is {TODAY_STR}. Flag any data older than 6 months."
    )

    answer = call_groq(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]
    )

    # Deduplicate sources
    sources: list[dict] = []
    seen: set[str] = set()
    for r in raw_results:
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            sources.append({"title": r.get("title", url), "url": url})

    return answer, sources


# FastAPI endpoint

@router.post("/query", response_model=ResearchResponse)
async def equity_research_query(body: ResearchQuery):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        answer, sources = run_equity_research_agent(body.question.strip())
        return ResearchResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Research agent error: {str(e)}"
        )