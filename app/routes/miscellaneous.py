"""
Market Trends : yfinance (Yahoo Finance) — free, no API key needed.
IPO           : finapi.upvaly.com/api/ipo — free REST API, no key required.

API contract (from official docs / live responses):
  GET https://finapi.upvaly.com/api/ipo
  Optional query params:
    status  — LIVE | UPCOMING | CLOSED
    type    — Mainboard | SME

  Response shape:
    {
      "status":     "success",
      "statusCode": 200,
      "message":    "IPO details fetched successfully",
      "data": [
        {
          "symbol":     "ADISOFT",
          "name":       "Adisoft Technologies",
          "type":       "SME",
          "priceRange": "₹163 – ₹172",
          "status":     "LIVE",           ← LIVE | UPCOMING | CLOSED
          "schedule": {
            "startDate": "2026-04-23",    ← ISO date (open date)
            "endDate":   "2026-04-27",    ← ISO date (close date)
            ...
          },
          "issueSize": {
            "totalIssueSize": "74.10",    ← crores
            ...
          },
          ...
        },
        ...
      ]
    }

Strategy: fetch all IPOs in one call (no status filter), classify by the
`status` field already present in each record.  Results are cached 10 min.
"""

from __future__ import annotations

import logging
import time
import warnings
from datetime import date, datetime
from typing import Any, Optional

import pandas as pd
import requests
import yfinance as yf
from fastapi import APIRouter, HTTPException

# -- Suppress noisy third-party loggers --
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/misc", tags=["miscellaneous"])

# -- Nifty 50 tickers on Yahoo Finance (.NS suffix = NSE) --
NIFTY_50: list[str] = [
    "RELIANCE.NS",  "TCS.NS",        "HDFCBANK.NS",   "INFY.NS",      "ICICIBANK.NS",
    "HINDUNILVR.NS","ITC.NS",         "SBIN.NS",        "BHARTIARTL.NS","KOTAKBANK.NS",
    "AXISBANK.NS",  "LT.NS",          "BAJFINANCE.NS",  "ASIANPAINT.NS","MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS",       "ULTRACEMCO.NS",  "ONGC.NS",      "WIPRO.NS",
    "NTPC.NS",      "POWERGRID.NS",   "M&M.NS",         "HCLTECH.NS",   "JSWSTEEL.NS",
    "ADANIENT.NS",  "ADANIPORTS.NS",  "BAJAJFINSV.NS",  "COALINDIA.NS", "DIVISLAB.NS",
    "DRREDDY.NS",   "EICHERMOT.NS",   "GRASIM.NS",      "HDFCLIFE.NS",  "HEROMOTOCO.NS",
    "HINDALCO.NS",  "INDUSINDBK.NS",  "LTIM.NS",        "NESTLEIND.NS", "SBILIFE.NS",
    "SHRIRAMFIN.NS","TATACONSUM.NS",  "TATAMOTORS.NS",  "TATASTEEL.NS", "TECHM.NS",
    "TORNTPHARM.NS","TRENT.NS",       "BPCL.NS",        "CIPLA.NS",     "VEDL.NS",
]

# Human-readable display names
_NAMES: dict[str, str] = {
    "RELIANCE":  "Reliance Industries",   "TCS":       "Tata Consultancy Svcs",
    "HDFCBANK":  "HDFC Bank",             "INFY":      "Infosys",
    "ICICIBANK": "ICICI Bank",            "HINDUNILVR":"Hindustan Unilever",
    "ITC":       "ITC Ltd",               "SBIN":      "State Bank of India",
    "BHARTIARTL":"Bharti Airtel",         "KOTAKBANK": "Kotak Mahindra Bank",
    "AXISBANK":  "Axis Bank",             "LT":        "Larsen & Toubro",
    "BAJFINANCE":"Bajaj Finance",         "ASIANPAINT":"Asian Paints",
    "MARUTI":    "Maruti Suzuki",         "SUNPHARMA": "Sun Pharmaceutical",
    "TITAN":     "Titan Company",         "ULTRACEMCO":"UltraTech Cement",
    "ONGC":      "Oil & Natural Gas",     "WIPRO":     "Wipro",
    "NTPC":      "NTPC",                  "POWERGRID": "Power Grid Corp",
    "M&M":       "Mahindra & Mahindra",  "HCLTECH":   "HCL Technologies",
    "JSWSTEEL":  "JSW Steel",             "ADANIENT":  "Adani Enterprises",
    "ADANIPORTS":"Adani Ports",           "BAJAJFINSV":"Bajaj Finserv",
    "COALINDIA": "Coal India",            "DIVISLAB":  "Divi's Laboratories",
    "DRREDDY":   "Dr Reddy's Labs",       "EICHERMOT": "Eicher Motors",
    "GRASIM":    "Grasim Industries",     "HDFCLIFE":  "HDFC Life Insurance",
    "HEROMOTOCO":"Hero MotoCorp",         "HINDALCO":  "Hindalco Industries",
    "INDUSINDBK":"IndusInd Bank",         "LTIM":      "LTIMindtree",
    "NESTLEIND": "Nestlé India",          "SBILIFE":   "SBI Life Insurance",
    "SHRIRAMFIN":"Shriram Finance",       "TATACONSUM":"Tata Consumer Products",
    "TATAMOTORS":"Tata Motors",           "TATASTEEL": "Tata Steel",
    "TECHM":     "Tech Mahindra",         "TORNTPHARM":"Torrent Pharma",
    "TRENT":     "Trent",                "BPCL":      "Bharat Petroleum",
    "CIPLA":     "Cipla",                "VEDL":      "Vedanta",
}


# -- Simple in-memory TTL cache --
class _TTLCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str, ttl: int = 300) -> Any | None:
        entry = self._store.get(key)
        if entry and (time.time() - entry[1]) < ttl:
            return entry[0]
        return None

    def set(self, key: str, val: Any) -> None:
        self._store[key] = (val, time.time())


_cache = _TTLCache()


# -- yfinance download helpers (Market Trends — unchanged) --

def _download(period: str) -> pd.DataFrame:
    """Download OHLCV for all Nifty 50 stocks. Cached to limit Yahoo Finance calls."""
    ttl = 3600 if period == "1y" else 300
    cached = _cache.get(f"yf_{period}", ttl=ttl)
    if cached is not None:
        return cached

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(
            NIFTY_50,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    _cache.set(f"yf_{period}", df)
    return df


def _records_short(df: pd.DataFrame) -> list[dict]:
    """Build records (LTP, % change, volume) from 5d data."""
    records: list[dict] = []
    for sym in NIFTY_50:
        try:
            close  = df[sym]["Close"].dropna()
            volume = df[sym]["Volume"].dropna()
            if len(close) < 2:
                continue
            ltp    = float(close.iloc[-1])
            prev   = float(close.iloc[-2])
            vol    = int(volume.iloc[-1]) if len(volume) else 0
            label  = sym.replace(".NS", "")
            records.append({
                "symbol": label,
                "name":   _NAMES.get(label, label),
                "ltp":    round(ltp, 2),
                "change": round(((ltp - prev) / prev) * 100, 2),
                "volume": vol,
                "index":  "NIFTY 50",
            })
        except Exception:
            continue
    return records


def _records_52w(df: pd.DataFrame) -> list[dict]:
    """Build records with 52-week high/low proximity from 1y data."""
    records: list[dict] = []
    for sym in NIFTY_50:
        try:
            high_s = df[sym]["High"].dropna()
            low_s  = df[sym]["Low"].dropna()
            close  = df[sym]["Close"].dropna()
            vol_s  = df[sym]["Volume"].dropna()
            if len(close) < 5:
                continue
            ltp   = float(close.iloc[-1])
            prev  = float(close.iloc[-2])
            h52   = float(high_s.max())
            l52   = float(low_s.min())
            label = sym.replace(".NS", "")
            records.append({
                "symbol":    label,
                "name":      _NAMES.get(label, label),
                "ltp":       round(ltp, 2),
                "change":    round(((ltp - prev) / prev) * 100, 2),
                "volume":    int(vol_s.iloc[-1]) if len(vol_s) else 0,
                "high52":    round(h52, 2),
                "low52":     round(l52, 2),
                "near_52wh": round(((ltp - h52) / h52) * 100, 2),
                "near_52wl": round(((ltp - l52) / l52) * 100, 2),
                "index":     "NIFTY 50",
            })
        except Exception:
            continue
    return records


# -- finapi.upvaly.com IPO helpers --

_FINAPI_IPO_URL = "https://finapi.upvaly.com/api/ipo"
_FINAPI_HEADERS = {
    "Accept": "application/json",
}

# Maps the API's `status` field values → our internal bucket names
_STATUS_MAP: dict[str, str] = {
    "LIVE":     "open",
    "UPCOMING": "upcoming",
    "CLOSED":   "closed",
}


def _normalise_finapi_record(raw: dict) -> dict:
    """
    Map a single finapi.upvaly.com record to the internal schema expected by
    the frontend's renderIpoTable().

    Includes both the minimal flat fields (for backward compatibility) and the
    full nested data for the expanded detail view.
    """
    schedule   = raw.get("schedule")   or {}
    issue_size = raw.get("issueSize")  or {}
    gmp        = raw.get("greyMarketPremium") or {}

    # price_band: use priceRange; fall back to "—" when missing / bare dash
    price_band = str(raw.get("priceRange") or "—").strip()
    if price_band in ("", "-", "–", "—"):
        price_band = "—"

    # issue_size summary: totalIssueSize is in crores (e.g. "74.10")
    total_size = str(issue_size.get("totalIssueSize") or "—").strip()
    if total_size not in ("", "None", "null"):
        issue_size_str = f"₹{total_size} Cr"
    else:
        issue_size_str = "—"

    # Build clean schedule dict (only non-null values)
    schedule_clean: dict = {}
    _schedule_labels = {
        "startDate":                  "Open Date",
        "endDate":                    "Close Date",
        "listingDate":                "Listing Date",
        "upiMandateDeadline":         "UPI Mandate Deadline",
        "allotmentFinalization":      "Allotment Finalisation",
        "refundInitiation":           "Refund Initiation",
        "shareCredit":                "Share Credit",
        "mandateEndDate":             "Mandate End Date",
        "lockInEndDateAnchor50":      "Lock-in End (Anchor 50%)",
        "lockInEndDateAnchorRemaining": "Lock-in End (Anchor Remaining)",
    }
    for key, label in _schedule_labels.items():
        val = schedule.get(key)
        if val:
            schedule_clean[label] = str(val)

    # Build clean issue size dict
    issue_size_clean: dict = {}
    _issue_labels = {
        "totalIssueSize": "Total Issue Size",
        "freshIssue":     "Fresh Issue",
        "offerForSale":   "Offer for Sale",
    }
    for key, label in _issue_labels.items():
        val = issue_size.get(key)
        if val is not None:
            issue_size_clean[label] = f"₹{val} Cr" if val else "—"

    # GMP trends list  (may be absent for UPCOMING)
    gmp_trends = gmp.get("gmpTrends") or []

    # Subscription numbers (may be absent for UPCOMING)
    sub_numbers = raw.get("subscriptionNumbers") or {}

    return {
        # ── Flat / legacy fields (backward-compat) ──
        "name":        str(raw.get("name")    or "—").strip(),
        "symbol":      str(raw.get("symbol")  or "").strip(),
        "ipo_type":    str(raw.get("type")    or "").strip(),
        "logo_url":    str(raw.get("logoUrl") or "").strip(),
        "price_band":  price_band,
        "open_date":   str(schedule.get("startDate")  or "—").strip(),
        "close_date":  str(schedule.get("endDate")    or "—").strip(),
        "lot_size":    "—",            # not provided by finapi.upvaly.com
        "issue_size":  issue_size_str,
        # ── Rich detail fields ──
        "schedule":            schedule_clean,
        "issue_size_detail":   issue_size_clean,
        "about_company":       str(raw.get("aboutCompany") or "").strip(),
        "gmp_trends":          gmp_trends,
        "subscription":        sub_numbers,
        "strengths":           raw.get("strengths") or [],
        "risks":               raw.get("risks")     or [],
        # Internal — used for bucket classification, removed before returning
        "_api_status": str(raw.get("status") or "").upper(),
    }


def _fetch_all_ipos() -> list[dict]:
    """
    Fetch all IPOs from finapi.upvaly.com/api/ipo in a single call (no status
    filter) so all three buckets are populated from one network round-trip.

    Raises requests.RequestException on network failure or RuntimeError if the
    response shape is unexpected.
    """
    resp = requests.get(_FINAPI_IPO_URL, headers=_FINAPI_HEADERS, timeout=15)
    resp.raise_for_status()

    payload = resp.json()

    # Verify the API-level status
    if payload.get("status") != "success":
        raise RuntimeError(
            f"finapi.upvaly.com returned non-success status: "
            f"{payload.get('status')} — {payload.get('message', '')}"
        )

    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError(
            f"finapi.upvaly.com response missing 'data' list. "
            f"Keys present: {list(payload.keys())}"
        )

    logger.info("finapi.upvaly.com /api/ipo returned %d records", len(data))
    return data


# -- Endpoints --

@router.get("/market-trends/{category}")
async def get_market_trends(category: str):
    """
    Live Nifty 50 market trend data via Yahoo Finance (yfinance).

    category options:
        gainers  — top gainers by % change
        losers   — top losers  by % change
        volume   — top stocks  by traded volume
        52w_high — stocks nearest to 52-week high
        52w_low  — stocks nearest to 52-week low
    """
    valid = {"gainers", "losers", "volume", "52w_high", "52w_low"}
    if category not in valid:
        raise HTTPException(400, detail=f"Invalid category. Valid: {sorted(valid)}")

    try:
        if category in ("52w_high", "52w_low"):
            df      = _download("1y")
            records = _records_52w(df)
            if category == "52w_high":
                records.sort(key=lambda x: x["near_52wh"], reverse=True)
            else:
                records.sort(key=lambda x: x["near_52wl"])
        else:
            df      = _download("5d")
            records = _records_short(df)
            if category == "gainers":
                records.sort(key=lambda x: x["change"], reverse=True)
            elif category == "losers":
                records.sort(key=lambda x: x["change"])
            elif category == "volume":
                records.sort(key=lambda x: x["volume"], reverse=True)

        return {"category": category, "data": records[:15]}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("market-trends [%s] error: %s", category, exc)
        raise HTTPException(502, detail=f"Failed to fetch market data: {exc}")


@router.get("/ipo")
async def get_ipo():
    """
    IPO data from finapi.upvaly.com — open / upcoming / closed buckets.
    Cached for 10 minutes.

    Response schema (frontend-compatible, unchanged from previous version):
        {
            "open":        [ { name, symbol, open_date, close_date,
                               price_band, lot_size, issue_size }, … ],
            "upcoming":    [ … ],
            "closed":      [ … ],   ← max 15 entries
            "unavailable": false
        }

    On any failure returns the same shape with unavailable:true and empty
    lists so Market Trends keeps working normally.
    """
    cached = _cache.get("ipo_data", ttl=600)
    if cached is not None:
        return cached

    # -- Fetch from finapi.upvaly.com/api/ipo --
    raw_list: list[dict] = []
    try:
        raw_list = _fetch_all_ipos()
    except Exception as exc:
        logger.error("finapi.upvaly.com IPO fetch failed: %s", exc)

    # -- Graceful empty — market trends keeps working even if IPO fails --
    if not raw_list:
        logger.error("IPO: finapi.upvaly.com returned no records. Serving empty buckets.")
        empty: dict = {"open": [], "closed": [], "upcoming": [], "unavailable": True}
        _cache.set("ipo_data", empty)
        return empty

    # -- Normalise & classify into open / upcoming / closed --
    open_ipos:     list[dict] = []
    closed_ipos:   list[dict] = []
    upcoming_ipos: list[dict] = []

    for raw in raw_list:
        rec        = _normalise_finapi_record(raw)
        api_status = rec.pop("_api_status", "")   # strip internal tag

        bucket = _STATUS_MAP.get(api_status)

        if bucket == "open":
            open_ipos.append(rec)
        elif bucket == "upcoming":
            upcoming_ipos.append(rec)
        elif bucket == "closed":
            closed_ipos.append(rec)
        else:
            # Unknown status — log and skip rather than silently misclassify
            logger.debug(
                "finapi IPO: unrecognised status '%s' for symbol '%s' — skipped",
                api_status, rec.get("symbol", "?"),
            )

    result = {
        "open":        open_ipos,
        "closed":      closed_ipos[:15],
        "upcoming":    upcoming_ipos,
        "unavailable": False,
    }
    _cache.set("ipo_data", result)
    return result