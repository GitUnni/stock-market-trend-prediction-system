"""
Route: /api/predict

Background-job architecture — prevents server timeouts on the 2-5 minute
ML pipeline (Prophet × 2 + XGBoost + LSTM + Ensemble).

Flow
────
  POST /api/predict/{symbol}?target_date=YYYY-MM-DD
      → validates input, enqueues job, returns { job_id, status:"queued" }

  GET  /api/predict/status/{job_id}
      → returns { job_id, status, progress_pct, progress_msg, result? }
        status: "queued" | "running" | "done" | "error"

  DELETE /api/predict/cancel/{job_id}
      → cancels a queued/running job

Full 4-model pipeline (mirrors the notebook exactly):
  1. Prophet   — macro trend + price forecast with 95 % CI
  2. XGBoost   — hierarchical entry-timing classifier
  3. LSTM      — sequence-memory entry-timing classifier (30-day lookback)
  4. Ensemble  — weighted average XGBoost(0.60) + LSTM(0.40),
                 gated by Prophet trend direction
                 STRONG = both agree · NORMAL = weighted avg exceeds threshold

Holiday calendar: pandas_market_calendars (NSE) — auto-detected with
                  hardcoded 2023-2026 fallback if library not installed.

FIX: simulate_volatile_forecast() — Prophet's raw yhat is a smooth
     expectation curve (trend + seasonality only).  After generating the
     forecast we inject AR(1)-modelled residual noise so the chart shows
     realistic price oscillations instead of a flat line.
"""

import warnings
warnings.filterwarnings("ignore")

import logging
import threading
import uuid
import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from upstash_redis import Redis

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, date
from typing import Optional
from urllib.parse import urlparse

from app.deps import get_db
from app import models

router = APIRouter(prefix="/api/predict", tags=["Prediction"])
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Redis-backed job store
# ─────────────────────────────────────────────────────────────────────
def _clean_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value.strip() if value else None


def _is_valid_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


redis_url = _clean_env("UPSTASH_REDIS_REST_URL")
redis_token = _clean_env("UPSTASH_REDIS_REST_TOKEN")
redis_client: Optional[Redis] = None

if redis_url and redis_token:
    if not _is_valid_http_url(redis_url):
        logger.error(
            "Invalid UPSTASH_REDIS_REST_URL. "
            "Use the Upstash REST URL and include https://"
        )
    else:
        redis_client = Redis(url=redis_url.rstrip("/"), token=redis_token)
else:
    logger.warning(
        "Prediction Redis is not configured. Set UPSTASH_REDIS_REST_URL "
        "and UPSTASH_REDIS_REST_TOKEN."
    )

_JOB_LOCK = threading.Lock()

MAX_STORED_JOBS = 200   # retained for backward compatibility
JOB_TTL_SECONDS = int(os.getenv("PREDICTION_JOB_TTL_SECONDS", "7200"))

# Result cache: keyed by (symbol, target_date) — 1 hour TTL so repeated
# requests within the hour are served instantly from Redis without re-running
# the 2-5 minute ML pipeline.
RESULT_CACHE_TTL_SECONDS = int(os.getenv("PREDICTION_CACHE_TTL_SECONDS", "3600"))

def _require_job_store():
    if redis_client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction jobs require Redis. Configure valid "
                "UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN. "
                "UPSTASH_REDIS_REST_URL must start with https://"
            ),
        )

def _job_key(job_id: str) -> str:
    return f"predict_job:{job_id}"

# -- Result-cache helpers --

def _cache_key(symbol: str, target_date: str) -> str:
    """Redis key for a finished prediction result, scoped by (symbol, target_date)."""
    return f"predict_result:v11:{symbol.upper()}:{target_date}"

def _cache_get(symbol: str, target_date: str) -> Optional[dict]:
    """Return the cached result dict, or None on miss / Redis unavailable."""
    if redis_client is None:
        return None
    try:
        payload = redis_client.get(_cache_key(symbol, target_date))
        if not payload:
            return None
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        return json.loads(payload)
    except Exception as exc:
        logger.warning(f"[PREDICT] Cache GET failed for {symbol}/{target_date}: {exc}")
        return None

def _cache_set(symbol: str, target_date: str, result: dict):
    """Persist a finished result in Redis with a 1-hour TTL."""
    if redis_client is None:
        return
    try:
        redis_client.setex(
            _cache_key(symbol, target_date),
            RESULT_CACHE_TTL_SECONDS,
            json.dumps(result),
        )
        logger.info(
            f"[PREDICT] Result cached for {symbol}/{target_date} "
            f"(TTL={RESULT_CACHE_TTL_SECONDS}s)"
        )
    except Exception as exc:
        logger.warning(f"[PREDICT] Cache SET failed for {symbol}/{target_date}: {exc}")

def _cache_ttl_remaining(symbol: str, target_date: str) -> Optional[int]:
    """Return the seconds remaining on the cache entry, or None if not found."""
    if redis_client is None:
        return None
    try:
        ttl = redis_client.ttl(_cache_key(symbol, target_date))
        return int(ttl) if ttl and int(ttl) > 0 else None
    except Exception as exc:
        logger.warning(f"[PREDICT] Cache TTL check failed for {symbol}/{target_date}: {exc}")
        return None

def _load_job(job_id: str) -> Optional[dict]:
    _require_job_store()
    payload = redis_client.get(_job_key(job_id))
    if not payload:
        return None
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(payload)

def _save_job(job_id: str, job_data: dict):
    _require_job_store()
    redis_client.setex(_job_key(job_id), JOB_TTL_SECONDS, json.dumps(job_data))

def _new_job(job_id: str, symbol: str, target_date: str) -> dict:
    return {
        "job_id"       : job_id,
        "symbol"       : symbol,
        "target_date"  : target_date,
        "status"       : "queued",      # queued | running | done | error
        "progress_pct" : 0,
        "progress_msg" : "Queued — waiting to start",
        "result"       : None,
        "error"        : None,
        "created_at"   : datetime.utcnow().isoformat(),
        "_cancel"      : False,
    }

def _update_job(job_id: str, **kwargs):
    with _JOB_LOCK:
        job = _load_job(job_id)
        if job is not None:
            job.update(kwargs)
            _save_job(job_id, job)

def _evict_old_jobs():
    """No-op when using TTL-based Redis eviction."""
    return

# --Constants --
ML_LOOKBACK_YEARS    = 8
SEQUENCE_LENGTH      = 30
XGB_WEIGHT           = float(os.getenv("PREDICTION_XGB_WEIGHT", "0.65"))
XGB_WEIGHT           = max(0.0, min(1.0, XGB_WEIGHT))
LSTM_WEIGHT          = round(1.0 - XGB_WEIGHT, 4)
CONFIDENCE_THRESHOLD = float(os.getenv("PREDICTION_ENTRY_THRESHOLD", "0.52"))
CONFIDENCE_THRESHOLD = max(0.0, min(1.0, CONFIDENCE_THRESHOLD))
ENSEMBLE_THRESHOLD   = CONFIDENCE_THRESHOLD

# Backtesting/strategy-layer settings. XGBoost and LSTM produce entry-quality
# probabilities; the strategy layer below controls exits separately.
PREFERRED_BACKTEST_DAYS = int(os.getenv("PREDICTION_BACKTEST_DAYS", "252"))
MIN_BACKTEST_DAYS       = int(os.getenv("PREDICTION_MIN_BACKTEST_DAYS", "180"))
EXIT_PROB_THRESHOLD     = float(os.getenv("PREDICTION_EXIT_PROB_THRESHOLD", "0.45"))
STOP_LOSS_PCT           = float(os.getenv("PREDICTION_STOP_LOSS_PCT", "0.05"))
TAKE_PROFIT_PCT         = float(os.getenv("PREDICTION_TAKE_PROFIT_PCT", "0.12"))
MAX_HOLD_DAYS           = int(os.getenv("PREDICTION_MAX_HOLD_DAYS", "30"))

PREFERRED_BACKTEST_DAYS = max(60, min(504, PREFERRED_BACKTEST_DAYS))
MIN_BACKTEST_DAYS       = max(60, min(PREFERRED_BACKTEST_DAYS, MIN_BACKTEST_DAYS))
EXIT_PROB_THRESHOLD     = max(0.0, min(CONFIDENCE_THRESHOLD, EXIT_PROB_THRESHOLD))
STOP_LOSS_PCT           = max(0.0, min(0.50, STOP_LOSS_PCT))
TAKE_PROFIT_PCT         = max(0.0, min(1.00, TAKE_PROFIT_PCT))
MAX_HOLD_DAYS           = max(5, min(252, MAX_HOLD_DAYS))

# Prophet gate quality control. Prophet is useful for stable large-cap stocks,
# but it can become a poor hard gate for explosive/volatile stocks. When the
# held-out Prophet error or uncertainty is high, ML and trend-investor logic
# treat Prophet direction as context rather than a hard BUY/SELL blocker.
PROPHET_HARD_GATE_MAX_MAPE = float(os.getenv("PREDICTION_PROPHET_GATE_MAX_MAPE", "8.0"))
PROPHET_HARD_GATE_MIN_DIR_ACC = float(os.getenv("PREDICTION_PROPHET_GATE_MIN_DIR_ACC", "55.0"))
PROPHET_HARD_GATE_MAX_CI_PCT = float(os.getenv("PREDICTION_PROPHET_GATE_MAX_CI_PCT", "25.0"))
PROPHET_HARD_GATE_MAX_MAPE = max(1.0, min(50.0, PROPHET_HARD_GATE_MAX_MAPE))
PROPHET_HARD_GATE_MIN_DIR_ACC = max(40.0, min(90.0, PROPHET_HARD_GATE_MIN_DIR_ACC))
PROPHET_HARD_GATE_MAX_CI_PCT = max(5.0, min(100.0, PROPHET_HARD_GATE_MAX_CI_PCT))

# Trend Investor mode replaces the old narrow long-only mode. Instead of
# waiting only for a fresh ML BUY timing signal, it asks the beginner-friendly
# question: “was the stock in a strong enough uptrend to hold?”
TREND_INVESTOR_EXIT_CONFIRM_DAYS = int(os.getenv("PREDICTION_TREND_EXIT_CONFIRM_DAYS", "5"))
TREND_INVESTOR_EXIT_CONFIRM_DAYS = max(1, min(30, TREND_INVESTOR_EXIT_CONFIRM_DAYS))

# Bullish-regime protection for the advanced hypothetical long/short backtest.
# A bearish/overbought signal should not automatically become a SHORT when the
# stock is in a strong bullish regime. This is especially important for beginner
# interpretation: SELL can mean Avoid/Exit, while shorting remains only an
# advanced hypothetical simulation.
BULL_REGIME_FILTER_ENABLED = os.getenv("PREDICTION_BULL_REGIME_FILTER", "1").strip().lower() not in {"0", "false", "no", "off"}
BULL_REGIME_SMA_WINDOW      = int(os.getenv("PREDICTION_BULL_REGIME_SMA_WINDOW", "100"))
BULL_REGIME_MOM_WINDOW      = int(os.getenv("PREDICTION_BULL_REGIME_MOM_WINDOW", "126"))
BULL_REGIME_FAST_MOM_WINDOW = int(os.getenv("PREDICTION_BULL_REGIME_FAST_MOM_WINDOW", "63"))
BULL_REGIME_MOM_PCT         = float(os.getenv("PREDICTION_BULL_REGIME_MOM_PCT", "0.25"))
BULL_REGIME_FAST_MOM_PCT    = float(os.getenv("PREDICTION_BULL_REGIME_FAST_MOM_PCT", "0.18"))
BULL_REGIME_SMA_SLOPE_PCT   = float(os.getenv("PREDICTION_BULL_REGIME_SMA_SLOPE_PCT", "0.02"))
BULL_REGIME_SMA_WINDOW      = max(40, min(220, BULL_REGIME_SMA_WINDOW))
BULL_REGIME_MOM_WINDOW      = max(60, min(252, BULL_REGIME_MOM_WINDOW))
BULL_REGIME_FAST_MOM_WINDOW = max(20, min(BULL_REGIME_MOM_WINDOW, BULL_REGIME_FAST_MOM_WINDOW))
BULL_REGIME_MOM_PCT         = max(0.05, min(2.00, BULL_REGIME_MOM_PCT))
BULL_REGIME_FAST_MOM_PCT    = max(0.03, min(1.00, BULL_REGIME_FAST_MOM_PCT))
BULL_REGIME_SMA_SLOPE_PCT   = max(0.00, min(0.50, BULL_REGIME_SMA_SLOPE_PCT))

FLAT_TREND_PCT       = float(os.getenv("PREDICTION_FLAT_TREND_PCT", "0.002"))
LSTM_MIN_AUC         = float(os.getenv("PREDICTION_LSTM_MIN_AUC", "0.58"))
LSTM_MIN_ACCURACY    = float(os.getenv("PREDICTION_LSTM_MIN_ACCURACY", "55.0"))

# Global LSTM tuning knobs. The earlier single-stock LSTM target was too noisy and
# collapsed into near-constant probabilities. This version trains a directional
# 3-class sequence model: CASH/no-trade, LONG setup, SHORT setup. The displayed
# ROC-AUC is the held-out active-vs-cash AUC, while the model also learns
# direction explicitly instead of depending only on Prophet for direction.
LSTM_SEQUENCE_LENGTHS = [int(x.strip()) for x in os.getenv("PREDICTION_LSTM_SEQUENCES", "15,30,45").split(",") if x.strip().isdigit()]
LSTM_SEQUENCE_LENGTHS = sorted({x for x in LSTM_SEQUENCE_LENGTHS if 5 <= x <= 90}) or [SEQUENCE_LENGTH]
LSTM_MAX_EPOCHS       = int(os.getenv("PREDICTION_LSTM_MAX_EPOCHS", "70"))
LSTM_PATIENCE         = int(os.getenv("PREDICTION_LSTM_PATIENCE", "10"))
LSTM_SEEDS            = [int(x.strip()) for x in os.getenv("PREDICTION_LSTM_SEEDS", "42,99").split(",") if x.strip().lstrip("-").isdigit()]
LSTM_SEEDS            = LSTM_SEEDS[:3] or [42]
LSTM_MIN_VAL_AUC      = float(os.getenv("PREDICTION_LSTM_MIN_VAL_AUC", "0.55"))
LSTM_INVERT_MARGIN    = float(os.getenv("PREDICTION_LSTM_INVERT_MARGIN", "0.03"))
LSTM_MAX_EPOCHS       = max(20, min(160, LSTM_MAX_EPOCHS))
LSTM_PATIENCE         = max(5, min(30, LSTM_PATIENCE))
LSTM_MIN_VAL_AUC      = max(0.50, min(0.75, LSTM_MIN_VAL_AUC))
LSTM_INVERT_MARGIN    = max(0.00, min(0.20, LSTM_INVERT_MARGIN))

LSTM_LABEL_HORIZON    = int(os.getenv("PREDICTION_LSTM_LABEL_HORIZON", "5"))
LSTM_ACTIVE_QUANTILE  = float(os.getenv("PREDICTION_LSTM_ACTIVE_QUANTILE", "0.62"))
LSTM_MIN_ACTIVE_RATE  = float(os.getenv("PREDICTION_LSTM_MIN_ACTIVE_RATE", "0.18"))
LSTM_MAX_ACTIVE_RATE  = float(os.getenv("PREDICTION_LSTM_MAX_ACTIVE_RATE", "0.48"))
LSTM_FOCAL_GAMMA      = float(os.getenv("PREDICTION_LSTM_FOCAL_GAMMA", "1.5"))
LSTM_TEMPERATURE_MAX  = float(os.getenv("PREDICTION_LSTM_TEMPERATURE_MAX", "3.0"))
LSTM_LABEL_HORIZON    = max(3, min(15, LSTM_LABEL_HORIZON))
LSTM_ACTIVE_QUANTILE  = max(0.50, min(0.85, LSTM_ACTIVE_QUANTILE))
LSTM_MIN_ACTIVE_RATE  = max(0.05, min(0.45, LSTM_MIN_ACTIVE_RATE))
LSTM_MAX_ACTIVE_RATE  = max(LSTM_MIN_ACTIVE_RATE + 0.05, min(0.70, LSTM_MAX_ACTIVE_RATE))
LSTM_FOCAL_GAMMA      = max(0.0, min(4.0, LSTM_FOCAL_GAMMA))
LSTM_TEMPERATURE_MAX  = max(1.0, min(8.0, LSTM_TEMPERATURE_MAX))

REGRESSOR_COLS = ["rsi","macd","bb_width","vol_change","daily_return","sma_20","sma_50"]

FALLBACK_STOCKS = {
    "RELIANCE":"RELIANCE.NS","TCS":"TCS.NS","HDFCBANK":"HDFCBANK.NS",
    "INFY":"INFY.NS","ICICIBANK":"ICICIBANK.NS","SBIN":"SBIN.NS",
    "BHARTIARTL":"BHARTIARTL.NS","ITC":"ITC.NS",
    "KOTAKBANK":"KOTAKBANK.NS","LT":"LT.NS","SOUTHBANK":"SOUTHBANK.NS",
}

XGB_ALL_FEATURES = [
    "prophet_trend_slope","prophet_ci_width","price_vs_prophet","prophet_is_uptrend",
    "daily_return","return_2d","return_5d","return_10d","return_21d",
    "hl_range","oc_range","gap",
    "close_vs_sma5","close_vs_sma10","close_vs_sma20","close_vs_sma50",
    "sma5_vs_sma20","sma20_vs_sma50","close_vs_ema9","close_vs_ema21",
    "rsi_7","rsi_14","rsi_21","rsi_oversold","rsi_overbought",
    "macd","macd_signal","macd_hist","macd_cross","bb_width","bb_pct","atr_pct",
    "stoch_k","stoch_d","vol_ratio","vol_change","obv_signal",
    "roc_5","roc_10","roc_21","lag_return_1","lag_return_2","lag_return_3",
    "lag_return_5","lag_return_10","volatility_5d","volatility_21d",
    "day_of_week","month","is_monday","is_friday",
    "is_budget_month","is_earnings_month","price_above_sma20","price_above_sma50",
]

LSTM_FEATURE_COLS = [
    "prophet_trend_slope","prophet_ci_width","price_vs_prophet","prophet_is_uptrend",
    "daily_return","return_5d","return_10d","hl_range","oc_range",
    "close_vs_sma5","close_vs_sma20","close_vs_sma50","sma5_vs_sma20","sma20_vs_sma50",
    "rsi_14","macd_hist","bb_pct","stoch_k","vol_ratio","obv_signal",
    "atr_pct","volatility_21d","day_of_week","is_earnings_month",
]


# Global Nifty-50 LSTM settings. This replaces the old single-stock LSTM idea:
# instead of learning from only ~1,500-2,000 rows of one stock, the LSTM learns
# reusable sequence patterns from many large, liquid Indian stocks plus market
# context. The target stock's own history is still included for adaptation, but
# the model is no longer starved of examples.
GLOBAL_LSTM_ENABLED = os.getenv("PREDICTION_GLOBAL_LSTM_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
GLOBAL_LSTM_MAX_TICKERS = int(os.getenv("PREDICTION_GLOBAL_LSTM_MAX_TICKERS", "50"))
GLOBAL_LSTM_CACHE_TTL_SECONDS = int(os.getenv("PREDICTION_GLOBAL_LSTM_CACHE_TTL_SECONDS", "21600"))
GLOBAL_LSTM_MAX_TRAIN_SAMPLES = int(os.getenv("PREDICTION_GLOBAL_LSTM_MAX_TRAIN_SAMPLES", "35000"))
GLOBAL_LSTM_MAX_VAL_SAMPLES = int(os.getenv("PREDICTION_GLOBAL_LSTM_MAX_VAL_SAMPLES", "9000"))
GLOBAL_LSTM_LABEL_HORIZON = int(os.getenv("PREDICTION_GLOBAL_LSTM_LABEL_HORIZON", "10"))
GLOBAL_LSTM_VOL_MULT = float(os.getenv("PREDICTION_GLOBAL_LSTM_VOL_MULT", "0.85"))
GLOBAL_LSTM_MIN_MOVE_PCT = float(os.getenv("PREDICTION_GLOBAL_LSTM_MIN_MOVE_PCT", "0.025"))
GLOBAL_LSTM_MAX_MOVE_PCT = float(os.getenv("PREDICTION_GLOBAL_LSTM_MAX_MOVE_PCT", "0.12"))
GLOBAL_LSTM_MAX_TICKERS = max(5, min(60, GLOBAL_LSTM_MAX_TICKERS))
GLOBAL_LSTM_MAX_TRAIN_SAMPLES = max(5000, min(120000, GLOBAL_LSTM_MAX_TRAIN_SAMPLES))
GLOBAL_LSTM_MAX_VAL_SAMPLES = max(1000, min(30000, GLOBAL_LSTM_MAX_VAL_SAMPLES))
GLOBAL_LSTM_LABEL_HORIZON = max(5, min(30, GLOBAL_LSTM_LABEL_HORIZON))
GLOBAL_LSTM_VOL_MULT = max(0.20, min(3.0, GLOBAL_LSTM_VOL_MULT))
GLOBAL_LSTM_MIN_MOVE_PCT = max(0.005, min(0.10, GLOBAL_LSTM_MIN_MOVE_PCT))
GLOBAL_LSTM_MAX_MOVE_PCT = max(GLOBAL_LSTM_MIN_MOVE_PCT, min(0.35, GLOBAL_LSTM_MAX_MOVE_PCT))

NIFTY50_YAHOO_SYMBOLS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS",
    "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS",
    "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
    "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS",
    "SBILIFE.NS", "SHRIRAMFIN.NS", "SBIN.NS", "SUNPHARMA.NS", "TATACONSUM.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS",
    "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS",
]

NIFTY50_SECTOR_MAP = {
    "ADANIENT.NS":"ENERGY", "ADANIPORTS.NS":"INFRA", "APOLLOHOSP.NS":"HEALTHCARE",
    "ASIANPAINT.NS":"CONSUMER", "AXISBANK.NS":"BANK", "BAJAJ-AUTO.NS":"AUTO",
    "BAJFINANCE.NS":"FINANCE", "BAJAJFINSV.NS":"FINANCE", "BEL.NS":"DEFENCE",
    "BHARTIARTL.NS":"TELECOM", "CIPLA.NS":"PHARMA", "COALINDIA.NS":"ENERGY",
    "DRREDDY.NS":"PHARMA", "EICHERMOT.NS":"AUTO", "GRASIM.NS":"MATERIALS",
    "HCLTECH.NS":"IT", "HDFCBANK.NS":"BANK", "HDFCLIFE.NS":"FINANCE",
    "HEROMOTOCO.NS":"AUTO", "HINDALCO.NS":"METAL", "HINDUNILVR.NS":"FMCG",
    "ICICIBANK.NS":"BANK", "INDUSINDBK.NS":"BANK", "INFY.NS":"IT", "ITC.NS":"FMCG",
    "JSWSTEEL.NS":"METAL", "KOTAKBANK.NS":"BANK", "LT.NS":"INFRA", "M&M.NS":"AUTO",
    "MARUTI.NS":"AUTO", "NESTLEIND.NS":"FMCG", "NTPC.NS":"ENERGY", "ONGC.NS":"ENERGY",
    "POWERGRID.NS":"ENERGY", "RELIANCE.NS":"ENERGY", "SBILIFE.NS":"FINANCE",
    "SHRIRAMFIN.NS":"FINANCE", "SBIN.NS":"BANK", "SUNPHARMA.NS":"PHARMA",
    "TATACONSUM.NS":"FMCG", "TATAMOTORS.NS":"AUTO", "TATASTEEL.NS":"METAL",
    "TCS.NS":"IT", "TECHM.NS":"IT", "TITAN.NS":"CONSUMER", "TRENT.NS":"CONSUMER",
    "ULTRACEMCO.NS":"MATERIALS", "WIPRO.NS":"IT",
}

GLOBAL_LSTM_FEATURE_COLS = [
    "daily_return", "return_2d", "return_5d", "return_10d", "return_20d", "return_60d",
    "close_vs_sma20", "close_vs_sma50", "close_vs_sma100", "close_vs_sma200",
    "sma20_vs_sma50", "sma50_vs_sma200", "ema12_vs_ema50",
    "rsi_14", "macd_scaled", "macd_signal_scaled", "macd_hist_scaled",
    "atr_pct", "bb_width", "bb_pct", "volatility_20d", "volatility_60d", "volume_ratio",
    "nifty_return_1d", "nifty_return_5d", "nifty_return_20d", "nifty_close_vs_sma50",
    "nifty_close_vs_sma200", "nifty_volatility_20d", "sector_return_5d", "sector_return_20d",
    "day_of_week", "month_sin", "month_cos",
]

_GLOBAL_LSTM_DATA_CACHE = {"created_at": None, "frames": None, "market_context": None}


# --NSE Holiday calendar --

def get_nse_holidays() -> pd.DataFrame:
    """
    Auto-detect NSE holidays via pandas_market_calendars.
    Falls back to a hardcoded 2023-2026 list if the library is absent.
    """
    horizon = (datetime.today() + timedelta(days=365)).strftime("%Y-%m-%d")
    try:
        import pandas_market_calendars as mcal
        cal      = mcal.get_calendar("NSE")
        schedule = cal.schedule(start_date="2000-01-01", end_date=horizon)
        all_bdays = pd.date_range(start="2000-01-01", end=horizon, freq="B")
        trading   = pd.DatetimeIndex(schedule.index).normalize()
        holidays  = all_bdays[~all_bdays.normalize().isin(trading)]
        logger.info(f"[PREDICT] NSE holidays via pandas_market_calendars: {len(holidays)}")
        return pd.DataFrame({"holiday":"market_holiday",
                              "ds":pd.to_datetime(holidays),
                              "lower_window":0,"upper_window":1})
    except Exception as exc:
        logger.warning(f"[PREDICT] mcal unavailable ({exc}), using fallback list")
        return pd.DataFrame({
            "holiday":"market_holiday",
            "ds": pd.to_datetime([
                "2023-01-26","2023-03-07","2023-03-30","2023-04-04",
                "2023-04-07","2023-04-14","2023-05-01","2023-06-28",
                "2023-08-15","2023-09-19","2023-10-02","2023-10-24",
                "2023-11-14","2023-11-27","2023-12-25",
                "2024-01-22","2024-01-26","2024-03-25","2024-03-29",
                "2024-04-11","2024-04-14","2024-04-17","2024-05-01",
                "2024-05-23","2024-06-17","2024-07-17","2024-08-15",
                "2024-10-02","2024-11-01","2024-11-15","2024-11-20","2024-12-25",
                "2025-02-26","2025-03-14","2025-03-31","2025-04-10",
                "2025-04-14","2025-04-18","2025-05-01","2025-08-15",
                "2025-08-27","2025-10-02","2025-10-21","2025-10-22",
                "2025-11-05","2025-12-25",
                "2026-01-26","2026-03-20","2026-04-02","2026-04-10",
                "2026-04-14","2026-05-01","2026-08-15","2026-10-02",
                "2026-10-20","2026-12-25",
            ]),
            "lower_window":0,"upper_window":1,
        })


# -- Data helpers --

def resolve_yahoo_symbol(symbol: str, db: Session) -> str:
    try:
        info = db.query(models.StockInfo).filter(
            models.StockInfo.symbol == symbol.upper()).first()
        if info:
            return info.yahoo_symbol
    except Exception as exc:
        logger.warning(f"[PREDICT] DB lookup failed for {symbol}: {exc}")
    return FALLBACK_STOCKS.get(symbol.upper(), f"{symbol.upper()}.NS")


def fetch_stock_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2000-01-01",
                     end=datetime.today().strftime("%Y-%m-%d"),
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.columns = [str(c).split("_")[0] if "_" in str(c) else str(c) for c in df.columns]
    needed = ["Open","High","Low","Close","Volume"]
    df = df[[c for c in needed if c in df.columns]]
    if "Close" not in df.columns or df["Close"].isna().all():
        raise ValueError("Close price missing or all-NaN.")
    df.reset_index(inplace=True)
    df.rename(columns={"Date":"date","index":"date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df.dropna(subset=["Close"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close  = pd.Series(df["Close"].values, dtype=float)
    volume = pd.Series(df["Volume"].values, dtype=float)
    df = df.copy(); df["Close"] = close.values; df["Volume"] = volume.values

    df["sma_20"] = close.rolling(20).mean().values
    df["sma_50"] = close.rolling(50).mean().values
    df["ema_12"] = close.ewm(span=12, adjust=False).mean().values
    df["ema_26"] = close.ewm(span=26, adjust=False).mean().values
    df["macd"]   = df["ema_12"] - df["ema_26"]

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]          = (100 - 100/(1+gain/loss)).values
    sma20 = close.rolling(20).mean(); std20 = close.rolling(20).std()
    df["bb_upper"]     = (sma20+2*std20).values
    df["bb_lower"]     = (sma20-2*std20).values
    df["bb_width"]     = ((df["bb_upper"]-df["bb_lower"])/sma20).values
    df["vol_change"]   = volume.pct_change().values
    df["daily_return"] = close.pct_change().values

    df.replace([np.inf,-np.inf], np.nan, inplace=True)
    df[["vol_change","daily_return"]] = df[["vol_change","daily_return"]].fillna(0)
    df.dropna(subset=["sma_50","rsi","macd","bb_width"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# -- Prophet pipeline --

def prepare_prophet_df(df: pd.DataFrame, test_days: int) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    pdf = pd.DataFrame({
        "ds": pd.to_datetime(df["date"].values),
        "y":  df["Close"].values.astype(float),
        **{c: df[c].values.astype(float) for c in REGRESSOR_COLS},
    })
    pdf.replace([np.inf,-np.inf], np.nan, inplace=True)
    split = len(pdf) - test_days
    for col in REGRESSOR_COLS:
        arr  = pdf[col].iloc[:split].to_numpy(dtype=float, na_value=0.0)
        mean = float(np.nanmean(arr)); std = float(np.nanstd(arr)) or 1.0
        pdf[col] = (pdf[col].to_numpy(dtype=float, na_value=0.0) - mean) / std
    pdf["y"] = pdf["y"].ffill()
    pdf.dropna(subset=["y"], inplace=True)
    pdf.reset_index(drop=True, inplace=True)
    return pdf


def build_prophet_model(holidays_df: pd.DataFrame):
    from prophet import Prophet
    m = Prophet(
        changepoint_prior_scale=0.5, changepoint_range=0.95, n_changepoints=35,
        seasonality_prior_scale=10, holidays_prior_scale=10,
        daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True,
        seasonality_mode="multiplicative", holidays=holidays_df, interval_width=0.95,
    )
    m.add_seasonality(name="quarterly", period=91.25, fourier_order=5)
    for col in REGRESSOR_COLS:
        m.add_regressor(col)
    return m


def evaluate_prophet_on_test(model, test_df: pd.DataFrame) -> dict:
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                  mean_absolute_percentage_error)
    fc = model.predict(test_df)
    actual = test_df["y"].values; pred = fc["yhat"].values
    series = [{"date": row["ds"].strftime("%Y-%m-%d"),
               "actual": round(float(actual[i]),2),
               "predicted": round(float(row["yhat"]),2),
               "lower": round(float(row["yhat_lower"]),2),
               "upper": round(float(row["yhat_upper"]),2)}
              for i, row in fc.iterrows()]
    return {
        "mae":  round(float(mean_absolute_error(actual, pred)),2),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, pred))),2),
        "mape": round(float(mean_absolute_percentage_error(actual, pred)*100),2),
        "direction_accuracy": round(
            float(np.mean(np.sign(np.diff(actual))==np.sign(np.diff(pred)))*100),1),
        "backtest_series": series,
    }


def make_future_df(model, prophet_df: pd.DataFrame, forecast_days: int) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=forecast_days, freq="B")
    known  = prophet_df[["ds"]+REGRESSOR_COLS].copy()
    future = future.merge(known, on="ds", how="left")
    future[REGRESSOR_COLS] = future[REGRESSOR_COLS].ffill().bfill()
    return future

# -- Volatile forecast simulation --

def simulate_volatile_forecast(prophet_df: pd.DataFrame,
                                final_forecast: pd.DataFrame,
                                seed: int = 42) -> pd.DataFrame:
    """
    Replace Prophet's smooth future yhat with a realistically volatile price
    path so the chart shows genuine ups and downs instead of a flat line.

    Why Prophet produces a flat line
    ─────────────────────────────────
    Prophet's yhat is the *conditional expectation* of price given the learned
    trend and seasonality.  It deliberately averages out all day-to-day noise.
    For a stock in sideways consolidation the result is nearly a horizontal line,
    which looks wrong on a chart even though it is mathematically correct as an
    expected value.

    Fix — AR(1) residual injection
    ───────────────────────────────
    1. Compute in-sample percentage residuals:
           r_t = (actual_t − yhat_t) / |yhat_t|
       These capture all the volatility Prophet ignored.

    2. Fit an AR(1) to the residuals:
           r_t = φ · r_{t−1} + ε_t,   ε_t ~ N(0, σ²)
       φ carries the day-to-day autocorrelation (momentum).
       σ is calibrated from historical residual innovation variance.

    3. Simulate n_future steps forward, seeded from the last observed residual
       so the path starts smoothly from the current price.

    4. Multiply the simulated residuals back onto Prophet's smooth yhat:
           new_yhat_t = prophet_yhat_t × (1 + sim_r_t)
       The overall trend shape is preserved; realistic noise is layered on top.

    5. Re-centre Prophet's original CI half-width on the new path so the
       confidence band stays meaningful and proportionate.

    Parameters
    ──────────
    prophet_df      : DataFrame with 'ds' and 'y' columns (historical actuals)
    final_forecast  : Full Prophet forecast DataFrame (in-sample + future)
    seed            : Random seed for reproducibility (same input → same chart)

    Returns
    ───────
    A copy of final_forecast with yhat / yhat_upper / yhat_lower updated for
    all future dates.  In-sample rows are unchanged.
    """
    np.random.seed(seed)

    cutoff      = prophet_df["ds"].max()
    future_mask = final_forecast["ds"] > cutoff
    future_fc   = final_forecast[future_mask].copy()
    n_future    = len(future_fc)

    if n_future == 0:
        return final_forecast

    # -- Step 1: in-sample percentage residuals --
    hist = (
        prophet_df[["ds", "y"]]
        .merge(final_forecast[["ds", "yhat"]], on="ds", how="inner")
        .dropna()
    )
    if len(hist) < 20:
        logger.warning("[PREDICT] simulate_volatile_forecast: too few in-sample points, skipping")
        return final_forecast

    pct_res = (
        (hist["y"].values - hist["yhat"].values)
        / np.abs(hist["yhat"].values + 1e-9)
    )

    # Use at most the most recent ~504 trading days (≈ 2 years) so that
    # recent regime volatility dominates, not decade-old history.
    pct_res = pct_res[-504:]
    pct_res = pct_res[np.isfinite(pct_res)]

    if len(pct_res) < 10:
        logger.warning("[PREDICT] simulate_volatile_forecast: not enough finite residuals, skipping")
        return final_forecast

    # -- Step 2: AR(1) parameter estimation --
    # φ  = lag-1 autocorrelation of residuals (momentum persistence)
    # σ² = innovation variance (unconditional var × (1 − φ²))
    if len(pct_res) > 2:
        phi = float(np.corrcoef(pct_res[:-1], pct_res[1:])[0, 1])
    else:
        phi = 0.0
    phi = float(np.clip(phi, -0.95, 0.95))          # keep AR(1) stationary

    unconditional_std  = float(np.std(pct_res))
    innovation_std     = float(unconditional_std * np.sqrt(max(1.0 - phi ** 2, 0.01)))

    logger.info(
        f"[PREDICT] volatile_forecast: n={n_future}  phi={phi:.3f}  "
        f"innov_std={innovation_std:.4f}  uncond_std={unconditional_std:.4f}"
    )

    # -- Step 3: simulate future percentage residuals --
    eps     = np.random.normal(0.0, innovation_std, n_future)
    sim_pct = np.zeros(n_future)

    # Seed from last observed residual so the path starts from current price
    # without a discontinuous jump.
    sim_pct[0] = phi * float(pct_res[-1]) + eps[0]
    for i in range(1, n_future):
        sim_pct[i] = phi * sim_pct[i - 1] + eps[i]

    # -- Step 4: apply simulated residuals to Prophet's smooth yhat --
    prophet_yhats = future_fc["yhat"].values.copy()
    new_yhat      = prophet_yhats * (1.0 + sim_pct)

    # Floor at 1 rupee - prices can't go negative
    new_yhat = np.maximum(new_yhat, 1.0)

    # -- Step 5: re-centre Prophet CI on new path --
    # Keep the half-width (uncertainty spread) from Prophet unchanged;
    # just shift the band so it surrounds the new volatile path.
    ci_half = (future_fc["yhat_upper"].values - future_fc["yhat_lower"].values) / 2.0

    result = final_forecast.copy()
    idx    = result[future_mask].index
    result.loc[idx, "yhat"]       = new_yhat
    result.loc[idx, "yhat_upper"] = new_yhat + ci_half
    result.loc[idx, "yhat_lower"] = new_yhat - ci_half

    return result

# -- XGBoost feature engineering + pipeline --

def build_xgb_features(df: pd.DataFrame, prophet_forecast: pd.DataFrame) -> pd.DataFrame:
    """52-feature engineering for XGBoost — exact mirror of notebook Cell 20."""
    df = df.copy()
    close  = pd.Series(df["Close"].values,  dtype=float)
    high   = pd.Series(df["High"].values,   dtype=float)
    low    = pd.Series(df["Low"].values,    dtype=float)
    volume = pd.Series(df["Volume"].values, dtype=float)
    open_  = pd.Series(df["Open"].values,   dtype=float)

    pc = prophet_forecast[["ds","trend","yhat","yhat_lower","yhat_upper"]].copy()
    pc["ds"] = pd.to_datetime(pc["ds"]).dt.tz_localize(None)
    df["date_dt"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.merge(pc, left_on="date_dt", right_on="ds", how="left")
    for c in ["trend","yhat","yhat_lower","yhat_upper"]:
        df[c] = df[c].ffill().bfill()
    df["prophet_trend_slope"] = df["trend"].diff(5)
    df["prophet_ci_width"]    = (df["yhat_upper"]-df["yhat_lower"])/(df["yhat"]+1e-9)
    df["price_vs_prophet"]    = (close.values-df["yhat"].values)/(df["yhat"].values+1e-9)
    df["prophet_is_uptrend"]  = (df["prophet_trend_slope"]>0).astype(int)

    df["daily_return"] = close.pct_change(); df["return_2d"]  = close.pct_change(2)
    df["return_5d"]    = close.pct_change(5); df["return_10d"] = close.pct_change(10)
    df["return_21d"]   = close.pct_change(21)
    df["hl_range"]     = (high-low)/close; df["oc_range"] = (close-open_)/open_
    df["gap"]          = (open_-close.shift(1))/close.shift(1)

    for w in [5,10,20,50]:
        df[f"sma_{w}"] = close.rolling(w).mean()
        df[f"close_vs_sma{w}"] = (close-df[f"sma_{w}"])/df[f"sma_{w}"]
    df["sma5_vs_sma20"]  = (df["sma_5"]-df["sma_20"])/df["sma_20"]
    df["sma20_vs_sma50"] = (df["sma_20"]-df["sma_50"])/df["sma_50"]
    df["ema_9"]  = close.ewm(span=9,  adjust=False).mean()
    df["ema_21"] = close.ewm(span=21, adjust=False).mean()
    df["close_vs_ema9"]  = (close-df["ema_9"])/df["ema_9"]
    df["close_vs_ema21"] = (close-df["ema_21"])/df["ema_21"]

    for p in [7,14,21]:
        d = close.diff(); g = d.clip(lower=0).rolling(p).mean()
        lo = (-d.clip(upper=0)).rolling(p).mean()
        df[f"rsi_{p}"] = 100-100/(1+g/(lo+1e-9))
    df["rsi_oversold"]   = (df["rsi_14"]<30).astype(int)
    df["rsi_overbought"] = (df["rsi_14"]>70).astype(int)

    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = e12-e26; df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]  = df["macd"]-df["macd_signal"]
    df["macd_cross"] = ((df["macd"]>df["macd_signal"]) &
                         (df["macd"].shift(1)<=df["macd_signal"].shift(1))).astype(int)

    sma20 = close.rolling(20).mean(); std20 = close.rolling(20).std()
    df["bb_upper"] = sma20+2*std20; df["bb_lower"] = sma20-2*std20
    df["bb_width"] = (df["bb_upper"]-df["bb_lower"])/sma20
    df["bb_pct"]   = (close-df["bb_lower"])/(df["bb_upper"]-df["bb_lower"]+1e-9)

    tr = pd.concat([high-low,(high-close.shift(1)).abs(),(low-close.shift(1)).abs()],axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean(); df["atr_pct"] = df["atr_14"]/close

    lo14 = low.rolling(14).min(); hi14 = high.rolling(14).max()
    df["stoch_k"] = 100*(close-lo14)/(hi14-lo14+1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    df["vol_ma20"]  = volume.rolling(20).mean()
    df["vol_ratio"] = volume/(df["vol_ma20"]+1e-9); df["vol_change"] = volume.pct_change()
    df["obv"]       = (np.sign(close.diff())*volume).cumsum()
    df["obv_ma20"]  = df["obv"].rolling(20).mean()
    df["obv_signal"]= (df["obv"]>df["obv_ma20"]).astype(int)

    for w in [5,10,21]: df[f"roc_{w}"] = close.pct_change(w)*100
    for lag in [1,2,3,5,10]: df[f"lag_return_{lag}"] = close.pct_change().shift(lag)

    df["volatility_5d"]  = close.pct_change().rolling(5).std()
    df["volatility_21d"] = close.pct_change().rolling(21).std()
    df["day_of_week"]       = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"]             = pd.to_datetime(df["date"]).dt.month
    df["is_monday"]         = (df["day_of_week"]==0).astype(int)
    df["is_friday"]         = (df["day_of_week"]==4).astype(int)
    df["is_budget_month"]   = (df["month"]==2).astype(int)
    df["is_earnings_month"] = df["month"].isin([1,4,7,10]).astype(int)
    df["price_above_sma20"] = (close>df["sma_20"]).astype(int)
    df["price_above_sma50"] = (close>df["sma_50"]).astype(int)

    sma5 = close.rolling(5).mean(); future_3d = close.shift(-3)
    in_up = df["prophet_is_uptrend"]==1; in_dn = df["prophet_is_uptrend"]==0
    df["target"] = 0
    df.loc[in_up & (close<sma5) & (future_3d>close), "target"] = 1
    df.loc[in_dn & (close>sma5) & (future_3d<close), "target"] = 1

    df.replace([np.inf,-np.inf], np.nan, inplace=True)
    df = df.iloc[:-3]
    df.dropna(subset=["target","sma_50","rsi_14","macd","atr_14","prophet_trend_slope"],
              inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _prophet_trend_state(honest_fc: pd.DataFrame, dates_series,
                         lookback: int = 5,
                         flat_pct: float = FLAT_TREND_PCT) -> np.ndarray:
    """
    Convert Prophet trend into a three-state gate: up, down, or flat.

    A flat trend should not be forced into BUY. This avoids the old binary
    gate where tiny/sideways Prophet movement could still push an active
    BUY/SELL signal.
    """
    pc = honest_fc[["ds", "trend"]].copy()
    pc["ds"] = pd.to_datetime(pc["ds"]).dt.tz_localize(None)

    dt = pd.to_datetime(pd.Series(dates_series))
    if dt.dt.tz is not None:
        dt = dt.dt.tz_localize(None)

    tm = dict(zip(pc["ds"], pc["trend"]))
    tv = dt.map(tm).ffill().bfill()

    trend_change_pct = (tv.diff(lookback) / tv.shift(lookback)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.where(
        trend_change_pct > flat_pct, "up",
        np.where(trend_change_pct < -flat_pct, "down", "flat")
    )




def _safe_metric_float(value) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _prophet_gate_quality(prophet_metrics: dict,
                          future_forecast: Optional[pd.DataFrame],
                          current_price: float) -> dict:
    """
    Decide whether Prophet direction is reliable enough to be used as a hard
    gate for ML trading signals.

    For volatile bullish stocks, Prophet can have high MAPE or a very wide
    confidence band. In those cases it should be displayed as context, not used
    to block every ML/trend signal.
    """
    mape = _safe_metric_float((prophet_metrics or {}).get("mape"))
    dir_acc = _safe_metric_float((prophet_metrics or {}).get("direction_accuracy"))
    ci_width_pct = None

    try:
        if future_forecast is not None and not future_forecast.empty:
            ff = future_forecast.copy()
            if {"yhat_upper", "yhat_lower"}.issubset(ff.columns):
                width = pd.to_numeric(ff["yhat_upper"], errors="coerce") - pd.to_numeric(ff["yhat_lower"], errors="coerce")
                ci_width_pct = float(np.nanmedian(width) / max(float(current_price), 1e-9) * 100.0)
    except Exception:
        ci_width_pct = None

    reasons = []
    if mape is not None and mape > PROPHET_HARD_GATE_MAX_MAPE:
        reasons.append(f"Prophet MAPE {mape:.2f}% is above {PROPHET_HARD_GATE_MAX_MAPE:.1f}%")
    if dir_acc is not None and dir_acc < PROPHET_HARD_GATE_MIN_DIR_ACC:
        reasons.append(f"Prophet direction accuracy {dir_acc:.1f}% is below {PROPHET_HARD_GATE_MIN_DIR_ACC:.0f}%")
    if ci_width_pct is not None and ci_width_pct > PROPHET_HARD_GATE_MAX_CI_PCT:
        reasons.append(f"Prophet forecast band is wide ({ci_width_pct:.1f}% of price)")

    return {
        "hard_gate_enabled": len(reasons) == 0,
        "quality_poor": len(reasons) > 0,
        "reason": "; ".join(reasons) if reasons else "Prophet validation is acceptable for hard gating",
        "mape": None if mape is None else round(mape, 2),
        "direction_accuracy": None if dir_acc is None else round(dir_acc, 1),
        "median_ci_width_pct": None if ci_width_pct is None else round(ci_width_pct, 2),
        "max_mape_threshold": round(PROPHET_HARD_GATE_MAX_MAPE, 2),
        "min_direction_accuracy_threshold": round(PROPHET_HARD_GATE_MIN_DIR_ACC, 1),
        "max_ci_width_pct_threshold": round(PROPHET_HARD_GATE_MAX_CI_PCT, 2),
    }


def _relax_flat_prophet_gate_when_unreliable(signals_df: Optional[pd.DataFrame],
                                             prob_col: str,
                                             prophet_gate_quality: Optional[dict]) -> Optional[pd.DataFrame]:
    """
    When Prophet quality is poor, do not let a flat/poor Prophet trend suppress
    an otherwise valid bullish-regime long setup. This does not force a BUY for
    every stock; it only converts high-probability FLAT-GATE rows into BUY when
    the price itself is in a strong bullish regime.
    """
    if signals_df is None or signals_df.empty:
        return signals_df
    if prophet_gate_quality is None or prophet_gate_quality.get("hard_gate_enabled", True):
        return signals_df
    if prob_col not in signals_df.columns:
        return signals_df

    df = _attach_bullish_regime_columns(signals_df.copy(), "Close")
    probs = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0).clip(0, 1)
    bullish = df.get("bullish_regime", pd.Series([False] * len(df), index=df.index)).fillna(False).astype(bool)
    sig = df.get("signal", pd.Series(["HOLD"] * len(df), index=df.index)).fillna("HOLD").astype(str).str.upper()
    strength = df.get("strength", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.upper()

    # Relax only the *flat Prophet gate* case. A raw bearish/SELL model signal
    # inside a strong bullish regime is still shown as WAIT / OVEREXTENDED by
    # the decision layer; Trend Investor mode may hold the trend, but the raw
    # short-term caution is not silently converted into a normal BUY signal.
    convert_to_buy = bullish & (probs >= CONFIDENCE_THRESHOLD) & (sig == "HOLD")
    if "strength" in df.columns:
        convert_to_buy = convert_to_buy & strength.isin(["FLAT GATE", "LOW PROB", ""])
    if "prophet_trend_state" in df.columns:
        trend_state = df["prophet_trend_state"].astype(str).str.lower()
        convert_to_buy = convert_to_buy & trend_state.isin(["flat", "nan", "none", ""])

    df.loc[convert_to_buy, "signal"] = "BUY"
    df.loc[convert_to_buy, "strength"] = "REGIME BUY"
    df.loc[convert_to_buy, "prophet_gate_relaxed"] = True
    df.loc[convert_to_buy, "prophet_gate_relax_reason"] = prophet_gate_quality.get("reason", "Prophet hard gate disabled")
    df["prophet_hard_gate_enabled"] = False
    df["prophet_gate_quality_reason"] = prophet_gate_quality.get("reason", "Prophet hard gate disabled")
    return df

def _display_signal_confidence(signal: str, entry_prob: Optional[float],
                               signal_source: str,
                               signal_strength: Optional[str] = None) -> Optional[float]:
    """
    entry_prob is the probability of a good active entry.

    BUY/SELL:
        display confidence = entry_prob.

    HOLD caused by low probability:
        display confidence = 1 - entry_prob.

    HOLD caused by Prophet's flat-trend gate:
        display confidence = None, because this is not a confident "no-trade"
        probability. It means the timing models may be active, but the direction
        gate blocked BUY/SELL because the trend is sideways.
    """
    if signal_source in {"prophet_fallback", "no_ml"} or entry_prob is None:
        return None

    try:
        p = float(entry_prob)
    except Exception:
        return None

    p = max(0.0, min(1.0, p))
    strength = str(signal_strength or "").upper()

    if signal == "HOLD" and (strength == "FLAT GATE" or p >= CONFIDENCE_THRESHOLD):
        return None
    if signal == "WAIT":
        # WAIT / OVEREXTENDED is a caution label, so show the active timing
        # probability as caution confidence instead of pretending it is BUY/SELL.
        return p
    return 1.0 - p if signal == "HOLD" else p


def _build_signal_reason(signal: str,
                         signal_strength: Optional[str],
                         entry_prob: Optional[float],
                         signal_source: str) -> str:
    """Short human-readable reason for the top signal card."""
    if signal_source == "prophet_fallback":
        return "Prophet-only directional fallback"
    if signal_source == "no_ml":
        return "ML timing confidence unavailable"

    strength = str(signal_strength or "").upper()
    if signal == "HOLD" and strength == "FLAT GATE":
        return "Timing probability crossed the threshold, but Prophet trend was flat"
    if signal == "HOLD" and strength == "LOW PROB":
        return "Timing probability stayed below the entry threshold"
    if signal == "WAIT" and strength in {"OVEREXTENDED", "DO NOT CHASE"}:
        return "Strong bullish regime detected; the model is cautious but not calling for a normal exit"
    if signal in {"BUY", "SELL", "WAIT"} and entry_prob is not None:
        return f"Active timing probability: {float(entry_prob) * 100:.1f}%"
    return ""




def _latest_bullish_regime_snapshot(*candidate_frames: Optional[pd.DataFrame]) -> dict:
    """
    Return the latest available bullish-regime snapshot from model signal frames.

    This is used only by the user-facing decision layer. The underlying model
    signals/backtests are preserved, but a bearish signal in a strong bullish
    regime is displayed as WAIT / OVEREXTENDED instead of AVOID / EXIT.
    """
    empty = {
        "current_bullish_regime": False,
        "current_bullish_regime_momentum_pct": None,
        "current_bullish_regime_fast_momentum_pct": None,
        "current_bullish_regime_sma_slope_pct": None,
    }
    for frame in candidate_frames:
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        df = frame.copy()
        if "bullish_regime" not in df.columns:
            if "Close" not in df.columns:
                continue
            try:
                df = _attach_bullish_regime_columns(df, "Close")
            except Exception:
                continue
        if "bullish_regime" not in df.columns or df.empty:
            continue
        row = df.iloc[-1]

        def _safe_float(col: str):
            value = row.get(col, None)
            try:
                if value is None or pd.isna(value):
                    return None
                return round(float(value), 2)
            except Exception:
                return None

        return {
            "current_bullish_regime": bool(row.get("bullish_regime", False)),
            "current_bullish_regime_momentum_pct": _safe_float("bullish_regime_momentum_pct"),
            "current_bullish_regime_fast_momentum_pct": _safe_float("bullish_regime_fast_momentum_pct"),
            "current_bullish_regime_sma_slope_pct": _safe_float("bullish_regime_sma_slope_pct"),
        }
    return empty


def _apply_bullish_regime_display_adjustment(signal: str,
                                             strength: Optional[str],
                                             entry_prob: Optional[float],
                                             signal_source: str,
                                             prophet_direction_signal: Optional[str],
                                             market_regime: dict) -> dict:
    """
    Convert aggressive bearish display labels into beginner-safe wording when
    the stock is still in a strong bullish regime.

    A strong bullish regime does not mean "BUY now". It means the broader trend
    is healthy enough that a short-term bearish/overbought signal should usually
    be read as WAIT / DO NOT CHASE, not AVOID / EXIT or short-selling.
    """
    raw_signal = str(signal or "HOLD").upper()
    raw_strength = str(strength or "—")
    bullish_now = bool(market_regime.get("current_bullish_regime"))

    adjusted = {
        "signal": raw_signal,
        "signal_strength": raw_strength,
        "signal_display_label": None,
        "raw_ml_signal": raw_signal,
        "raw_ml_signal_strength": raw_strength,
        "signal_adjusted_by_regime": False,
        "regime_adjustment_reason": None,
    }

    if not bullish_now:
        return adjusted

    # Strong bullish regime + bearish timing = overextended/wait.
    if raw_signal == "SELL":
        adjusted.update({
            "signal": "WAIT",
            "signal_strength": "OVEREXTENDED",
            "signal_display_label": "WAIT / OVEREXTENDED",
            "signal_adjusted_by_regime": True,
            "regime_adjustment_reason": (
                "Strong bullish regime detected. The bearish/overbought setup is "
                "shown as WAIT / OVEREXTENDED instead of AVOID / EXIT. This means "
                "avoid chasing a fresh entry, not blindly exit a strong trend."
            ),
        })
        return adjusted

    # Strong bullish regime + no entry setup = wait/do not chase rather than a
    # plain HOLD that hides the regime context. Only apply this when Prophet is
    # also leaning down/sideways or the ML strength is not a clean BUY.
    if raw_signal == "HOLD" and str(prophet_direction_signal or "HOLD").upper() in {"SELL", "HOLD"}:
        adjusted.update({
            "signal": "WAIT",
            "signal_strength": "DO NOT CHASE",
            "signal_display_label": "WAIT / DO NOT CHASE",
            "signal_adjusted_by_regime": True,
            "regime_adjustment_reason": (
                "Strong bullish regime detected, but the model does not see a clean "
                "fresh BUY entry. The safer beginner interpretation is WAIT / DO NOT CHASE."
            ),
        })
        return adjusted

    return adjusted

def _hierarchical_signals(proba: np.ndarray, dates_series,
                           honest_fc: pd.DataFrame,
                           price_col_series,
                           threshold: float = CONFIDENCE_THRESHOLD) -> pd.DataFrame:
    """Convert model probabilities → BUY/SELL/HOLD using Prophet trend gate."""
    trend_state = _prophet_trend_state(honest_fc, dates_series)

    sigs, strengths = [], []
    for i, p in enumerate(proba):
        if p >= threshold and trend_state[i] == "up":
            sigs.append("BUY")
            strengths.append("ACTIVE")
        elif p >= threshold and trend_state[i] == "down":
            sigs.append("SELL")
            strengths.append("ACTIVE")
        elif p >= threshold and trend_state[i] == "flat":
            sigs.append("HOLD")
            strengths.append("FLAT GATE")
        else:
            sigs.append("HOLD")
            strengths.append("LOW PROB")

    return pd.DataFrame({
        "date"                : dates_series,
        "Close"               : price_col_series,
        "prob_good_entry"     : proba.round(4),
        "prophet_uptrend"     : (trend_state == "up").astype(int),
        "prophet_trend_state" : trend_state,
        "signal"              : sigs,
        "strength"            : strengths,
    })


def run_xgboost_pipeline(xgb_df: pd.DataFrame, test_days: int,
                          honest_fc: pd.DataFrame) -> dict:
    """Train XGBoost, return signals + metrics (notebook Cells 22-29)."""
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, roc_auc_score

    split = len(xgb_df) - test_days
    tr = xgb_df.iloc[:split-3].copy(); te = xgb_df.iloc[split:].copy()
    feat_cols = [c for c in XGB_ALL_FEATURES if c in xgb_df.columns]

    # Feature selection — top-20 by importance (notebook Cell 22)
    sel = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                              eval_metric="logloss", random_state=42, verbosity=0)
    sel.fit(tr[feat_cols].values, tr["target"].values, verbose=False)
    top20 = (pd.DataFrame({"f": feat_cols, "i": sel.feature_importances_})
             .nlargest(20,"i")["f"].tolist())

    Xtr = tr[top20].values; ytr = tr["target"].values
    Xte = te[top20].values; yte = te["target"].values
    pos_w = float((ytr==0).sum() / max((ytr==1).sum(),1))

    model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                eval_metric="logloss", random_state=42, verbosity=0,
                                scale_pos_weight=pos_w)
    model.fit(Xtr, ytr, eval_set=[(Xtr,ytr),(Xte,yte)], verbose=False)

    yproba = model.predict_proba(Xte)[:,1]
    ypred  = (yproba >= 0.5).astype(int)
    acc    = float(accuracy_score(yte, ypred)*100)
    try:    auc = float(roc_auc_score(yte, yproba))
    except: auc = 0.5

    sigs = _hierarchical_signals(yproba, te["date"].values,
                                  honest_fc, te["Close"].values)
    regime_cols = [
        "date", "bullish_regime", "bullish_regime_momentum_pct",
        "bullish_regime_fast_momentum_pct", "bullish_regime_sma_slope_pct",
    ]
    regime = _attach_bullish_regime_columns(xgb_df[["date", "Close"]].copy(), "Close")
    sigs = sigs.merge(regime[[c for c in regime_cols if c in regime.columns]], on="date", how="left")

    return {"signals": sigs, "metrics": {"accuracy": round(acc,1),
                                          "roc_auc":  round(auc,4)}}

# -- LSTM pipeline --

def _extract_ohlcv_from_download(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a standard OHLCV frame from a yfinance single/multi-ticker download."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = list(raw.columns.get_level_values(0).unique())
            lvl1 = list(raw.columns.get_level_values(1).unique())
            if ticker in lvl0:
                sub = raw[ticker].copy()
            elif ticker in lvl1:
                sub = raw.xs(ticker, axis=1, level=1).copy()
            else:
                return pd.DataFrame()
        else:
            sub = raw.copy()
        sub = sub.reset_index()
        sub.rename(columns={"Date": "date", "Datetime": "date", "index": "date"}, inplace=True)
        needed = ["date", "Open", "High", "Low", "Close", "Volume"]
        keep = [c for c in needed if c in sub.columns]
        sub = sub[keep].copy()
        if "date" not in sub.columns or "Close" not in sub.columns:
            return pd.DataFrame()
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c not in sub.columns:
                sub[c] = sub["Close"] if c != "Volume" else 0
        sub["date"] = pd.to_datetime(sub["date"], errors="coerce").dt.tz_localize(None)
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub = sub.dropna(subset=["date", "Close"]).sort_values("date").reset_index(drop=True)
        return sub
    except Exception as exc:
        logger.warning(f"[PREDICT] Could not parse yfinance data for {ticker}: {exc}")
        return pd.DataFrame()


def _make_market_context_from_ohlcv(market_df: pd.DataFrame) -> pd.DataFrame:
    """Create Nifty market-context features aligned by date."""
    if market_df is None or market_df.empty:
        return pd.DataFrame(columns=["date"])
    m = market_df.copy().sort_values("date").reset_index(drop=True)
    close = pd.Series(m["Close"].values, dtype=float)
    m["nifty_return_1d"] = close.pct_change()
    m["nifty_return_5d"] = close.pct_change(5)
    m["nifty_return_20d"] = close.pct_change(20)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    m["nifty_close_vs_sma50"] = close / sma50 - 1.0
    m["nifty_close_vs_sma200"] = close / sma200 - 1.0
    m["nifty_volatility_20d"] = close.pct_change().rolling(20).std()
    keep = [
        "date", "nifty_return_1d", "nifty_return_5d", "nifty_return_20d",
        "nifty_close_vs_sma50", "nifty_close_vs_sma200", "nifty_volatility_20d",
    ]
    return m[keep].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _compute_global_lstm_features(stock_df: pd.DataFrame,
                                  ticker: str,
                                  market_context: pd.DataFrame,
                                  sector_context: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build normalized, transferable sequence features for the global LSTM."""
    if stock_df is None or stock_df.empty:
        return pd.DataFrame()

    df = stock_df.copy().sort_values("date").reset_index(drop=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = df["Close"] if c != "Volume" else 0
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "Close"]).reset_index(drop=True)
    if len(df) < 260:
        return pd.DataFrame()

    close = pd.Series(df["Close"].values, dtype=float)
    high = pd.Series(df["High"].values, dtype=float)
    low = pd.Series(df["Low"].values, dtype=float)
    volume = pd.Series(df["Volume"].values, dtype=float)

    df["daily_return"] = close.pct_change()
    df["return_2d"] = close.pct_change(2)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["return_20d"] = close.pct_change(20)
    df["return_60d"] = close.pct_change(60)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma100 = close.rolling(100).mean()
    sma200 = close.rolling(200).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    df["close_vs_sma20"] = close / sma20 - 1.0
    df["close_vs_sma50"] = close / sma50 - 1.0
    df["close_vs_sma100"] = close / sma100 - 1.0
    df["close_vs_sma200"] = close / sma200 - 1.0
    df["sma20_vs_sma50"] = sma20 / sma50 - 1.0
    df["sma50_vs_sma200"] = sma50 / sma200 - 1.0
    df["ema12_vs_ema50"] = ema12 / ema50 - 1.0

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = (100 - 100 / (1 + rs)) / 100.0

    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    df["macd_scaled"] = macd / close.replace(0, np.nan)
    df["macd_signal_scaled"] = macd_signal / close.replace(0, np.nan)
    df["macd_hist_scaled"] = macd_hist / close.replace(0, np.nan)

    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df["atr_pct"] = atr / close.replace(0, np.nan)

    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    df["bb_width"] = (bb_upper - bb_lower) / sma20.replace(0, np.nan)
    df["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    ret1 = close.pct_change()
    df["volatility_20d"] = ret1.rolling(20).std()
    df["volatility_60d"] = ret1.rolling(60).std()
    df["volume_ratio"] = volume / volume.rolling(20).mean().replace(0, np.nan)

    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek / 4.0
    month = pd.to_datetime(df["date"]).dt.month.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    future_close = close.shift(-GLOBAL_LSTM_LABEL_HORIZON)
    future_return = future_close / close.replace(0, np.nan) - 1.0
    threshold = (df["atr_pct"].rolling(20).median() * GLOBAL_LSTM_VOL_MULT).clip(
        lower=GLOBAL_LSTM_MIN_MOVE_PCT,
        upper=GLOBAL_LSTM_MAX_MOVE_PCT,
    )
    df["future_return"] = future_return
    df["move_threshold"] = threshold
    df["direction_label"] = 0
    df.loc[future_return > threshold, "direction_label"] = 1
    df.loc[future_return < -threshold, "direction_label"] = 2
    df.loc[future_return.isna() | threshold.isna(), "direction_label"] = np.nan
    df["ticker"] = ticker
    df["sector"] = NIFTY50_SECTOR_MAP.get(ticker, "OTHER")

    if market_context is not None and not market_context.empty:
        df = df.merge(market_context, on="date", how="left")
    else:
        for c in ["nifty_return_1d", "nifty_return_5d", "nifty_return_20d", "nifty_close_vs_sma50", "nifty_close_vs_sma200", "nifty_volatility_20d"]:
            df[c] = 0.0
    if sector_context is not None and not sector_context.empty:
        df = df.merge(sector_context, on="date", how="left")
    if "sector_return_5d" not in df.columns:
        df["sector_return_5d"] = df.get("nifty_return_5d", 0.0)
    if "sector_return_20d" not in df.columns:
        df["sector_return_20d"] = df.get("nifty_return_20d", 0.0)

    for c in GLOBAL_LSTM_FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[GLOBAL_LSTM_FEATURE_COLS] = df[GLOBAL_LSTM_FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    for c in GLOBAL_LSTM_FEATURE_COLS:
        lo, hi = df[c].quantile(0.005), df[c].quantile(0.995)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            df[c] = df[c].clip(lo, hi)
    df[GLOBAL_LSTM_FEATURE_COLS] = df[GLOBAL_LSTM_FEATURE_COLS].fillna(0.0)
    return df


def _download_global_lstm_training_frames(start_date: str, market_context: pd.DataFrame) -> list:
    """Fetch and feature-engineer Nifty-50 frames, using a short in-process cache."""
    now = datetime.utcnow()
    cached_at = _GLOBAL_LSTM_DATA_CACHE.get("created_at")
    cached_frames = _GLOBAL_LSTM_DATA_CACHE.get("frames")
    if cached_at and cached_frames is not None:
        try:
            if (now - cached_at).total_seconds() < GLOBAL_LSTM_CACHE_TTL_SECONDS:
                return cached_frames
        except Exception:
            pass
    tickers = NIFTY50_YAHOO_SYMBOLS[:GLOBAL_LSTM_MAX_TICKERS]
    frames = []
    try:
        logger.info(f"[PREDICT] Downloading global LSTM Nifty universe: {len(tickers)} tickers")
        raw = yf.download(tickers, start=start_date, end=datetime.today().strftime("%Y-%m-%d"),
                          auto_adjust=True, progress=False, threads=True, group_by="ticker")
        for ticker in tickers:
            sub = _extract_ohlcv_from_download(raw, ticker)
            feat = _compute_global_lstm_features(sub, ticker, market_context)
            if not feat.empty:
                frames.append(feat)
    except Exception as exc:
        logger.warning(f"[PREDICT] Global LSTM multi-download failed: {exc}; trying smaller loop")
        for ticker in tickers[:min(20, len(tickers))]:
            try:
                raw_one = yf.download(ticker, start=start_date, end=datetime.today().strftime("%Y-%m-%d"),
                                      auto_adjust=True, progress=False, threads=False)
                sub = _extract_ohlcv_from_download(raw_one, ticker)
                feat = _compute_global_lstm_features(sub, ticker, market_context)
                if not feat.empty:
                    frames.append(feat)
            except Exception as exc_one:
                logger.warning(f"[PREDICT] Global LSTM ticker failed {ticker}: {exc_one}")
    _GLOBAL_LSTM_DATA_CACHE["created_at"] = now
    _GLOBAL_LSTM_DATA_CACHE["frames"] = frames
    return frames


def run_lstm_pipeline(xgb_df: pd.DataFrame, test_days: int, honest_fc: pd.DataFrame) -> dict:
    """Global Nifty-50 LSTM sequence model replacing the old per-stock LSTM."""
    if not GLOBAL_LSTM_ENABLED:
        return {"signals": None, "metrics": {"accuracy": None, "roc_auc": None}, "lstm_reliable": False}
    try:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss
        from sklearn.utils.class_weight import compute_class_weight
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    except Exception as exc:
        logger.warning(f"[PREDICT] Global LSTM dependencies unavailable: {exc}")
        return {"signals": None, "metrics": {"accuracy": None, "roc_auc": None}, "lstm_reliable": False}

    if xgb_df is None or xgb_df.empty or len(xgb_df) < (test_days + max(LSTM_SEQUENCE_LENGTHS) + 260):
        return {"signals": None, "metrics": {"accuracy": None, "roc_auc": None}, "lstm_reliable": False}
    df = xgb_df.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = df["Close"] if c != "Volume" else 0
    df = df.dropna(subset=["date", "Close"]).reset_index(drop=True)
    test_days = min(test_days, max(30, len(df) - max(LSTM_SEQUENCE_LENGTHS) - GLOBAL_LSTM_LABEL_HORIZON - 5))
    split = len(df) - test_days
    if split <= max(LSTM_SEQUENCE_LENGTHS) + 260:
        return {"signals": None, "metrics": {"accuracy": None, "roc_auc": None}, "lstm_reliable": False}

    start_date = (df["date"].min() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    test_start_date = df["date"].iloc[split]
    val_start_date = test_start_date - pd.Timedelta(days=365)

    try:
        nifty_raw = yf.download("^NSEI", start=start_date, end=datetime.today().strftime("%Y-%m-%d"),
                                auto_adjust=True, progress=False, threads=False)
        nifty_df = _extract_ohlcv_from_download(nifty_raw, "^NSEI")
        market_context = _make_market_context_from_ohlcv(nifty_df)
    except Exception as exc:
        logger.warning(f"[PREDICT] Could not fetch Nifty context for global LSTM: {exc}")
        market_context = pd.DataFrame(columns=["date"])

    target_feat = _compute_global_lstm_features(df[["date", "Open", "High", "Low", "Close", "Volume"]].copy(), "__TARGET__", market_context)
    global_frames = _download_global_lstm_training_frames(start_date, market_context)
    all_frames = [f for f in global_frames if f is not None and not f.empty]
    if not target_feat.empty:
        all_frames.append(target_feat)
    if len(all_frames) < 5:
        logger.warning("[PREDICT] Global LSTM has too few stock frames; falling back to no LSTM")
        return {"signals": None, "metrics": {"accuracy": None, "roc_auc": None}, "lstm_reliable": False}

    def _safe_auc_local(y_true, y_score) -> float:
        try:
            y_true = np.asarray(y_true)
            y_score = np.asarray(y_score)
            if len(np.unique(y_true[~pd.isna(y_true)])) < 2:
                return 0.5
            return float(roc_auc_score(y_true, y_score))
        except Exception:
            return 0.5

    def _direction_accuracy_local(y_true, probs3) -> float:
        y_true = np.asarray(y_true).astype(int)
        probs3 = np.asarray(probs3, dtype=float)
        active = y_true != 0
        if active.sum() == 0:
            return 0.0
        pred_dir = np.where(probs3[:, 1] >= probs3[:, 2], 1, 2)
        return float((pred_dir[active] == y_true[active]).mean() * 100.0)

    def _make_sequences_from_frame(frame: pd.DataFrame, seq_len: int, mode: str):
        f = frame.sort_values("date").reset_index(drop=True)
        X_rows, y_rows, d_rows = [], [], []
        feat = f[GLOBAL_LSTM_FEATURE_COLS].to_numpy(dtype=float)
        labels = f["direction_label"].to_numpy()
        dates = pd.to_datetime(f["date"]).to_numpy()
        for i in range(seq_len, len(f)):
            dt = pd.Timestamp(dates[i])
            if mode == "train":
                if not (dt < val_start_date) or pd.isna(labels[i]):
                    continue
            elif mode == "val":
                if not (val_start_date <= dt < test_start_date) or pd.isna(labels[i]):
                    continue
            elif mode == "test_target":
                if not (dt >= test_start_date):
                    continue
            X_rows.append(feat[i-seq_len:i])
            y_rows.append(-1 if pd.isna(labels[i]) else int(labels[i]))
            d_rows.append(pd.Timestamp(dates[i]))
        if not X_rows:
            return np.empty((0, seq_len, len(GLOBAL_LSTM_FEATURE_COLS))), np.array([]), []
        return np.asarray(X_rows, dtype=np.float32), np.asarray(y_rows, dtype=int), d_rows

    def _class_balanced_limit(X, y, max_samples: int, seed: int = 42):
        if len(y) <= max_samples:
            return X, y
        rng = np.random.default_rng(seed)
        indices = []
        classes = np.unique(y)
        per_class = max(1, max_samples // max(1, len(classes)))
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            take = min(len(cls_idx), per_class)
            if take > 0:
                indices.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
        if len(indices) < max_samples:
            remaining = np.setdiff1d(np.arange(len(y)), np.asarray(indices, dtype=int), assume_unique=False)
            take = min(len(remaining), max_samples - len(indices))
            if take > 0:
                indices.extend(rng.choice(remaining, size=take, replace=False).tolist())
        indices = np.asarray(indices, dtype=int)
        rng.shuffle(indices)
        return X[indices], y[indices]

    class SeqDS(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
        def __len__(self):
            return len(self.y)
        def __getitem__(self, i):
            return self.X[i], self.y[i]

    class GlobalAttentionLSTM(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_features, hidden_size=48, num_layers=1,
                                batch_first=True, dropout=0.0, bidirectional=True)
            self.attn = nn.Sequential(nn.Linear(96, 32), nn.Tanh(), nn.Linear(32, 1))
            self.head = nn.Sequential(nn.LayerNorm(96), nn.Dropout(0.25), nn.Linear(96, 48),
                                      nn.ReLU(), nn.Dropout(0.20), nn.Linear(48, 3))
        def forward(self, x):
            out, _ = self.lstm(x)
            w = torch.softmax(self.attn(out).squeeze(-1), dim=1).unsqueeze(-1)
            return self.head((out * w).sum(dim=1))

    def _predict_logits(net, X, batch_size=512):
        device = next(net.parameters()).device
        outs = []
        net.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
                outs.append(net(xb).detach().cpu().numpy())
        return np.vstack(outs) if outs else np.empty((0, 3))

    def _softmax(logits, temperature=1.0):
        z = np.asarray(logits, dtype=float) / max(float(temperature), 1e-6)
        z = z - np.nanmax(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)

    def _fit_temperature(logits, y):
        y = np.asarray(y, dtype=int)
        if len(y) < 50:
            return 1.0
        best_t, best_loss = 1.0, float("inf")
        for t in np.linspace(0.75, LSTM_TEMPERATURE_MAX, 18):
            p = _softmax(logits, t)
            try:
                loss = log_loss(y, np.clip(p, 1e-6, 1-1e-6), labels=[0, 1, 2])
            except Exception:
                continue
            if loss < best_loss:
                best_loss, best_t = loss, float(t)
        return best_t

    def _train_candidate(seq_len: int, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        X_train_parts, y_train_parts, X_val_parts, y_val_parts = [], [], [], []
        for frame in all_frames:
            Xtr, ytr, _ = _make_sequences_from_frame(frame, seq_len, "train")
            Xva, yva, _ = _make_sequences_from_frame(frame, seq_len, "val")
            if len(ytr):
                X_train_parts.append(Xtr); y_train_parts.append(ytr)
            if len(yva):
                X_val_parts.append(Xva); y_val_parts.append(yva)
        if not X_train_parts or not X_val_parts:
            return None
        X_train = np.vstack(X_train_parts)
        y_train = np.concatenate(y_train_parts).astype(int)
        X_val = np.vstack(X_val_parts)
        y_val = np.concatenate(y_val_parts).astype(int)
        X_train, y_train = X_train[y_train >= 0], y_train[y_train >= 0]
        X_val, y_val = X_val[y_val >= 0], y_val[y_val >= 0]
        if len(y_train) < 1500 or len(y_val) < 200 or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            return None
        flat = X_train.reshape(-1, X_train.shape[-1])
        mu = np.nanmean(flat, axis=0)
        sd = np.nanstd(flat, axis=0)
        sd = np.where((sd < 1e-6) | ~np.isfinite(sd), 1.0, sd)
        mu = np.where(~np.isfinite(mu), 0.0, mu)
        X_train = np.clip((X_train - mu) / sd, -6.0, 6.0)
        X_val = np.clip((X_val - mu) / sd, -6.0, 6.0)
        X_train, y_train = _class_balanced_limit(X_train, y_train, GLOBAL_LSTM_MAX_TRAIN_SAMPLES, seed)
        X_val, y_val = _class_balanced_limit(X_val, y_val, GLOBAL_LSTM_MAX_VAL_SAMPLES, seed + 7)
        classes = np.array([0, 1, 2])
        try:
            cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        except Exception:
            counts = np.bincount(y_train, minlength=3).astype(float)
            cw = len(y_train) / (3.0 * np.maximum(counts, 1.0))
        cw = np.clip(cw, 0.50, 4.00).astype(np.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = GlobalAttentionLSTM(X_train.shape[-1]).to(device)
        opt = optim.AdamW(net.parameters(), lr=0.0012, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=2, factor=0.6)
        ce = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device), reduction="none")
        def focal_loss(logits, yb):
            base = ce(logits, yb)
            if LSTM_FOCAL_GAMMA <= 0:
                return base.mean()
            pt = torch.exp(-base).clamp(1e-5, 1.0)
            return (((1.0 - pt) ** LSTM_FOCAL_GAMMA) * base).mean()
        sample_weights = cw[y_train]
        sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double),
                                        num_samples=len(sample_weights), replacement=True)
        loader = DataLoader(SeqDS(X_train, y_train), batch_size=256, sampler=sampler)
        best_state, best_score, best_loss, patience = None, -1.0, float("inf"), 0
        for _epoch in range(min(LSTM_MAX_EPOCHS, 70)):
            net.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = focal_loss(net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            val_logits = _predict_logits(net, X_val)
            val_probs = _softmax(val_logits)
            active_prob = val_probs[:, 1] + val_probs[:, 2]
            active_true = (y_val != 0).astype(int)
            val_auc = _safe_auc_local(active_true, active_prob)
            dir_acc = _direction_accuracy_local(y_val, val_probs) / 100.0
            try:
                val_loss = log_loss(y_val, np.clip(val_probs, 1e-6, 1-1e-6), labels=[0, 1, 2])
            except Exception:
                val_loss = 99.0
            score = val_auc + 0.10 * max(0.0, dir_acc - 0.50)
            scheduler.step(score)
            improved = (score > best_score + 1e-4) or (abs(score - best_score) < 1e-4 and val_loss < best_loss)
            if improved:
                best_score, best_loss, patience = score, val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                patience += 1
            if patience >= LSTM_PATIENCE:
                break
        if best_state is not None:
            net.load_state_dict(best_state)
        val_logits = _predict_logits(net, X_val)
        temp = _fit_temperature(val_logits, y_val)
        val_probs = _softmax(val_logits, temp)
        val_active_auc = _safe_auc_local((y_val != 0).astype(int), val_probs[:, 1] + val_probs[:, 2])
        val_dir_acc = _direction_accuracy_local(y_val, val_probs)
        X_test_raw, y_test, test_dates = _make_sequences_from_frame(target_feat, seq_len, "test_target")
        if len(X_test_raw) == 0:
            return None
        X_test = np.clip((X_test_raw - mu) / sd, -6.0, 6.0)
        test_probs = _softmax(_predict_logits(net, X_test), temp)
        return {
            "seq_len": int(seq_len), "seed": int(seed), "val_auc": round(float(val_active_auc), 4),
            "val_direction_accuracy": round(float(val_dir_acc), 1), "temperature": round(float(temp), 3),
            "test_probs3": np.asarray(test_probs, dtype=float), "test_labels3": np.asarray(y_test, dtype=int),
            "test_dates": test_dates, "train_samples": int(len(y_train)), "val_samples": int(len(y_val)),
            "class_weights": [round(float(x), 3) for x in cw],
        }

    candidate_lengths = LSTM_SEQUENCE_LENGTHS[:2] if len(LSTM_SEQUENCE_LENGTHS) > 2 else LSTM_SEQUENCE_LENGTHS
    candidate_seeds = LSTM_SEEDS[:1]
    candidates = []
    for seq_len in candidate_lengths:
        for seed in candidate_seeds:
            try:
                cand = _train_candidate(seq_len, seed)
                if cand is not None:
                    candidates.append(cand)
            except Exception as exc:
                logger.warning(f"[PREDICT] Global LSTM candidate failed seq={seq_len} seed={seed}: {exc}")
    if not candidates:
        logger.warning("[PREDICT] Global LSTM produced no valid candidates")
        return {"signals": None, "metrics": {"accuracy": None, "roc_auc": None}, "lstm_reliable": False}

    best_val_auc = max(c["val_auc"] for c in candidates)
    selected = [c for c in candidates if c["val_auc"] >= max(0.50, best_val_auc - 0.025)]
    selected = sorted(selected, key=lambda c: (c["val_auc"], c["val_direction_accuracy"]), reverse=True)[:3]
    all_dates = sorted(set.intersection(*[set(pd.Timestamp(d).date() for d in c["test_dates"]) for c in selected]))
    if not all_dates:
        return {"signals": None, "metrics": {"accuracy": None, "roc_auc": None}, "lstm_reliable": False}
    weights = np.asarray([max(c["val_auc"] - 0.49, 0.01) for c in selected], dtype=float)
    weights = weights / weights.sum()
    prob_by_date, label_by_date = {}, {}
    mappings = [{pd.Timestamp(dt).date(): i for i, dt in enumerate(c["test_dates"])} for c in selected]
    for d in all_dates:
        parts, labels_for_d = [], []
        for w, c, mapping in zip(weights, selected, mappings):
            i = mapping.get(d)
            if i is not None:
                parts.append(w * c["test_probs3"][i])
                labels_for_d.append(c["test_labels3"][i])
        if parts:
            p = np.sum(parts, axis=0)
            p = np.clip(p, 1e-6, 1.0)
            prob_by_date[d] = p / p.sum()
            good_labels = [x for x in labels_for_d if x >= 0]
            label_by_date[d] = int(good_labels[0]) if good_labels else -1

    te_slice = df.iloc[split:].copy().reset_index(drop=True)
    te_slice["date_key"] = pd.to_datetime(te_slice["date"]).dt.date
    probs_rows, labels_rows = [], []
    for d in te_slice["date_key"]:
        probs_rows.append(prob_by_date.get(d, np.array([1.0, 0.0, 0.0])))
        labels_rows.append(label_by_date.get(d, -1))
    probs3 = np.asarray(probs_rows, dtype=float)
    labels3 = np.asarray(labels_rows, dtype=int)
    active_prob = probs3[:, 1] + probs3[:, 2]
    valid_metric = labels3 >= 0
    if valid_metric.sum() >= 20 and len(np.unique((labels3[valid_metric] != 0).astype(int))) > 1:
        active_true = (labels3[valid_metric] != 0).astype(int)
        active_pred = (active_prob[valid_metric] >= CONFIDENCE_THRESHOLD).astype(int)
        active_acc = float(accuracy_score(active_true, active_pred) * 100.0)
        try:
            balanced_acc = float(balanced_accuracy_score(active_true, active_pred) * 100.0)
        except Exception:
            balanced_acc = active_acc
        active_auc = _safe_auc_local(active_true, active_prob[valid_metric])
        direction_acc = _direction_accuracy_local(labels3[valid_metric], probs3[valid_metric])
    else:
        active_acc = balanced_acc = direction_acc = 0.0
        active_auc = 0.5

    trend_state = _prophet_trend_state(honest_fc, te_slice["date"].values)
    model_direction = np.where(probs3[:, 1] >= probs3[:, 2], "BUY", "SELL")
    sigs, strengths = [], []
    for p, md, trend in zip(active_prob, model_direction, trend_state):
        if p < CONFIDENCE_THRESHOLD:
            sigs.append("HOLD"); strengths.append("LOW PROB")
        elif trend == "flat":
            sigs.append("HOLD"); strengths.append("FLAT GATE")
        elif md == "BUY" and trend == "up":
            sigs.append("BUY"); strengths.append("ACTIVE")
        elif md == "SELL" and trend == "down":
            sigs.append("SELL"); strengths.append("ACTIVE")
        else:
            sigs.append("HOLD"); strengths.append("DIRECTION CONFLICT")

    sigs_df = pd.DataFrame({
        "date": te_slice["date"].values, "Close": te_slice["Close"].values,
        "prob_good_entry": np.round(active_prob, 4),
        "lstm_cash_prob": np.round(probs3[:, 0], 4), "lstm_long_prob": np.round(probs3[:, 1], 4),
        "lstm_short_prob": np.round(probs3[:, 2], 4), "model_direction": model_direction,
        "prophet_uptrend": (trend_state == "up").astype(int), "prophet_trend_state": trend_state,
        "signal": sigs, "strength": strengths,
    })
    regime_cols = ["date", "bullish_regime", "bullish_regime_momentum_pct", "bullish_regime_fast_momentum_pct", "bullish_regime_sma_slope_pct"]
    regime = _attach_bullish_regime_columns(xgb_df[["date", "Close"]].copy(), "Close")
    sigs_df = sigs_df.merge(regime[[c for c in regime_cols if c in regime.columns]], on="date", how="left")

    train_counts_total = np.zeros(3, dtype=int)
    val_counts_total = np.zeros(3, dtype=int)
    for frame in all_frames:
        if "direction_label" in frame.columns:
            dates = pd.to_datetime(frame["date"])
            tr = frame[dates < val_start_date]["direction_label"].dropna().astype(int)
            va = frame[(dates >= val_start_date) & (dates < test_start_date)]["direction_label"].dropna().astype(int)
            train_counts_total += np.bincount(tr, minlength=3)[:3]
            val_counts_total += np.bincount(va, minlength=3)[:3]
    test_counts = np.bincount(labels3[labels3 >= 0].astype(int), minlength=3)
    has_enough_global_data = sum(int(c.get("train_samples", 0)) for c in selected) >= 3000
    lstm_reliable = bool(has_enough_global_data and active_auc >= LSTM_MIN_AUC and
                         balanced_acc >= max(50.0, LSTM_MIN_ACCURACY - 8.0) and
                         direction_acc >= 52.0 and best_val_auc >= LSTM_MIN_VAL_AUC)
    selected_summary = [{
        "seq_len": int(c["seq_len"]), "seed": int(c["seed"]), "val_auc": float(c["val_auc"]),
        "val_direction_accuracy": float(c["val_direction_accuracy"]), "temperature": float(c["temperature"]),
        "train_samples": int(c.get("train_samples", 0)), "val_samples": int(c.get("val_samples", 0)),
    } for c in selected]
    logger.info("[PREDICT] Global LSTM selected=%s active_auc=%.4f active_acc=%.1f bal_acc=%.1f dir_acc=%.1f reliable=%s",
                selected_summary, active_auc, active_acc, balanced_acc, direction_acc, lstm_reliable)
    return {
        "signals": sigs_df,
        "metrics": {
            "accuracy": round(float(active_acc), 1), "balanced_accuracy": round(float(balanced_acc), 1),
            "roc_auc": round(float(active_auc), 4), "direction_accuracy": round(float(direction_acc), 1),
            "validation_auc": round(float(best_val_auc), 4), "selected_candidates": selected_summary,
            "sequence_lengths": candidate_lengths, "label_mode": "global_nifty50_directional_3class_volatility_adjusted",
            "label_horizon_days": int(GLOBAL_LSTM_LABEL_HORIZON), "global_universe": "Nifty 50 + target stock history",
            "global_tickers_requested": int(min(GLOBAL_LSTM_MAX_TICKERS, len(NIFTY50_YAHOO_SYMBOLS))),
            "global_tickers_used": int(max(0, len(all_frames) - 1)), "feature_count": int(len(GLOBAL_LSTM_FEATURE_COLS)),
            "train_label_counts": {"cash": int(train_counts_total[0]), "long": int(train_counts_total[1]), "short": int(train_counts_total[2])},
            "validation_label_counts": {"cash": int(val_counts_total[0]), "long": int(val_counts_total[1]), "short": int(val_counts_total[2])},
            "test_label_counts": {"cash": int(test_counts[0]), "long": int(test_counts[1]), "short": int(test_counts[2])},
        },
        "lstm_reliable": lstm_reliable,
    }

# -- Ensemble --

def generate_ensemble_signals(xgb_sigs: pd.DataFrame,
                               lstm_sigs: pd.DataFrame,
                               honest_fc: pd.DataFrame,
                               lstm_auc: float = 0.5,
                               lstm_reliable: bool = False) -> pd.DataFrame:
    xgb_cols = ["date", "Close", "prob_good_entry", "prophet_uptrend"]
    regime_cols = [
        "bullish_regime", "bullish_regime_momentum_pct",
        "bullish_regime_fast_momentum_pct", "bullish_regime_sma_slope_pct",
    ]
    xgb_cols += [c for c in regime_cols if c in xgb_sigs.columns]
    xgb = xgb_sigs[xgb_cols].copy()
    xgb.rename(columns={"prob_good_entry": "xgb_prob"}, inplace=True)
    lst = lstm_sigs[["date","prob_good_entry"]].rename(
        columns={"prob_good_entry":"lstm_prob"})
    df  = xgb.merge(lst, on="date", how="inner")

    # Adaptive weighting: if LSTM is still weak after the v3 directional-label improvements,
    # exclude it from the ensemble instead of letting a bad sequence model drag
    # down a strong XGBoost signal. When LSTM clears reliability gates, restore
    # the configured XGBoost/LSTM weights.
    if not lstm_reliable or lstm_auc < LSTM_MIN_AUC:
        eff_xgb, eff_lstm = 1.0, 0.0
        logger.info(
            f"[PREDICT] LSTM AUC={lstm_auc:.4f} unreliable — "
            "using XGBoost-only ensemble weight for this run"
        )
    else:
        eff_xgb, eff_lstm = XGB_WEIGHT, LSTM_WEIGHT

    df["ensemble_prob"] = (eff_xgb*df["xgb_prob"] + eff_lstm*df["lstm_prob"]).round(4)
    df["xgb_agrees"]    = (df["xgb_prob"] >= ENSEMBLE_THRESHOLD).astype(int)
    df["lstm_agrees"]   = (df["lstm_prob"] >= ENSEMBLE_THRESHOLD).astype(int)
    df["both_agree"]    = ((df["xgb_agrees"]==1)&(df["lstm_agrees"]==1)).astype(int)

    trend_state = _prophet_trend_state(honest_fc, df["date"].values)
    df["prophet_trend_state"] = trend_state
    df["prophet_up"] = (trend_state == "up").astype(int)

    sigs, strs = [], []
    for i, row in df.iterrows():
        act = row["ensemble_prob"] >= ENSEMBLE_THRESHOLD
        trend = row["prophet_trend_state"]
        if act and trend == "up":
            sigs.append("BUY")
            strs.append("STRONG" if row["both_agree"] else "NORMAL")
        elif act and trend == "down":
            sigs.append("SELL")
            strs.append("STRONG" if row["both_agree"] else "NORMAL")
        elif act and trend == "flat":
            # Important for UI: this is not a low-confidence HOLD.
            # The timing models are active, but Prophet's direction gate says the
            # price trend is sideways, so the active trade is blocked.
            sigs.append("HOLD")
            strs.append("FLAT GATE")
        else:
            sigs.append("HOLD")
            strs.append("LOW PROB")
    df["signal"] = sigs; df["strength"] = strs
    return df

# -- Strategy conversion + diagnostics --

def _normalise_trend_state(df: pd.DataFrame) -> pd.Series:
    """Return a clean up/down/flat trend-state series for strategy rules."""
    if "prophet_trend_state" in df.columns:
        state = df["prophet_trend_state"].astype(str).str.lower()
        return state.where(state.isin(["up", "down", "flat"]), "flat")
    if "prophet_uptrend" in df.columns:
        return np.where(pd.to_numeric(df["prophet_uptrend"], errors="coerce").fillna(0).astype(int) == 1, "up", "down")
    if "prophet_up" in df.columns:
        return np.where(pd.to_numeric(df["prophet_up"], errors="coerce").fillna(0).astype(int) == 1, "up", "down")
    return pd.Series(["flat"] * len(df), index=df.index)


def _attach_bullish_regime_columns(df: pd.DataFrame,
                                   price_col: str = "Close") -> pd.DataFrame:
    """
    Add explainable bullish-regime columns for the advanced hypothetical
    long/short mode.

    Strong bullish regimes block fresh SHORT entries. In that case, a bearish
    or overbought signal is treated as Avoid/Exit rather than "open a short".
    """
    if df is None or df.empty or price_col not in df.columns:
        return df

    out = df.copy()
    if "bullish_regime" in out.columns:
        out["bullish_regime"] = out["bullish_regime"].fillna(False).astype(bool)
        for col in [
            "bullish_regime_momentum_pct",
            "bullish_regime_fast_momentum_pct",
            "bullish_regime_sma_slope_pct",
        ]:
            if col not in out.columns:
                out[col] = np.nan
        return out

    close = pd.to_numeric(out[price_col], errors="coerce")
    min_sma = max(20, min(BULL_REGIME_SMA_WINDOW, 60))

    sma = close.rolling(BULL_REGIME_SMA_WINDOW, min_periods=min_sma).mean()
    mom = close.pct_change(BULL_REGIME_MOM_WINDOW)
    fast_mom = close.pct_change(BULL_REGIME_FAST_MOM_WINDOW)
    sma_slope = sma.pct_change(20)

    # Early-window fallback: the held-out window may not yet have 126 days of
    # local momentum, so use 63-day momentum until the longer measure exists.
    effective_mom = mom.where(mom.notna(), fast_mom)

    strong_bull = (
        bool(BULL_REGIME_FILTER_ENABLED)
        & (close > sma)
        & (
            (effective_mom >= BULL_REGIME_MOM_PCT)
            | ((fast_mom >= BULL_REGIME_FAST_MOM_PCT) & (sma_slope >= 0))
            | ((effective_mom >= BULL_REGIME_MOM_PCT * 0.60) & (sma_slope >= BULL_REGIME_SMA_SLOPE_PCT))
        )
    )

    out["bullish_regime"] = strong_bull.fillna(False).astype(bool)
    out["bullish_regime_momentum_pct"] = (effective_mom * 100).replace([np.inf, -np.inf], np.nan).round(2)
    out["bullish_regime_fast_momentum_pct"] = (fast_mom * 100).replace([np.inf, -np.inf], np.nan).round(2)
    out["bullish_regime_sma_slope_pct"] = (sma_slope * 100).replace([np.inf, -np.inf], np.nan).round(2)
    return out


def build_entry_exit_strategy_signals(signals_df: pd.DataFrame,
                                      prob_col: str,
                                      model_name: str,
                                      price_col: str = "Close",
                                      entry_threshold: float = CONFIDENCE_THRESHOLD,
                                      exit_threshold: float = EXIT_PROB_THRESHOLD,
                                      stop_loss_pct: float = STOP_LOSS_PCT,
                                      take_profit_pct: float = TAKE_PROFIT_PCT,
                                      max_hold_days: int = MAX_HOLD_DAYS) -> pd.DataFrame:
    """
    Convert entry-quality probabilities into a real long-only strategy.

    XGBoost/LSTM/Ensemble probabilities answer only one question:
        "Is this a good active entry/timing setup?"

    They are not complete trading systems by themselves. This function adds a
    separate exit layer so backtesting can produce multiple completed trades:
      - enter only when probability >= entry_threshold and Prophet trend is up
      - exit when probability falls below exit_threshold
      - exit when Prophet trend becomes flat/down
      - exit on stop-loss, take-profit, or max holding period
    """
    if signals_df is None or signals_df.empty:
        return pd.DataFrame()

    df = signals_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["date", price_col]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return df

    if prob_col not in df.columns:
        # Without a probability column we cannot build an entry-quality strategy.
        df["entry_prob"] = np.nan
        df["signal"] = "HOLD"
        df["strategy_signal"] = "HOLD"
        df["trade_reason"] = "NO_PROBABILITY"
        return df

    df["entry_prob"] = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0).clip(0, 1)
    df["prophet_trend_state"] = list(_normalise_trend_state(df))
    df["entry_quality"] = df["entry_prob"] >= entry_threshold
    if "signal" in df.columns:
        df["raw_advisory_signal"] = df["signal"].astype(str).str.upper()
    else:
        df["raw_advisory_signal"] = np.where(
            (df["entry_prob"] >= entry_threshold) & (df["prophet_trend_state"] == "up"), "BUY",
            np.where((df["entry_prob"] >= entry_threshold) & (df["prophet_trend_state"] == "down"), "SELL", "HOLD")
        )

    strategy_signals, reasons = [], []
    position = "OUT"
    entry_price = None
    entry_date = None

    for _, row in df.iterrows():
        px = float(row[price_col])
        prob = float(row["entry_prob"])
        trend = str(row["prophet_trend_state"]).lower()
        dt = row["date"]
        sig = "HOLD"
        reason = "WAITING_FOR_ENTRY"

        raw_sig = str(row.get("raw_advisory_signal", "HOLD")).upper()

        if position == "OUT":
            if raw_sig == "BUY" and prob >= entry_threshold:
                sig = "BUY"
                reason = "ENTRY_MODEL_BUY_SIGNAL"
                position = "IN"
                entry_price = px
                entry_date = dt
            elif raw_sig == "SELL" and prob >= entry_threshold:
                reason = "BEARISH_SIGNAL_AVOID_LONG_ENTRY"
            elif prob >= entry_threshold and trend == "flat":
                reason = "ENTRY_BLOCKED_FLAT_TREND"
            elif prob >= entry_threshold and trend == "down":
                reason = "ENTRY_BLOCKED_DOWNTREND"
            else:
                reason = "LOW_ENTRY_PROBABILITY"
        else:
            hold_days = int((dt - entry_date).days) if entry_date is not None else 0
            ret = (px / entry_price - 1.0) if entry_price else 0.0

            if raw_sig == "SELL":
                sig = "SELL"
                reason = "EXIT_MODEL_BEARISH_SIGNAL"
            elif trend != "up":
                sig = "SELL"
                reason = "EXIT_TREND_NOT_UP"
            elif prob <= exit_threshold:
                sig = "SELL"
                reason = "EXIT_PROB_DROPPED"
            elif stop_loss_pct > 0 and ret <= -stop_loss_pct:
                sig = "SELL"
                reason = "EXIT_STOP_LOSS"
            elif take_profit_pct > 0 and ret >= take_profit_pct:
                sig = "SELL"
                reason = "EXIT_TAKE_PROFIT"
            elif hold_days >= max_hold_days:
                sig = "SELL"
                reason = "EXIT_MAX_HOLD_DAYS"
            else:
                reason = "HOLDING_POSITION"

            if sig == "SELL":
                position = "OUT"
                entry_price = None
                entry_date = None

        strategy_signals.append(sig)
        reasons.append(reason)

    df["strategy_signal"] = strategy_signals
    df["trade_reason"] = reasons
    # run_backtest reads the 'signal' column, so replace the model's raw
    # advisory signal with the complete strategy signal in this returned copy.
    df["raw_model_signal"] = df.get("signal", "HOLD")
    df["signal"] = df["strategy_signal"]
    df["model_name"] = model_name
    return df





def build_trend_investor_strategy_signals(signals_df: pd.DataFrame,
                                          prob_col: Optional[str],
                                          model_name: str,
                                          price_col: str = "Close",
                                          exit_confirm_days: int = TREND_INVESTOR_EXIT_CONFIRM_DAYS) -> pd.DataFrame:
    """
    Beginner-friendly long-only Trend Investor strategy.

    This replaces the old strict long-only strategy. The old strategy waited for
    a fresh ML BUY timing signal, so strong momentum stocks often stayed cash.
    Trend Investor mode instead enters/holds when the stock is in a strong
    bullish regime and exits only after the trend weakens for a few days.

    BUY  = enter/hold the uptrend
    SELL = exit when the bullish regime breaks
    HOLD = no action
    """
    if signals_df is None or signals_df.empty:
        return pd.DataFrame()

    df = signals_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["date", price_col]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return df

    df = _attach_bullish_regime_columns(df, price_col)
    df["prophet_trend_state"] = list(_normalise_trend_state(df))
    if prob_col and prob_col in df.columns:
        df["entry_prob"] = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0).clip(0, 1)
    else:
        df["entry_prob"] = np.nan

    if "signal" in df.columns:
        df["raw_advisory_signal"] = df["signal"].astype(str).str.upper()
    else:
        df["raw_advisory_signal"] = "HOLD"

    signals, reasons = [], []
    position = "OUT"
    weak_trend_streak = 0

    for _, row in df.iterrows():
        bullish = bool(row.get("bullish_regime", False))
        raw_sig = str(row.get("raw_advisory_signal", "HOLD")).upper()
        prob = row.get("entry_prob", np.nan)
        try:
            prob_val = float(prob)
        except Exception:
            prob_val = np.nan

        sig = "HOLD"
        reason = "WAITING_FOR_BULLISH_REGIME"

        if position == "OUT":
            if bullish:
                sig = "BUY"
                reason = "TREND_INVESTOR_BULLISH_REGIME_ENTRY"
                position = "IN"
                weak_trend_streak = 0
            else:
                reason = "NO_BULLISH_REGIME"
        else:
            if bullish:
                weak_trend_streak = 0
                if raw_sig == "SELL":
                    reason = "HOLDING_BULLISH_REGIME_MODEL_CAUTION"
                elif np.isfinite(prob_val) and prob_val >= CONFIDENCE_THRESHOLD:
                    reason = "HOLDING_BULLISH_REGIME_CONFIRMED"
                else:
                    reason = "HOLDING_BULLISH_REGIME"
            else:
                weak_trend_streak += 1
                if weak_trend_streak >= exit_confirm_days:
                    sig = "SELL"
                    reason = "TREND_INVESTOR_TREND_BREAK_EXIT"
                    position = "OUT"
                    weak_trend_streak = 0
                else:
                    reason = "TREND_WEAKENING_WAITING_FOR_CONFIRMATION"

        signals.append(sig)
        reasons.append(reason)

    df["strategy_signal"] = signals
    df["trade_reason"] = reasons
    df["raw_model_signal"] = df.get("signal", "HOLD")
    df["signal"] = df["strategy_signal"]
    df["model_name"] = model_name
    return df


def build_prophet_trend_investor_strategy_signals(backtest_series: list,
                                                  price_col: str = "Close") -> pd.DataFrame:
    """Trend Investor strategy using Prophet validation prices as the price series."""
    if not backtest_series:
        return pd.DataFrame()
    df = pd.DataFrame(backtest_series).copy()
    if df.empty or "actual" not in df.columns:
        return pd.DataFrame()
    df.rename(columns={"actual": price_col}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["date", price_col]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return df
    df["signal"] = "HOLD"
    return build_trend_investor_strategy_signals(df, None, "Prophet", price_col=price_col)

def build_long_short_strategy_signals(signals_df: pd.DataFrame,
                                      prob_col: str,
                                      model_name: str,
                                      price_col: str = "Close",
                                      entry_threshold: float = CONFIDENCE_THRESHOLD,
                                      exit_threshold: float = EXIT_PROB_THRESHOLD,
                                      stop_loss_pct: float = STOP_LOSS_PCT,
                                      take_profit_pct: float = TAKE_PROFIT_PCT,
                                      max_hold_days: int = MAX_HOLD_DAYS) -> pd.DataFrame:
    """
    Convert entry-quality probabilities into a directional long/short strategy.

    This mode is intentionally separate from the beginner-friendly long-only
    mode. It lets the backtest answer a different question:
        "If SELL meant opening a short position instead of only exiting a long,
         what would the strategy have done?"

    Direction rules:
      - probability >= entry_threshold and Prophet trend is up   -> LONG / BUY
      - probability >= entry_threshold and Prophet trend is down -> SHORT / SELL
      - flat trend blocks fresh directional entries

    Risk/exit rules:
      - exit when probability falls below exit_threshold
      - exit when Prophet trend turns flat
      - reverse when a strong opposite signal appears
      - stop-loss, take-profit, or max holding period
    """
    if signals_df is None or signals_df.empty:
        return pd.DataFrame()

    df = signals_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["date", price_col]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return df

    df = _attach_bullish_regime_columns(df, price_col)

    if prob_col not in df.columns:
        df["entry_prob"] = np.nan
        df["signal"] = "HOLD"
        df["strategy_signal"] = "HOLD"
        df["trade_reason"] = "NO_PROBABILITY"
        df["position_after_signal"] = "CASH"
        df["model_name"] = model_name
        return df

    df["entry_prob"] = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0).clip(0, 1)
    df["prophet_trend_state"] = list(_normalise_trend_state(df))
    df["entry_quality"] = df["entry_prob"] >= entry_threshold
    if "signal" in df.columns:
        df["raw_advisory_signal"] = df["signal"].astype(str).str.upper()
    else:
        df["raw_advisory_signal"] = np.where(
            (df["entry_prob"] >= entry_threshold) & (df["prophet_trend_state"] == "up"), "BUY",
            np.where((df["entry_prob"] >= entry_threshold) & (df["prophet_trend_state"] == "down"), "SELL", "HOLD")
        )

    signals, reasons, positions = [], [], []
    position = "CASH"  # CASH | LONG | SHORT
    entry_price = None
    entry_date = None

    for _, row in df.iterrows():
        px = float(row[price_col])
        prob = float(row["entry_prob"])
        trend = str(row["prophet_trend_state"]).lower()
        dt = row["date"]
        sig = "HOLD"
        reason = "WAITING_FOR_DIRECTIONAL_ENTRY"

        raw_sig = str(row.get("raw_advisory_signal", "HOLD")).upper()
        desired = "CASH"
        bullish_regime = bool(row.get("bullish_regime", False))
        if raw_sig == "BUY" and prob >= entry_threshold:
            desired = "LONG"
        elif raw_sig == "SELL" and prob >= entry_threshold:
            if bullish_regime:
                desired = "CASH"
                reason = "SHORT_BLOCKED_BULLISH_REGIME"
            else:
                desired = "SHORT"
        elif prob >= entry_threshold and trend == "flat":
            desired = "CASH"
            reason = "ENTRY_BLOCKED_FLAT_TREND"
        elif prob >= entry_threshold:
            desired = "CASH"
            reason = "DIRECTION_CONFLICT_OR_GATE_BLOCK"
        else:
            reason = "LOW_ENTRY_PROBABILITY"

        if position == "CASH":
            if desired == "LONG":
                sig = "BUY"
                reason = "LONG_ENTRY_PROB_AND_UPTREND"
                position = "LONG"
                entry_price = px
                entry_date = dt
            elif desired == "SHORT":
                sig = "SELL"
                reason = "SHORT_ENTRY_PROB_AND_DOWNTREND"
                position = "SHORT"
                entry_price = px
                entry_date = dt

        elif position == "LONG":
            hold_days = int((dt - entry_date).days) if entry_date is not None else 0
            ret = (px / entry_price - 1.0) if entry_price else 0.0

            if desired == "SHORT":
                sig = "SELL"
                reason = "REVERSE_LONG_TO_SHORT"
                position = "SHORT"
                entry_price = px
                entry_date = dt
            elif raw_sig == "SELL" and bullish_regime:
                sig = "CASH"
                reason = "EXIT_LONG_BEARISH_SIGNAL_IN_BULLISH_REGIME"
                position = "CASH"
                entry_price = None
                entry_date = None
            elif trend == "flat":
                sig = "CASH"
                reason = "EXIT_LONG_FLAT_TREND"
                position = "CASH"
                entry_price = None
                entry_date = None
            elif prob <= exit_threshold:
                sig = "CASH"
                reason = "EXIT_LONG_PROB_DROPPED"
                position = "CASH"
                entry_price = None
                entry_date = None
            elif stop_loss_pct > 0 and ret <= -stop_loss_pct:
                sig = "CASH"
                reason = "EXIT_LONG_STOP_LOSS"
                position = "CASH"
                entry_price = None
                entry_date = None
            elif take_profit_pct > 0 and ret >= take_profit_pct:
                sig = "CASH"
                reason = "EXIT_LONG_TAKE_PROFIT"
                position = "CASH"
                entry_price = None
                entry_date = None
            elif hold_days >= max_hold_days:
                sig = "CASH"
                reason = "EXIT_LONG_MAX_HOLD_DAYS"
                position = "CASH"
                entry_price = None
                entry_date = None
            else:
                reason = "HOLDING_LONG"

        elif position == "SHORT":
            hold_days = int((dt - entry_date).days) if entry_date is not None else 0
            ret = ((entry_price - px) / entry_price) if entry_price and px > 0 else 0.0

            if desired == "LONG":
                sig = "BUY"
                reason = "REVERSE_SHORT_TO_LONG"
                position = "LONG"
                entry_price = px
                entry_date = dt
            elif bullish_regime:
                sig = "CASH"
                reason = "EXIT_SHORT_BULLISH_REGIME"
                position = "CASH"
                entry_price = None
                entry_date = None
            elif trend == "flat":
                sig = "CASH"
                reason = "EXIT_SHORT_FLAT_TREND"
                position = "CASH"
                entry_price = None
                entry_date = None
            elif prob <= exit_threshold:
                sig = "CASH"
                reason = "EXIT_SHORT_PROB_DROPPED"
                position = "CASH"
                entry_price = None
                entry_date = None
            elif stop_loss_pct > 0 and ret <= -stop_loss_pct:
                sig = "CASH"
                reason = "EXIT_SHORT_STOP_LOSS"
                position = "CASH"
                entry_price = None
                entry_date = None
            elif take_profit_pct > 0 and ret >= take_profit_pct:
                sig = "CASH"
                reason = "EXIT_SHORT_TAKE_PROFIT"
                position = "CASH"
                entry_price = None
                entry_date = None
            elif hold_days >= max_hold_days:
                sig = "CASH"
                reason = "EXIT_SHORT_MAX_HOLD_DAYS"
                position = "CASH"
                entry_price = None
                entry_date = None
            else:
                reason = "HOLDING_SHORT"

        signals.append(sig)
        reasons.append(reason)
        positions.append(position)

    df["raw_model_signal"] = df.get("signal", "HOLD")
    df["strategy_signal"] = signals
    df["signal"] = df["strategy_signal"]
    df["trade_reason"] = reasons
    df["position_after_signal"] = positions
    df["model_name"] = model_name
    return df

def build_prophet_strategy_signals(backtest_series: list,
                                   entry_threshold_pct: float = 0.005,
                                   exit_threshold_pct: float = 0.000,
                                   price_col: str = "Close") -> pd.DataFrame:
    """Create a Prophet-only strategy with separate entry and exit rules."""
    if not backtest_series:
        return pd.DataFrame()

    df = pd.DataFrame(backtest_series).copy()
    if df.empty or "actual" not in df.columns or "predicted" not in df.columns:
        return pd.DataFrame()

    df.rename(columns={"actual": price_col}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
    df = df.dropna(subset=["date", price_col, "predicted"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return df

    df["predicted_edge_pct"] = (df["predicted"] / df[price_col] - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    signals, reasons = [], []
    position = "OUT"
    entry_price = None
    entry_date = None

    for _, row in df.iterrows():
        px = float(row[price_col])
        edge = float(row["predicted_edge_pct"])
        dt = row["date"]
        sig = "HOLD"
        reason = "WAITING_FOR_ENTRY"

        if position == "OUT":
            if edge >= entry_threshold_pct:
                sig = "BUY"
                reason = "PROPHET_POSITIVE_EDGE"
                position = "IN"
                entry_price = px
                entry_date = dt
            else:
                reason = "NO_POSITIVE_EDGE"
        else:
            hold_days = int((dt - entry_date).days) if entry_date is not None else 0
            ret = (px / entry_price - 1.0) if entry_price else 0.0
            if edge <= exit_threshold_pct:
                sig = "SELL"
                reason = "PROPHET_EDGE_FADED"
            elif STOP_LOSS_PCT > 0 and ret <= -STOP_LOSS_PCT:
                sig = "SELL"
                reason = "EXIT_STOP_LOSS"
            elif TAKE_PROFIT_PCT > 0 and ret >= TAKE_PROFIT_PCT:
                sig = "SELL"
                reason = "EXIT_TAKE_PROFIT"
            elif hold_days >= MAX_HOLD_DAYS:
                sig = "SELL"
                reason = "EXIT_MAX_HOLD_DAYS"
            else:
                reason = "HOLDING_POSITION"

            if sig == "SELL":
                position = "OUT"
                entry_price = None
                entry_date = None

        signals.append(sig)
        reasons.append(reason)

    df["signal"] = signals
    df["strategy_signal"] = signals
    df["trade_reason"] = reasons
    return df



def build_prophet_long_short_strategy_signals(backtest_series: list,
                                              entry_threshold_pct: float = 0.005,
                                              exit_threshold_pct: float = 0.000,
                                              price_col: str = "Close") -> pd.DataFrame:
    """Create a Prophet-only long/short strategy from predicted price edge."""
    if not backtest_series:
        return pd.DataFrame()

    df = pd.DataFrame(backtest_series).copy()
    if df.empty or "actual" not in df.columns or "predicted" not in df.columns:
        return pd.DataFrame()

    df.rename(columns={"actual": price_col}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
    df = df.dropna(subset=["date", price_col, "predicted"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return df

    df = _attach_bullish_regime_columns(df, price_col)
    df["predicted_edge_pct"] = (df["predicted"] / df[price_col] - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    signals, reasons, positions = [], [], []
    position = "CASH"
    entry_price = None
    entry_date = None

    for _, row in df.iterrows():
        px = float(row[price_col])
        edge = float(row["predicted_edge_pct"])
        dt = row["date"]
        sig = "HOLD"
        reason = "WAITING_FOR_DIRECTIONAL_EDGE"

        desired = "CASH"
        bullish_regime = bool(row.get("bullish_regime", False))
        if edge >= entry_threshold_pct:
            desired = "LONG"
        elif edge <= -entry_threshold_pct:
            if bullish_regime:
                desired = "CASH"
                reason = "SHORT_BLOCKED_BULLISH_REGIME"
            else:
                desired = "SHORT"
        else:
            reason = "NO_DIRECTIONAL_EDGE"

        if position == "CASH":
            if desired == "LONG":
                sig = "BUY"
                reason = "PROPHET_LONG_EDGE"
                position = "LONG"
                entry_price = px
                entry_date = dt
            elif desired == "SHORT":
                sig = "SELL"
                reason = "PROPHET_SHORT_EDGE"
                position = "SHORT"
                entry_price = px
                entry_date = dt

        elif position == "LONG":
            hold_days = int((dt - entry_date).days) if entry_date is not None else 0
            ret = (px / entry_price - 1.0) if entry_price else 0.0
            if desired == "SHORT":
                sig = "SELL"
                reason = "REVERSE_LONG_TO_SHORT"
                position = "SHORT"
                entry_price = px
                entry_date = dt
            elif edge <= -entry_threshold_pct and bullish_regime:
                sig = "CASH"
                reason = "EXIT_LONG_BEARISH_EDGE_IN_BULLISH_REGIME"
                position = "CASH"
                entry_price = None
                entry_date = None
            elif edge <= exit_threshold_pct:
                sig = "CASH"
                reason = "EXIT_LONG_EDGE_FADED"
                position = "CASH"
                entry_price = None
                entry_date = None
            elif STOP_LOSS_PCT > 0 and ret <= -STOP_LOSS_PCT:
                sig = "CASH"
                reason = "EXIT_LONG_STOP_LOSS"
                position = "CASH"
                entry_price = None
                entry_date = None
            elif TAKE_PROFIT_PCT > 0 and ret >= TAKE_PROFIT_PCT:
                sig = "CASH"
                reason = "EXIT_LONG_TAKE_PROFIT"
                position = "CASH"
                entry_price = None
                entry_date = None
            elif hold_days >= MAX_HOLD_DAYS:
                sig = "CASH"
                reason = "EXIT_LONG_MAX_HOLD_DAYS"
                position = "CASH"
                entry_price = None
                entry_date = None
            else:
                reason = "HOLDING_LONG"

        elif position == "SHORT":
            hold_days = int((dt - entry_date).days) if entry_date is not None else 0
            ret = ((entry_price - px) / entry_price) if entry_price and px > 0 else 0.0
            if desired == "LONG":
                sig = "BUY"
                reason = "REVERSE_SHORT_TO_LONG"
                position = "LONG"
                entry_price = px
                entry_date = dt
            elif bullish_regime:
                sig = "CASH"
                reason = "EXIT_SHORT_BULLISH_REGIME"
                position = "CASH"
                entry_price = None
                entry_date = None
            elif edge >= -exit_threshold_pct:
                sig = "CASH"
                reason = "EXIT_SHORT_EDGE_FADED"
                position = "CASH"
                entry_price = None
                entry_date = None
            elif STOP_LOSS_PCT > 0 and ret <= -STOP_LOSS_PCT:
                sig = "CASH"
                reason = "EXIT_SHORT_STOP_LOSS"
                position = "CASH"
                entry_price = None
                entry_date = None
            elif TAKE_PROFIT_PCT > 0 and ret >= TAKE_PROFIT_PCT:
                sig = "CASH"
                reason = "EXIT_SHORT_TAKE_PROFIT"
                position = "CASH"
                entry_price = None
                entry_date = None
            elif hold_days >= MAX_HOLD_DAYS:
                sig = "CASH"
                reason = "EXIT_SHORT_MAX_HOLD_DAYS"
                position = "CASH"
                entry_price = None
                entry_date = None
            else:
                reason = "HOLDING_SHORT"

        signals.append(sig)
        reasons.append(reason)
        positions.append(position)

    df["signal"] = signals
    df["strategy_signal"] = signals
    df["trade_reason"] = reasons
    df["position_after_signal"] = positions
    return df

def signal_diagnostics(signals_df: pd.DataFrame,
                       prob_col: Optional[str] = None,
                       backtest_result: Optional[dict] = None,
                       mode: str = "long_only") -> dict:
    """Small, UI-friendly explanation of why a model did or did not trade."""
    if signals_df is None or signals_df.empty:
        return {
            "days": 0,
            "high_probability_days": 0,
            "exit_probability_days": 0,
            "low_probability_days": 0,
            "flat_gate_days": 0,
            "bullish_regime_days": 0,
            "shorts_blocked_bullish_regime": 0,
            "uptrend_days": 0,
            "downtrend_days": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "completed_trades": 0,
            "entry_threshold": round(CONFIDENCE_THRESHOLD, 3),
            "exit_threshold": round(EXIT_PROB_THRESHOLD, 3),
            "message": "No signal data available",
            "mode": mode,
        }

    df = signals_df.copy()
    trend_state = _normalise_trend_state(df)
    sigs = df.get("signal", pd.Series(["HOLD"] * len(df))).fillna("HOLD").astype(str).str.upper()

    if prob_col and prob_col in df.columns:
        probs = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0).clip(0, 1)
        high_probability_days = int((probs >= CONFIDENCE_THRESHOLD).sum())
        exit_probability_days = int((probs <= EXIT_PROB_THRESHOLD).sum())
        low_probability_days = int((probs < CONFIDENCE_THRESHOLD).sum())
        flat_gate_days = int(((probs >= CONFIDENCE_THRESHOLD) & (trend_state == "flat")).sum())
    else:
        high_probability_days = 0
        exit_probability_days = 0
        low_probability_days = 0
        flat_gate_days = 0

    bullish_regime_days = int(pd.Series(df.get("bullish_regime", pd.Series([False] * len(df))), index=df.index).fillna(False).astype(bool).sum())
    trade_reasons = df.get("trade_reason", pd.Series([""] * len(df))).fillna("").astype(str).str.upper()
    shorts_blocked_bullish = int((trade_reasons == "SHORT_BLOCKED_BULLISH_REGIME").sum())

    bt = backtest_result or {}
    buy_signals = int((sigs == "BUY").sum())
    sell_signals = int((sigs == "SELL").sum())
    completed = int(bt.get("n_trades", 0) or 0)

    mode_l = str(mode).lower()
    is_long_short = mode_l == "long_short"
    is_trend_investor = mode_l == "trend_investor"
    directional_signals = buy_signals + sell_signals if is_long_short else buy_signals

    if completed == 0 and is_trend_investor and bullish_regime_days == 0:
        message = "No trend entry: the stock did not meet the strong bullish-regime rules."
    elif completed == 0 and shorts_blocked_bullish > 0:
        message = "Short entries were blocked by the bullish-regime filter."
    elif completed == 0 and high_probability_days == 0:
        message = "No entries: probability never crossed the entry threshold."
    elif completed == 0 and directional_signals == 0 and flat_gate_days > 0:
        message = "Entries were blocked because Prophet trend was flat."
    elif completed == 0 and directional_signals == 0 and is_long_short:
        message = "No long/short entries: timing was high only when the trend gate did not allow a directional trade."
    elif completed == 0 and buy_signals == 0 and is_trend_investor:
        message = "No trend entries: bullish-regime conditions were not strong enough."
    elif completed == 0 and buy_signals == 0:
        message = "No long entries: timing was high only during flat/downtrend days."
    elif completed < 3:
        message = "Low sample: the strategy traded, but not enough times for strong statistical confidence."
    else:
        message = "Enough completed trades to make the backtest more readable; still not a guarantee."

    return {
        "days": int(len(df)),
        "high_probability_days": high_probability_days,
        "exit_probability_days": exit_probability_days,
        "low_probability_days": low_probability_days,
        "flat_gate_days": flat_gate_days,
        "bullish_regime_days": bullish_regime_days,
        "shorts_blocked_bullish_regime": shorts_blocked_bullish,
        "uptrend_days": int((trend_state == "up").sum()),
        "downtrend_days": int((trend_state == "down").sum()),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "hold_signals": int((sigs == "HOLD").sum()),
        "completed_trades": completed,
        "entry_threshold": round(CONFIDENCE_THRESHOLD, 3),
        "exit_threshold": round(EXIT_PROB_THRESHOLD, 3),
        "stop_loss_pct": round(STOP_LOSS_PCT * 100, 2),
        "take_profit_pct": round(TAKE_PROFIT_PCT * 100, 2),
        "max_hold_days": int(MAX_HOLD_DAYS),
        "message": message,
        "mode": mode,
    }

# -- Generic backtester  --

def run_backtest(signals_df: pd.DataFrame,
                 price_col: str = "Close",
                 initial_capital: float = 100_000) -> dict:
    """
    Long-only signal backtest.

    BUY  = enter with full available cash
    SELL = exit to cash
    HOLD = do nothing

    The function intentionally returns explicit no-trade metadata. Earlier the
    UI displayed many 0.00 values, which looked like a broken table. In reality,
    0.00 often means the strategy stayed in cash because no completed BUY→SELL
    cycle occurred in the held-out test window.
    """
    if signals_df is None or signals_df.empty or price_col not in signals_df.columns:
        return {
            "total_return": None,
            "bh_total_return": None,
            "sharpe": None,
            "bh_sharpe": None,
            "max_drawdown": None,
            "bh_max_drawdown": None,
            "win_rate": None,
            "profit_factor": None,
            "n_trades": 0,
            "n_entries": 0,
            "n_exits": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "status": "NO_DATA",
            "status_reason": "Backtest data unavailable",
            "portfolio_timeline": [],
        }

    df = signals_df.copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", price_col]).sort_values("date").reset_index(drop=True)

    if df.empty:
        return {
            "total_return": None,
            "bh_total_return": None,
            "sharpe": None,
            "bh_sharpe": None,
            "max_drawdown": None,
            "bh_max_drawdown": None,
            "win_rate": None,
            "profit_factor": None,
            "n_trades": 0,
            "n_entries": 0,
            "n_exits": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "status": "NO_DATA",
            "status_reason": "Backtest data unavailable",
            "portfolio_timeline": [],
        }

    prices = df[price_col].astype(float).to_numpy()
    signals = df.get("signal", pd.Series(["HOLD"] * len(df))).fillna("HOLD").astype(str).str.upper().to_numpy()
    dates = pd.to_datetime(df["date"].values)

    buy_signals = int(np.sum(signals == "BUY"))
    sell_signals = int(np.sum(signals == "SELL"))
    hold_signals = int(np.sum(signals == "HOLD"))

    cash, shares, position = float(initial_capital), 0.0, "OUT"
    pvals, trades, completed = [], [], []
    buy_px, buy_date = None, None

    for i, (px, sig) in enumerate(zip(prices, signals)):
        px = float(px)
        if not np.isfinite(px) or px <= 0:
            pvals.append(cash if position == "OUT" else cash + shares * max(px, 0))
            continue

        if sig == "BUY" and position == "OUT":
            shares = cash / px
            cash = 0.0
            position = "IN"
            buy_px, buy_date = px, dates[i]
            trades.append({"date": dates[i], "action": "BUY", "price": px, "shares": shares})
        elif sig == "SELL" and position == "IN":
            cash = shares * px
            pnl_pct = (px / buy_px - 1.0) * 100 if buy_px else 0.0
            completed.append({"buy_date": buy_date, "sell_date": dates[i], "buy": buy_px, "sell": px, "pnl_pct": pnl_pct})
            shares = 0.0
            position = "OUT"
            buy_px, buy_date = None, None
            trades.append({"date": dates[i], "action": "SELL", "price": px, "shares": 0.0})

        pvals.append(cash + shares * px)

    if position == "IN":
        px = float(prices[-1])
        cash = shares * px
        pnl_pct = (px / buy_px - 1.0) * 100 if buy_px else 0.0
        completed.append({"buy_date": buy_date, "sell_date": dates[-1], "buy": buy_px, "sell": px, "pnl_pct": pnl_pct})
        trades.append({"date": dates[-1], "action": "SELL (close)", "price": px, "shares": 0.0})
        shares = 0.0
        position = "OUT"
        pvals[-1] = cash

    pvals = np.array(pvals, dtype=float)
    bh = (initial_capital / prices[0]) * prices

    def safe_sharpe(rets: np.ndarray) -> Optional[float]:
        if rets is None or len(rets) < 2 or not np.isfinite(rets).any():
            return None
        ex = rets - 0.065 / 252
        s = float(np.nanstd(ex))
        if s <= 1e-8:
            return None
        return float(np.nanmean(ex) / s * np.sqrt(252))

    def max_dd(v: np.ndarray) -> Optional[float]:
        if v is None or len(v) == 0:
            return None
        pk = float(v[0])
        w = 0.0
        for x in v:
            x = float(x)
            pk = max(pk, x)
            w = max(w, (pk - x) / (pk + 1e-9))
        return float(w * 100)

    dr = np.diff(pvals) / (pvals[:-1] + 1e-9) if len(pvals) > 1 else np.array([])
    bdr = np.diff(bh) / (bh[:-1] + 1e-9) if len(bh) > 1 else np.array([])

    n_trades = len(completed)
    pnl_pcts = [float(t["pnl_pct"]) for t in completed]
    wins = [p for p in pnl_pcts if p > 0]
    losses = [abs(p) for p in pnl_pcts if p < 0]

    if n_trades == 0:
        if buy_signals == 0:
            status_reason = "No BUY signal triggered"
        elif sell_signals == 0:
            status_reason = "No completed BUY → SELL cycle"
        else:
            status_reason = "No completed trade cycle"
        status = "NO_TRADES"
        sharpe = None
        win_rate = None
        profit_factor = None
    else:
        status = "ACTIVE"
        status_reason = f"{n_trades} completed trade{'s' if n_trades != 1 else ''}"
        sharpe = safe_sharpe(dr)
        win_rate = float(len(wins) / n_trades * 100)
        if losses:
            profit_factor = float(sum(wins) / (sum(losses) + 1e-9))
        elif wins:
            profit_factor = None  # mathematically infinite; leave UI to show —
        else:
            profit_factor = 0.0

    def rnd(x, nd=2):
        return None if x is None or not np.isfinite(float(x)) else round(float(x), nd)

    return {
        "total_return": rnd((pvals[-1] / initial_capital - 1) * 100, 2),
        "bh_total_return": rnd((bh[-1] / initial_capital - 1) * 100, 2),
        "sharpe": rnd(sharpe, 2),
        "bh_sharpe": rnd(safe_sharpe(bdr), 2),
        "max_drawdown": rnd(max_dd(pvals), 2),
        "bh_max_drawdown": rnd(max_dd(bh), 2),
        "win_rate": rnd(win_rate, 1),
        "profit_factor": rnd(profit_factor, 2),
        "n_trades": int(n_trades),
        "n_entries": int(sum(1 for t in trades if t["action"] == "BUY")),
        "n_exits": int(sum(1 for t in trades if "SELL" in t["action"])),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "hold_signals": hold_signals,
        "status": status,
        "status_reason": status_reason,
        "portfolio_timeline": [
            {"date": str(d.date()), "strategy": round(float(v), 2), "bh": round(float(b), 2)}
            for d, v, b in zip(dates, pvals, bh)
        ],
    }


def run_long_short_backtest(signals_df: pd.DataFrame,
                            price_col: str = "Close",
                            initial_capital: float = 100_000) -> dict:
    """
    Directional long/short backtest.

    BUY  = be long / reverse from short to long
    SELL = be short / reverse from long to short
    CASH = close any open position and stay in cash
    HOLD = keep the current state unchanged

    This is shown separately from the long-only backtest because short-selling
    changes the meaning of SELL. In the beginner long-only mode, SELL means
    "exit/avoid". In this mode, SELL means an active short position that gains
    if the price falls and loses if the price rises.
    """
    if signals_df is None or signals_df.empty or price_col not in signals_df.columns:
        return {
            "total_return": None,
            "bh_total_return": None,
            "sharpe": None,
            "bh_sharpe": None,
            "max_drawdown": None,
            "bh_max_drawdown": None,
            "win_rate": None,
            "profit_factor": None,
            "n_trades": 0,
            "n_entries": 0,
            "n_exits": 0,
            "long_entries": 0,
            "short_entries": 0,
            "cash_exits": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "status": "NO_DATA",
            "status_reason": "Backtest data unavailable",
            "portfolio_timeline": [],
            "mode": "long_short",
        }

    df = signals_df.copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", price_col]).sort_values("date").reset_index(drop=True)

    if df.empty:
        return {
            "total_return": None,
            "bh_total_return": None,
            "sharpe": None,
            "bh_sharpe": None,
            "max_drawdown": None,
            "bh_max_drawdown": None,
            "win_rate": None,
            "profit_factor": None,
            "n_trades": 0,
            "n_entries": 0,
            "n_exits": 0,
            "long_entries": 0,
            "short_entries": 0,
            "cash_exits": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "status": "NO_DATA",
            "status_reason": "Backtest data unavailable",
            "portfolio_timeline": [],
            "mode": "long_short",
        }

    prices = df[price_col].astype(float).to_numpy()
    signals = df.get("signal", pd.Series(["HOLD"] * len(df))).fillna("HOLD").astype(str).str.upper().to_numpy()
    dates = pd.to_datetime(df["date"].values)

    buy_signals = int(np.sum(signals == "BUY"))
    sell_signals = int(np.sum(signals == "SELL"))
    hold_signals = int(np.sum(signals == "HOLD"))
    cash_signals = int(np.sum(signals == "CASH"))

    equity = float(initial_capital)
    position = "CASH"  # CASH | LONG | SHORT
    entry_px = None
    entry_date = None
    entry_equity = None
    pvals, trades, completed = [], [], []

    def current_value(px: float) -> float:
        nonlocal equity, position, entry_px, entry_equity
        if position == "LONG" and entry_px:
            return float(entry_equity * (px / entry_px))
        if position == "SHORT" and entry_px:
            # Full-equity short model. Value rises when price falls and can fall
            # sharply if price rises; clamp only to avoid nonsensical negative
            # chart values after extreme moves.
            return float(max(entry_equity * (2.0 - px / entry_px), 0.0))
        return float(equity)

    def close_position(px: float, dt, action: str):
        nonlocal equity, position, entry_px, entry_date, entry_equity, completed, trades
        if position == "CASH":
            return
        exit_value = current_value(px)
        if position == "LONG":
            pnl_pct = (px / entry_px - 1.0) * 100 if entry_px else 0.0
        else:
            pnl_pct = ((entry_px - px) / entry_px) * 100 if entry_px and px > 0 else 0.0
        completed.append({
            "side": position,
            "entry_date": entry_date,
            "exit_date": dt,
            "entry": entry_px,
            "exit": px,
            "pnl_pct": pnl_pct,
        })
        trades.append({"date": dt, "action": action, "price": px, "side": position})
        equity = exit_value
        position = "CASH"
        entry_px = None
        entry_date = None
        entry_equity = None

    def open_position(side: str, px: float, dt):
        nonlocal position, entry_px, entry_date, entry_equity, trades, equity
        position = side
        entry_px = px
        entry_date = dt
        entry_equity = equity
        trades.append({"date": dt, "action": "BUY" if side == "LONG" else "SELL_SHORT", "price": px, "side": side})

    for i, (px, sig) in enumerate(zip(prices, signals)):
        px = float(px)
        dt = dates[i]
        if not np.isfinite(px) or px <= 0:
            pvals.append(current_value(px if np.isfinite(px) else 0.0))
            continue

        if sig == "BUY":
            if position == "SHORT":
                close_position(px, dt, "COVER_SHORT")
            if position == "CASH":
                open_position("LONG", px, dt)
        elif sig == "SELL":
            if position == "LONG":
                close_position(px, dt, "SELL_LONG")
            if position == "CASH":
                open_position("SHORT", px, dt)
        elif sig == "CASH":
            if position == "LONG":
                close_position(px, dt, "SELL_LONG")
            elif position == "SHORT":
                close_position(px, dt, "COVER_SHORT")

        pvals.append(current_value(px))

    if position != "CASH":
        px = float(prices[-1])
        dt = dates[-1]
        close_position(px, dt, "CLOSE_END")
        pvals[-1] = equity

    pvals = np.array(pvals, dtype=float)
    bh = (initial_capital / prices[0]) * prices

    def safe_sharpe(rets: np.ndarray) -> Optional[float]:
        if rets is None or len(rets) < 2 or not np.isfinite(rets).any():
            return None
        ex = rets - 0.065 / 252
        s = float(np.nanstd(ex))
        if s <= 1e-8:
            return None
        return float(np.nanmean(ex) / s * np.sqrt(252))

    def max_dd(v: np.ndarray) -> Optional[float]:
        if v is None or len(v) == 0:
            return None
        pk = float(v[0])
        w = 0.0
        for x in v:
            x = float(x)
            pk = max(pk, x)
            w = max(w, (pk - x) / (pk + 1e-9))
        return float(w * 100)

    dr = np.diff(pvals) / (pvals[:-1] + 1e-9) if len(pvals) > 1 else np.array([])
    bdr = np.diff(bh) / (bh[:-1] + 1e-9) if len(bh) > 1 else np.array([])

    n_trades = len(completed)
    pnl_pcts = [float(t["pnl_pct"]) for t in completed]
    wins = [p for p in pnl_pcts if p > 0]
    losses = [abs(p) for p in pnl_pcts if p < 0]
    long_entries = int(sum(1 for t in trades if t.get("action") == "BUY"))
    short_entries = int(sum(1 for t in trades if t.get("action") == "SELL_SHORT"))
    cash_exits = int(sum(1 for t in trades if t.get("action") in {"SELL_LONG", "COVER_SHORT", "CLOSE_END"}))

    if n_trades == 0:
        if buy_signals == 0 and sell_signals == 0:
            status_reason = "No BUY or SHORT signal triggered"
        else:
            status_reason = "No completed directional trade cycle"
        status = "NO_TRADES"
        sharpe = None
        win_rate = None
        profit_factor = None
    else:
        status = "ACTIVE"
        status_reason = f"{n_trades} completed directional trade{'s' if n_trades != 1 else ''}"
        sharpe = safe_sharpe(dr)
        win_rate = float(len(wins) / n_trades * 100)
        if losses:
            profit_factor = float(sum(wins) / (sum(losses) + 1e-9))
        elif wins:
            profit_factor = None
        else:
            profit_factor = 0.0

    def rnd(x, nd=2):
        return None if x is None or not np.isfinite(float(x)) else round(float(x), nd)

    return {
        "total_return": rnd((pvals[-1] / initial_capital - 1) * 100, 2),
        "bh_total_return": rnd((bh[-1] / initial_capital - 1) * 100, 2),
        "sharpe": rnd(sharpe, 2),
        "bh_sharpe": rnd(safe_sharpe(bdr), 2),
        "max_drawdown": rnd(max_dd(pvals), 2),
        "bh_max_drawdown": rnd(max_dd(bh), 2),
        "win_rate": rnd(win_rate, 1),
        "profit_factor": rnd(profit_factor, 2),
        "n_trades": int(n_trades),
        "n_entries": int(long_entries + short_entries),
        "n_exits": int(cash_exits),
        "long_entries": long_entries,
        "short_entries": short_entries,
        "cash_exits": cash_exits,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "hold_signals": hold_signals,
        "cash_signals": cash_signals,
        "status": status,
        "status_reason": status_reason,
        "mode": "long_short",
        "portfolio_timeline": [
            {"date": str(d.date()), "strategy": round(float(v), 2), "bh": round(float(b), 2)}
            for d, v, b in zip(dates, pvals, bh)
        ],
    }

# -- Background worker --

def _run_pipeline(job_id: str, symbol: str, target_date: str,
                  yahoo_symbol: str, forecast_days: int):
    """
    Executed in a background thread.  Updates _JOB_STORE[job_id] at each step.
    If _JOB_STORE[job_id]['_cancel'] is set to True the thread stops cleanly.
    """
    def progress(pct: int, msg: str):
        _update_job(job_id, status="running", progress_pct=pct, progress_msg=msg)
        logger.info(f"[PREDICT {job_id}] {pct}%  {msg}")

    def cancelled() -> bool:
        with _JOB_LOCK:
            job = _load_job(job_id)
            return bool(job and job.get("_cancel", False))

    for lg in ["prophet","cmdstanpy","numexpr","h5py","absl","tensorflow"]:
        logging.getLogger(lg).setLevel(logging.ERROR)

    try:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()

        progress(2, "Fetching historical price data from Yahoo Finance…")
        raw_df = fetch_stock_data(yahoo_symbol)
        if cancelled(): return

        progress(6, "Computing technical indicators…")
        data_df    = add_technical_indicators(raw_df.copy())
        total_days = len(data_df)

        # Use a meaningful held-out window for backtesting. 60 trading days was
        # too short and regularly produced 0-1 trades. Prefer ~1 trading year
        # when enough history exists, fall back safely for newer stocks.
        if total_days >= PREFERRED_BACKTEST_DAYS + 300:
            TEST_DAYS = PREFERRED_BACKTEST_DAYS
        elif total_days >= MIN_BACKTEST_DAYS + 300:
            TEST_DAYS = MIN_BACKTEST_DAYS
        else:
            TEST_DAYS = min(90, max(30, total_days // 10))

        if total_days < TEST_DAYS + 100:
            raise ValueError(f"Not enough history ({total_days} days). Need ≥ {TEST_DAYS+100}.")

        INDIAN_HOLIDAYS = get_nse_holidays()
        if cancelled(): return

        # -- Prophet on full history --
        progress(10, "Preparing Prophet dataset (normalising regressors)…")
        prophet_df = prepare_prophet_df(data_df, TEST_DAYS)
        split_idx  = len(prophet_df)-TEST_DAYS
        train_df   = prophet_df.iloc[:split_idx].copy()
        test_df    = prophet_df.iloc[split_idx:].copy()
        if cancelled(): return

        progress(14, "Training Prophet model on full history (30–60 s)…")
        model_p = build_prophet_model(INDIAN_HOLIDAYS)
        model_p.fit(train_df)
        if cancelled(): return

        progress(28, "Evaluating Prophet on held-out test set…")
        try:
            prophet_val = evaluate_prophet_on_test(model_p, test_df)
        except Exception:
            prophet_val = {"mae":None,"rmse":None,"mape":None,
                           "direction_accuracy":None,"backtest_series":[]}

        progress(32, "Retraining Prophet on full data → generating forecast…")
        final_model = build_prophet_model(INDIAN_HOLIDAYS)
        final_model.fit(prophet_df)
        future_df      = make_future_df(final_model, prophet_df, forecast_days)
        final_forecast = final_model.predict(future_df)

        # -- Detect flat trend BEFORE adding volatility --
        # forecast_flat should reflect Prophet's underlying trend, not the
        # volatility we are about to inject, so we measure it on the raw
        # smooth forecast here and set the flag now.
        cutoff_date   = prophet_df["ds"].max()
        current_price = float(prophet_df["y"].iloc[-1])
        _smooth_future = final_forecast[final_forecast["ds"] > cutoff_date].copy()

        forecast_flat      = False
        forecast_range_pct = 0.0
        if not _smooth_future.empty:
            _yhats_smooth      = _smooth_future["yhat"].values
            _range_smooth      = float(np.max(_yhats_smooth) - np.min(_yhats_smooth))
            forecast_range_pct = round((_range_smooth / max(current_price, 1e-9)) * 100, 2)
            forecast_flat      = forecast_range_pct < 2.0

        # -- Inject realistic volatility into the smooth forecast --
        # This is the core fix: Prophet's raw yhat is the conditional
        # expectation (trend + seasonality only) and looks flat on a chart.
        # simulate_volatile_forecast() overlays AR(1)-modelled residual
        # noise so the forecast oscillates like real price data.
        progress(34, "Simulating realistic price path (AR-1 residual model)…")
        try:
            final_forecast = simulate_volatile_forecast(prophet_df, final_forecast)
            logger.info(f"[PREDICT {job_id}] Volatile forecast applied successfully")
        except Exception as vfe:
            logger.warning(f"[PREDICT {job_id}] simulate_volatile_forecast failed ({vfe}), "
                           "falling back to smooth Prophet forecast")

        future_only = final_forecast[final_forecast["ds"] > cutoff_date].copy()
        prophet_gate_quality = _prophet_gate_quality(prophet_val, future_only, current_price)
        if cancelled(): return

        # -- ML-windowed data --
        progress(38, f"Slicing last {ML_LOOKBACK_YEARS} years for XGBoost/LSTM…")
        ml_cutoff  = pd.Timestamp.today()-pd.DateOffset(years=ML_LOOKBACK_YEARS)
        data_df_ml = data_df[pd.to_datetime(data_df["date"])>=ml_cutoff].reset_index(drop=True)
        prophet_ml = prophet_df[pd.to_datetime(prophet_df["ds"])>=ml_cutoff].reset_index(drop=True)
        ml_split   = len(prophet_ml)-TEST_DAYS
        train_ml   = prophet_ml.iloc[:ml_split].copy()
        test_ml    = prophet_ml.iloc[ml_split:].copy()

        progress(42, "Training ML-window Prophet (honest no-leakage forecast for features)…")
        try:
            ml_p  = build_prophet_model(INDIAN_HOLIDAYS)
            ml_p.fit(train_ml)
            fc_tr = ml_p.predict(train_ml)[["ds","trend","yhat","yhat_lower","yhat_upper"]]
            fc_te = ml_p.predict(test_ml) [["ds","trend","yhat","yhat_lower","yhat_upper"]]
            honest_fc = (pd.concat([fc_tr,fc_te],ignore_index=True)
                         .drop_duplicates("ds").sort_values("ds").reset_index(drop=True))
        except Exception as e:
            logger.warning(f"[PREDICT {job_id}] ML-window Prophet failed: {e}, using full fc")
            honest_fc = final_forecast[["ds","trend","yhat","yhat_lower","yhat_upper"]].copy()
        if cancelled(): return

        # -- XGBoost --
        xgb_result = {"signals":None,"metrics":{"accuracy":None,"roc_auc":None}}
        xgb_bt     = None
        xgb_bt_long_short = None
        xgb_strategy_sigs = None
        xgb_long_short_sigs = None
        xgb_diag   = {}
        xgb_diag_long_short = {}
        xgb_df_feat= None
        try:
            import xgboost  # noqa — check availability
            progress(48, "Engineering XGBoost features (52 indicators + Prophet context)…")
            xgb_df_feat = build_xgb_features(data_df_ml.copy(), honest_fc)
            if len(xgb_df_feat) >= TEST_DAYS+50:
                if cancelled(): return
                progress(54, "Training XGBoost classifier (feature selection + fit)…")
                xgb_result = run_xgboost_pipeline(xgb_df_feat, TEST_DAYS, honest_fc)
                xgb_result["signals"] = _relax_flat_prophet_gate_when_unreliable(xgb_result.get("signals"), "prob_good_entry", prophet_gate_quality)
                xgb_strategy_sigs = build_trend_investor_strategy_signals(
                    _relax_flat_prophet_gate_when_unreliable(xgb_result["signals"], "prob_good_entry", prophet_gate_quality),
                    prob_col="prob_good_entry",
                    model_name="XGBoost",
                )
                xgb_bt = run_backtest(xgb_strategy_sigs)
                xgb_diag = signal_diagnostics(xgb_strategy_sigs, "entry_prob", xgb_bt, mode="trend_investor")

                xgb_long_short_sigs = build_long_short_strategy_signals(
                    xgb_result["signals"],
                    prob_col="prob_good_entry",
                    model_name="XGBoost",
                )
                xgb_bt_long_short = run_long_short_backtest(xgb_long_short_sigs)
                xgb_diag_long_short = signal_diagnostics(
                    xgb_long_short_sigs, "entry_prob", xgb_bt_long_short, mode="long_short"
                )
        except Exception as e:
            logger.warning(f"[PREDICT {job_id}] XGBoost error (non-fatal): {e}")
        if cancelled(): return

        # -- LSTM --
        lstm_result = {"signals":None,"metrics":{"accuracy":None,"roc_auc":None},"lstm_reliable":False}
        lstm_bt     = None
        lstm_bt_long_short = None
        lstm_strategy_sigs = None
        lstm_long_short_sigs = None
        lstm_diag   = {}
        lstm_diag_long_short = {}
        try:
            import torch  # noqa — check availability
            if xgb_df_feat is not None and len(xgb_df_feat)>=SEQUENCE_LENGTH+TEST_DAYS+20:
                if cancelled(): return
                progress(62, "Training Global Nifty-50 LSTM sequence model…")
                lstm_result = run_lstm_pipeline(xgb_df_feat, TEST_DAYS, honest_fc)
                lstm_result["signals"] = _relax_flat_prophet_gate_when_unreliable(lstm_result.get("signals"), "prob_good_entry", prophet_gate_quality)
                if lstm_result["signals"] is not None:
                    lstm_strategy_sigs = build_trend_investor_strategy_signals(
                        _relax_flat_prophet_gate_when_unreliable(lstm_result["signals"], "prob_good_entry", prophet_gate_quality),
                        prob_col="prob_good_entry",
                        model_name="LSTM",
                    )
                    lstm_bt = run_backtest(lstm_strategy_sigs)
                    lstm_diag = signal_diagnostics(lstm_strategy_sigs, "entry_prob", lstm_bt, mode="trend_investor")

                    lstm_long_short_sigs = build_long_short_strategy_signals(
                        lstm_result["signals"],
                        prob_col="prob_good_entry",
                        model_name="LSTM",
                    )
                    lstm_bt_long_short = run_long_short_backtest(lstm_long_short_sigs)
                    lstm_diag_long_short = signal_diagnostics(
                        lstm_long_short_sigs, "entry_prob", lstm_bt_long_short, mode="long_short"
                    )
        except Exception as e:
            logger.warning(f"[PREDICT {job_id}] LSTM error (non-fatal): {e}")
        if cancelled(): return

        # -- Ensemble --
        progress(78, "Combining XGBoost + LSTM into Ensemble (tiered signals)…")
        ens_sigs         = None; ens_bt = None
        ens_strategy_sigs = None
        ens_long_short_sigs = None
        ens_bt_long_short = None
        ens_diag         = {}
        ens_diag_long_short = {}
        current_signal   = "HOLD"
        current_strength = "—"
        current_entry_probability = None
        signal_source = "no_ml"

        try:
            xs = xgb_result["signals"]; ls = lstm_result["signals"]
            lstm_auc_val = lstm_result.get("metrics", {}).get("roc_auc") or 0.5
            lstm_rel_val = bool(lstm_result.get("lstm_reliable", False))

            # Do not suppress the ensemble just because LSTM is weak. Instead,
            # generate it and let generate_ensemble_signals() downweight LSTM.
            # This keeps the Ensemble tab populated while still being honest
            # about LSTM reliability in the UI.
            if xs is not None and ls is not None:
                ens_sigs = generate_ensemble_signals(
                    xs, ls, honest_fc,
                    lstm_auc=float(lstm_auc_val),
                    lstm_reliable=lstm_rel_val,
                )
                ens_sigs = _relax_flat_prophet_gate_when_unreliable(ens_sigs, "ensemble_prob", prophet_gate_quality)
                ens_strategy_sigs = build_trend_investor_strategy_signals(
                    _relax_flat_prophet_gate_when_unreliable(ens_sigs, "ensemble_prob", prophet_gate_quality),
                    prob_col="ensemble_prob",
                    model_name="Ensemble",
                )
                ens_bt = run_backtest(ens_strategy_sigs)
                ens_diag = signal_diagnostics(ens_strategy_sigs, "entry_prob", ens_bt, mode="trend_investor")

                ens_long_short_sigs = build_long_short_strategy_signals(
                    ens_sigs,
                    prob_col="ensemble_prob",
                    model_name="Ensemble",
                )
                ens_bt_long_short = run_long_short_backtest(ens_long_short_sigs)
                ens_diag_long_short = signal_diagnostics(
                    ens_long_short_sigs, "entry_prob", ens_bt_long_short, mode="long_short"
                )

                last = ens_sigs.iloc[-1]
                current_signal   = str(last["signal"])
                current_strength = str(last["strength"])
                current_entry_probability = float(last["ensemble_prob"])
                signal_source    = "ensemble"
            elif xs is not None:
                last             = xs.iloc[-1]
                current_signal   = str(last["signal"])
                current_entry_probability = float(last["prob_good_entry"])
                xgb_strength = str(last.get("strength", ""))
                if current_signal == "HOLD" and xgb_strength in {"FLAT GATE", "LOW PROB"}:
                    current_strength = xgb_strength
                else:
                    current_strength = "XGBoost only"
                signal_source    = "xgboost_only"
        except Exception as e:
            logger.warning(f"[PREDICT {job_id}] Ensemble error (non-fatal): {e}")

        # -- Prophet directional view, kept separate from the ML/ensemble signal --
        # The old logic overwrote HOLD with Prophet-only BUY/SELL. That made the
        # final signal look like Prophet dominated 90% of the time. Now Prophet's
        # price-direction opinion is returned as a separate informational field.
        prophet_direction_signal = "HOLD"
        prophet_direction_change_pct = None
        if len(future_only):
            prophet_direction_change_pct = round(
                (float(future_only["yhat"].iloc[-1]) - current_price) / current_price * 100,
                2,
            )
            if prophet_direction_change_pct > 0.5:
                prophet_direction_signal = "BUY"
            elif prophet_direction_change_pct < -0.5:
                prophet_direction_signal = "SELL"

        # -- Beginner-safe signal display for strong bullish regimes --
        # A strongly bullish stock can still look short-term overextended. In that
        # case, do not present the main user-facing signal as AVOID / EXIT. Show
        # WAIT / OVEREXTENDED instead, while keeping the raw ML signal for audit
        # and keeping advanced hypothetical backtests separate.
        market_regime = _latest_bullish_regime_snapshot(ens_sigs, xgb_result.get("signals"), lstm_result.get("signals"), xgb_df_feat)
        display_adjustment = _apply_bullish_regime_display_adjustment(
            current_signal,
            current_strength,
            current_entry_probability,
            signal_source,
            prophet_direction_signal,
            market_regime,
        )
        raw_ml_signal = display_adjustment["raw_ml_signal"]
        raw_ml_signal_strength = display_adjustment["raw_ml_signal_strength"]
        current_signal = display_adjustment["signal"]
        current_strength = display_adjustment["signal_strength"]

        # -- Prophet-only backtest --
        progress(84, "Running Prophet strategy backtest…")
        prophet_bt: dict = {}
        prophet_bt_long_short: dict = {}
        prophet_diag: dict = {}
        prophet_diag_long_short: dict = {}
        if prophet_val.get("backtest_series"):
            prophet_strategy_sigs = build_prophet_trend_investor_strategy_signals(prophet_val["backtest_series"])
            prophet_bt = run_backtest(prophet_strategy_sigs)
            prophet_diag = signal_diagnostics(prophet_strategy_sigs, None, prophet_bt, mode="trend_investor")

            prophet_long_short_sigs = build_prophet_long_short_strategy_signals(prophet_val["backtest_series"])
            prophet_bt_long_short = run_long_short_backtest(prophet_long_short_sigs)
            prophet_diag_long_short = signal_diagnostics(
                prophet_long_short_sigs, None, prophet_bt_long_short, mode="long_short"
            )

        # -- Assemble response --
        progress(90, "Assembling forecast arrays…")
        hist_cut   = cutoff_date-timedelta(days=180)
        historical = [{"date":r["ds"].strftime("%Y-%m-%d"),"price":round(float(r["y"]),2)}
                       for _,r in prophet_df[prophet_df["ds"]>=hist_cut].iterrows()]
        in_sample  = [{"date":r["ds"].strftime("%Y-%m-%d"),"yhat":round(float(r["yhat"]),2)}
                       for _,r in final_forecast[
                           (final_forecast["ds"]>hist_cut)&
                           (final_forecast["ds"]<=cutoff_date)].iterrows()]
        forecast   = [{"date":r["ds"].strftime("%Y-%m-%d"),
                        "yhat": round(float(r["yhat"]),2),
                        "lower":round(float(r["yhat_lower"]),2),
                        "upper":round(float(r["yhat_upper"]),2)}
                       for _,r in future_only.iterrows()]

        # NOTE: forecast_flat was computed from the smooth Prophet forecast
        # before volatility injection (see above), so it still correctly
        # reflects whether Prophet's trend is essentially flat.

        target_pred = None
        if forecast:
            target_pred = min(forecast, key=lambda x:
                abs(datetime.strptime(x["date"],"%Y-%m-%d").date()-target_dt))

        checkpoints = []
        for d in [7,14,30,60,90]:
            if d<=forecast_days and forecast:
                r   = forecast[min(d-1, len(forecast)-1)]
                pct = round((r["yhat"]-current_price)/current_price*100, 2)
                checkpoints.append({"days":d,"date":r["date"],"price":r["yhat"],
                                     "lower":r["lower"],"upper":r["upper"],
                                     "change_pct":pct,"direction":"▲" if pct>=0 else "▼"})

        ens_table = []
        if ens_sigs is not None:
            ens_table_cols = ["date", "xgb_prob", "lstm_prob", "ensemble_prob", "signal", "strength"]
            if "bullish_regime" in ens_sigs.columns:
                ens_table_cols.append("bullish_regime")
            ens_table = (ens_sigs[ens_table_cols]
                         .tail(15).assign(date=lambda df: df["date"].astype(str))
                         .to_dict("records"))

        display_confidence = _display_signal_confidence(
            current_signal, current_entry_probability, signal_source, current_strength
        )
        signal_reason = _build_signal_reason(
            current_signal, current_strength, current_entry_probability, signal_source
        )

        result = {
            "symbol": symbol.upper(), "yahoo_symbol": yahoo_symbol,
            "target_date": target_date, "forecast_days": forecast_days,
            "current_price": round(current_price,2),
            "signal": current_signal, "signal_strength": current_strength,
            "signal_display_label": display_adjustment.get("signal_display_label"),
            "raw_ml_signal": raw_ml_signal,
            "raw_ml_signal_strength": raw_ml_signal_strength,
            "signal_adjusted_by_regime": bool(display_adjustment.get("signal_adjusted_by_regime", False)),
            "regime_adjustment_reason": display_adjustment.get("regime_adjustment_reason"),
            "signal_confidence": None if display_confidence is None else round(display_confidence, 3),
            "entry_probability": None if current_entry_probability is None else round(float(current_entry_probability), 3),
            "signal_source": signal_source,
            "signal_reason": signal_reason,
            "prophet_direction_signal": prophet_direction_signal,
            "prophet_direction_change_pct": prophet_direction_change_pct,
            "prophet_gate_quality": prophet_gate_quality,
            "target_pred": target_pred,
            "historical": historical, "in_sample_fit": in_sample,
            "forecast": forecast, "checkpoints": checkpoints,
            "forecast_flat": forecast_flat,
            "forecast_range_pct": forecast_range_pct,
            "prophet_metrics": {k:v for k,v in prophet_val.items() if k!="backtest_series"},
            "xgb_metrics":  xgb_result["metrics"],
            "lstm_metrics": lstm_result["metrics"],
            # Backward-compatible default: Trend Investor mode.
            "backtest": {
                "prophet":  {k:v for k,v in prophet_bt.items()  if k!="portfolio_timeline"},
                "xgboost":  {k:v for k,v in xgb_bt.items()      if k!="portfolio_timeline"} if xgb_bt  else {},
                "lstm":     {k:v for k,v in lstm_bt.items()      if k!="portfolio_timeline"} if lstm_bt  else {},
                "ensemble": {k:v for k,v in ens_bt.items()       if k!="portfolio_timeline"} if ens_bt  else {},
                "buy_hold_return": (xgb_bt or lstm_bt or ens_bt or prophet_bt or {}).get("bh_total_return"),
                "mode": "trend_investor",
            },
            "backtest_timelines": {
                "prophet":  prophet_bt.get("portfolio_timeline",[]),
                "xgboost":  xgb_bt.get("portfolio_timeline",[])  if xgb_bt  else [],
                "lstm":     lstm_bt.get("portfolio_timeline",[])  if lstm_bt  else [],
                "ensemble": ens_bt.get("portfolio_timeline",[])   if ens_bt  else [],
            },
            "signal_diagnostics": {
                "prophet": prophet_diag,
                "xgboost": xgb_diag,
                "lstm": lstm_diag,
                "ensemble": ens_diag,
            },

            # Two-mode backtest payload for the dashboard toggle.
            "backtest_modes": {
                "trend_investor": {
                    "prophet":  {k:v for k,v in prophet_bt.items()  if k!="portfolio_timeline"},
                    "xgboost":  {k:v for k,v in xgb_bt.items()      if k!="portfolio_timeline"} if xgb_bt  else {},
                    "lstm":     {k:v for k,v in lstm_bt.items()      if k!="portfolio_timeline"} if lstm_bt  else {},
                    "ensemble": {k:v for k,v in ens_bt.items()       if k!="portfolio_timeline"} if ens_bt  else {},
                    "buy_hold_return": (xgb_bt or lstm_bt or ens_bt or prophet_bt or {}).get("bh_total_return"),
                    "mode": "trend_investor",
                },
                # Backward-compatible alias for older dashboard code.
                "long_only": {
                    "prophet":  {k:v for k,v in prophet_bt.items()  if k!="portfolio_timeline"},
                    "xgboost":  {k:v for k,v in xgb_bt.items()      if k!="portfolio_timeline"} if xgb_bt  else {},
                    "lstm":     {k:v for k,v in lstm_bt.items()      if k!="portfolio_timeline"} if lstm_bt  else {},
                    "ensemble": {k:v for k,v in ens_bt.items()       if k!="portfolio_timeline"} if ens_bt  else {},
                    "buy_hold_return": (xgb_bt or lstm_bt or ens_bt or prophet_bt or {}).get("bh_total_return"),
                    "mode": "trend_investor",
                },
                "long_short": {
                    "prophet":  {k:v for k,v in prophet_bt_long_short.items()  if k!="portfolio_timeline"},
                    "xgboost":  {k:v for k,v in xgb_bt_long_short.items()      if k!="portfolio_timeline"} if xgb_bt_long_short  else {},
                    "lstm":     {k:v for k,v in lstm_bt_long_short.items()      if k!="portfolio_timeline"} if lstm_bt_long_short  else {},
                    "ensemble": {k:v for k,v in ens_bt_long_short.items()       if k!="portfolio_timeline"} if ens_bt_long_short  else {},
                    "buy_hold_return": (xgb_bt_long_short or lstm_bt_long_short or ens_bt_long_short or prophet_bt_long_short or {}).get("bh_total_return"),
                    "mode": "long_short",
                },
            },
            "backtest_timelines_modes": {
                "trend_investor": {
                    "prophet":  prophet_bt.get("portfolio_timeline",[]),
                    "xgboost":  xgb_bt.get("portfolio_timeline",[])  if xgb_bt  else [],
                    "lstm":     lstm_bt.get("portfolio_timeline",[])  if lstm_bt  else [],
                    "ensemble": ens_bt.get("portfolio_timeline",[])   if ens_bt  else [],
                },
                "long_only": {
                    "prophet":  prophet_bt.get("portfolio_timeline",[]),
                    "xgboost":  xgb_bt.get("portfolio_timeline",[])  if xgb_bt  else [],
                    "lstm":     lstm_bt.get("portfolio_timeline",[])  if lstm_bt  else [],
                    "ensemble": ens_bt.get("portfolio_timeline",[])   if ens_bt  else [],
                },
                "long_short": {
                    "prophet":  prophet_bt_long_short.get("portfolio_timeline",[]),
                    "xgboost":  xgb_bt_long_short.get("portfolio_timeline",[])  if xgb_bt_long_short  else [],
                    "lstm":     lstm_bt_long_short.get("portfolio_timeline",[])  if lstm_bt_long_short  else [],
                    "ensemble": ens_bt_long_short.get("portfolio_timeline",[])   if ens_bt_long_short  else [],
                },
            },
            "signal_diagnostics_modes": {
                "trend_investor": {
                    "prophet": prophet_diag,
                    "xgboost": xgb_diag,
                    "lstm": lstm_diag,
                    "ensemble": ens_diag,
                },
                "long_only": {
                    "prophet": prophet_diag,
                    "xgboost": xgb_diag,
                    "lstm": lstm_diag,
                    "ensemble": ens_diag,
                },
                "long_short": {
                    "prophet": prophet_diag_long_short,
                    "xgboost": xgb_diag_long_short,
                    "lstm": lstm_diag_long_short,
                    "ensemble": ens_diag_long_short,
                },
            },
            "strategy_settings": {
                "entry_threshold": round(CONFIDENCE_THRESHOLD, 3),
                "exit_threshold": round(EXIT_PROB_THRESHOLD, 3),
                "stop_loss_pct": round(STOP_LOSS_PCT * 100, 2),
                "take_profit_pct": round(TAKE_PROFIT_PCT * 100, 2),
                "max_hold_days": int(MAX_HOLD_DAYS),
                "preferred_backtest_days": int(PREFERRED_BACKTEST_DAYS),
                "trend_investor_exit_confirm_days": int(TREND_INVESTOR_EXIT_CONFIRM_DAYS),
                "prophet_hard_gate_enabled": bool(prophet_gate_quality.get("hard_gate_enabled", True)),
                "prophet_gate_quality_reason": prophet_gate_quality.get("reason"),
                "bullish_regime_filter_enabled": bool(BULL_REGIME_FILTER_ENABLED),
                "bullish_regime_sma_window": int(BULL_REGIME_SMA_WINDOW),
                "bullish_regime_momentum_window": int(BULL_REGIME_MOM_WINDOW),
                "bullish_regime_momentum_pct": round(BULL_REGIME_MOM_PCT * 100, 2),
                "bullish_regime_fast_momentum_pct": round(BULL_REGIME_FAST_MOM_PCT * 100, 2),
            },
            "market_regime": market_regime,
            "ensemble_table": ens_table,
            "prophet_backtest_series": prophet_val.get("backtest_series",[]),
            "lstm_reliable":       lstm_result.get("lstm_reliable",False),
            "total_training_days": total_days,
            "test_days": TEST_DAYS, "ml_lookback_years": ML_LOOKBACK_YEARS,
            "entry_threshold": ENSEMBLE_THRESHOLD,
            "xgb_weight": XGB_WEIGHT, "lstm_weight": LSTM_WEIGHT,
            "train_cutoff": train_df["ds"].max().strftime("%Y-%m-%d"),
            "holiday_source": "pandas_market_calendars (NSE auto)"
                               if len(INDIAN_HOLIDAYS)>100 else "hardcoded fallback",
        }

        progress(100, "Done!")
        _update_job(job_id, status="done", progress_pct=100,
                    progress_msg="Prediction complete!", result=result)

        # -- Persist result in the 1-hour result cache --
        # Future requests for the same (symbol, target_date) within 1 hour
        # will be served directly from Redis without re-running the pipeline.
        _cache_set(symbol, target_date, result)

    except Exception as exc:
        logger.exception(f"[PREDICT {job_id}] Pipeline error: {exc}")
        _update_job(job_id, status="error", progress_msg=str(exc), error=str(exc))

# -- API endpoints --

@router.post("/{symbol}", status_code=202)
async def start_prediction(
    symbol      : str,
    target_date : str = Query(..., description="Target date YYYY-MM-DD (future only)"),
    db          : Session = Depends(get_db),
):
    """
    Start a background prediction job.
    Returns immediately with { job_id, status: 'queued' }.
    Poll GET /api/predict/status/{job_id} to track progress.
    """
    try:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "Invalid date format — use YYYY-MM-DD.")
    if target_dt <= date.today():
        raise HTTPException(400, "target_date must be a future date.")
    forecast_days = (target_dt - date.today()).days
    if forecast_days > 365:
        raise HTTPException(400, "Target date must be within 365 days from today.")

    yahoo_symbol = resolve_yahoo_symbol(symbol, db)
    _require_job_store()

    # -- Result-cache fast path --
    # If an identical (symbol, target_date) request completed within the
    # last hour, synthesise a pre-filled "done" job in Redis and return
    # it immediately — no ML pipeline needed.
    cached_result = _cache_get(symbol.upper(), target_date)
    if cached_result is not None:
        cache_job_id  = str(uuid.uuid4())
        ttl_remaining = _cache_ttl_remaining(symbol.upper(), target_date) or RESULT_CACHE_TTL_SECONDS
        cached_job    = {
            **_new_job(cache_job_id, symbol.upper(), target_date),
            "status"      : "done",
            "progress_pct": 100,
            "progress_msg": "Served from cache.",
            "result"      : cached_result,
            "cached"      : True,
            "cache_ttl_remaining_seconds": ttl_remaining,
        }
        with _JOB_LOCK:
            _save_job(cache_job_id, cached_job)
        logger.info(
            f"[PREDICT] Cache HIT for {symbol.upper()}/{target_date} "
            f"(TTL remaining: {ttl_remaining}s) → synthetic job {cache_job_id}"
        )
        return {
            "job_id" : cache_job_id,
            "status" : "done",
            "cached" : True,
            "cache_ttl_remaining_seconds": ttl_remaining,
            "message": "Result served from cache. "
                       "Poll /api/predict/status/{job_id} to retrieve it.",
        }

    # -- Cache miss — run the full pipeline --
    job_id = str(uuid.uuid4())
    with _JOB_LOCK:
        _save_job(job_id, _new_job(job_id, symbol.upper(), target_date))
    _evict_old_jobs()

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, symbol.upper(), target_date, yahoo_symbol, forecast_days),
        daemon=True,
        name=f"predict-{job_id[:8]}",
    )
    thread.start()

    return {"job_id": job_id, "status": "queued",
            "cached": False,
            "message": "Prediction started. Poll /api/predict/status/{job_id} for updates."}


@router.get("/status/{job_id}")
async def get_prediction_status(job_id: str):
    """
    Poll prediction job status.

    Response shape:
      { job_id, status, progress_pct, progress_msg, result?, error? }

    status values:
      "queued"  — waiting to start
      "running" — pipeline executing
      "done"    — result is populated
      "error"   — error message is populated
    """
    with _JOB_LOCK:
        job = _load_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' not found. "
                            "Jobs are cleared on server restart.")
    return {
        "job_id"      : job["job_id"],
        "symbol"      : job["symbol"],
        "target_date" : job["target_date"],
        "status"      : job["status"],
        "progress_pct": job["progress_pct"],
        "progress_msg": job["progress_msg"],
        "result"      : job["result"],
        "error"       : job["error"],
        # Cache metadata — present only on cache-hit synthetic jobs
        "cached"      : job.get("cached", False),
        "cache_ttl_remaining_seconds": job.get("cache_ttl_remaining_seconds"),
    }


@router.delete("/cancel/{job_id}", status_code=200)
async def cancel_prediction(job_id: str):
    """Request cancellation of a running prediction job."""
    with _JOB_LOCK:
        job = _load_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    if job["status"] in ("done", "error"):
        return {"message": f"Job already {job['status']} — nothing to cancel."}
    _update_job(job_id, _cancel=True, status="error",
                progress_msg="Cancelled by user.", error="Cancelled by user.")
    return {"message": "Cancellation requested.", "job_id": job_id}