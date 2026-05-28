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
    return f"predict_result:v4:{symbol.upper()}:{target_date}"

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

FLAT_TREND_PCT       = float(os.getenv("PREDICTION_FLAT_TREND_PCT", "0.002"))
LSTM_MIN_AUC         = float(os.getenv("PREDICTION_LSTM_MIN_AUC", "0.58"))
LSTM_MIN_ACCURACY    = float(os.getenv("PREDICTION_LSTM_MIN_ACCURACY", "55.0"))

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
    if signal in {"BUY", "SELL"} and entry_prob is not None:
        return f"Active timing probability: {float(entry_prob) * 100:.1f}%"
    return ""


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
    return {"signals": sigs, "metrics": {"accuracy": round(acc,1),
                                          "roc_auc":  round(auc,4)}}

# -- LSTM pipeline --
def run_lstm_pipeline(xgb_df: pd.DataFrame, test_days: int,
                      honest_fc: pd.DataFrame) -> dict:
    """Train LSTM, return signals + metrics. Gracefully skips if torch absent."""
    try:
        import torch, torch.nn as nn, torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, roc_auc_score
    except ImportError as e:
        logger.warning(f"[PREDICT] torch/sklearn unavailable: {e}")
        return {"signals": None, "metrics": {"accuracy": None, "roc_auc": None},
                "lstm_reliable": False}

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_cols = [c for c in LSTM_FEATURE_COLS if c in xgb_df.columns]
    split     = len(xgb_df) - test_days

    tr_raw = xgb_df.iloc[:split][feat_cols].values
    te_raw = xgb_df.iloc[split:][feat_cols].values
    tr_tgt = xgb_df.iloc[:split]["target"].values
    te_tgt = xgb_df.iloc[split:]["target"].values

    scaler   = StandardScaler()
    tr_sc    = scaler.fit_transform(tr_raw); te_sc = scaler.transform(te_raw)
    comb_sc  = np.vstack([tr_sc, te_sc])
    comb_tgt = np.concatenate([tr_tgt, te_tgt])

    SEQ = SEQUENCE_LENGTH
    def make_seq(feats, tgts, L):
        X, y = [], []
        for i in range(L, len(feats)): X.append(feats[i-L:i]); y.append(tgts[i])
        return np.array(X,dtype=np.float32), np.array(y,dtype=np.float32)

    X_tr, y_tr   = make_seq(tr_sc, tr_tgt, SEQ)
    X_all, y_all = make_seq(comb_sc, comb_tgt, SEQ)
    X_te = X_all[-test_days:]; y_te = y_all[-test_days:]

    lstm_reliable = len(X_tr) >= 500
    vsplit  = int(len(X_tr)*0.85)
    X_val, y_val = X_tr[vsplit:], y_tr[vsplit:]
    X_tr,  y_tr  = X_tr[:vsplit], y_tr[:vsplit]
    n_feat  = len(feat_cols)

    class LSTMNet(nn.Module):
        def __init__(self, nf, h=64, layers=2, drop=0.4):
            super().__init__()
            self.lstm = nn.LSTM(nf, h, layers, batch_first=True,
                                dropout=drop if layers>1 else 0)
            self.fc   = nn.Sequential(
                nn.Linear(h,64),nn.ReLU(),nn.Dropout(drop),
                nn.Linear(64,32),nn.ReLU(),nn.Dropout(drop/2),nn.Linear(32,1))
        def forward(self, x):
            o, _ = self.lstm(x); return self.fc(o[:,-1,:]).squeeze(1)
        def predict_proba(self, x):
            with torch.no_grad(): return torch.sigmoid(self.forward(x))

    net  = LSTMNet(n_feat).to(device)
    posw = torch.tensor([(y_tr==0).sum()/max((y_tr==1).sum(),1)],
                         dtype=torch.float32).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=posw)
    opt  = optim.Adam(net.parameters(), lr=0.0005, weight_decay=1e-4)
    sch  = optim.lr_scheduler.ReduceLROnPlateau(opt,"min",patience=7,factor=0.5)

    class DS(Dataset):
        def __init__(self,X,y): self.X=torch.tensor(X); self.y=torch.tensor(y)
        def __len__(self): return len(self.X)
        def __getitem__(self,i): return self.X[i], self.y[i]

    tr_ld = DataLoader(DS(X_tr, y_tr),   batch_size=64, shuffle=True)   # shuffle=True for better generalisation
    va_ld = DataLoader(DS(X_val, y_val), batch_size=64, shuffle=False)
    te_ld = DataLoader(DS(X_te, y_te),   batch_size=64, shuffle=False)

    best_val, best_w, pat, PATIENCE = float("inf"), None, 0, 15
    for epoch in range(80):
        net.train()
        for Xb, yb in tr_ld:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(net(Xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        vl = float(np.mean([crit(net(Xb.to(device)), yb.to(device)).item()
                              for Xb, yb in va_ld]))
        sch.step(vl)
        if vl < best_val:
            best_val = vl; best_w = {k:v.clone() for k,v in net.state_dict().items()}; pat = 0
        else: pat += 1
        if pat >= PATIENCE: break

    net.load_state_dict(best_w); net.eval()
    probs_all, lbls_all = [], []
    with torch.no_grad():
        for Xb, yb in te_ld:
            probs_all.extend(net.predict_proba(Xb.to(device)).cpu().numpy())
            lbls_all.extend(yb.numpy().astype(int))

    all_probs = np.array(probs_all); all_lbls = np.array(lbls_all)
    acc = float(accuracy_score(all_lbls, (all_probs>=0.5).astype(int))*100)
    try:    auc = float(roc_auc_score(all_lbls, all_probs))
    except: auc = 0.5

    # Sequence count alone is not enough. A weak validation AUC can drag the
    # ensemble down, so mark LSTM reliable only when it clears performance gates.
    lstm_reliable = bool(
        lstm_reliable
        and auc >= LSTM_MIN_AUC
        and acc >= LSTM_MIN_ACCURACY
    )

    te_slice = xgb_df.iloc[split:].copy()
    sigs = _hierarchical_signals(all_probs, te_slice["date"].values,
                                  honest_fc, te_slice["Close"].values)
    return {"signals": sigs, "metrics": {"accuracy": round(acc,1),
                                          "roc_auc":  round(auc,4)},
            "lstm_reliable": lstm_reliable}

# -- Ensemble --

def generate_ensemble_signals(xgb_sigs: pd.DataFrame,
                               lstm_sigs: pd.DataFrame,
                               honest_fc: pd.DataFrame,
                               lstm_auc: float = 0.5,
                               lstm_reliable: bool = False) -> pd.DataFrame:
    xgb = xgb_sigs[["date","Close","prob_good_entry","prophet_uptrend"]].copy()
    xgb.columns = ["date","Close","xgb_prob","prophet_uptrend"]
    lst = lstm_sigs[["date","prob_good_entry"]].rename(
        columns={"prob_good_entry":"lstm_prob"})
    df  = xgb.merge(lst, on="date", how="inner")

    # Adaptive weighting: when LSTM validation is weak, reduce its influence.
    # The caller may also choose XGBoost-only when LSTM is below reliability gates.
    if not lstm_reliable or lstm_auc < 0.60:
        eff_xgb  = min(XGB_WEIGHT + 0.15, 0.85)
        eff_lstm = 1.0 - eff_xgb
        logger.info(
            f"[PREDICT] LSTM AUC={lstm_auc:.4f} unreliable/weak — "
            f"weights adjusted to XGB={eff_xgb:.2f} LSTM={eff_lstm:.2f}"
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

        if position == "OUT":
            if prob >= entry_threshold and trend == "up":
                sig = "BUY"
                reason = "ENTRY_PROB_AND_UPTREND"
                position = "IN"
                entry_price = px
                entry_date = dt
            elif prob >= entry_threshold and trend == "flat":
                reason = "ENTRY_BLOCKED_FLAT_TREND"
            elif prob >= entry_threshold and trend == "down":
                reason = "ENTRY_BLOCKED_DOWNTREND"
            else:
                reason = "LOW_ENTRY_PROBABILITY"
        else:
            hold_days = int((dt - entry_date).days) if entry_date is not None else 0
            ret = (px / entry_price - 1.0) if entry_price else 0.0

            if trend != "up":
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


def signal_diagnostics(signals_df: pd.DataFrame,
                       prob_col: Optional[str] = None,
                       backtest_result: Optional[dict] = None) -> dict:
    """Small, UI-friendly explanation of why a model did or did not trade."""
    if signals_df is None or signals_df.empty:
        return {
            "days": 0,
            "high_probability_days": 0,
            "exit_probability_days": 0,
            "low_probability_days": 0,
            "flat_gate_days": 0,
            "uptrend_days": 0,
            "downtrend_days": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "completed_trades": 0,
            "entry_threshold": round(CONFIDENCE_THRESHOLD, 3),
            "exit_threshold": round(EXIT_PROB_THRESHOLD, 3),
            "message": "No signal data available",
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

    bt = backtest_result or {}
    buy_signals = int((sigs == "BUY").sum())
    sell_signals = int((sigs == "SELL").sum())
    completed = int(bt.get("n_trades", 0) or 0)

    if completed == 0 and high_probability_days == 0:
        message = "No entries: probability never crossed the entry threshold."
    elif completed == 0 and buy_signals == 0 and flat_gate_days > 0:
        message = "Entries were blocked because Prophet trend was flat."
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
        xgb_strategy_sigs = None
        xgb_diag   = {}
        xgb_df_feat= None
        try:
            import xgboost  # noqa — check availability
            progress(48, "Engineering XGBoost features (52 indicators + Prophet context)…")
            xgb_df_feat = build_xgb_features(data_df_ml.copy(), honest_fc)
            if len(xgb_df_feat) >= TEST_DAYS+50:
                if cancelled(): return
                progress(54, "Training XGBoost classifier (feature selection + fit)…")
                xgb_result = run_xgboost_pipeline(xgb_df_feat, TEST_DAYS, honest_fc)
                xgb_strategy_sigs = build_entry_exit_strategy_signals(
                    xgb_result["signals"],
                    prob_col="prob_good_entry",
                    model_name="XGBoost",
                )
                xgb_bt = run_backtest(xgb_strategy_sigs)
                xgb_diag = signal_diagnostics(xgb_strategy_sigs, "entry_prob", xgb_bt)
        except Exception as e:
            logger.warning(f"[PREDICT {job_id}] XGBoost error (non-fatal): {e}")
        if cancelled(): return

        # -- LSTM --
        lstm_result = {"signals":None,"metrics":{"accuracy":None,"roc_auc":None},"lstm_reliable":False}
        lstm_bt     = None
        lstm_strategy_sigs = None
        lstm_diag   = {}
        try:
            import torch  # noqa — check availability
            if xgb_df_feat is not None and len(xgb_df_feat)>=SEQUENCE_LENGTH+TEST_DAYS+20:
                if cancelled(): return
                progress(62, "Training LSTM (2-layer, early-stopping, up to 80 epochs)…")
                lstm_result = run_lstm_pipeline(xgb_df_feat, TEST_DAYS, honest_fc)
                if lstm_result["signals"] is not None:
                    lstm_strategy_sigs = build_entry_exit_strategy_signals(
                        lstm_result["signals"],
                        prob_col="prob_good_entry",
                        model_name="LSTM",
                    )
                    lstm_bt = run_backtest(lstm_strategy_sigs)
                    lstm_diag = signal_diagnostics(lstm_strategy_sigs, "entry_prob", lstm_bt)
        except Exception as e:
            logger.warning(f"[PREDICT {job_id}] LSTM error (non-fatal): {e}")
        if cancelled(): return

        # -- Ensemble --
        progress(78, "Combining XGBoost + LSTM into Ensemble (tiered signals)…")
        ens_sigs         = None; ens_bt = None
        ens_strategy_sigs = None
        ens_diag         = {}
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
                ens_strategy_sigs = build_entry_exit_strategy_signals(
                    ens_sigs,
                    prob_col="ensemble_prob",
                    model_name="Ensemble",
                )
                ens_bt = run_backtest(ens_strategy_sigs)
                ens_diag = signal_diagnostics(ens_strategy_sigs, "entry_prob", ens_bt)

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

        # -- Prophet-only backtest --
        progress(84, "Running Prophet strategy backtest…")
        prophet_bt: dict = {}
        prophet_diag: dict = {}
        if prophet_val.get("backtest_series"):
            prophet_strategy_sigs = build_prophet_strategy_signals(prophet_val["backtest_series"])
            prophet_bt = run_backtest(prophet_strategy_sigs)
            prophet_diag = signal_diagnostics(prophet_strategy_sigs, None, prophet_bt)

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
            ens_table = (ens_sigs[["date","xgb_prob","lstm_prob","ensemble_prob","signal","strength"]]
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
            "signal_confidence": None if display_confidence is None else round(display_confidence, 3),
            "entry_probability": None if current_entry_probability is None else round(float(current_entry_probability), 3),
            "signal_source": signal_source,
            "signal_reason": signal_reason,
            "prophet_direction_signal": prophet_direction_signal,
            "prophet_direction_change_pct": prophet_direction_change_pct,
            "target_pred": target_pred,
            "historical": historical, "in_sample_fit": in_sample,
            "forecast": forecast, "checkpoints": checkpoints,
            "forecast_flat": forecast_flat,
            "forecast_range_pct": forecast_range_pct,
            "prophet_metrics": {k:v for k,v in prophet_val.items() if k!="backtest_series"},
            "xgb_metrics":  xgb_result["metrics"],
            "lstm_metrics": lstm_result["metrics"],
            "backtest": {
                "prophet":  {k:v for k,v in prophet_bt.items()  if k!="portfolio_timeline"},
                "xgboost":  {k:v for k,v in xgb_bt.items()      if k!="portfolio_timeline"} if xgb_bt  else {},
                "lstm":     {k:v for k,v in lstm_bt.items()      if k!="portfolio_timeline"} if lstm_bt  else {},
                "ensemble": {k:v for k,v in ens_bt.items()       if k!="portfolio_timeline"} if ens_bt  else {},
                "buy_hold_return": (xgb_bt or lstm_bt or ens_bt or prophet_bt or {}).get("bh_total_return"),
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
            "strategy_settings": {
                "entry_threshold": round(CONFIDENCE_THRESHOLD, 3),
                "exit_threshold": round(EXIT_PROB_THRESHOLD, 3),
                "stop_loss_pct": round(STOP_LOSS_PCT * 100, 2),
                "take_profit_pct": round(TAKE_PROFIT_PCT * 100, 2),
                "max_hold_days": int(MAX_HOLD_DAYS),
                "preferred_backtest_days": int(PREFERRED_BACKTEST_DAYS),
            },
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