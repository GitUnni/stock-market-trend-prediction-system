"""
Global LSTM service for the prediction module.

Place this file at:
    app/ml/global_lstm_service.py

Purpose
-------
This replaces the older request-time / single-stock LSTM approach with a saved
Global LSTM trained across a large NSE/Nifty-style universe.

Training should be done separately by running:
    python -m app.ml.train_global_lstm

Prediction code should only load the saved artifact from:
    app/static/models/global_lstm/

The model is NOT blindly deleted after 3 days. Instead, when retrained, the new
candidate replaces the old model only if validation quality is better. This
prevents accidentally replacing a decent model with a worse one.
"""

from __future__ import annotations

import io
import json
import math
import os
import shutil
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# sklearn is already used by the prediction pipeline.
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score


# ─────────────────────────────────────────────────────────────────────
# Paths / persistence
# ─────────────────────────────────────────────────────────────────────
APP_ROOT = Path(os.getenv("PREDICTION_APP_ROOT", ".")).resolve()
MODEL_ROOT = Path(os.getenv("PREDICTION_MODEL_ROOT", "app/static/models")).resolve()
GLOBAL_LSTM_DIR = Path(os.getenv("PREDICTION_GLOBAL_LSTM_DIR", str(MODEL_ROOT / "global_lstm"))).resolve()

MODEL_FILE = GLOBAL_LSTM_DIR / "model.pt"
SCALER_FILE = GLOBAL_LSTM_DIR / "scaler.npz"
THRESHOLDS_FILE = GLOBAL_LSTM_DIR / "thresholds.json"
FEATURES_FILE = GLOBAL_LSTM_DIR / "feature_columns.json"
METADATA_FILE = GLOBAL_LSTM_DIR / "metadata.json"

MODEL_MAX_AGE_DAYS = int(os.getenv("PREDICTION_GLOBAL_LSTM_MAX_AGE_DAYS", "3"))
MODEL_MAX_AGE_DAYS = max(1, min(30, MODEL_MAX_AGE_DAYS))

# Important: by default prediction does NOT train inside the request.
# Set to 1 only if you accept slow first prediction calls.
AUTO_TRAIN_IF_MISSING = os.getenv("PREDICTION_AUTO_TRAIN_GLOBAL_LSTM", "0").strip().lower() in {"1", "true", "yes", "on"}


# ─────────────────────────────────────────────────────────────────────
# Training configuration
# ─────────────────────────────────────────────────────────────────────
MARKET_TICKER = os.getenv("PREDICTION_MARKET_TICKER", "^NSEI")
DATA_START_DATE = os.getenv("PREDICTION_GLOBAL_LSTM_START_DATE", "2014-01-01")
SEQUENCE_LENGTH = int(os.getenv("PREDICTION_GLOBAL_LSTM_SEQUENCE_LENGTH", "45"))
LABEL_HORIZON = int(os.getenv("PREDICTION_GLOBAL_LSTM_LABEL_HORIZON", "15"))
MAX_TICKERS = int(os.getenv("PREDICTION_GLOBAL_LSTM_MAX_TICKERS", "160"))
MAX_TRAIN_SAMPLES = int(os.getenv("PREDICTION_GLOBAL_LSTM_MAX_TRAIN_SAMPLES", "120000"))
MAX_VAL_SAMPLES = int(os.getenv("PREDICTION_GLOBAL_LSTM_MAX_VAL_SAMPLES", "30000"))
EPOCHS = int(os.getenv("PREDICTION_GLOBAL_LSTM_EPOCHS", "45"))
PATIENCE = int(os.getenv("PREDICTION_GLOBAL_LSTM_PATIENCE", "8"))
BATCH_SIZE = int(os.getenv("PREDICTION_GLOBAL_LSTM_BATCH_SIZE", "256"))
WALK_FOLDS = int(os.getenv("PREDICTION_GLOBAL_LSTM_WALK_FOLDS", "2"))
VOL_MULT = float(os.getenv("PREDICTION_GLOBAL_LSTM_VOL_MULT", "0.85"))
MIN_MOVE_PCT = float(os.getenv("PREDICTION_GLOBAL_LSTM_MIN_MOVE_PCT", "0.018"))
MAX_MOVE_PCT = float(os.getenv("PREDICTION_GLOBAL_LSTM_MAX_MOVE_PCT", "0.12"))
RANDOM_SEED = int(os.getenv("PREDICTION_GLOBAL_LSTM_SEED", "42"))

SEQUENCE_LENGTH = max(15, min(90, SEQUENCE_LENGTH))
LABEL_HORIZON = max(5, min(45, LABEL_HORIZON))
MAX_TICKERS = max(20, min(500, MAX_TICKERS))
EPOCHS = max(10, min(160, EPOCHS))
PATIENCE = max(3, min(30, PATIENCE))
WALK_FOLDS = max(1, min(4, WALK_FOLDS))
MIN_MOVE_PCT = max(0.005, min(0.10, MIN_MOVE_PCT))
MAX_MOVE_PCT = max(MIN_MOVE_PCT, min(0.40, MAX_MOVE_PCT))
VOL_MULT = max(0.15, min(3.0, VOL_MULT))

# Reliability gates used by predict.py to decide whether LSTM should influence
# the ensemble. These are intentionally conservative.
MIN_ACTIVE_AUC_FOR_ENSEMBLE = float(os.getenv("PREDICTION_GLOBAL_LSTM_MIN_ACTIVE_AUC", "0.58"))
MIN_BALANCED_ACC_FOR_ENSEMBLE = float(os.getenv("PREDICTION_GLOBAL_LSTM_MIN_BALANCED_ACC", "52.0"))
MIN_DIRECTION_ACC_FOR_ENSEMBLE = float(os.getenv("PREDICTION_GLOBAL_LSTM_MIN_DIRECTION_ACC", "53.0"))


# ─────────────────────────────────────────────────────────────────────
# Universe
# ─────────────────────────────────────────────────────────────────────
# Built-in fallback Nifty 50 universe. The service now tries to download the
# current official Nifty 50 constituents CSV at runtime. This fallback is used
# only if NSE/niftyindices is temporarily unavailable.
#
# Why this matters:
# - Symbols like TATAMOTORS.NS and PEL.NS can become invalid after index changes,
#   delistings, renames or demergers.
# - The live CSV keeps the training universe aligned with the current index.
NIFTY50_YAHOO_SYMBOLS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDIGO.NS", "INFY.NS", "ITC.NS",
    "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "MAXHEALTH.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SHRIRAMFIN.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TATACONSUM.NS", "TMPV.NS", "TATASTEEL.NS", "TCS.NS",
    "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS",
]

# Official CSV endpoints used for live constituent refresh. NSE occasionally
# changes hostnames, so keep multiple mirrors.
NIFTY50_CSV_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
]

NIFTY50_LIVE_REFRESH_ENABLED = os.getenv(
    "PREDICTION_GLOBAL_LSTM_LIVE_NIFTY50", "1"
).strip().lower() not in {"0", "false", "no", "off"}

INCLUDE_EXTRA_LIQUID_SYMBOLS = os.getenv(
    "PREDICTION_GLOBAL_LSTM_INCLUDE_EXTRAS", "0"
).strip().lower() in {"1", "true", "yes", "on"}

# Known Yahoo/NSE symbol changes or common user aliases. These are intentionally
# conservative; official CSV symbols still take priority.
YAHOO_SYMBOL_ALIASES = {
    "TATAMOTORS": "TMPV.NS",          # Tata Motors PV after demerger / Yahoo replacement
    "TATAMOTORS.NS": "TMPV.NS",
    "ZOMATO": "ETERNAL.NS",
    "ZOMATO.NS": "ETERNAL.NS",
    "MCDOWELL-N": "UNITDSPR.NS",
    "MCDOWELL-N.NS": "UNITDSPR.NS",
    "GMRINFRA": "GMRAIRPORT.NS",
    "GMRINFRA.NS": "GMRAIRPORT.NS",
    "PEL": "PIRAMAL.NS",
    "PEL.NS": "PIRAMAL.NS",
}

_NIFTY50_LIVE_CACHE: Optional[List[str]] = None

EXTRA_LIQUID_NSE_SYMBOLS = [
    "ABB.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", "AIAENG.NS", "ALKEM.NS", "AMBUJACEM.NS",
    "ASHOKLEY.NS", "ASTRAL.NS", "AUROPHARMA.NS", "AUBANK.NS", "BANDHANBNK.NS", "BANKBARODA.NS",
    "BANKINDIA.NS", "BATAINDIA.NS", "BERGEPAINT.NS", "BHARATFORG.NS", "BHEL.NS", "BIOCON.NS",
    "BOSCHLTD.NS", "BPCL.NS", "BRITANNIA.NS", "CANBK.NS", "CHOLAFIN.NS", "COFORGE.NS",
    "COLPAL.NS", "CONCOR.NS", "CROMPTON.NS", "CUMMINSIND.NS", "DABUR.NS", "DALBHARAT.NS",
    "DEEPAKNTR.NS", "DIVISLAB.NS", "DLF.NS", "DIXON.NS", "FEDERALBNK.NS", "GAIL.NS",
    "GLAND.NS", "GLAXO.NS", "GMRINFRA.NS", "GODREJCP.NS", "GODREJPROP.NS", "HAL.NS",
    "HAVELLS.NS", "HDFCAMC.NS", "HINDPETRO.NS", "HONAUT.NS", "ICICIGI.NS", "ICICIPRULI.NS",
    "IDFCFIRSTB.NS", "IGL.NS", "INDHOTEL.NS", "INDIGO.NS", "INDUSTOWER.NS", "IOC.NS",
    "IPCALAB.NS", "IRCTC.NS", "JINDALSTEL.NS", "JUBLFOOD.NS", "LALPATHLAB.NS", "LAURUSLABS.NS",
    "LICHSGFIN.NS", "LODHA.NS", "LUPIN.NS", "MANAPPURAM.NS", "MCDOWELL-N.NS", "MFSL.NS",
    "MGL.NS", "MOTHERSON.NS", "MPHASIS.NS", "MRF.NS", "MUTHOOTFIN.NS", "NAM-INDIA.NS",
    "NAUKRI.NS", "NMDC.NS", "OBEROIRLTY.NS", "OFSS.NS", "PAGEIND.NS", "PEL.NS",
    "PERSISTENT.NS", "PETRONET.NS", "PIIND.NS", "PNB.NS", "POLYCAB.NS", "PVRINOX.NS",
    "RAMCOCEM.NS", "RBLBANK.NS", "RECLTD.NS", "SAIL.NS", "SBICARD.NS", "SIEMENS.NS",
    "SRF.NS", "SUPREMEIND.NS", "TATACHEM.NS", "TATACOMM.NS", "TATAELXSI.NS", "TATAPOWER.NS",
    "TORNTPHARM.NS", "TORNTPOWER.NS", "TVSMOTOR.NS", "UBL.NS", "UNIONBANK.NS", "VBL.NS",
    "VEDL.NS", "VOLTAS.NS", "ZEEL.NS", "ZYDUSLIFE.NS", "PAYTM.NS", "POLICYBZR.NS",
]

SECTOR_MAP: Dict[str, str] = {
    # Nifty 50 map
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
    # Common extras
    "RECLTD.NS":"FINANCE", "HAL.NS":"DEFENCE", "BHEL.NS":"CAPITAL_GOODS", "SIEMENS.NS":"CAPITAL_GOODS",
    "DIXON.NS":"CONSUMER", "PERSISTENT.NS":"IT", "COFORGE.NS":"IT", "MPHASIS.NS":"IT",
    "TORNTPHARM.NS":"PHARMA", "ZYDUSLIFE.NS":"PHARMA", "LUPIN.NS":"PHARMA", "AUROPHARMA.NS":"PHARMA",
    "TVSMOTOR.NS":"AUTO", "BHARATFORG.NS":"AUTO", "ASHOKLEY.NS":"AUTO", "FEDERALBNK.NS":"BANK",
    "BANKBARODA.NS":"BANK", "CANBK.NS":"BANK", "PNB.NS":"BANK", "IDFCFIRSTB.NS":"BANK",
    # Current/renamed Nifty constituents and common Yahoo replacements
    "ETERNAL.NS":"CONSUMER", "INDIGO.NS":"SERVICES", "JIOFIN.NS":"FINANCE",
    "MAXHEALTH.NS":"HEALTHCARE", "TMPV.NS":"AUTO", "TMCV.NS":"AUTO",
    "PIRAMAL.NS":"PHARMA", "UNITDSPR.NS":"CONSUMER", "GMRAIRPORT.NS":"INFRA",
}

FEATURE_COLUMNS = [
    "daily_return", "return_2d", "return_5d", "return_10d", "return_20d", "return_60d",
    "relative_return_5d", "relative_return_20d", "sector_relative_return_20d",
    "close_vs_sma20", "close_vs_sma50", "close_vs_sma100", "close_vs_sma200",
    "sma20_vs_sma50", "sma50_vs_sma200", "ema12_vs_ema50",
    "rsi_14", "macd_scaled", "macd_signal_scaled", "macd_hist_scaled",
    "atr_pct", "bb_width", "bb_pct", "volatility_20d", "volatility_60d", "volume_ratio",
    "nifty_return_1d", "nifty_return_5d", "nifty_return_20d", "nifty_close_vs_sma50",
    "nifty_close_vs_sma200", "nifty_volatility_20d", "sector_return_5d", "sector_return_20d",
    "day_of_week", "month_sin", "month_cos",
]


# ─────────────────────────────────────────────────────────────────────
# Small utilities
# ─────────────────────────────────────────────────────────────────────
def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def _log(msg: str, logger=None):
    if logger:
        try:
            logger.info(msg)
            return
        except Exception:
            pass
    print(msg)


def _warn(msg: str, logger=None):
    if logger:
        try:
            logger.warning(msg)
            return
        except Exception:
            pass
    print("[WARN]", msg)


def normalise_yahoo_symbol(symbol: str) -> str:
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return symbol
    symbol = YAHOO_SYMBOL_ALIASES.get(symbol, symbol)
    if symbol.startswith("^"):
        return symbol
    if symbol.endswith(".BO") or symbol.endswith(".NS"):
        return symbol
    if "." in symbol:
        return symbol
    return f"{symbol}.NS"


def _read_csv_from_url(url: str, timeout: int = 15) -> pd.DataFrame:
    """Read CSV with browser-like headers; NSE blocks plain Python clients sometimes."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        ),
        "Accept": "text/csv,application/csv,application/octet-stream,*/*;q=0.8",
        "Referer": "https://www.nseindia.com/",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return pd.read_csv(io.BytesIO(raw))


def fetch_current_nifty50_symbols(logger=None) -> List[str]:
    """Return current Nifty 50 Yahoo symbols from the official NSE/Nifty CSV.

    Falls back to the built-in list if the live source is unavailable. This
    prevents stale names such as TATAMOTORS.NS, PEL.NS, etc. from entering the
    default training universe.
    """
    global _NIFTY50_LIVE_CACHE

    if _NIFTY50_LIVE_CACHE:
        return list(_NIFTY50_LIVE_CACHE)

    if not NIFTY50_LIVE_REFRESH_ENABLED:
        _log("[GLOBAL_LSTM] Live Nifty 50 refresh disabled; using fallback list", logger)
        return list(NIFTY50_YAHOO_SYMBOLS)

    for url in NIFTY50_CSV_URLS:
        try:
            df = _read_csv_from_url(url)
            cols = {str(c).strip().lower(): c for c in df.columns}
            symbol_col = cols.get("symbol") or cols.get("symbols")
            if symbol_col is None:
                continue
            symbols = []
            for raw_symbol in df[symbol_col].dropna().astype(str).tolist():
                s = normalise_yahoo_symbol(raw_symbol)
                if s and s != MARKET_TICKER:
                    symbols.append(s)
            # De-duplicate while preserving order.
            seen, clean = set(), []
            for s in symbols:
                if s not in seen:
                    seen.add(s)
                    clean.append(s)
            if len(clean) >= 45:
                _NIFTY50_LIVE_CACHE = clean[:50]
                _log(f"[GLOBAL_LSTM] Loaded {len(_NIFTY50_LIVE_CACHE)} current Nifty 50 symbols from {url}", logger)
                return list(_NIFTY50_LIVE_CACHE)
        except Exception as exc:
            _warn(f"Could not load live Nifty 50 list from {url}: {exc}", logger)

    _warn("Live Nifty 50 list unavailable; using updated built-in fallback list", logger)
    _NIFTY50_LIVE_CACHE = list(NIFTY50_YAHOO_SYMBOLS)
    return list(_NIFTY50_LIVE_CACHE)


def extract_ohlcv_from_download(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return standard OHLCV frame from yfinance output."""
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
        return sub.dropna(subset=["date", "Close"]).sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def download_one(ticker: str, start: str = DATA_START_DATE, end: Optional[str] = None) -> pd.DataFrame:
    end = end or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        return extract_ohlcv_from_download(raw, ticker)
    except Exception:
        return pd.DataFrame()


def make_market_context(market_df: pd.DataFrame) -> pd.DataFrame:
    if market_df is None or market_df.empty:
        return pd.DataFrame(columns=["date"])
    m = market_df.copy().sort_values("date").reset_index(drop=True)
    close = pd.Series(m["Close"].values, dtype=float)
    m["nifty_close"] = close
    m["nifty_return_1d"] = close.pct_change()
    m["nifty_return_5d"] = close.pct_change(5)
    m["nifty_return_20d"] = close.pct_change(20)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    m["nifty_close_vs_sma50"] = close / sma50 - 1.0
    m["nifty_close_vs_sma200"] = close / sma200 - 1.0
    m["nifty_volatility_20d"] = close.pct_change().rolling(20).std()
    keep = [
        "date", "nifty_close", "nifty_return_1d", "nifty_return_5d", "nifty_return_20d",
        "nifty_close_vs_sma50", "nifty_close_vs_sma200", "nifty_volatility_20d",
    ]
    return m[keep].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_sector_context(raw_frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Build simple equal-weight sector return context from downloaded stocks."""
    rows = []
    for ticker, df in raw_frames.items():
        if df is None or df.empty or len(df) < 80:
            continue
        sector = SECTOR_MAP.get(ticker, "OTHER")
        tmp = df[["date", "Close"]].copy().sort_values("date")
        c = pd.Series(tmp["Close"].values, dtype=float)
        tmp["sector"] = sector
        tmp["ret_1d"] = c.pct_change()
        tmp["ret_5d"] = c.pct_change(5)
        tmp["ret_20d"] = c.pct_change(20)
        rows.append(tmp[["date", "sector", "ret_1d", "ret_5d", "ret_20d"]])
    if not rows:
        return {}
    all_ret = pd.concat(rows, ignore_index=True)
    out = {}
    for sector, g in all_ret.groupby("sector"):
        sec = g.groupby("date", as_index=False).agg(
            sector_return_1d=("ret_1d", "mean"),
            sector_return_5d=("ret_5d", "mean"),
            sector_return_20d=("ret_20d", "mean"),
        )
        out[sector] = sec.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def compute_features(stock_df: pd.DataFrame,
                     ticker: str,
                     market_context: pd.DataFrame,
                     sector_context: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build normalized transferable features and volatility-adjusted relative labels."""
    if stock_df is None or stock_df.empty:
        return pd.DataFrame()
    df = stock_df.copy().sort_values("date").reset_index(drop=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = df["Close"] if c != "Volume" else 0
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "Close"]).reset_index(drop=True)
    if len(df) < max(260, SEQUENCE_LENGTH + LABEL_HORIZON + 60):
        return pd.DataFrame()

    close = pd.Series(df["Close"].values, dtype=float)
    high = pd.Series(df["High"].values, dtype=float)
    low = pd.Series(df["Low"].values, dtype=float)
    volume = pd.Series(df["Volume"].values, dtype=float)

    # Stock-only normalized features
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
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
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

    # Market and sector context
    if market_context is not None and not market_context.empty:
        df = df.merge(market_context, on="date", how="left")
    else:
        for c in ["nifty_close", "nifty_return_1d", "nifty_return_5d", "nifty_return_20d", "nifty_close_vs_sma50", "nifty_close_vs_sma200", "nifty_volatility_20d"]:
            df[c] = 0.0
    if sector_context is not None and not sector_context.empty:
        df = df.merge(sector_context, on="date", how="left")
    for c in ["sector_return_1d", "sector_return_5d", "sector_return_20d"]:
        if c not in df.columns:
            # Fallback to broad market context when real sector context is absent.
            suffix = c.replace("sector_", "nifty_")
            df[c] = df.get(suffix, 0.0)

    df["relative_return_5d"] = df["return_5d"] - df["nifty_return_5d"]
    df["relative_return_20d"] = df["return_20d"] - df["nifty_return_20d"]
    df["sector_relative_return_20d"] = df["return_20d"] - df["sector_return_20d"]

    # Relative-return labels: learn whether the stock is an opportunity vs the market,
    # not merely whether the entire market drifted up/down.
    future_close = close.shift(-LABEL_HORIZON)
    stock_future_return = future_close / close.replace(0, np.nan) - 1.0

    if "nifty_close" in df.columns and pd.to_numeric(df["nifty_close"], errors="coerce").notna().sum() > 50:
        nifty_close = pd.Series(pd.to_numeric(df["nifty_close"], errors="coerce").values, dtype=float)
        future_nifty_return = nifty_close.shift(-LABEL_HORIZON) / nifty_close.replace(0, np.nan) - 1.0
    else:
        future_nifty_return = pd.Series(np.zeros(len(df)), index=df.index)

    relative_future_return = stock_future_return - future_nifty_return
    threshold = (df["atr_pct"].rolling(20).median() * VOL_MULT).clip(MIN_MOVE_PCT, MAX_MOVE_PCT)

    df["future_return"] = stock_future_return
    df["future_relative_return"] = relative_future_return
    df["move_threshold"] = threshold
    df["direction_label"] = 0
    df.loc[(relative_future_return > threshold) & (stock_future_return > 0), "direction_label"] = 1  # LONG
    df.loc[(relative_future_return < -threshold) | (stock_future_return < -threshold), "direction_label"] = 2  # EXIT/AVOID
    df.loc[stock_future_return.isna() | threshold.isna(), "direction_label"] = np.nan

    df["ticker"] = ticker
    df["sector"] = SECTOR_MAP.get(ticker, "OTHER")

    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    for c in FEATURE_COLUMNS:
        # Clip outliers using per-stock percentiles. This makes the global model less
        # sensitive to splits, corporate actions, or data-provider spikes.
        lo, hi = df[c].quantile(0.005), df[c].quantile(0.995)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            df[c] = df[c].clip(lo, hi)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0.0)
    return df.reset_index(drop=True)


def make_sequences_from_frame(frame: pd.DataFrame, seq_len: int = SEQUENCE_LENGTH) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str]]:
    if frame is None or frame.empty:
        return np.empty((0, seq_len, len(FEATURE_COLUMNS)), dtype=np.float32), np.array([], dtype=int), [], []
    f = frame.sort_values("date").reset_index(drop=True)
    values = f[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    labels = pd.to_numeric(f["direction_label"], errors="coerce").to_numpy()
    dates = pd.to_datetime(f["date"]).to_list()
    tickers = f.get("ticker", pd.Series(["UNKNOWN"] * len(f))).astype(str).to_list()
    X, y, ds, ts = [], [], [], []
    for i in range(seq_len, len(f)):
        label = labels[i]
        if not np.isfinite(label):
            continue
        window = values[i - seq_len:i]
        if not np.isfinite(window).all():
            continue
        X.append(window)
        y.append(int(label))
        ds.append(dates[i])
        ts.append(tickers[i])
    if not X:
        return np.empty((0, seq_len, len(FEATURE_COLUMNS)), dtype=np.float32), np.array([], dtype=int), [], []
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int), ds, ts


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, score))
    except Exception:
        return 0.5


def _direction_accuracy(y_true3: np.ndarray, probs3: np.ndarray) -> float:
    active = y_true3 != 0
    if active.sum() == 0:
        return 0.0
    pred_dir = np.where(probs3[:, 1] >= probs3[:, 2], 1, 2)
    return float((pred_dir[active] == y_true3[active]).mean() * 100.0)


def _active_metrics(y_true3: np.ndarray, probs3: np.ndarray, thresholds: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    active_score = probs3[:, 1] + probs3[:, 2]
    y_active = (y_true3 != 0).astype(int)
    if thresholds:
        pred_active = ((probs3[:, 1] >= float(thresholds.get("long_entry_threshold", 0.60))) |
                       (probs3[:, 2] >= float(thresholds.get("avoid_entry_threshold", 0.60)))).astype(int)
    else:
        pred_active = (active_score >= 0.55).astype(int)
    return {
        "active_accuracy": float(accuracy_score(y_active, pred_active) * 100.0) if len(y_active) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_active, pred_active) * 100.0) if len(np.unique(y_active)) > 1 else 50.0,
        "active_roc_auc": _safe_auc(y_active, active_score),
        "direction_accuracy": _direction_accuracy(y_true3, probs3),
    }


# ─────────────────────────────────────────────────────────────────────
# PyTorch model
# ─────────────────────────────────────────────────────────────────────
def _require_torch():
    import torch
    import torch.nn as nn
    return torch, nn


class _TorchModelFactory:
    @staticmethod
    def build(n_features: int, hidden_size: int = 64, dropout: float = 0.20):
        torch, nn = _require_torch()

        class GlobalAttentionLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=hidden_size,
                    num_layers=2,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=False,
                )
                self.attn = nn.Sequential(
                    nn.Linear(hidden_size, 32),
                    nn.Tanh(),
                    nn.Linear(32, 1),
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 32),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, 3),
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                weights = torch.softmax(self.attn(out).squeeze(-1), dim=1).unsqueeze(-1)
                ctx = (out * weights).sum(dim=1)
                return self.head(ctx)

        return GlobalAttentionLSTM()


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / max(float(temperature), 1e-6)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-9, None)


def _predict_logits(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    torch, _ = _require_torch()
    device = next(model.parameters()).device
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            outs.append(model(xb).detach().cpu().numpy())
    return np.vstack(outs) if outs else np.empty((0, 3))


def _fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    # Simple grid search: enough for this project and avoids extra dependencies.
    from sklearn.metrics import log_loss
    best_t, best_loss = 1.0, float("inf")
    for t in np.linspace(0.7, 4.0, 24):
        p = _softmax(logits, t)
        try:
            loss = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1, 2])
        except Exception:
            loss = float("inf")
        if loss < best_loss:
            best_loss, best_t = loss, float(t)
    return best_t


def _class_balanced_limit(X: np.ndarray, y: np.ndarray, max_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(y) <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    indices = []
    classes = np.unique(y)
    per_class = max(1, max_samples // max(1, len(classes)))
    for cls in classes:
        idx = np.where(y == cls)[0]
        take = min(len(idx), per_class)
        indices.extend(rng.choice(idx, take, replace=False).tolist())
    if len(indices) < max_samples:
        remaining = np.setdiff1d(np.arange(len(y)), np.asarray(indices), assume_unique=False)
        take = min(len(remaining), max_samples - len(indices))
        if take > 0:
            indices.extend(rng.choice(remaining, take, replace=False).tolist())
    rng.shuffle(indices)
    return X[indices], y[indices]


def train_torch_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, seed: int = RANDOM_SEED):
    torch, nn = _require_torch()
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    from sklearn.utils.class_weight import compute_class_weight

    torch.manual_seed(seed)
    np.random.seed(seed)

    classes = np.array([0, 1, 2])
    try:
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    except Exception:
        counts = np.bincount(y_train, minlength=3).astype(float)
        cw = len(y_train) / (3.0 * np.maximum(counts, 1.0))
    cw = np.clip(cw, 0.45, 5.0).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _TorchModelFactory.build(X_train.shape[-1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=0.0010, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=2, factor=0.6)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device), reduction="none")

    def focal_loss(logits, yb):
        base = ce(logits, yb)
        pt = torch.exp(-base).clamp(1e-5, 1.0)
        gamma = 1.5
        return (((1.0 - pt) ** gamma) * base).mean()

    sample_weights = cw[y_train]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)

    best_state, best_score, best_epoch, patience = None, -1.0, 0, 0
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = focal_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        logits = _predict_logits(model, X_val)
        probs = _softmax(logits)
        active_auc = _safe_auc((y_val != 0).astype(int), probs[:, 1] + probs[:, 2])
        dir_acc = _direction_accuracy(y_val, probs) / 100.0
        bal = balanced_accuracy_score((y_val != 0).astype(int), (probs[:, 1] + probs[:, 2] >= 0.55).astype(int)) if len(np.unique(y_val != 0)) > 1 else 0.5
        score = active_auc + 0.08 * max(0.0, dir_acc - 0.50) + 0.05 * max(0.0, bal - 0.50)
        scheduler.step(score)
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch
            patience = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    logits = _predict_logits(model, X_val)
    temp = _fit_temperature(logits, y_val)
    probs = _softmax(logits, temp)
    metrics = _active_metrics(y_val, probs)
    metrics["temperature"] = float(temp)
    metrics["best_epoch"] = int(best_epoch)
    metrics["class_weights"] = [float(x) for x in cw]
    return model, metrics


def fit_scaler(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = X.reshape(-1, X.shape[-1])
    mu = np.nanmean(flat, axis=0)
    sd = np.nanstd(flat, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where((sd > 1e-6) & np.isfinite(sd), sd, 1.0)
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_scaler(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return np.clip((X - mu.reshape(1, 1, -1)) / sd.reshape(1, 1, -1), -6.0, 6.0).astype(np.float32)


def build_universe(extra_symbols: Optional[Sequence[str]] = None, max_tickers: int = MAX_TICKERS, logger=None) -> List[str]:
    env_extra = os.getenv("PREDICTION_GLOBAL_LSTM_EXTRA_TICKERS", "")
    env_symbols = [normalise_yahoo_symbol(x) for x in env_extra.replace(";", ",").split(",") if x.strip()]

    nifty_symbols = fetch_current_nifty50_symbols(logger=logger)
    symbols = list(nifty_symbols)

    # Extras are disabled by default because stale non-index tickers were causing
    # noisy 404/no-timezone errors. Enable explicitly when you want a larger
    # universe after maintaining that list.
    if INCLUDE_EXTRA_LIQUID_SYMBOLS:
        symbols += EXTRA_LIQUID_NSE_SYMBOLS

    symbols += list(extra_symbols or []) + env_symbols

    seen, out = set(), []
    for s in symbols:
        s = normalise_yahoo_symbol(s)
        if not s or s in seen or s == MARKET_TICKER:
            continue
        seen.add(s)
        out.append(s)

    _log(f"[GLOBAL_LSTM] Universe size requested: {min(len(out), max_tickers)} (base Nifty50={len(nifty_symbols)}, extras={'ON' if INCLUDE_EXTRA_LIQUID_SYMBOLS else 'OFF'})", logger)
    return out[:max_tickers]


def download_universe(symbols: Sequence[str], start_date: str = DATA_START_DATE, logger=None) -> Dict[str, pd.DataFrame]:
    raw_frames: Dict[str, pd.DataFrame] = {}
    # Download in chunks. Huge multi-ticker downloads are fragile on Yahoo.
    chunk_size = int(os.getenv("PREDICTION_GLOBAL_LSTM_DOWNLOAD_CHUNK", "30"))
    chunk_size = max(5, min(60, chunk_size))
    end_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    for start in range(0, len(symbols), chunk_size):
        chunk = list(symbols[start:start + chunk_size])
        try:
            _log(f"[GLOBAL_LSTM] Downloading {start + 1}-{start + len(chunk)} / {len(symbols)}", logger)
            raw = yf.download(chunk, start=start_date, end=end_date, auto_adjust=True, progress=False, threads=True, group_by="ticker")
            for ticker in chunk:
                df = extract_ohlcv_from_download(raw, ticker)
                if not df.empty and len(df) > 350:
                    raw_frames[ticker] = df
        except Exception as exc:
            _warn(f"Chunk download failed ({exc}); falling back to one-by-one", logger)
            for ticker in chunk:
                df = download_one(ticker, start=start_date, end=end_date)
                if not df.empty and len(df) > 350:
                    raw_frames[ticker] = df
        # Be gentle with data provider.
        time.sleep(0.25)
    return raw_frames


def build_training_frames(symbols: Optional[Sequence[str]] = None, target_symbols: Optional[Sequence[str]] = None, logger=None) -> List[pd.DataFrame]:
    universe = build_universe(extra_symbols=target_symbols, max_tickers=MAX_TICKERS if symbols is None else len(symbols), logger=logger)
    if symbols is not None:
        universe = [normalise_yahoo_symbol(x) for x in symbols]
    _log(f"[GLOBAL_LSTM] Universe size requested: {len(universe)}", logger)

    market_df = download_one(MARKET_TICKER, DATA_START_DATE)
    market_context = make_market_context(market_df)
    raw_frames = download_universe(universe, DATA_START_DATE, logger)
    sector_contexts = build_sector_context(raw_frames)

    frames = []
    for ticker, raw_df in raw_frames.items():
        sector = SECTOR_MAP.get(ticker, "OTHER")
        sec_ctx = sector_contexts.get(sector)
        feat = compute_features(raw_df, ticker, market_context, sec_ctx)
        if not feat.empty:
            frames.append(feat)
    _log(f"[GLOBAL_LSTM] Usable feature frames: {len(frames)}", logger)
    return frames


def sequences_from_frames(frames: Sequence[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_parts, y_parts, date_parts, ticker_parts = [], [], [], []
    for frame in frames:
        X, y, dates, tickers = make_sequences_from_frame(frame, SEQUENCE_LENGTH)
        if len(y) > 0:
            X_parts.append(X)
            y_parts.append(y)
            date_parts.extend(pd.to_datetime(dates).to_numpy())
            ticker_parts.extend(tickers)
    if not X_parts:
        return np.empty((0, SEQUENCE_LENGTH, len(FEATURE_COLUMNS))), np.array([]), np.array([]), np.array([])
    return np.vstack(X_parts), np.concatenate(y_parts), np.asarray(date_parts), np.asarray(ticker_parts)


def choose_thresholds(y_true3: np.ndarray, probs3: np.ndarray) -> Dict[str, float]:
    """Choose model-specific thresholds from validation data."""
    def best_for_class(cls: int, grid: np.ndarray) -> float:
        y = (y_true3 == cls).astype(int)
        if y.sum() < 10:
            return 0.62
        best_t, best_score = 0.62, -1.0
        for t in grid:
            pred = (probs3[:, cls] >= t).astype(int)
            # F1 rewards actionable precision/recall better than raw accuracy.
            score = f1_score(y, pred, zero_division=0)
            active_rate = pred.mean()
            # Penalize extreme over-trading.
            if active_rate > 0.35:
                score *= 0.80
            if score > best_score:
                best_score, best_t = score, float(t)
        return float(best_t)

    grid = np.linspace(0.42, 0.78, 25)
    long_t = best_for_class(1, grid)
    avoid_t = best_for_class(2, grid)
    active_t = float(min(max(0.50, min(long_t, avoid_t)), 0.70))
    return {
        "long_entry_threshold": round(long_t, 4),
        "avoid_entry_threshold": round(avoid_t, 4),
        "active_entry_threshold": round(active_t, 4),
    }


def train_and_maybe_save_global_lstm(symbols: Optional[Sequence[str]] = None,
                                     target_symbols: Optional[Sequence[str]] = None,
                                     force: bool = False,
                                     logger=None) -> Dict:
    """Train global LSTM and replace existing saved model only if the new one is better."""
    import torch

    np.random.seed(RANDOM_SEED)
    frames = build_training_frames(symbols=symbols, target_symbols=target_symbols, logger=logger)
    X_all, y_all, dates_all, tickers_all = sequences_from_frames(frames)
    if len(y_all) < 5000 or len(np.unique(y_all)) < 2:
        raise RuntimeError(f"Not enough LSTM training sequences. Got {len(y_all)} sequences.")

    date_series = pd.to_datetime(dates_all)
    latest_year = int(pd.Timestamp(date_series.max()).year)
    candidate_val_years = list(range(latest_year - WALK_FOLDS, latest_year))
    fold_metrics = []
    best_fold_model = None
    best_fold_scaler = None
    best_fold_thresholds = None
    best_fold_score = -1.0

    for fold_i, val_year in enumerate(candidate_val_years):
        val_start = pd.Timestamp(f"{val_year}-01-01")
        val_end = pd.Timestamp(f"{val_year + 1}-01-01")
        train_mask = date_series < val_start
        val_mask = (date_series >= val_start) & (date_series < val_end)
        if train_mask.sum() < 4000 or val_mask.sum() < 600:
            continue
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_val, y_val = X_all[val_mask], y_all[val_mask]
        X_train, y_train = _class_balanced_limit(X_train, y_train, MAX_TRAIN_SAMPLES, RANDOM_SEED + fold_i)
        X_val, y_val = _class_balanced_limit(X_val, y_val, MAX_VAL_SAMPLES, RANDOM_SEED + 100 + fold_i)
        mu, sd = fit_scaler(X_train)
        X_train_s = apply_scaler(X_train, mu, sd)
        X_val_s = apply_scaler(X_val, mu, sd)
        _log(f"[GLOBAL_LSTM] Fold {fold_i + 1}: train={len(y_train)} val={len(y_val)} year={val_year}", logger)
        model, metrics = train_torch_model(X_train_s, y_train, X_val_s, y_val, RANDOM_SEED + fold_i)
        logits = _predict_logits(model, X_val_s)
        probs = _softmax(logits, metrics.get("temperature", 1.0))
        thresholds = choose_thresholds(y_val, probs)
        metrics2 = _active_metrics(y_val, probs, thresholds)
        metrics.update(metrics2)
        metrics.update({"val_year": int(val_year), "train_samples": int(len(y_train)), "val_samples": int(len(y_val)), "thresholds": thresholds})
        fold_metrics.append(metrics)
        score = metrics["active_roc_auc"] + 0.006 * max(0.0, metrics["balanced_accuracy"] - 50.0) + 0.004 * max(0.0, metrics["direction_accuracy"] - 50.0)
        if score > best_fold_score:
            best_fold_score = score
            best_fold_model = model
            best_fold_scaler = (mu, sd)
            best_fold_thresholds = thresholds

    if not fold_metrics or best_fold_model is None:
        raise RuntimeError("Walk-forward validation produced no valid folds.")

    # Train final model on all data except the most recent label horizon. Use the last
    # 6 months as final validation for calibration/thresholds.
    last_date = pd.Timestamp(date_series.max())
    final_val_start = last_date - pd.DateOffset(months=6)
    final_train_mask = date_series < final_val_start
    final_val_mask = date_series >= final_val_start
    if final_train_mask.sum() >= 5000 and final_val_mask.sum() >= 300:
        X_train, y_train = X_all[final_train_mask], y_all[final_train_mask]
        X_val, y_val = X_all[final_val_mask], y_all[final_val_mask]
        X_train, y_train = _class_balanced_limit(X_train, y_train, MAX_TRAIN_SAMPLES, RANDOM_SEED + 999)
        X_val, y_val = _class_balanced_limit(X_val, y_val, MAX_VAL_SAMPLES, RANDOM_SEED + 1999)
        mu, sd = fit_scaler(X_train)
        X_train_s, X_val_s = apply_scaler(X_train, mu, sd), apply_scaler(X_val, mu, sd)
        _log(f"[GLOBAL_LSTM] Final train={len(y_train)} val={len(y_val)}", logger)
        final_model, final_metrics = train_torch_model(X_train_s, y_train, X_val_s, y_val, RANDOM_SEED + 777)
        final_probs = _softmax(_predict_logits(final_model, X_val_s), final_metrics.get("temperature", 1.0))
        thresholds = choose_thresholds(y_val, final_probs)
        final_metrics.update(_active_metrics(y_val, final_probs, thresholds))
        final_metrics.update({"thresholds": thresholds, "final_validation_start": str(final_val_start.date()), "train_samples": int(len(y_train)), "val_samples": int(len(y_val))})
        model_to_save = final_model
        scaler_to_save = (mu, sd)
        thresholds_to_save = thresholds
    else:
        _warn("Using best walk-forward fold as saved model because final split was too small.", logger)
        model_to_save = best_fold_model
        scaler_to_save = best_fold_scaler
        thresholds_to_save = best_fold_thresholds or {"long_entry_threshold": 0.60, "avoid_entry_threshold": 0.60, "active_entry_threshold": 0.55}
        final_metrics = dict(max(fold_metrics, key=lambda m: m.get("active_roc_auc", 0.0)))

    active_auc = float(final_metrics.get("active_roc_auc", 0.5))
    balanced_acc = float(final_metrics.get("balanced_accuracy", 50.0))
    direction_acc = float(final_metrics.get("direction_accuracy", 0.0))
    reliable = bool(
        active_auc >= MIN_ACTIVE_AUC_FOR_ENSEMBLE
        and balanced_acc >= MIN_BALANCED_ACC_FOR_ENSEMBLE
        and direction_acc >= MIN_DIRECTION_ACC_FOR_ENSEMBLE
    )

    old_meta = load_metadata()
    old_auc = float(old_meta.get("active_roc_auc", 0.0) or 0.0) if old_meta else 0.0
    should_replace = bool(force or not MODEL_FILE.exists() or active_auc >= old_auc + 0.003)

    candidate_meta = {
        "model_name": "global_lstm_nse_relative_return_v1",
        "trained_at": _now_iso(),
        "expires_at": (datetime.now(tz=timezone.utc) + timedelta(days=MODEL_MAX_AGE_DAYS)).replace(microsecond=0).isoformat(),
        "sequence_length": int(SEQUENCE_LENGTH),
        "label_horizon_days": int(LABEL_HORIZON),
        "feature_count": int(len(FEATURE_COLUMNS)),
        "universe_requested": int(len(build_universe(max_tickers=MAX_TICKERS))),
        "frames_used": int(len(frames)),
        "total_sequences": int(len(y_all)),
        "active_roc_auc": round(active_auc, 4),
        "balanced_accuracy": round(balanced_acc, 1),
        "direction_accuracy": round(direction_acc, 1),
        "active_accuracy": round(float(final_metrics.get("active_accuracy", 0.0)), 1),
        "temperature": round(float(final_metrics.get("temperature", 1.0)), 4),
        "thresholds": thresholds_to_save,
        "walk_forward_metrics": fold_metrics,
        "reliable_for_ensemble": reliable,
        "status": "candidate_replaced_old" if should_replace else "candidate_rejected_old_kept",
        "old_active_roc_auc": round(old_auc, 4),
    }

    if should_replace:
        save_artifacts(model_to_save, scaler_to_save[0], scaler_to_save[1], thresholds_to_save, candidate_meta)
        _log(f"[GLOBAL_LSTM] Saved model to {GLOBAL_LSTM_DIR} (AUC={active_auc:.4f})", logger)
    else:
        _log(f"[GLOBAL_LSTM] Kept existing model. Candidate AUC={active_auc:.4f}, old AUC={old_auc:.4f}", logger)

    return candidate_meta


def save_artifacts(model, mu: np.ndarray, sd: np.ndarray, thresholds: Dict[str, float], metadata: Dict):
    import torch
    GLOBAL_LSTM_DIR.mkdir(parents=True, exist_ok=True)
    arch = {"n_features": len(FEATURE_COLUMNS), "hidden_size": 64, "dropout": 0.20}
    torch.save({"state_dict": model.state_dict(), "arch": arch}, MODEL_FILE)
    np.savez(SCALER_FILE, mean=np.asarray(mu, dtype=np.float32), scale=np.asarray(sd, dtype=np.float32))
    THRESHOLDS_FILE.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    FEATURES_FILE.write_text(json.dumps({"feature_columns": FEATURE_COLUMNS}, indent=2), encoding="utf-8")
    METADATA_FILE.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def load_metadata() -> Dict:
    try:
        if METADATA_FILE.exists():
            return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def is_model_fresh(metadata: Optional[Dict] = None) -> bool:
    metadata = metadata or load_metadata()
    if not MODEL_FILE.exists() or not SCALER_FILE.exists() or not THRESHOLDS_FILE.exists():
        return False
    trained_at = metadata.get("trained_at")
    if not trained_at:
        # Fallback to file modification time.
        return (datetime.now(tz=timezone.utc) - datetime.fromtimestamp(MODEL_FILE.stat().st_mtime, tz=timezone.utc)).days < MODEL_MAX_AGE_DAYS
    try:
        ts = datetime.fromisoformat(str(trained_at).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return datetime.now(tz=timezone.utc) - ts <= timedelta(days=MODEL_MAX_AGE_DAYS)
    except Exception:
        return False


def load_artifacts(logger=None):
    metadata = load_metadata()
    if not is_model_fresh(metadata):
        raise FileNotFoundError(
            "Global LSTM model is missing or stale. Run: python -m app.ml.train_global_lstm"
        )
    torch, _ = _require_torch()
    payload = torch.load(MODEL_FILE, map_location="cpu")
    arch = payload.get("arch", {"n_features": len(FEATURE_COLUMNS), "hidden_size": 64, "dropout": 0.20})
    model = _TorchModelFactory.build(
        int(arch.get("n_features", len(FEATURE_COLUMNS))),
        int(arch.get("hidden_size", 64)),
        float(arch.get("dropout", 0.20)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    scaler = np.load(SCALER_FILE)
    mu = scaler["mean"].astype(np.float32)
    sd = scaler["scale"].astype(np.float32)
    thresholds = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))
    return model, mu, sd, thresholds, metadata


def cleanup_rejected_candidates(logger=None):
    """Reserved for future candidate folders. Keeps current production model intact."""
    # Deliberately no automatic deletion of the production model here.
    return


def predict_saved_global_lstm_for_target(target_ohlcv_df: pd.DataFrame,
                                         test_days: int,
                                         confidence_threshold: float = 0.52,
                                         target_ticker: str = "__TARGET__",
                                         prophet_trend_state: Optional[Sequence[str]] = None,
                                         logger=None,
                                         auto_train_if_missing: Optional[bool] = None) -> Dict:
    """Return signals/metrics for target stock using the saved global LSTM artifact."""
    auto_train = AUTO_TRAIN_IF_MISSING if auto_train_if_missing is None else bool(auto_train_if_missing)
    try:
        model, mu, sd, thresholds, meta = load_artifacts(logger=logger)
    except Exception as exc:
        if auto_train:
            _warn(f"Saved Global LSTM unavailable ({exc}). Training now because auto-train is enabled.", logger)
            train_and_maybe_save_global_lstm(target_symbols=[target_ticker], force=False, logger=logger)
            model, mu, sd, thresholds, meta = load_artifacts(logger=logger)
        else:
            _warn(str(exc), logger)
            return {
                "signals": None,
                "metrics": {
                    "accuracy": None,
                    "balanced_accuracy": None,
                    "roc_auc": None,
                    "direction_accuracy": None,
                    "validation_auc": None,
                    "model_status": "missing_or_stale_saved_model",
                    "message": "Run python -m app.ml.train_global_lstm to train the saved Global LSTM.",
                },
                "lstm_reliable": False,
            }

    if target_ohlcv_df is None or target_ohlcv_df.empty:
        return {"signals": None, "metrics": {"accuracy": None, "roc_auc": None}, "lstm_reliable": False}

    start = pd.to_datetime(target_ohlcv_df["date"]).min().strftime("%Y-%m-%d") if "date" in target_ohlcv_df.columns else DATA_START_DATE
    market_df = download_one(MARKET_TICKER, start)
    market_context = make_market_context(market_df)
    feat = compute_features(target_ohlcv_df, normalise_yahoo_symbol(target_ticker), market_context, None)
    if feat.empty:
        return {"signals": None, "metrics": {"accuracy": None, "roc_auc": None, "model_status": "feature_build_failed"}, "lstm_reliable": False}

    X, y, dates, _tickers = make_sequences_from_frame(feat, int(meta.get("sequence_length", SEQUENCE_LENGTH)))
    if len(y) == 0:
        return {"signals": None, "metrics": {"accuracy": None, "roc_auc": None, "model_status": "no_sequences"}, "lstm_reliable": False}
    Xs = apply_scaler(X, mu, sd)
    logits = _predict_logits(model, Xs)
    probs3 = _softmax(logits, float(meta.get("temperature", 1.0)))

    # Align to the same last N test days used by the rest of the prediction pipeline.
    seq_df = pd.DataFrame({"date": pd.to_datetime(dates), "label": y})
    seq_df["row_i"] = np.arange(len(seq_df))
    seq_df = seq_df.sort_values("date").reset_index(drop=True)
    take = seq_df.tail(max(1, int(test_days)))
    idx = take["row_i"].to_numpy(dtype=int)
    dates_take = take["date"].to_numpy()
    labels3 = y[idx].astype(int)
    probs_take = probs3[idx]

    long_t = float(thresholds.get("long_entry_threshold", max(0.58, confidence_threshold)))
    avoid_t = float(thresholds.get("avoid_entry_threshold", max(0.58, confidence_threshold)))
    active_t = float(thresholds.get("active_entry_threshold", min(long_t, avoid_t)))
    exit_t = min(float(confidence_threshold), max(0.05, active_t * 0.85))

    active_score = np.maximum(probs_take[:, 1], probs_take[:, 2])
    active_prob = probs_take[:, 1] + probs_take[:, 2]
    model_direction = np.where(probs_take[:, 1] >= probs_take[:, 2], "BUY", "SELL")

    # LSTM now emits its own threshold-calibrated raw signal. Prophet trend is
    # kept as context, not as a hard blocker inside the LSTM service. This lets
    # the strategy layer and ensemble decide how to use a reliable LSTM setup
    # score without forcing it through the old XGBoost-style 52% gate.
    trend_state = np.asarray(prophet_trend_state if prophet_trend_state is not None else ["flat"] * len(dates_take), dtype=object)
    if len(trend_state) != len(dates_take):
        trend_state = np.asarray(["flat"] * len(dates_take), dtype=object)

    signals, strengths, context_flags, threshold_used = [], [], [], []
    for lp, ap, md, trend in zip(probs_take[:, 1], probs_take[:, 2], model_direction, trend_state):
        trend_l = str(trend).lower()
        if lp >= long_t and lp >= ap:
            signals.append("BUY")
            strengths.append("ACTIVE")
            threshold_used.append(long_t)
            context_flags.append("DIRECTION_CONFLICT" if trend_l == "down" else "OK")
        elif ap >= avoid_t and ap > lp:
            signals.append("SELL")
            strengths.append("ACTIVE")
            threshold_used.append(avoid_t)
            context_flags.append("DIRECTION_CONFLICT" if trend_l == "up" else "OK")
        elif active_score[len(signals)] >= active_t:
            # Active setup is present but neither class-specific threshold wins
            # cleanly. Keep HOLD, but mark it as near-threshold so diagnostics
            # do not call this a generic low-probability day.
            signals.append("HOLD")
            strengths.append("ACTIVE_UNCERTAIN")
            threshold_used.append(active_t)
            context_flags.append("CLASS_UNCERTAIN")
        else:
            signals.append("HOLD")
            strengths.append("LOW PROB")
            threshold_used.append(active_t)
            context_flags.append("LOW_PROB")

    # Pull close price for corresponding dates.
    close_map = feat.set_index(pd.to_datetime(feat["date"]))["Close"].to_dict()
    close_vals = [float(close_map.get(pd.Timestamp(d), np.nan)) for d in dates_take]
    sigs_df = pd.DataFrame({
        "date": pd.to_datetime(dates_take),
        "Close": close_vals,
        "prob_good_entry": np.round(active_score, 4),
        "prob_active_setup": np.round(active_prob, 4),
        "lstm_cash_prob": np.round(probs_take[:, 0], 4),
        "lstm_long_prob": np.round(probs_take[:, 1], 4),
        "lstm_short_prob": np.round(probs_take[:, 2], 4),
        "model_direction": model_direction,
        "prophet_uptrend": (trend_state == "up").astype(int),
        "prophet_trend_state": trend_state,
        "signal": signals,
        "raw_lstm_signal": signals,
        "strength": strengths,
        "lstm_context_flag": context_flags,
        "entry_threshold_used": np.round(threshold_used, 4),
        "long_entry_threshold": round(long_t, 4),
        "avoid_entry_threshold": round(avoid_t, 4),
        "active_entry_threshold": round(active_t, 4),
        "exit_threshold_used": round(exit_t, 4),
    })

    valid_metric = labels3 >= 0
    if valid_metric.sum() >= 20:
        metrics = _active_metrics(labels3[valid_metric], probs_take[valid_metric], thresholds)
    else:
        metrics = {"active_accuracy": 0.0, "balanced_accuracy": 50.0, "active_roc_auc": 0.5, "direction_accuracy": 0.0}

    reliable = bool(
        float(meta.get("active_roc_auc", metrics["active_roc_auc"])) >= MIN_ACTIVE_AUC_FOR_ENSEMBLE
        and float(meta.get("balanced_accuracy", metrics["balanced_accuracy"])) >= MIN_BALANCED_ACC_FOR_ENSEMBLE
        and float(meta.get("direction_accuracy", metrics["direction_accuracy"])) >= MIN_DIRECTION_ACC_FOR_ENSEMBLE
    )

    return {
        "signals": sigs_df,
        "metrics": {
            "accuracy": round(float(metrics.get("active_accuracy", 0.0)), 1),
            "balanced_accuracy": round(float(metrics.get("balanced_accuracy", 50.0)), 1),
            "roc_auc": round(float(metrics.get("active_roc_auc", 0.5)), 4),
            "direction_accuracy": round(float(metrics.get("direction_accuracy", 0.0)), 1),
            "validation_auc": round(float(meta.get("active_roc_auc", 0.5)), 4),
            "label_mode": "saved_global_lstm_relative_return_3class",
            "label_horizon_days": int(meta.get("label_horizon_days", LABEL_HORIZON)),
            "global_universe": "Saved global NSE/Nifty-style universe",
            "global_tickers_used": int(meta.get("frames_used", 0)),
            "feature_count": int(meta.get("feature_count", len(FEATURE_COLUMNS))),
            "trained_at": meta.get("trained_at"),
            "model_status": "loaded_saved_model",
            "thresholds": thresholds,
            "reliable_for_ensemble": reliable,
        },
        "lstm_reliable": reliable,
    }