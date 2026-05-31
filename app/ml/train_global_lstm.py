"""
Train the saved Global LSTM model for the prediction module.

Place this file at:
    app/ml/train_global_lstm.py

Run from your project root:
    python -m app.ml.train_global_lstm

Optional examples:
    PREDICTION_GLOBAL_LSTM_MAX_TICKERS=200 python -m app.ml.train_global_lstm
    PREDICTION_GLOBAL_LSTM_EXTRA_TICKERS="CUPID.NS,RECLTD.NS" python -m app.ml.train_global_lstm

Artifacts are saved to:
    app/static/models/global_lstm/

A newly trained candidate replaces the old saved model only when its validation
Active ROC-AUC is better, unless you pass --force.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.ml.global_lstm_service import train_and_maybe_save_global_lstm, GLOBAL_LSTM_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Train Global LSTM for prediction module")
    parser.add_argument("--force", action="store_true", help="Replace the existing saved model even if candidate metrics are not better")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated Yahoo symbols to train on instead of the built-in universe")
    parser.add_argument("--target-symbols", type=str, default="", help="Extra target symbols to add to the built-in universe, e.g. CUPID.NS,RECLTD.NS")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("train_global_lstm")
    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()] or None
    target_symbols = [x.strip() for x in args.target_symbols.split(",") if x.strip()] or None
    logger.info("Training Global LSTM. Output folder: %s", GLOBAL_LSTM_DIR)
    meta = train_and_maybe_save_global_lstm(
        symbols=symbols,
        target_symbols=target_symbols,
        force=bool(args.force),
        logger=logger,
    )
    print(json.dumps(meta, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
