import yfinance as yf
import pandas as pd
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import StockPrice

def fetch_stock_data(symbol: str, period="1y", interval="1d"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        return None

    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Symbol"] = symbol

    return df


def store_stock_data(df: pd.DataFrame):
    db: Session = SessionLocal()

    for _, row in df.iterrows():
        record = StockPrice(
            symbol=row["Symbol"],
            date=row["Date"],
            open=row["Open"],
            high=row["High"],
            low=row["Low"],
            close=row["Close"],
            volume=row["Volume"],
        )

        try:
            db.add(record)
            db.commit()
        except:
            db.rollback()  # duplicate date → skip

    db.close()
