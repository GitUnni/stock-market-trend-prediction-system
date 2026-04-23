"""
Daily Stock Metrics Update Script
Run this script daily after market close (6 PM IST) to update all stock metrics

Usage: python -m app.update_stock_metrics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models import StockInfo, StockMetrics
import yfinance as yf
from datetime import datetime
import time


def update_stock_metrics():
    """
    Fetch and update metrics for all stocks in database
    """
    db = SessionLocal()
    
    print("=" * 60)
    print("STOCK METRICS UPDATE - Starting")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get all stocks
    stocks = db.query(StockInfo).all()
    total = len(stocks)
    print(f"Total stocks to update: {total}")
    print("-" * 60)
    
    success_count = 0
    error_count = 0
    
    for idx, stock in enumerate(stocks, 1):
        try:
            print(f"[{idx}/{total}] Updating {stock.symbol} ({stock.name})...", end=" ")
            
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(stock.yahoo_symbol)
            info = ticker.info
            
            # Check if we got valid data
            if not info or 'currentPrice' not in info:
                print("❌ No data")
                error_count += 1
                continue
            
            # Check if metrics exist
            metrics = db.query(StockMetrics).filter(
                StockMetrics.symbol == stock.symbol
            ).first()
            
            if not metrics:
                # Create new metrics
                metrics = StockMetrics(symbol=stock.symbol)
                db.add(metrics)
            
            # Update all metrics
            # Price Data
            metrics.current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            metrics.previous_close = info.get('previousClose')
            metrics.open_price = info.get('open') or info.get('regularMarketOpen')
            metrics.day_high = info.get('dayHigh') or info.get('regularMarketDayHigh')
            metrics.day_low = info.get('dayLow') or info.get('regularMarketDayLow')
            metrics.week_52_high = info.get('fiftyTwoWeekHigh')
            metrics.week_52_low = info.get('fiftyTwoWeekLow')
            
            # Volume & Trading
            metrics.volume = info.get('volume') or info.get('regularMarketVolume')
            metrics.avg_volume = info.get('averageVolume') or info.get('averageVolume10days')
            
            # Market Metrics
            metrics.market_cap = info.get('marketCap')
            
            # Valuation Ratios
            metrics.pe_ratio = info.get('trailingPE') or info.get('forwardPE')
            metrics.pb_ratio = info.get('priceToBook')
            metrics.price_to_sales = info.get('priceToSalesTrailing12Months')
            metrics.peg_ratio = info.get('pegRatio')
            
            # Profitability
            metrics.profit_margin = info.get('profitMargins')
            metrics.operating_margin = info.get('operatingMargins')
            metrics.return_on_assets = info.get('returnOnAssets')
            metrics.return_on_equity = info.get('returnOnEquity')
            
            # Dividends
            metrics.dividend_yield = info.get('dividendYield')
            metrics.dividend_rate = info.get('dividendRate')
            metrics.payout_ratio = info.get('payoutRatio')
            
            # Growth
            metrics.revenue_growth = info.get('revenueGrowth')
            metrics.earnings_growth = info.get('earningsGrowth')
            
            # Financial Health
            metrics.current_ratio = info.get('currentRatio')
            metrics.quick_ratio = info.get('quickRatio')
            metrics.debt_to_equity = info.get('debtToEquity')
            
            # Per Share Data
            metrics.earnings_per_share = info.get('trailingEps') or info.get('forwardEps')
            metrics.book_value_per_share = info.get('bookValue')
            
            # Beta
            metrics.beta = info.get('beta')
            
            # Sector & Industry
            metrics.sector = info.get('sector')
            metrics.industry = info.get('industry')
            
            # Update timestamp
            metrics.updated_at = datetime.now()
            
            db.commit()
            print("✓")
            success_count += 1
            
            # Small delay to avoid rate limiting
            time.sleep(0.3)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            error_count += 1
            db.rollback()
            continue
    
    db.close()
    
    print("-" * 60)
    print(f"Update completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Success: {success_count} stocks")
    print(f"Errors: {error_count} stocks")
    print("=" * 60)


def quick_update_sample_stocks():
    """
    Quick update for testing - updates only top 50 stocks
    """
    db = SessionLocal()
    
    print("QUICK UPDATE - Top 50 stocks")
    print("-" * 60)
    
    # Get top 50 stocks by alphabetical order
    stocks = db.query(StockInfo).limit(50).all()
    
    success_count = 0
    for idx, stock in enumerate(stocks, 1):
        try:
            print(f"[{idx}/50] {stock.symbol}...", end=" ")
            
            ticker = yf.Ticker(stock.yahoo_symbol)
            info = ticker.info
            
            if not info or 'currentPrice' not in info:
                print("❌")
                continue
            
            metrics = db.query(StockMetrics).filter(
                StockMetrics.symbol == stock.symbol
            ).first()
            
            if not metrics:
                metrics = StockMetrics(symbol=stock.symbol)
                db.add(metrics)
            
            # Update key metrics only
            metrics.current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            metrics.market_cap = info.get('marketCap')
            metrics.pe_ratio = info.get('trailingPE')
            metrics.pb_ratio = info.get('priceToBook')
            metrics.volume = info.get('volume')
            metrics.dividend_yield = info.get('dividendYield')
            metrics.beta = info.get('beta')
            metrics.sector = info.get('sector')
            metrics.updated_at = datetime.now()
            
            db.commit()
            print("✓")
            success_count += 1
            time.sleep(0.3)
            
        except Exception as e:
            print(f"❌ {str(e)}")
            db.rollback()
    
    db.close()
    print(f"Quick update complete: {success_count}/50 stocks updated")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Update stock metrics from Yahoo Finance')
    parser.add_argument('--quick', action='store_true', help='Quick update for top 50 stocks only')
    args = parser.parse_args()
    
    if args.quick:
        quick_update_sample_stocks()
    else:
        update_stock_metrics()