from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import yfinance as yf
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging

from app.deps import get_db
from app import models

router = APIRouter(prefix="/api/stocks", tags=["Stocks"])

logger = logging.getLogger(__name__)


def sync_nse_stocks_from_csv():
    """
    Download and sync NSE stock list from NSE website
    NSE provides equity list as CSV file
    """
    try:
        # NSE equity list URL (updated periodically by NSE)
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        
        # Download CSV
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            # Expected columns: SYMBOL, NAME OF COMPANY, etc.
            stocks = []
            for _, row in df.iterrows():
                symbol = str(row.get('SYMBOL', '')).strip()
                name = str(row.get(' NAME OF COMPANY', '')).strip()  # Note: space before NAME
                
                if symbol and name:
                    stocks.append({
                        'symbol': symbol,
                        'name': name,
                        'yahoo_symbol': f"{symbol}.NS"
                    })
            
            logger.info(f"Successfully synced {len(stocks)} NSE stocks")
            return stocks
        else:
            logger.error(f"Failed to download NSE stock list: {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Error syncing NSE stocks: {str(e)}")
        return []


def get_stocks_from_database(db: Session):
    """
    Get all stocks from database
    Returns list of stocks with symbol and name
    """
    try:
        # Try to get from StockInfo table (we'll create this)
        stocks = db.query(models.StockInfo).all()
        return [{"symbol": s.symbol, "name": s.name, "yahoo_symbol": s.yahoo_symbol} for s in stocks]
    except Exception as e:
        logger.error(f"Error fetching from database: {str(e)}")
        return []


def search_stocks_in_list(stock_list: List[dict], query: str, limit: int = 20):
    """
    Search stocks in a given list
    """
    query_lower = query.lower()
    
    # Filter stocks
    results = []
    for stock in stock_list:
        # Search in both name and symbol
        if (query_lower in stock['name'].lower() or 
            query_lower in stock['symbol'].lower()):
            results.append(stock)
    
    # Limit results
    return results[:limit]


# Fallback minimal stock list (in case API/database fails)
FALLBACK_STOCKS = {
    "RELIANCE": {"name": "Reliance Industries", "symbol": "RELIANCE.NS"},
    "TCS": {"name": "Tata Consultancy Services", "symbol": "TCS.NS"},
    "HDFCBANK": {"name": "HDFC Bank", "symbol": "HDFCBANK.NS"},
    "INFY": {"name": "Infosys", "symbol": "INFY.NS"},
    "ICICIBANK": {"name": "ICICI Bank", "symbol": "ICICIBANK.NS"},
    "SBIN": {"name": "State Bank of India", "symbol": "SBIN.NS"},
    "BHARTIARTL": {"name": "Bharti Airtel", "symbol": "BHARTIARTL.NS"},
    "ITC": {"name": "ITC Limited", "symbol": "ITC.NS"},
    "KOTAKBANK": {"name": "Kotak Mahindra Bank", "symbol": "KOTAKBANK.NS"},
    "LT": {"name": "Larsen & Toubro", "symbol": "LT.NS"},
    "SOUTHBANK": {"name": "South Indian Bank", "symbol": "SOUTHBANK.NS"},
}


@router.get("/search")
async def search_stocks(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=50, description="Maximum number of results"),
    db: Session = Depends(get_db)
):
    """
    Search for stocks by name or symbol
    
    This endpoint searches through ALL NSE-listed stocks stored in the database.
    If database is empty, it searches through a fallback list.
    
    Parameters:
    - q: Search query (minimum 1 character)
    - limit: Maximum number of results to return (default: 20, max: 50)
    
    Returns:
    - List of matching stocks with name and symbol
    """
    query = q.strip()
    
    # Try to get stocks from database first
    try:
        stock_list = get_stocks_from_database(db)
        
        if not stock_list:
            # Database is empty, use fallback
            stock_list = [
                {"symbol": k, "name": v["name"], "yahoo_symbol": v["symbol"]}
                for k, v in FALLBACK_STOCKS.items()
            ]
            logger.warning("Using fallback stock list - database is empty")
        
        # Search in the list
        results = search_stocks_in_list(stock_list, query, limit)
        
        return {
            "query": q,
            "count": len(results),
            "stocks": results,
            "source": "database" if stock_list else "fallback"
        }
        
    except Exception as e:
        logger.error(f"Error in stock search: {str(e)}")
        
        # Final fallback
        stock_list = [
            {"symbol": k, "name": v["name"], "yahoo_symbol": v["symbol"]}
            for k, v in FALLBACK_STOCKS.items()
        ]
        results = search_stocks_in_list(stock_list, query, limit)
        
        return {
            "query": q,
            "count": len(results),
            "stocks": results,
            "source": "fallback",
            "error": "Database error - using fallback list"
        }


@router.post("/sync-nse")
async def sync_nse_stocks(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Sync NSE stock list from NSE website
    This should be called periodically (e.g., daily) to update stock list
    
    Admin-only endpoint (add authentication in production)
    """
    def sync_task():
        try:
            # Download NSE stock list
            stocks = sync_nse_stocks_from_csv()
            
            if stocks:
                # Clear existing stocks
                db.query(models.StockInfo).delete()
                
                # Add new stocks
                for stock_data in stocks:
                    stock = models.StockInfo(
                        symbol=stock_data['symbol'],
                        name=stock_data['name'],
                        yahoo_symbol=stock_data['yahoo_symbol'],
                        exchange='NSE'
                    )
                    db.add(stock)
                
                db.commit()
                logger.info(f"Successfully synced {len(stocks)} stocks to database")
            else:
                logger.error("No stocks retrieved from NSE")
                
        except Exception as e:
            logger.error(f"Error in sync task: {str(e)}")
            db.rollback()
    
    # Run sync in background
    background_tasks.add_task(sync_task)
    
    return {
        "message": "Stock sync initiated in background",
        "status": "processing"
    }


@router.get("/quote/{symbol}")
async def get_stock_quote(symbol: str, db: Session = Depends(get_db)):
    """
    Get current stock quote for a symbol
    
    Parameters:
    - symbol: Stock symbol (e.g., RELIANCE, TCS, SOUTHBANK)
    
    Returns:
    - Current price, change, volume, and other quote data
    """
    # Try to find stock in database
    try:
        stock_info = db.query(models.StockInfo).filter(
            models.StockInfo.symbol == symbol.upper()
        ).first()
        
        if stock_info:
            yahoo_symbol = stock_info.yahoo_symbol
            stock_name = stock_info.name
        else:
            # Check fallback
            if symbol.upper() in FALLBACK_STOCKS:
                yahoo_symbol = FALLBACK_STOCKS[symbol.upper()]["symbol"]
                stock_name = FALLBACK_STOCKS[symbol.upper()]["name"]
            else:
                raise HTTPException(status_code=404, detail="Stock symbol not found")
    except Exception as e:
        # Fallback check
        if symbol.upper() in FALLBACK_STOCKS:
            yahoo_symbol = FALLBACK_STOCKS[symbol.upper()]["symbol"]
            stock_name = FALLBACK_STOCKS[symbol.upper()]["name"]
        else:
            raise HTTPException(status_code=404, detail="Stock symbol not found")
    
    try:
        # Fetch data from Yahoo Finance
        ticker = yf.Ticker(yahoo_symbol)
        info = ticker.info
        
        # Get current price and change
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        previous_close = info.get('previousClose')
        
        if current_price and previous_close:
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100
        else:
            change = None
            change_percent = None
        
        return {
            "symbol": symbol.upper(),
            "name": stock_name,
            "current_price": current_price,
            "previous_close": previous_close,
            "change": change,
            "change_percent": change_percent,
            "volume": info.get('volume'),
            "market_cap": info.get('marketCap'),
            "day_high": info.get('dayHigh'),
            "day_low": info.get('dayLow'),
            "fifty_two_week_high": info.get('fiftyTwoWeekHigh'),
            "fifty_two_week_low": info.get('fiftyTwoWeekLow'),
            "pe_ratio": info.get('trailingPE'),
            "dividend_yield": info.get('dividendYield')
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching stock data: {str(e)}"
        )


@router.get("/history/{symbol}")
async def get_stock_history(
    symbol: str,
    period: str = Query("1mo", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    interval: str = Query("1d", description="Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)"),
    db: Session = Depends(get_db)
):
    """
    Get historical stock data
    
    Parameters:
    - symbol: Stock symbol
    - period: Time period for historical data
    - interval: Data interval
    
    Returns:
    - Historical price data
    """
    # Get yahoo symbol
    try:
        stock_info = db.query(models.StockInfo).filter(
            models.StockInfo.symbol == symbol.upper()
        ).first()
        
        if stock_info:
            yahoo_symbol = stock_info.yahoo_symbol
            stock_name = stock_info.name
        else:
            if symbol.upper() in FALLBACK_STOCKS:
                yahoo_symbol = FALLBACK_STOCKS[symbol.upper()]["symbol"]
                stock_name = FALLBACK_STOCKS[symbol.upper()]["name"]
            else:
                raise HTTPException(status_code=404, detail="Stock symbol not found")
    except Exception as e:
        if symbol.upper() in FALLBACK_STOCKS:
            yahoo_symbol = FALLBACK_STOCKS[symbol.upper()]["symbol"]
            stock_name = FALLBACK_STOCKS[symbol.upper()]["name"]
        else:
            raise HTTPException(status_code=404, detail="Stock symbol not found")
    
    try:
        # Fetch historical data
        ticker = yf.Ticker(yahoo_symbol)
        # Use auto_adjust=False to get separate 'Adj Close' column
        hist = ticker.history(period=period, interval=interval, auto_adjust=False)
        
        # Convert to list of dictionaries
        history_data = []
        for index, row in hist.iterrows():
            # Get adjusted close and regular close
            try:
                adj_close = float(row['Adj Close'])
                close = float(row['Close'])
            except (KeyError, TypeError):
                # Fallback to regular close if Adj Close not available
                adj_close = float(row['Close'])
                close = float(row['Close'])
            
            history_data.append({
                "date": index.strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": close,
                "adj_close": adj_close,
                "volume": int(row['Volume'])
            })
        
        return {
            "symbol": symbol.upper(),
            "name": stock_name,
            "period": period,
            "interval": interval,
            "data": history_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching historical data: {str(e)}"
        )


@router.get("/list")
async def list_all_stocks(
    limit: int = Query(None, description="Limit number of results"),
    db: Session = Depends(get_db)
):
    """
    Get list of all available stocks
    
    Returns:
    - Complete list of all stocks in the database
    """
    try:
        stock_list = get_stocks_from_database(db)
        
        if not stock_list:
            # Use fallback
            stock_list = [
                {"symbol": k, "name": v["name"], "yahoo_symbol": v["symbol"]}
                for k, v in FALLBACK_STOCKS.items()
            ]
        
        # Sort by name
        stock_list.sort(key=lambda x: x["name"])
        
        if limit:
            stock_list = stock_list[:limit]
        
        return {
            "count": len(stock_list),
            "stocks": stock_list
        }
    except Exception as e:
        logger.error(f"Error listing stocks: {str(e)}")
        
        # Fallback
        stock_list = [
            {"symbol": k, "name": v["name"], "yahoo_symbol": v["symbol"]}
            for k, v in FALLBACK_STOCKS.items()
        ]
        stock_list.sort(key=lambda x: x["name"])
        
        return {
            "count": len(stock_list),
            "stocks": stock_list,
            "source": "fallback"
        }


@router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """
    Get statistics about the stock database
    """
    try:
        total_stocks = db.query(models.StockInfo).count()
        
        return {
            "total_stocks": total_stocks,
            "database_populated": total_stocks > 0,
            "last_sync": None  # Can add a sync_log table to track this
        }
    except Exception as e:
        return {
            "total_stocks": 0,
            "database_populated": False,
            "error": str(e)
        }


@router.get("/metrics/{symbol}")
async def get_stock_metrics(symbol: str, db: Session = Depends(get_db)):
    """
    Get detailed stock metrics from database
    Returns comprehensive financial data for stock details display
    Returns 404 if metrics not found (will be handled gracefully by frontend)
    """
    try:
        # Get stock metrics from database
        metrics = db.query(models.StockMetrics).filter(
            models.StockMetrics.symbol == symbol.upper()
        ).first()
        
        if not metrics:
            # Return 404 instead of 500 - this is expected for some stocks
            raise HTTPException(status_code=404, detail=f"Metrics not found for {symbol}")
        
        # Format and return metrics
        return {
            "symbol": metrics.symbol,
            "market_cap": metrics.market_cap,
            "volume": metrics.volume,
            "sector": metrics.sector,
            "industry": metrics.industry,
            "dividend_yield": metrics.dividend_yield,
            "pe_ratio": metrics.pe_ratio,
            "pb_ratio": metrics.pb_ratio,
            "profit_margin": metrics.profit_margin,
            "revenue_growth": metrics.revenue_growth,
            "debt_to_equity": metrics.debt_to_equity,
            "book_value_per_share": metrics.book_value_per_share,
            "week_52_high": metrics.week_52_high,
            "week_52_low": metrics.week_52_low,
            "updated_at": metrics.updated_at
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (404)
        raise
    except Exception as e:
        # Log the actual error for debugging
        logger.error(f"Error fetching metrics for {symbol}: {str(e)}")
        # Return 404 to frontend (graceful degradation)
        raise HTTPException(
            status_code=404,
            detail=f"Metrics not available for {symbol}"
        )


@router.get("/logo/{symbol}")
async def get_stock_logo(symbol: str):
    """
    Proxy endpoint to fetch company logo URL via TickerLogos API.

    The AllinvestView logo-search API blocks browser requests (no CORS header),
    so we call it server-side and return just the CDN logo URL to the frontend.

    Parameters:
    - symbol: NSE stock symbol (e.g. RELIANCE, TCS)

    Returns:
    - logo_url: CDN URL for the company logo (ready to use in <img src>)
    - domain:   Company website domain resolved from the ticker
    - name:     Company name returned by the logo API
    """
    # Strip exchange suffix just in case the frontend passes e.g. RELIANCE.NS
    ticker = symbol.upper().split(".")[0]

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(
            f"https://www.allinvestview.com/api/logo-search/?q={ticker}",
            headers=headers,
            timeout=8
        )

        if resp.status_code != 200:
            raise HTTPException(status_code=404, detail="Logo not found")

        data = resp.json()
        results = data.get("results", [])

        if not results or not results[0].get("website"):
            raise HTTPException(status_code=404, detail="Logo not found")

        result = results[0]
        # Strip protocol to get a clean domain for the CDN
        domain = (
            result["website"]
            .replace("http://", "")
            .replace("https://", "")
            .rstrip("/")
            .split("/")[0]
        )

        return {
            "symbol": ticker,
            "name": result.get("name", ""),
            "domain": domain,
            "logo_url": f"https://cdn.tickerlogos.com/{domain}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching logo for {symbol}: {str(e)}")
        raise HTTPException(status_code=404, detail="Logo lookup failed")