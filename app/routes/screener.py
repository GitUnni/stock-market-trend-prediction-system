from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import List, Optional
from pydantic import BaseModel

from app.deps import get_db
from app import models

router = APIRouter(prefix="/api/screener", tags=["Screener"])


class ScreenerResult(BaseModel):
    symbol: str
    name: str
    current_price: Optional[float]
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    volume: Optional[float]
    dividend_yield: Optional[float]
    roe: Optional[float]
    debt_to_equity: Optional[float]
    beta: Optional[float]
    sector: Optional[str]
    industry: Optional[str]


def parse_and_execute_query(query_string: str, db: Session):
    """
    Parse screening query and execute against database
    
    Supported operators: >, <, >=, <=, =, AND, OR
    Supported parameters: see StockMetrics model
    
    Example: "market_cap > 100000 AND pe_ratio < 20"
    """
    
    # Start with base query joining StockMetrics and StockInfo
    query = db.query(
        models.StockMetrics,
        models.StockInfo.name
    ).join(
        models.StockInfo,
        models.StockMetrics.symbol == models.StockInfo.symbol
    )
    
    # Parse query string
    query_string = query_string.strip()
    
    if not query_string:
        return query.limit(100).all()
    
    # Convert to lowercase for parsing
    query_lower = query_string.lower()
    
    # Build filter conditions
    filters = []
    
    # Split by AND/OR
    if ' and ' in query_lower:
        conditions = query_string.split(' AND ')
        use_and = True
    elif ' or ' in query_lower:
        conditions = query_string.split(' OR ')
        use_and = False
    else:
        conditions = [query_string]
        use_and = True
    
    # Parse each condition
    for condition in conditions:
        condition = condition.strip()
        
        # Parse operator
        if '>=' in condition:
            parts = condition.split('>=')
            operator = '>='
        elif '<=' in condition:
            parts = condition.split('<=')
            operator = '<='
        elif '>' in condition:
            parts = condition.split('>')
            operator = '>'
        elif '<' in condition:
            parts = condition.split('<')
            operator = '<'
        elif '=' in condition:
            parts = condition.split('=')
            operator = '='
        else:
            continue
        
        if len(parts) != 2:
            continue
        
        field = parts[0].strip().lower()
        try:
            value = float(parts[1].strip())
        except:
            continue
        
        # Special handling for market_cap (convert crores to actual value)
        # User enters: market_cap > 10000 (meaning 10000 crores)
        # We need: market_cap > 100000000000 (actual value)
        if field == 'market_cap':
            value = value * 10000000  # Convert crores to actual value
        
        # Map field names to model columns
        field_mapping = {
            'price': models.StockMetrics.current_price,
            'current_price': models.StockMetrics.current_price,
            'market_cap': models.StockMetrics.market_cap,
            'pe_ratio': models.StockMetrics.pe_ratio,
            'pe': models.StockMetrics.pe_ratio,
            'pb_ratio': models.StockMetrics.pb_ratio,
            'pb': models.StockMetrics.pb_ratio,
            'volume': models.StockMetrics.volume,
            'dividend_yield': models.StockMetrics.dividend_yield,
            'dividend': models.StockMetrics.dividend_yield,
            'roe': models.StockMetrics.return_on_equity,
            'roa': models.StockMetrics.return_on_assets,
            'debt_to_equity': models.StockMetrics.debt_to_equity,
            'beta': models.StockMetrics.beta,
            'eps': models.StockMetrics.earnings_per_share,
            'profit_margin': models.StockMetrics.profit_margin,
            'revenue_growth': models.StockMetrics.revenue_growth,
            'earnings_growth': models.StockMetrics.earnings_growth,
            'current_ratio': models.StockMetrics.current_ratio,
            'peg_ratio': models.StockMetrics.peg_ratio,
        }
        
        column = field_mapping.get(field)
        if not column:
            continue
        
        # Apply operator and also filter out NULL values
        if operator == '>':
            filters.append(and_(column.isnot(None), column > value))
        elif operator == '<':
            filters.append(and_(column.isnot(None), column < value))
        elif operator == '>=':
            filters.append(and_(column.isnot(None), column >= value))
        elif operator == '<=':
            filters.append(and_(column.isnot(None), column <= value))
        elif operator == '=':
            filters.append(and_(column.isnot(None), column == value))
    
    # Apply filters
    if filters:
        if use_and:
            query = query.filter(and_(*filters))
        else:
            query = query.filter(or_(*filters))
    
    # Limit results to 200
    return query.limit(200).all()


@router.get("/run")
async def run_screener(
    query: str = Query(..., description="Screening query"),
    db: Session = Depends(get_db)
):
    """
    Execute screening query against stock metrics database
    
    Example queries:
    - market_cap > 100000 AND pe_ratio < 20
    - price < 500 AND volume > 1000000
    - dividend_yield > 3 AND pe_ratio < 15
    - roe > 15 AND debt_to_equity < 1
    
    Returns list of matching stocks with their metrics
    """
    
    try:
        results = parse_and_execute_query(query, db)
        
        # Format results
        formatted_results = []
        for metrics, stock_name in results:
            formatted_results.append({
                "symbol": metrics.symbol,
                "name": stock_name,
                "current_price": metrics.current_price,
                "market_cap": metrics.market_cap,
                "pe_ratio": metrics.pe_ratio,
                "pb_ratio": metrics.pb_ratio,
                "volume": metrics.volume,
                "dividend_yield": metrics.dividend_yield,
                "roe": metrics.return_on_equity,
                "debt_to_equity": metrics.debt_to_equity,
                "beta": metrics.beta,
                "sector": metrics.sector,
                "industry": metrics.industry
            })
        
        return {
            "query": query,
            "count": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error executing query: {str(e)}"
        )


@router.get("/metrics")
async def get_available_metrics():
    """
    Get list of available parameters for screening
    """
    return {
        "parameters": [
            {
                "name": "current_price",
                "aliases": ["price"],
                "description": "Current stock price in ₹",
                "type": "number"
            },
            {
                "name": "market_cap",
                "aliases": [],
                "description": "Market capitalization",
                "type": "number"
            },
            {
                "name": "pe_ratio",
                "aliases": ["pe"],
                "description": "Price to Earnings ratio",
                "type": "number"
            },
            {
                "name": "pb_ratio",
                "aliases": ["pb"],
                "description": "Price to Book ratio",
                "type": "number"
            },
            {
                "name": "volume",
                "aliases": [],
                "description": "Trading volume",
                "type": "number"
            },
            {
                "name": "dividend_yield",
                "aliases": ["dividend"],
                "description": "Dividend yield %",
                "type": "number"
            },
            {
                "name": "roe",
                "aliases": [],
                "description": "Return on Equity %",
                "type": "number"
            },
            {
                "name": "roa",
                "aliases": [],
                "description": "Return on Assets %",
                "type": "number"
            },
            {
                "name": "debt_to_equity",
                "aliases": [],
                "description": "Debt to Equity ratio",
                "type": "number"
            },
            {
                "name": "beta",
                "aliases": [],
                "description": "Stock volatility (beta)",
                "type": "number"
            },
            {
                "name": "eps",
                "aliases": [],
                "description": "Earnings Per Share",
                "type": "number"
            },
            {
                "name": "profit_margin",
                "aliases": [],
                "description": "Profit margin %",
                "type": "number"
            },
            {
                "name": "revenue_growth",
                "aliases": [],
                "description": "Revenue growth %",
                "type": "number"
            },
            {
                "name": "earnings_growth",
                "aliases": [],
                "description": "Earnings growth %",
                "type": "number"
            },
            {
                "name": "current_ratio",
                "aliases": [],
                "description": "Current ratio (liquidity)",
                "type": "number"
            },
            {
                "name": "peg_ratio",
                "aliases": [],
                "description": "PEG ratio",
                "type": "number"
            }
        ],
        "operators": [">", "<", ">=", "<=", "="],
        "logical": ["AND", "OR"]
    }