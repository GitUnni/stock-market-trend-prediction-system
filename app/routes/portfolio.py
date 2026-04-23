from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
import yfinance as yf
from jose import jwt, JWTError

from app.deps import get_db
from app import models
from app.auth import SECRET_KEY, ALGORITHM  # reuse the same key/algo used when signing tokens

router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])

# ── Auth dependency ────────────────────────────────────────────────────────────
_bearer = HTTPBearer(auto_error=False)

def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> int:
    """
    Extract the authenticated user's ID from the Bearer JWT token.
    The token is signed by app/auth.py's create_access_token with sub=str(user.id).
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated — Bearer token missing",
        )
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )
        user_id_str = payload.get("sub")
        if user_id_str is None:
            raise ValueError("sub claim missing")
        return int(user_id_str)
    except (JWTError, ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


class AddHoldingRequest(BaseModel):
    symbol: str
    stock_name: str
    quantity: int
    avg_price: float


class HoldingResponse(BaseModel):
    id: int
    symbol: str
    stock_name: str
    quantity: int
    avg_price: float
    invested_amount: float
    current_value: float
    returns: float
    returns_percent: float


class PortfolioSummaryResponse(BaseModel):
    total_holdings: int
    total_invested: float
    total_current_value: float
    total_returns: float
    total_returns_percent: float
    day_returns: float
    day_returns_percent: float


@router.post("/holdings")
async def add_holding(
    holding: AddHoldingRequest,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    """
    Add a stock to user's portfolio
    If stock already exists, update quantity and average price
    """
    # Check if holding already exists
    existing_holding = db.query(models.Portfolio).filter(
        models.Portfolio.user_id == user_id,
        models.Portfolio.symbol == holding.symbol
    ).first()
    
    if existing_holding:
        # Update existing holding
        # Calculate new average price
        total_invested = (existing_holding.quantity * existing_holding.avg_price) + \
                        (holding.quantity * holding.avg_price)
        total_quantity = existing_holding.quantity + holding.quantity
        new_avg_price = total_invested / total_quantity
        
        existing_holding.quantity = total_quantity
        existing_holding.avg_price = new_avg_price
        
        db.commit()
        db.refresh(existing_holding)
        
        return {
            "message": "Holding updated successfully",
            "holding": {
                "id": existing_holding.id,
                "symbol": existing_holding.symbol,
                "stock_name": existing_holding.stock_name,
                "quantity": existing_holding.quantity,
                "avg_price": existing_holding.avg_price
            }
        }
    else:
        # Create new holding
        new_holding = models.Portfolio(
            user_id=user_id,
            symbol=holding.symbol,
            stock_name=holding.stock_name,
            quantity=holding.quantity,
            avg_price=holding.avg_price
        )
        
        db.add(new_holding)
        db.commit()
        db.refresh(new_holding)
        
        return {
            "message": "Holding added successfully",
            "holding": {
                "id": new_holding.id,
                "symbol": new_holding.symbol,
                "stock_name": new_holding.stock_name,
                "quantity": new_holding.quantity,
                "avg_price": new_holding.avg_price
            }
        }


@router.get("/holdings")
async def get_holdings(
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    """
    Get all holdings for a user with current values
    """
    holdings = db.query(models.Portfolio).filter(
        models.Portfolio.user_id == user_id
    ).all()
    
    if not holdings:
        return {
            "holdings": [],
            "summary": {
                "total_holdings": 0,
                "total_invested": 0,
                "total_current_value": 0,
                "total_returns": 0,
                "total_returns_percent": 0,
                "day_returns": 0,
                "day_returns_percent": 0
            }
        }
    
    # Fetch current prices for all holdings
    holdings_with_values = []
    total_invested = 0
    total_current_value = 0
    
    for holding in holdings:
        invested_amount = holding.quantity * holding.avg_price
        total_invested += invested_amount
        
        # Get current price from Yahoo Finance
        try:
            stock_info = db.query(models.StockInfo).filter(
                models.StockInfo.symbol == holding.symbol
            ).first()
            
            if stock_info:
                ticker = yf.Ticker(stock_info.yahoo_symbol)
                info = ticker.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice') or holding.avg_price
            else:
                current_price = holding.avg_price
        except:
            current_price = holding.avg_price
        
        current_value = holding.quantity * current_price
        total_current_value += current_value
        
        returns = current_value - invested_amount
        returns_percent = (returns / invested_amount) * 100 if invested_amount > 0 else 0
        
        holdings_with_values.append({
            "id": holding.id,
            "symbol": holding.symbol,
            "stock_name": holding.stock_name,
            "quantity": holding.quantity,
            "avg_price": holding.avg_price,
            "current_price": current_price,
            "invested_amount": invested_amount,
            "current_value": current_value,
            "returns": returns,
            "returns_percent": returns_percent
        })
    
    # Calculate summary
    total_returns = total_current_value - total_invested
    total_returns_percent = (total_returns / total_invested) * 100 if total_invested > 0 else 0
    
    # TODO: Calculate day returns (requires previous day's values)
    day_returns = 0
    day_returns_percent = 0
    
    return {
        "holdings": holdings_with_values,
        "summary": {
            "total_holdings": len(holdings),
            "total_invested": round(total_invested, 2),
            "total_current_value": round(total_current_value, 2),
            "total_returns": round(total_returns, 2),
            "total_returns_percent": round(total_returns_percent, 2),
            "day_returns": round(day_returns, 2),
            "day_returns_percent": round(day_returns_percent, 2)
        }
    }


@router.delete("/holdings/{holding_id}")
async def delete_holding(
    holding_id: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    """
    Delete a holding from portfolio
    """
    holding = db.query(models.Portfolio).filter(
        models.Portfolio.id == holding_id,
        models.Portfolio.user_id == user_id
    ).first()
    
    if not holding:
        raise HTTPException(status_code=404, detail="Holding not found")
    
    db.delete(holding)
    db.commit()
    
    return {"message": "Holding deleted successfully"}


@router.put("/holdings/{holding_id}")
async def update_holding(
    holding_id: int,
    quantity: int,
    avg_price: float,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    """
    Update quantity and average price of a holding
    """
    holding = db.query(models.Portfolio).filter(
        models.Portfolio.id == holding_id,
        models.Portfolio.user_id == user_id
    ).first()
    
    if not holding:
        raise HTTPException(status_code=404, detail="Holding not found")
    
    holding.quantity = quantity
    holding.avg_price = avg_price
    
    db.commit()
    db.refresh(holding)
    
    return {
        "message": "Holding updated successfully",
        "holding": {
            "id": holding.id,
            "symbol": holding.symbol,
            "stock_name": holding.stock_name,
            "quantity": holding.quantity,
            "avg_price": holding.avg_price
        }
    }