from sqlalchemy import Column, Integer, String, Boolean, Float, Date, DateTime, UniqueConstraint
from sqlalchemy.sql import func
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="CUSTOMER")  
    status = Column(String, default="PENDING_ADMIN_APPROVAL")
    is_email_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Institution-specific (nullable for customers)
    institution_name = Column(String, nullable=True)
    registration_number = Column(String, nullable=True)
    country = Column(String, nullable=True)
    contact_person = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)

class StockInfo(Base):
    """
    Table to store information about all available stocks
    Synced from NSE/BSE periodically
    """
    __tablename__ = "stock_info"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True, nullable=False)  # e.g., "RELIANCE", "SOUTHBANK"
    name = Column(String, nullable=False)  # e.g., "Reliance Industries", "South Indian Bank"
    yahoo_symbol = Column(String, nullable=False)  # e.g., "RELIANCE.NS", "SOUTHBANK.NS"
    exchange = Column(String, default="NSE")  # NSE or BSE
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Portfolio(Base):
    """
    Table to store user's stock holdings/portfolio
    """
    __tablename__ = "portfolio"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)  # Reference to User
    symbol = Column(String, nullable=False)  # Stock symbol (e.g., "RELIANCE")
    stock_name = Column(String, nullable=False)  # Stock name for display
    quantity = Column(Integer, nullable=False)  # Number of shares
    avg_price = Column(Float, nullable=False)  # Average purchase price per share
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("user_id", "symbol", name="uq_user_symbol"),
    )

class StockMetrics(Base):
    """
    Table to store stock metrics for screening
    Updated daily after market close
    """
    __tablename__ = "stock_metrics"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True, nullable=False)
    
    # Price Data
    current_price = Column(Float)
    previous_close = Column(Float)
    open_price = Column(Float)
    day_high = Column(Float)
    day_low = Column(Float)
    week_52_high = Column(Float)
    week_52_low = Column(Float)
    
    # Volume & Trading
    volume = Column(Float)
    avg_volume = Column(Float)
    
    # Market Metrics
    market_cap = Column(Float)
    
    # Valuation Ratios
    pe_ratio = Column(Float)  # Price to Earnings
    pb_ratio = Column(Float)  # Price to Book
    price_to_sales = Column(Float)
    peg_ratio = Column(Float)  # PEG Ratio
    
    # Profitability
    profit_margin = Column(Float)
    operating_margin = Column(Float)
    return_on_assets = Column(Float)  # ROA
    return_on_equity = Column(Float)  # ROE
    
    # Dividends
    dividend_yield = Column(Float)
    dividend_rate = Column(Float)
    payout_ratio = Column(Float)
    
    # Growth
    revenue_growth = Column(Float)
    earnings_growth = Column(Float)
    
    # Financial Health
    current_ratio = Column(Float)
    quick_ratio = Column(Float)
    debt_to_equity = Column(Float)
    
    # Per Share Data
    earnings_per_share = Column(Float)  # EPS
    book_value_per_share = Column(Float)
    
    # Beta (volatility)
    beta = Column(Float)
    
    # Metadata
    sector = Column(String)
    industry = Column(String)
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

class Broadcast(Base):
    """
    Table to store broadcast messages sent by admin to all customers
    """
    __tablename__ = "broadcasts"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())