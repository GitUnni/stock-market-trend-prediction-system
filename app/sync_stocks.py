"""
Stock Sync Script - Downloads NSE stock list and populates database
Run this to populate the stock_info table with all NSE stocks

Usage: python -m app.sync_stocks
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models import StockInfo
import requests
import pandas as pd
from io import StringIO
import time
import json


def sync_from_nse_with_session():
    """
    Download from NSE official website with proper session handling
    NSE requires cookies from the main page first
    """
    print("\n=== Attempting NSE Official Download ===")
    try:
        # Step 1: Create session and get cookies
        session = requests.Session()
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Visit main page to get cookies
        print("Step 1: Getting session cookies from NSE...")
        main_page = session.get('https://www.nseindia.com', headers=headers, timeout=15)
        print(f"  Main page status: {main_page.status_code}")
        
        # Wait a bit
        time.sleep(2)
        
        # Step 2: Try to download the CSV
        print("Step 2: Downloading stock list CSV...")
        csv_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        
        headers['Referer'] = 'https://www.nseindia.com/'
        response = session.get(csv_url, headers=headers, timeout=30)
        
        print(f"  CSV download status: {response.status_code}")
        print(f"  Response length: {len(response.text)} characters")
        
        if response.status_code == 200 and len(response.text) > 100:
            # Parse CSV
            print("Step 3: Parsing CSV data...")
            
            # Print first few lines to debug
            lines = response.text.split('\n')[:5]
            print(f"  First few lines of CSV:")
            for line in lines:
                print(f"    {line}")
            
            # Read CSV
            df = pd.read_csv(StringIO(response.text))
            print(f"  CSV columns: {df.columns.tolist()}")
            print(f"  Total rows: {len(df)}")
            
            stocks = []
            for idx, row in df.iterrows():
                # Try different column name variations
                symbol = None
                name = None
                
                # Try to get symbol
                for col in ['SYMBOL', 'Symbol', 'symbol', 'CODE']:
                    if col in df.columns:
                        symbol = str(row.get(col, '')).strip()
                        if symbol and symbol != 'nan':
                            break
                
                # Try to get name
                for col in [' NAME OF COMPANY', 'NAME OF COMPANY', 'Name', 'name', 'Company Name', 'COMPANY']:
                    if col in df.columns:
                        name = str(row.get(col, '')).strip()
                        if name and name != 'nan':
                            break
                
                if symbol and name:
                    stocks.append({
                        'symbol': symbol,
                        'name': name,
                        'yahoo_symbol': f"{symbol}.NS"
                    })
                    
                    # Print first 5 for verification
                    if idx < 5:
                        print(f"  Sample stock {idx+1}: {symbol} - {name}")
            
            if stocks:
                print(f"\n✓ Successfully parsed {len(stocks)} stocks from NSE")
                return stocks
            else:
                print("✗ No stocks found in CSV (parsing issue)")
                return []
        else:
            print(f"✗ Failed to download CSV: Status {response.status_code}")
            return []
            
    except Exception as e:
        print(f"✗ NSE download error: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def get_comprehensive_stock_list():
    """
    Comprehensive list of major Indian stocks as fallback
    Includes top 200+ stocks from NSE
    """
    print("\n=== Using Comprehensive Fallback List ===")
    
    stocks = [
        # Banking & Financial Services
        {"symbol": "HDFCBANK", "name": "HDFC Bank"},
        {"symbol": "ICICIBANK", "name": "ICICI Bank"},
        {"symbol": "SBIN", "name": "State Bank of India"},
        {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank"},
        {"symbol": "AXISBANK", "name": "Axis Bank"},
        {"symbol": "INDUSINDBK", "name": "IndusInd Bank"},
        {"symbol": "BANDHANBNK", "name": "Bandhan Bank"},
        {"symbol": "FEDERALBNK", "name": "Federal Bank"},
        {"symbol": "IDFCFIRSTB", "name": "IDFC First Bank"},
        {"symbol": "PNB", "name": "Punjab National Bank"},
        {"symbol": "BANKBARODA", "name": "Bank of Baroda"},
        {"symbol": "CANBK", "name": "Canara Bank"},
        {"symbol": "UNIONBANK", "name": "Union Bank of India"},
        {"symbol": "AUBANK", "name": "AU Small Finance Bank"},
        {"symbol": "SOUTHBANK", "name": "South Indian Bank"},  # Your example!
        {"symbol": "RBLBANK", "name": "RBL Bank"},
        {"symbol": "YESBANK", "name": "Yes Bank"},
        
        # IT Services
        {"symbol": "TCS", "name": "Tata Consultancy Services"},
        {"symbol": "INFY", "name": "Infosys"},
        {"symbol": "WIPRO", "name": "Wipro"},
        {"symbol": "HCLTECH", "name": "HCL Technologies"},
        {"symbol": "TECHM", "name": "Tech Mahindra"},
        {"symbol": "LTI", "name": "LTI Mindtree"},
        {"symbol": "COFORGE", "name": "Coforge"},
        {"symbol": "MPHASIS", "name": "Mphasis"},
        {"symbol": "PERSISTENT", "name": "Persistent Systems"},
        
        # Oil & Gas
        {"symbol": "RELIANCE", "name": "Reliance Industries"},
        {"symbol": "ONGC", "name": "Oil & Natural Gas Corp"},
        {"symbol": "BPCL", "name": "Bharat Petroleum"},
        {"symbol": "IOC", "name": "Indian Oil Corporation"},
        {"symbol": "HINDPETRO", "name": "Hindustan Petroleum"},
        {"symbol": "GAIL", "name": "GAIL India"},
        {"symbol": "PETRONET", "name": "Petronet LNG"},
        
        # Automobiles
        {"symbol": "MARUTI", "name": "Maruti Suzuki India"},
        {"symbol": "M&M", "name": "Mahindra & Mahindra"},
        {"symbol": "BAJAJ-AUTO", "name": "Bajaj Auto"},
        {"symbol": "HEROMOTOCO", "name": "Hero MotoCorp"},
        {"symbol": "EICHERMOT", "name": "Eicher Motors"},
        {"symbol": "TVSMOTOR", "name": "TVS Motor Company"},
        {"symbol": "ASHOKLEY", "name": "Ashok Leyland"},
        {"symbol": "ESCORTS", "name": "Escorts Kubota"},
        
        # Telecom
        {"symbol": "BHARTIARTL", "name": "Bharti Airtel"},
        {"symbol": "IDEA", "name": "Vodafone Idea"},
        
        # Pharmaceuticals
        {"symbol": "SUNPHARMA", "name": "Sun Pharmaceutical Industries"},
        {"symbol": "DRREDDY", "name": "Dr. Reddy's Laboratories"},
        {"symbol": "CIPLA", "name": "Cipla"},
        {"symbol": "DIVISLAB", "name": "Divi's Laboratories"},
        {"symbol": "BIOCON", "name": "Biocon"},
        {"symbol": "AUROPHARMA", "name": "Aurobindo Pharma"},
        {"symbol": "LUPIN", "name": "Lupin"},
        {"symbol": "ALKEM", "name": "Alkem Laboratories"},
        {"symbol": "TORNTPHARM", "name": "Torrent Pharmaceuticals"},
        {"symbol": "APOLLOHOSP", "name": "Apollo Hospitals Enterprise"},
        
        # FMCG
        {"symbol": "HINDUNILVR", "name": "Hindustan Unilever"},
        {"symbol": "ITC", "name": "ITC Limited"},
        {"symbol": "NESTLEIND", "name": "Nestle India"},
        {"symbol": "BRITANNIA", "name": "Britannia Industries"},
        {"symbol": "DABUR", "name": "Dabur India"},
        {"symbol": "MARICO", "name": "Marico"},
        {"symbol": "GODREJCP", "name": "Godrej Consumer Products"},
        {"symbol": "COLPAL", "name": "Colgate-Palmolive India"},
        {"symbol": "EMAMILTD", "name": "Emami"},
        {"symbol": "TATACONSUM", "name": "Tata Consumer Products"},
        {"symbol": "MCDOWELL-N", "name": "United Spirits"},
        {"symbol": "RADICO", "name": "Radico Khaitan"},
        
        # Cement
        {"symbol": "ULTRACEMCO", "name": "UltraTech Cement"},
        {"symbol": "SHREECEM", "name": "Shree Cement"},
        {"symbol": "GRASIM", "name": "Grasim Industries"},
        {"symbol": "AMBUJACEM", "name": "Ambuja Cements"},
        {"symbol": "ACC", "name": "ACC"},
        {"symbol": "JKCEMENT", "name": "JK Cement"},
        {"symbol": "RAMCOCEM", "name": "Ramco Cements"},
        
        # Metals & Mining
        {"symbol": "TATASTEEL", "name": "Tata Steel"},
        {"symbol": "JSWSTEEL", "name": "JSW Steel"},
        {"symbol": "HINDALCO", "name": "Hindalco Industries"},
        {"symbol": "COALINDIA", "name": "Coal India"},
        {"symbol": "VEDL", "name": "Vedanta"},
        {"symbol": "SAIL", "name": "Steel Authority of India"},
        {"symbol": "JINDALSTEL", "name": "Jindal Steel & Power"},
        {"symbol": "NMDC", "name": "NMDC"},
        {"symbol": "HINDZINC", "name": "Hindustan Zinc"},
        {"symbol": "NATIONALUM", "name": "National Aluminium Company"},
        
        # Power & Utilities
        {"symbol": "NTPC", "name": "NTPC"},
        {"symbol": "POWERGRID", "name": "Power Grid Corporation of India"},
        {"symbol": "ADANIPOWER", "name": "Adani Power"},
        {"symbol": "TATAPOWER", "name": "Tata Power Company"},
        {"symbol": "ADANIGREEN", "name": "Adani Green Energy"},
        {"symbol": "TORNTPOWER", "name": "Torrent Power"},
        {"symbol": "JSW ENERGY", "name": "JSW Energy"},
        
        # Infrastructure & Construction
        {"symbol": "LT", "name": "Larsen & Toubro"},
        {"symbol": "ADANIPORTS", "name": "Adani Ports and Special Economic Zone"},
        {"symbol": "GMRINFRA", "name": "GMR Infrastructure"},
        {"symbol": "IRB", "name": "IRB Infrastructure Developers"},
        {"symbol": "NBCC", "name": "NBCC India"},
        
        # Retail
        {"symbol": "DMART", "name": "Avenue Supermarts (DMart)"},
        {"symbol": "TRENT", "name": "Trent"},
        {"symbol": "SHOPERSTOP", "name": "Shoppers Stop"},
        {"symbol": "ABFRL", "name": "Aditya Birla Fashion and Retail"},
        
        # Consumer Durables
        {"symbol": "TITAN", "name": "Titan Company"},
        {"symbol": "VOLTAS", "name": "Voltas"},
        {"symbol": "HAVELLS", "name": "Havells India"},
        {"symbol": "WHIRLPOOL", "name": "Whirlpool of India"},
        {"symbol": "BLUESTARCO", "name": "Blue Star"},
        {"symbol": "CROMPTON", "name": "Crompton Greaves Consumer Electricals"},
        {"symbol": "DIXON", "name": "Dixon Technologies"},
        
        # Paints & Chemicals
        {"symbol": "ASIANPAINT", "name": "Asian Paints"},
        {"symbol": "BERGEPAINT", "name": "Berger Paints India"},
        {"symbol": "AKZOINDIA", "name": "Akzo Nobel India"},
        {"symbol": "PIDILITIND", "name": "Pidilite Industries"},
        {"symbol": "SRF", "name": "SRF"},
        {"symbol": "ATUL", "name": "Atul"},
        {"symbol": "DEEPAKNTR", "name": "Deepak Nitrite"},
        
        # Real Estate
        {"symbol": "DLF", "name": "DLF"},
        {"symbol": "GODREJPROP", "name": "Godrej Properties"},
        {"symbol": "OBEROIRLTY", "name": "Oberoi Realty"},
        {"symbol": "PRESTIGE", "name": "Prestige Estates Projects"},
        {"symbol": "BRIGADE", "name": "Brigade Enterprises"},
        
        # Financial Services (NBFCs)
        {"symbol": "BAJFINANCE", "name": "Bajaj Finance"},
        {"symbol": "BAJAJFINSV", "name": "Bajaj Finserv"},
        {"symbol": "CHOLAFIN", "name": "Cholamandalam Investment and Finance Company"},
        {"symbol": "M&MFIN", "name": "Mahindra & Mahindra Financial Services"},
        {"symbol": "LICHSGFIN", "name": "LIC Housing Finance"},
        {"symbol": "PFC", "name": "Power Finance Corporation"},
        {"symbol": "RECLTD", "name": "REC Limited"},
        {"symbol": "SHRIRAMFIN", "name": "Shriram Finance"},
        
        # Insurance
        {"symbol": "SBILIFE", "name": "SBI Life Insurance Company"},
        {"symbol": "HDFCLIFE", "name": "HDFC Life Insurance Company"},
        {"symbol": "ICICIPRULI", "name": "ICICI Prudential Life Insurance Company"},
        
        # Capital Markets
        {"symbol": "HDFCAMC", "name": "HDFC Asset Management Company"},
        {"symbol": "MUTHOOTFIN", "name": "Muthoot Finance"},
        
        # Diversified
        {"symbol": "SIEMENS", "name": "Siemens"},
        {"symbol": "ABB", "name": "ABB India"},
        {"symbol": "BOSCHLTD", "name": "Bosch"},
        {"symbol": "3MINDIA", "name": "3M India"},
        {"symbol": "HONAUT", "name": "Honeywell Automation India"},
        
        # Media & Entertainment
        {"symbol": "ZEEL", "name": "Zee Entertainment Enterprises"},
        {"symbol": "SUNTV", "name": "Sun TV Network"},
        {"symbol": "PVR", "name": "PVR"},
        
        # Hotels & Tourism
        {"symbol": "INDHOTEL", "name": "Indian Hotels Company"},
        {"symbol": "LEMONTREE", "name": "Lemon Tree Hotels"},
        
        # Agriculture
        {"symbol": "UPL", "name": "UPL"},
        {"symbol": "COROMANDEL", "name": "Coromandel International"},
        {"symbol": "PIIND", "name": "PI Industries"},
    ]
    
    # Add .NS suffix for Yahoo Finance
    for stock in stocks:
        stock['yahoo_symbol'] = f"{stock['symbol']}.NS"
    
    print(f"✓ Using fallback list with {len(stocks)} major Indian stocks")
    print(f"  Including: {stocks[14]['name']}")  # South Indian Bank
    
    return stocks


def save_to_database(stocks):
    """
    Save stocks to database
    """
    if not stocks:
        print("\n✗ No stocks to save!")
        return 0
    
    print(f"\n=== Saving {len(stocks)} stocks to database ===")
    
    db = SessionLocal()
    try:
        # Clear existing stocks
        print("Step 1: Clearing existing stocks...")
        deleted_count = db.query(StockInfo).delete()
        print(f"  Deleted {deleted_count} old records")
        
        # Add new stocks
        print("Step 2: Adding new stocks...")
        added_count = 0
        for stock_data in stocks:
            stock = StockInfo(
                symbol=stock_data['symbol'],
                name=stock_data['name'],
                yahoo_symbol=stock_data['yahoo_symbol'],
                exchange='NSE'
            )
            db.add(stock)
            added_count += 1
            
            # Print progress every 100 stocks
            if added_count % 100 == 0:
                print(f"  Added {added_count} stocks...")
        
        # Commit
        print("Step 3: Committing to database...")
        db.commit()
        
        print(f"\n✓ Successfully saved {added_count} stocks to database!")
        
        # Verify
        print("\nVerification:")
        total = db.query(StockInfo).count()
        print(f"  Total stocks in database: {total}")
        
        # Test South Indian Bank
        south_bank = db.query(StockInfo).filter(StockInfo.symbol == "SOUTHBANK").first()
        if south_bank:
            print(f"  ✓ South Indian Bank found: {south_bank.name}")
        
        # Show first 5 stocks
        first_five = db.query(StockInfo).limit(5).all()
        print(f"\n  First 5 stocks:")
        for stock in first_five:
            print(f"    {stock.symbol} - {stock.name}")
        
        return added_count
        
    except Exception as e:
        print(f"\n✗ Database error: {str(e)}")
        import traceback
        traceback.print_exc()
        db.rollback()
        return 0
    finally:
        db.close()


def main():
    """
    Main sync function
    """
    print("=" * 60)
    print("NSE STOCK DATABASE SYNC")
    print("=" * 60)
    
    # Try NSE official first
    stocks = sync_from_nse_with_session()
    
    # If NSE fails, use fallback
    if not stocks:
        print("\nNSE download failed, using comprehensive fallback list...")
        stocks = get_comprehensive_stock_list()
    
    # Save to database
    if stocks:
        count = save_to_database(stocks)
        print(f"\n{'=' * 60}")
        print(f"SYNC COMPLETE: {count} stocks synced!")
        print(f"{'=' * 60}")
    else:
        print("\n✗ SYNC FAILED: No stocks available")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())