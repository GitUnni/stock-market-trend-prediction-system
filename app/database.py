from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/app.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,    # Detects dropped connections and reconnects automatically
    pool_recycle=300,      # Recycle connections every 5 min (Neon drops idle after ~5 min)
    pool_size=5,           # Max persistent connections in the pool
    max_overflow=10,       # Extra connections allowed under heavy load
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()