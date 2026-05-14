from sqlalchemy import create_engine, MetaData
from dotenv import load_dotenv
import os

load_dotenv()

# SQLite source
sqlite_engine = create_engine("sqlite:///data/app.db")

# Neon destination
DATABASE_URL = os.getenv("DATABASE_URL")

postgres_engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

sqlite_meta = MetaData()
sqlite_meta.reflect(bind=sqlite_engine)

sqlite_conn = sqlite_engine.connect()
postgres_conn = postgres_engine.connect()

for table in sqlite_meta.sorted_tables:

    if table.name == "alembic_version":
        continue

    print(f"Migrating table: {table.name}")

    rows = sqlite_conn.execute(table.select()).fetchall()

    if rows:
        data = [dict(row._mapping) for row in rows]

        postgres_table = table.to_metadata(MetaData())
        postgres_table.create(bind=postgres_engine, checkfirst=True)

        postgres_conn.execute(postgres_table.insert(), data)
        postgres_conn.commit()

print("Migration completed!")