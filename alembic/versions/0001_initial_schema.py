"""Initial schema

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-05-09
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if "broadcasts" not in inspector.get_table_names():
        op.create_table(
        "broadcasts",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
    )
        op.create_index("ix_broadcasts_id", "broadcasts", ["id"])

    if "feedback" not in inspector.get_table_names():
        op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("user_name", sa.String(), nullable=False),
        sa.Column("user_email", sa.String(), nullable=False),
        sa.Column("feedback_type", sa.String(), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=True),
        sa.Column("subject", sa.String(), nullable=True),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("admin_reply", sa.String(), nullable=True),
        sa.Column("replied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
    )
        op.create_index("ix_feedback_id", "feedback", ["id"])
        op.create_index("ix_feedback_user_id", "feedback", ["user_id"])

    if "stock_info" not in inspector.get_table_names():
        op.create_table(
        "stock_info",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("yahoo_symbol", sa.String(), nullable=False),
        sa.Column("exchange", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
        op.create_index("ix_stock_info_id", "stock_info", ["id"])
        op.create_index("ix_stock_info_symbol", "stock_info", ["symbol"], unique=True)

    if "stock_metrics" not in inspector.get_table_names():
        op.create_table(
        "stock_metrics",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column("current_price", sa.Float(), nullable=True),
        sa.Column("previous_close", sa.Float(), nullable=True),
        sa.Column("open_price", sa.Float(), nullable=True),
        sa.Column("day_high", sa.Float(), nullable=True),
        sa.Column("day_low", sa.Float(), nullable=True),
        sa.Column("week_52_high", sa.Float(), nullable=True),
        sa.Column("week_52_low", sa.Float(), nullable=True),
        sa.Column("volume", sa.Float(), nullable=True),
        sa.Column("avg_volume", sa.Float(), nullable=True),
        sa.Column("market_cap", sa.Float(), nullable=True),
        sa.Column("pe_ratio", sa.Float(), nullable=True),
        sa.Column("pb_ratio", sa.Float(), nullable=True),
        sa.Column("price_to_sales", sa.Float(), nullable=True),
        sa.Column("peg_ratio", sa.Float(), nullable=True),
        sa.Column("profit_margin", sa.Float(), nullable=True),
        sa.Column("operating_margin", sa.Float(), nullable=True),
        sa.Column("return_on_assets", sa.Float(), nullable=True),
        sa.Column("return_on_equity", sa.Float(), nullable=True),
        sa.Column("dividend_yield", sa.Float(), nullable=True),
        sa.Column("dividend_rate", sa.Float(), nullable=True),
        sa.Column("payout_ratio", sa.Float(), nullable=True),
        sa.Column("revenue_growth", sa.Float(), nullable=True),
        sa.Column("earnings_growth", sa.Float(), nullable=True),
        sa.Column("current_ratio", sa.Float(), nullable=True),
        sa.Column("quick_ratio", sa.Float(), nullable=True),
        sa.Column("debt_to_equity", sa.Float(), nullable=True),
        sa.Column("earnings_per_share", sa.Float(), nullable=True),
        sa.Column("book_value_per_share", sa.Float(), nullable=True),
        sa.Column("beta", sa.Float(), nullable=True),
        sa.Column("sector", sa.String(), nullable=True),
        sa.Column("industry", sa.String(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
    )
        op.create_index("ix_stock_metrics_id", "stock_metrics", ["id"])
        op.create_index("ix_stock_metrics_symbol", "stock_metrics", ["symbol"], unique=True)

    if "users" not in inspector.get_table_names():
        op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("is_email_verified", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("institution_name", sa.String(), nullable=True),
        sa.Column("registration_number", sa.String(), nullable=True),
        sa.Column("country", sa.String(), nullable=True),
        sa.Column("contact_person", sa.String(), nullable=True),
        sa.Column("contact_phone", sa.String(), nullable=True),
    )
        op.create_index("ix_users_email", "users", ["email"], unique=True)
        op.create_index("ix_users_id", "users", ["id"])

    if "portfolio" not in inspector.get_table_names():
        op.create_table(
        "portfolio",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column("stock_name", sa.String(), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("avg_price", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("user_id", "symbol", name="uq_user_symbol"),
    )
        op.create_index("ix_portfolio_id", "portfolio", ["id"])
        op.create_index("ix_portfolio_user_id", "portfolio", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_portfolio_user_id", table_name="portfolio")
    op.drop_index("ix_portfolio_id", table_name="portfolio")
    op.drop_table("portfolio")
    op.drop_index("ix_users_id", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
    op.drop_index("ix_stock_metrics_symbol", table_name="stock_metrics")
    op.drop_index("ix_stock_metrics_id", table_name="stock_metrics")
    op.drop_table("stock_metrics")
    op.drop_index("ix_stock_info_symbol", table_name="stock_info")
    op.drop_index("ix_stock_info_id", table_name="stock_info")
    op.drop_table("stock_info")
    op.drop_index("ix_feedback_user_id", table_name="feedback")
    op.drop_index("ix_feedback_id", table_name="feedback")
    op.drop_table("feedback")
    op.drop_index("ix_broadcasts_id", table_name="broadcasts")
    op.drop_table("broadcasts")
