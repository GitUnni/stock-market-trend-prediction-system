# 📈 Stock Market Price Trend Prediction and Backtesting Strategy System

A comprehensive, AI-driven web application for stock market analytics, price trend prediction, portfolio management, and real-time market intelligence. Built with **FastAPI**, powered by ensemble machine learning models (**Prophet + XGBoost + LSTM**), and enriched with AI-based news sentiment analysis and equity research tools.

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [User Roles](#user-roles)
- [API Keys Guide](#api-keys-guide)
- [License](#license)

---

## About the Project

The **Stock Market Price Trend Prediction and Backtesting Strategy System** is a full-stack financial intelligence platform designed to bridge the gap between retail investors and institutional participants. The system provides real-time stock data, ensemble-based price prediction, portfolio tracking, AI-powered equity research, sentiment-driven news classification, and a query-based stock screener — all within a single unified platform.

The platform supports three distinct user roles: **Customer**, **Financial Institution**, and **Admin**, each with tailored capabilities and access controls.

---

## Key Features

### 👤 Customer
- Search stocks and view detailed price charts with full fundamental metrics
- **Portfolio Management**:
  - Number of holdings
  - Total portfolio value (invested amount + returns)
  - Total returns and total invested amount
  - Per-stock: quantity, invested amount, average buy price, total returns, current value
- Fundamental metrics: Today's Low/High, 52-Week Low/High, Market Cap, Volume, Sector, Industry, Dividend Yield, P/E Ratio, P/B Ratio, Profit Margin, Revenue Growth, Debt-to-Equity, Book Value/Share
- **Price Prediction** using an ensemble of:
  - Facebook Prophet (time-series decomposition)
  - XGBoost (gradient boosting)
  - LSTM (deep learning via PyTorch)
  - Final ensembled model combining all three
- Prediction metrics: MAE, RMSE, MAPE, Direction Accuracy, Timing Accuracy, ROC-AUC, Signal Strength, Confidence Score
- Simulated **Backtesting** results to validate strategy effectiveness
- **Stock Screener**: Query-based filtering (e.g., `dividend_yield > 0.03`)
- **News Section** with AI-generated summaries and sentiment classification (All, Indian, Global, Positive, Negative)

### 🏦 Financial Institution
- Search stocks and view detailed price charts with full fundamental metrics
- **Portfolio Management**:
  - Number of holdings
  - Total portfolio value (invested amount + returns)
  - Total returns and total invested amount
  - Per-stock: quantity, invested amount, average buy price, total returns, current value
  - Fundamental metrics: Today's Low/High, 52-Week Low/High, Market Cap, Volume, Sector, Industry, Dividend Yield, P/E Ratio, P/B Ratio, Profit Margin, Revenue Growth, Debt-to-Equity, Book Value/Share
- **Equity Research Tool**: AI-powered research interface for any stock market topic (powered by Groq + Tavily)
- **News Section** with AI-generated summaries and sentiment classification (All, Indian, Global, Positive, Negative)

### 🔐 Admin
- View all registered users in the system
- Delete users
- **Approve or Reject** Financial Institution registration requests (FIs cannot access the system until approved)

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | HTML, CSS, JavaScript, Jinja2 Templates |
| **Database / ORM** | SQLAlchemy, PostgreSQL via `psycopg` (production-ready), SQLite (local development) |
| **Database Migrations** | Alembic |
| **Caching / Ephemeral State** | Upstash Redis (`upstash-redis`) for verification/reset code storage with TTL |
| **Authentication & Security** | Passlib + bcrypt password hashing, Python-JOSE (JWT), HTTP Bearer auth, Gmail SMTP for OTP/verification emails |
| **ML Models** | Prophet, XGBoost, PyTorch (LSTM), Scikit-learn |
| **Data** | yfinance, pandas, pandas-market-calendars |
| **Visualisation** | Plotly |
| **AI / LLM Orchestration** | LangChain, LangChain-Community, LangChain-Groq, LangGraph |
| **News API** | MarketAux |
| **Research** | Groq API, Tavily API, DuckDuckGo Search (`duckduckgo-search`, `ddgs`), `curl-cffi` |
| **Validation & Config** | Pydantic schemas (FastAPI), python-dotenv, email-validator |
| **Utilities** | BeautifulSoup4, lxml, Requests |
| **Package Manager** | uv |
| **Python Version** | 3.12 |

---

## Project Structure

```
stock-market-trend-prediction-system/
├── alembic/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       └── 0001_initial_schema.py
├── app/
│   ├── .env                       # Environment variables (NOT committed — create manually)
│   ├── main.py                    # FastAPI app entry point
│   ├── auth.py                    # JWT authentication & database session
│   ├── models.py                  # SQLAlchemy database models
│   ├── database.py                # SQLAlchemy engine/session/base
│   ├── deps.py                    # Shared FastAPI dependencies
│   ├── models.py                  # SQLAlchemy models
│   ├── schemas.py                 # Pydantic request/response schemas
│   ├── manualadd.py               # Manual admin/bootstrap helpers
│   ├── sync_stocks.py             # Stock master sync job
│   ├── update_stock_metrics.py    # Stock metrics update job
│   ├── routes/
│   │   └── __init__.py
│   │   ├── auth.py           # Auth routes (login, register, email verification)
│   │   ├── broadcasts.py
│   │   ├── feedback.py
│   │   ├── miscellaneous.py
│   │   ├── news.py
│   │   ├── pages.py
│   │   ├── portfolio.py
│   │   ├── predict.py
│   │   ├── research.py
│   │   ├── screener.py
│   │   ├── stocks.py
│   │   ├── test.py
│   ├── templates/            # Jinja2 HTML templates
│   │   ├── dashboards/
│   │   │   ├── admin_dashboard.html
│   │   │   ├── customer_dashboard.html
│   │   │   └── fin_dashboard.html
│   └── static/
│       ├── css/
│       ├── js/
│       └── img/
├── .gitignore
├── alembic.ini
├── .python-version           # Python 3.12
├── pyproject.toml            # Project dependencies (uv)
└── README.md
```

---

## Prerequisites

Before you begin, make sure you have the following installed:

- **Python 3.12** — [Download](https://www.python.org/downloads/)
- **uv** (fast Python package manager) — Install with:
  ```bash
  pip install uv
  ```
  Or using the official installer(pip command above is faster and error free):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  > On Windows, use: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/GitUnni/stock-market-trend-prediction-system.git
cd stock-market-trend-prediction-system
```

### 2. Create a Virtual Environment using uv

```bash
uv venv
```

> No need to manually activate the virtual environment. `uv run` (used in the next steps) automatically resolves and uses the project's `.venv` for you without any conflicts. If you have installed python extension in vscode then it will automatically activate virtual environment each time a new terminal is opened. To deactivate it simply type the below command: 

```bash
deactivate
```
> This will deactivate any conflicting virtual env and run cleanly the `.venv` we will use `uv run` (used in the next steps)

### 3. Install Dependencies

```bash
uv sync
```

This will install all dependencies listed in `pyproject.toml`, including FastAPI, Prophet, XGBoost, PyTorch (LSTM), LangChain, and all other required packages.

### 4. Set Up the Environment Variables

Create a `.env` file inside the `app/` folder:

```bash
# Navigate to the app folder
cd app

# Create the .env file (Linux/macOS)
touch .env

# Or on Windows
type nul > .env
```

Then populate it with your keys (see [Environment Variables](#environment-variables) below).

---

## Environment Variables

Create a file at `app/.env` with the following contents. Replace each `your_key` / `your_gmail` with your actual credentials without any space and `""`:

```env
# Groq API key — used in research.py for AI-powered Equity Research Tool
GROQ_API_KEY=your_key

# Tavily API key — used in research.py for web search during equity research (monthly refresh)
TAVILY_API_KEY=your_key

# MarketAux API key — used in news.py for fetching Indian & Global market news (daily refresh)
MARKETAUX_API_KEY=your_key

# Gmail credentials — used in routes/auth.py for sending verification/notification emails
sender_email=your_gmail
sender_password=your_key

# Secret key — used in auth.py for JWT token signing and session security
SECRET_KEY=your_key

# Upstash Redis REST URL — used in routes/auth.py for distributed verification/reset code storage
UPSTASH_REDIS_REST_URL=your_upstash_redis_rest_url

# Upstash Redis REST token — used in routes/auth.py to authenticate REST calls to Upstash
UPSTASH_REDIS_REST_TOKEN=your_upstash_redis_rest_token

# Verification/reset code TTL (in seconds) — optional; default is 600 (10 minutes)
VERIFICATION_CODE_TTL_SECONDS=600

# Maximum verification/reset attempts before lockout — optional; default is 5
MAX_VERIFICATION_ATTEMPTS=5
```

> ⚠️ **Important:** Never commit your `.env` file to version control. It is already listed in `.gitignore` to prevent accidental exposure.

### Where to Get Each Key

| Key | Source |
|---|---|
| `GROQ_API_KEY` | [https://console.groq.com](https://console.groq.com) — Free tier available |
| `TAVILY_API_KEY` | [https://app.tavily.com](https://app.tavily.com) — Free tier available (monthly refresh) |
| `MARKETAUX_API_KEY` | [https://www.marketaux.com](https://www.marketaux.com) — Free tier available (daily refresh) |
| `sender_email` | Your Gmail address used to send system emails |
| `sender_password` | Gmail App Password (not your regular password) — Generate at [Google Account → Security → App Passwords](https://myaccount.google.com/apppasswords) |
| `SECRET_KEY` | Generate a random strong string locally (example: `python -c "import secrets; print(secrets.token_urlsafe(32))"`) |
| `UPSTASH_REDIS_REST_URL` | [https://console.upstash.com](https://console.upstash.com) → Redis DB → REST API section |
| `UPSTASH_REDIS_REST_TOKEN` | [https://console.upstash.com](https://console.upstash.com) → Redis DB → REST API section |
| `VERIFICATION_CODE_TTL_SECONDS` | Local config value (optional). Keep default `600` unless you need a different expiry window |
| `MAX_VERIFICATION_ATTEMPTS` | Local config value (optional). Keep default `5` unless you need stricter/looser lockout |

---

## Running the Application

From the **root directory** of the project (where `pyproject.toml` is located), run:

```bash
uv run uvicorn app.main:app --reload
```

The application will start at: **[http://localhost:8000](http://localhost:8000)**

- `--reload` enables hot-reloading during development (auto-restarts on code changes)
- To run on a different port: `uv run uvicorn app.main:app --reload --port 8080`

---

Individual scripts to run for populating the database after running 
```bash      
uv run python -m app.sync_stocks		#Scraping NSE's stock data into stock_info table daily update
uv run python -m app.manualadd			#Adding admin manually
uv run python -m app.update_stock_metrics	#Stock metrics daily update
```

## User Roles

### Customer
Registers directly and gets immediate access to stock search, price prediction, backtesting, screener, and news features after email verification.

### Financial Institution
Registers and submits a request for approval. **The account remains inactive until the Admin approves the request.** Upon approval, the Financial Institution gains access to portfolio management, equity research, and news features.

### Admin
A pre-configured admin (manualadd.py) account manages user oversight, approves or rejects Financial Institution registration requests, and can delete users from the system.

---

## API Keys Guide

This section explains how to generate each credential used by the project and where it maps in `app/.env`.

### Upstash Redis (Email Verification / Password Reset Code Storage)
1. Sign in at [console.upstash.com](https://console.upstash.com)
2. Create a **Redis** database in the region closest to your deployment
3. Open the database → **REST API** section
4. Copy:
   - `UPSTASH_REDIS_REST_URL`
   - `UPSTASH_REDIS_REST_TOKEN`
5. Add them to `app/.env`

### Gmail App Password Setup
1. Go to your [Google Account](https://myaccount.google.com/)
2. Navigate to **Security** → **2-Step Verification** (must be enabled)
3. At the bottom, click **App Passwords**
4. Select **Mail** and your device, then click **Generate**
5. Use the 16-character password as `sender_password` in your `.env`

### JWT Secret Key (`SECRET_KEY`)
- Add a random string or generate a secure random key locally by typing the below command in terminal:
  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```
- Paste it as `SECRET_KEY` in `app/.env`
- Keep this value private and rotate it if exposed

### Groq API (LLM for Equity Research)
- Sign up at [console.groq.com](https://console.groq.com)
- Create an API key from the dashboard
- Free tier is sufficient for development

### Tavily API (Web Search for Research)
- Sign up at [app.tavily.com](https://app.tavily.com)
- Get your API key from the dashboard
- Free tier refreshes monthly

### MarketAux API (News)
- Sign up at [marketaux.com](https://www.marketaux.com)
- Get your API key from the dashboard
- Free tier refreshes daily

### Optional Security Tuning Vars (No external provider required)
- `VERIFICATION_CODE_TTL_SECONDS`: default `600` (10 minutes)
- `MAX_VERIFICATION_ATTEMPTS`: default `5`
- Set these in `app/.env` only if you want behavior different from defaults

---

## License

This project is intended for academic and educational purposes.

---

> **Disclaimer:** This system is built for educational purposes only and should not be considered financial advice. Always consult a qualified financial advisor before making investment decisions.
---

## Database & Migrations (Production)

For production deployment (e.g., Render), use **PostgreSQL** instead of SQLite.

### 1) Set DATABASE_URL

Set `DATABASE_URL` in your deployment environment (example with Neon):

```env
DATABASE_URL=postgresql+psycopg://<user>:<password>@<host>/<db>?sslmode=require
```

> Local development can still use SQLite by leaving `DATABASE_URL` unset.

### 2) Run Alembic migrations

This project now includes Alembic config and an initial migration.

```bash
alembic upgrade head
```


### Existing SQLite database already has tables?

If your database was created before Alembic (tables already exist), run:

```bash
alembic stamp head
```

`stamp` marks the current schema version without trying to recreate tables.
After stamping, future schema changes should use normal migrations (`revision` + `upgrade`).

### 3) Create new migrations when models change

```bash
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```

Use migrations for schema changes instead of `Base.metadata.create_all(...)` in app startup.
