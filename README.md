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
- Search stocks and view detailed price charts
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
- **News Section** (identical to Customer): AI-summarised, sentiment-classified market news

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
| **Database** | SQLAlchemy (SQLite for local development) |
| **Authentication** | Passlib (bcrypt), Python-JOSE (JWT), Gmail SMTP |
| **ML Models** | Prophet, XGBoost, PyTorch (LSTM), Scikit-learn |
| **Data** | yfinance, pandas, pandas-market-calendars |
| **Visualisation** | Plotly |
| **AI / LLM** | LangChain, LangChain-Groq, LangGraph |
| **News API** | MarketAux |
| **Research** | Groq API, Tavily API, DuckDuckGo Search |
| **Utilities** | Python-dotenv, BeautifulSoup4, lxml, Requests |
| **Package Manager** | uv |
| **Python Version** | 3.12 |

---

## Project Structure

```
stock-market-trend-prediction-system/
├── app/
│   ├── .env                  # Environment variables (NOT committed — create manually)
│   ├── main.py               # FastAPI app entry point
│   ├── auth.py               # JWT authentication & database session
│   ├── models.py             # SQLAlchemy database models
│   ├── news.py               # News fetching & sentiment analysis (MarketAux)
│   ├── research.py           # Equity research tool (Groq + Tavily)
│   ├── routes/
│   │   ├── auth.py           # Auth routes (login, register, email verification)
│   │   ├── admin.py          # Admin routes
│   │   ├── customer.py       # Customer routes
│   │   └── financial.py      # Financial Institution routes
│   ├── templates/            # Jinja2 HTML templates
│   └── static/               # CSS and JavaScript assets
├── .gitignore
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
  Or using the official installer:
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

### 2. Create and Activate a Virtual Environment using uv

```bash
uv venv
```

Activate the virtual environment:

- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```
- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```

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

Create a file at `app/.env` with the following contents. Replace each `your_key` / `your_gmail` with your actual credentials without any space and "":

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
| `SECRET_KEY` | Any long random string — generate with: `python -c "import secrets; print(secrets.token_hex(32))"` |

---

## Running the Application

From the **root directory** of the project (where `pyproject.toml` is located), run:

```bash
uvicorn app.main:app --reload
```

The application will start at: **[http://localhost:8000](http://localhost:8000)**

- `--reload` enables hot-reloading during development (auto-restarts on code changes)
- To run on a different port: `uvicorn app.main:app --reload --port 8080`

---

## User Roles

### Customer
Registers directly and gets immediate access to stock search, price prediction, backtesting, screener, and news features after email verification.

### Financial Institution
Registers and submits a request for approval. **The account remains inactive until the Admin approves the request.** Upon approval, the Financial Institution gains access to portfolio management, equity research, and news features.

### Admin
A pre-configured admin account manages user oversight, approves or rejects Financial Institution registration requests, and can delete users from the system.

---

## API Keys Guide

### Gmail App Password Setup
1. Go to your [Google Account](https://myaccount.google.com/)
2. Navigate to **Security** → **2-Step Verification** (must be enabled)
3. At the bottom, click **App Passwords**
4. Select **Mail** and your device, then click **Generate**
5. Use the 16-character password as `sender_password` in your `.env`

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

---

## License

This project is intended for academic and educational purposes.

---

> **Disclaimer:** This system is built for educational purposes only and should not be considered financial advice. Always consult a qualified financial advisor before making investment decisions.