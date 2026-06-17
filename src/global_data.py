# global_data.py — Fetching global bonds, currencies, commodities, and FRED macro data
import logging
import io
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
import numpy as np
from datetime import datetime

log = logging.getLogger(__name__)

# Standard fallback baseline values (calibrated for current 2026 conditions)
FALLBACK_STATE = {
    "us_10y_yield": 4.43,
    "us_5y_yield": 4.16,
    "us_30y_yield": 4.93,
    "us_3m_yield": 3.63,
    "usd_inr": 94.50,
    "crude_oil_wti": 75.83,
    "crude_oil_brent": 79.61,
    "gold": 4353.10,
    "silver": 69.99,
    "copper": 6.51,
    "india_10y_yield": 7.02,
    "india_5y_yield": 6.87, # 10Y - 15 bps spread
    "india_3m_yield": 6.57, # 10Y - 45 bps spread
    "us_cpi_yoy": 4.17,
    "india_cpi_yoy": 4.20,  # Curated target (FRED is delayed)
    "us_interest_rate": 3.63,
    "india_repo_rate": 6.50, # Curated RBI policy repo rate
    
    # 1-day change percents for widgets
    "us_10y_yield_chg": 0.05,
    "usd_inr_chg": -0.11,
    "crude_oil_wti_chg": 1.38,
    "crude_oil_brent_chg": 1.49,
    "gold_chg": 0.04,
    "silver_chg": -0.33,
    "copper_chg": -0.16,
    "last_updated": "FALLBACK"
}

def _fetch_yf_ticker_price_and_change(ticker_sym: str, default_price: float) -> tuple:
    """Helper to fetch last price and percent change for a ticker from yfinance."""
    price = default_price
    pct_chg = 0.0
    try:
        t = yf.Ticker(ticker_sym)
        # Try fast_info first
        fi = t.fast_info
        price_val = getattr(fi, "last_price", None)
        prev_close_val = getattr(fi, "previous_close", None)
        
        # Fallback to history if fast_info is empty or NaN
        if price_val is None or pd.isna(price_val):
            hist = t.history(period="5d")
            if not hist.empty and len(hist) >= 2:
                price_val = hist["Close"].iloc[-1]
                prev_close_val = hist["Close"].iloc[-2]
            elif not hist.empty:
                price_val = hist["Close"].iloc[-1]
                
        if price_val is not None and not pd.isna(price_val):
            price = float(price_val)
            if prev_close_val is not None and not pd.isna(prev_close_val) and prev_close_val > 0:
                pct_chg = ((price - float(prev_close_val)) / float(prev_close_val)) * 100
    except Exception as e:
        log.warning(f"Error fetching ticker {ticker_sym} from Yahoo Finance: {e}")
    return price, pct_chg

def _get_fred_df(ticker_id: str) -> pd.DataFrame:
    """Download FRED CSV series data."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={ticker_id}"
    resp = requests.get(url, timeout=10)
    if resp.status_code == 200:
        df = pd.read_csv(io.StringIO(resp.text))
        df = df[df[ticker_id] != "."]
        df[ticker_id] = pd.to_numeric(df[ticker_id])
        return df
    raise ConnectionError(f"FRED status code {resp.status_code}")

@st.cache_data(ttl=900)  # 15 minutes cache
def get_global_market_state() -> dict:
    """
    Unified loader for all bond yields, exchange rates, commodities, and CPI macro data.
    Queries Yahoo Finance and FRED, fallback to baseline in case of offline/timeout.
    """
    state = FALLBACK_STATE.copy()
    success = False
    
    # ── 1. Fetch live Yahoo Finance data ──────────────────────────────────
    yf_mappings = {
        "us_10y_yield": ("^TNX", FALLBACK_STATE["us_10y_yield"]),
        "us_5y_yield": ("^FVX", FALLBACK_STATE["us_5y_yield"]),
        "us_30y_yield": ("^TYX", FALLBACK_STATE["us_30y_yield"]),
        "us_3m_yield": ("^IRX", FALLBACK_STATE["us_3m_yield"]),
        "usd_inr": ("USDINR=X", FALLBACK_STATE["usd_inr"]),
        "crude_oil_wti": ("CL=F", FALLBACK_STATE["crude_oil_wti"]),
        "crude_oil_brent": ("BZ=F", FALLBACK_STATE["crude_oil_brent"]),
        "gold": ("GC=F", FALLBACK_STATE["gold"]),
        "silver": ("SI=F", FALLBACK_STATE["silver"]),
        "copper": ("HG=F", FALLBACK_STATE["copper"]),
    }
    
    try:
        for key, (ticker, fallback_val) in yf_mappings.items():
            price, pct_chg = _fetch_yf_ticker_price_and_change(ticker, fallback_val)
            state[key] = price
            # Track change percentage too
            state[f"{key}_chg"] = pct_chg
        success = True
    except Exception as e:
        log.warning(f"Yahoo Finance global state fetch failed: {e}")

    # ── 2. Fetch live FRED macro data ──────────────────────────────────────
    try:
        # US Fed Funds Rate
        df_fed = _get_fred_df("FEDFUNDS")
        if df_fed is not None and not df_fed.empty:
            state["us_interest_rate"] = float(df_fed.iloc[-1]["FEDFUNDS"])
            
        # US Inflation calculation from CPI index
        df_uscpi = _get_fred_df("CPIAUCSL")
        if df_uscpi is not None and len(df_uscpi) > 12:
            latest_val = float(df_uscpi.iloc[-1]["CPIAUCSL"])
            ago_val = float(df_uscpi.iloc[-13]["CPIAUCSL"])
            state["us_cpi_yoy"] = ((latest_val - ago_val) / ago_val) * 100
            
        # India CPI calculation from CPI index
        df_incpi = _get_fred_df("INDCPIALLMINMEI")
        if df_incpi is not None and len(df_incpi) > 12:
            latest_val = float(df_incpi.iloc[-1]["INDCPIALLMINMEI"])
            ago_val = float(df_incpi.iloc[-13]["INDCPIALLMINMEI"])
            state["india_cpi_yoy"] = ((latest_val - ago_val) / ago_val) * 100
            
        # India 10Y Sovereign Bond Yield
        df_in10y = _get_fred_df("INDIRLTLT01STM")
        if df_in10y is not None and not df_in10y.empty:
            state["india_10y_yield"] = float(df_in10y.iloc[-1]["INDIRLTLT01STM"])
            
        # India Repo Rate (check if latest from FRED is available and reasonable, else use 6.50% RBI default)
        df_inrepo = _get_fred_df("IRSTCB01INM156N")
        if df_inrepo is not None and not df_inrepo.empty:
            fred_repo = float(df_inrepo.iloc[-1]["IRSTCB01INM156N"])
            # If the FRED data is not too stale, use it, else keep default 6.50%
            obs_date_str = str(df_inrepo.iloc[-1]["observation_date"])
            obs_yr = int(obs_date_str.split("-")[0])
            if obs_yr >= 2025:
                state["india_repo_rate"] = fred_repo
            else:
                state["india_repo_rate"] = 6.50 # default paused RBI rate
                
        # Construct yield spreads for India (3M and 5Y) based on 10Y yield
        state["india_5y_yield"] = state["india_10y_yield"] - 0.15 # typical -15 bps
        state["india_3m_yield"] = state["india_10y_yield"] - 0.45 # typical -45 bps
        
        success = True
    except Exception as e:
        log.warning(f"FRED macro state fetch failed: {e}")

    # Set timestamp
    try:
        import pytz
        ist = pytz.timezone('Asia/Kolkata')
        now_time = datetime.now(ist)
    except ImportError:
        now_time = datetime.now()
        
    state["last_updated"] = now_time.strftime("%d %b %H:%M IST") if success else "FALLBACK (OFFLINE)"
    return state
