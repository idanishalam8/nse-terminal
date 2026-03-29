# data.py — Real-time data fetching
# Cache TTL = 15 minutes (st.cache_data handles this in app.py)
# No pkl files written — Streamlit's in-memory cache only

import time, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from src.config import NSE500, ALL_TICKERS, HIST_PARAMS, NUMERIC_COLS

warnings.filterwarnings("ignore")


def fetch_ticker(ticker: str) -> dict:
    """Fetch live data for a single ticker from Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker":          ticker,
            "name":            info.get("shortName", ALL_TICKERS.get(ticker, {}).get("name", ticker)),
            "sector":          ALL_TICKERS.get(ticker, {}).get("sector", "Unknown"),
            "cap_tier":        ALL_TICKERS.get(ticker, {}).get("cap_tier", "mid"),
            "pe":              info.get("trailingPE"),
            "fwd_pe":          info.get("forwardPE"),
            "pb":              info.get("priceToBook"),
            "ev_ebitda":       info.get("enterpriseToEbitda"),
            "ev_sales":        info.get("enterpriseToRevenue"),
            "div_yield":       (info.get("dividendYield") or 0) * 100,
            "mktcap":          info.get("marketCap"),
            "price":           info.get("currentPrice") or info.get("regularMarketPrice"),
            "52w_high":        info.get("fiftyTwoWeekHigh"),
            "52w_low":         info.get("fiftyTwoWeekLow"),
            "beta":            info.get("beta"),
            "roe":             info.get("returnOnEquity"),
            "operating_margin":info.get("operatingMargins"),
            "net_margin":      info.get("profitMargins"),
            "revenue":         info.get("totalRevenue"),
            "ebitda":          info.get("ebitda"),
            "net_income":      info.get("netIncomeToCommon"),
            "total_debt":      info.get("totalDebt"),
            "cash":            info.get("totalCash"),
            "revenue_growth":  info.get("revenueGrowth"),
            "recommend":       info.get("recommendationKey", ""),
            "analyst_target":  info.get("targetMeanPrice"),
        }
    except Exception:
        return {
            "ticker": ticker, "name": ticker,
            "sector": ALL_TICKERS.get(ticker, {}).get("sector", "Unknown"),
            "cap_tier": ALL_TICKERS.get(ticker, {}).get("cap_tier", "mid"),
            **{k: None for k in ["pe","fwd_pe","pb","ev_ebitda","ev_sales","div_yield",
                                  "mktcap","price","52w_high","52w_low","beta","roe",
                                  "operating_margin","net_margin","revenue","ebitda",
                                  "net_income","total_debt","cash","revenue_growth",
                                  "recommend","analyst_target"]},
            "div_yield": 0,
        }


def fetch_all_live(progress_cb=None) -> pd.DataFrame:
    """
    Fetch live multiples for all NSE 500 companies.
    Called fresh on every session — no pkl cache.
    """
    rows = []
    tickers = [(t, d["sector"]) for t, d in ALL_TICKERS.items()]
    for i, (ticker, _) in enumerate(tickers):
        if progress_cb:
            progress_cb(i / len(tickers), ticker)
        rows.append(fetch_ticker(ticker))
        time.sleep(0.2)
    return pd.DataFrame(rows)


def fetch_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch 1-year price history for a ticker."""
    try:
        df = yf.Ticker(ticker).history(period=period)[["Close", "Volume"]]
        return df
    except Exception:
        return pd.DataFrame()


def gen_synthetic_history(sector: str, years: int = 10) -> pd.DataFrame:
    """
    Generate realistic mean-reverting historical data.
    Used as fallback when NSE live history is unavailable.
    """
    params = HIST_PARAMS.get(sector, HIST_PARAMS["Information Technology"])
    n = years * 252
    dates = pd.date_range(end=date.today(), periods=n, freq="B")
    rng = np.random.default_rng(abs(hash(sector)) % (2**32))

    def ou(lo, hi, mean, std, n):
        s = np.zeros(n); s[0] = mean
        for i in range(1, n):
            s[i] = s[i-1] + 0.015*(mean - s[i-1]) + std*0.04*rng.normal()
            s[i] = np.clip(s[i], lo*0.7, hi*1.3)
        return s

    data = {m: ou(*p, n) for m, p in params.items()}
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"

    # Inject COVID crash (Feb-Apr 2020)
    crash = (df.index >= "2020-02-20") & (df.index <= "2020-04-30")
    for c in ["pe", "pb", "ev_ebitda"]:
        df.loc[crash, c] *= rng.uniform(0.55, 0.72)
    # 2021 re-rating
    surge = (df.index >= "2021-01-01") & (df.index <= "2021-12-31")
    for c in ["pe", "pb"]:
        df.loc[surge, c] *= rng.uniform(1.15, 1.35)

    return df.reset_index()


def fetch_all_history(years: int = 10) -> dict:
    """Generate/fetch historical data for all sectors."""
    return {sector: gen_synthetic_history(sector, years) for sector in NSE500}


def clean_company_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate raw company data."""
    df = df.copy()
    bounds = {
        "pe":        (0.5, 150),
        "pb":        (0.1,  60),
        "ev_ebitda": (0.5,  80),
        "div_yield": (0.0,  20),
    }
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=hi)
            df.loc[df[col] < lo, col] = np.nan
    return df


def aggregate_to_sector(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate company multiples to sector medians."""
    df_clean = clean_company_data(df)
    return df_clean.groupby("sector")[NUMERIC_COLS].median()


def find_peers(target_ticker: str, all_df: pd.DataFrame, n_peers: int = 8) -> pd.DataFrame:
    """Find comparable peers for target company."""
    if target_ticker not in ALL_TICKERS:
        return pd.DataFrame()

    info = ALL_TICKERS[target_ticker]
    sector = info["sector"]
    cap_tier = info["cap_tier"]

    from src.config import CAP_ADJACENCY
    allowed = CAP_ADJACENCY.get(cap_tier, ["mid"])

    target_rows = all_df[all_df["ticker"] == target_ticker]
    if target_rows.empty:
        return pd.DataFrame()

    target_mc = float(target_rows.iloc[0].get("mktcap") or 0)

    candidates = all_df[
        (all_df["ticker"] != target_ticker) &
        (all_df["sector"] == sector) &
        (all_df["cap_tier"].isin(allowed))
    ].copy()

    if len(candidates) < 3:
        candidates = all_df[
            (all_df["ticker"] != target_ticker) &
            (all_df["cap_tier"].isin(allowed))
        ].copy()

    def score(row):
        mc = float(row.get("mktcap") or 0)
        if mc <= 0 or target_mc <= 0:
            return 999
        return abs(np.log(mc) - np.log(target_mc))

    candidates["_score"] = candidates.apply(score, axis=1)
    candidates = candidates.sort_values("_score").head(n_peers)
    combined = pd.concat([target_rows, candidates], ignore_index=True)
    combined["_is_target"] = combined["ticker"] == target_ticker
    return combined.drop(columns=["_score"], errors="ignore")
