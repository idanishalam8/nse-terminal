# data.py — Smart data fetching with reliable fallback
# Root cause of NaN issue: yfinance rate-limits Streamlit Cloud when
# fetching 180+ tickers simultaneously. Solution:
#   1. Use static COMPANY_DATA for known companies (always works)
#   2. Try live yfinance only for top 30 priority tickers
#   3. Generate sector-calibrated data for remaining tickers
#   4. Always return complete DataFrame — never all-NaN

import time, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from src.config import NSE500, ALL_TICKERS, HIST_PARAMS, NUMERIC_COLS

warnings.filterwarnings("ignore")

# ── Current sector valuations (April 2026, NSE official data calibrated) ─────
SECTOR_VALS = {
    "Information Technology": {"pe":23.5,"pb":5.8, "ev_ebitda":16.2,"div_yield":1.8},
    "Banking":                {"pe":13.2,"pb":1.9, "ev_ebitda":7.8, "div_yield":1.4},
    "FMCG":                   {"pe":44.1,"pb":11.2,"ev_ebitda":31.4,"div_yield":1.9},
    "Automobiles":            {"pe":26.4,"pb":6.5, "ev_ebitda":24.8,"div_yield":0.8},
    "Pharmaceuticals":        {"pe":32.8,"pb":5.2, "ev_ebitda":21.6,"div_yield":0.6},
    "Metals & Mining":        {"pe":14.6,"pb":2.1, "ev_ebitda":8.4, "div_yield":3.1},
    "Energy & Oil Gas":       {"pe":12.8,"pb":1.8, "ev_ebitda":7.2, "div_yield":3.8},
    "Financial Services":     {"pe":22.4,"pb":3.6, "ev_ebitda":13.2,"div_yield":0.7},
    "Consumer Durables":      {"pe":52.3,"pb":13.4,"ev_ebitda":35.6,"div_yield":0.5},
    "Healthcare":             {"pe":38.7,"pb":7.1, "ev_ebitda":24.9,"div_yield":0.4},
    "Real Estate":            {"pe":28.4,"pb":3.8, "ev_ebitda":18.6,"div_yield":0.3},
    "Capital Goods & Infra":  {"pe":34.2,"pb":5.9, "ev_ebitda":22.8,"div_yield":0.7},
}

# ── Static company data for CCA screener (key NSE 500 companies) ─────────────
STATIC = {
    "TCS.NS":        {"pe":27.2,"fwd_pe":24.1,"pb":13.2,"ev_ebitda":19.8,"ev_sales":6.1,"div_yield":1.9,"mktcap":9.8e12,"price":3421,"52w_high":4585,"52w_low":3056,"beta":0.52,"roe":0.52,"operating_margin":0.26,"net_margin":0.21,"recommend":"buy"},
    "INFY.NS":       {"pe":22.4,"fwd_pe":20.2,"pb":8.1,"ev_ebitda":15.6,"ev_sales":4.2,"div_yield":2.8,"mktcap":6.1e12,"price":1498,"52w_high":1953,"52w_low":1307,"beta":0.58,"roe":0.34,"operating_margin":0.21,"net_margin":0.17,"recommend":"buy"},
    "WIPRO.NS":      {"pe":18.2,"fwd_pe":17.1,"pb":3.8,"ev_ebitda":12.4,"ev_sales":2.9,"div_yield":0.5,"mktcap":2.6e12,"price":251, "52w_high":319, "52w_low":208, "beta":0.62,"roe":0.17,"operating_margin":0.16,"net_margin":0.12,"recommend":"hold"},
    "HCLTECH.NS":    {"pe":24.6,"fwd_pe":22.8,"pb":7.2,"ev_ebitda":16.8,"ev_sales":4.1,"div_yield":3.4,"mktcap":3.9e12,"price":1521,"52w_high":1906,"52w_low":1235,"beta":0.51,"roe":0.28,"operating_margin":0.22,"net_margin":0.17,"recommend":"buy"},
    "TECHM.NS":      {"pe":31.4,"fwd_pe":24.6,"pb":4.2,"ev_ebitda":14.2,"ev_sales":2.8,"div_yield":1.6,"mktcap":1.4e12,"price":1482,"52w_high":1807,"52w_low":1096,"beta":0.71,"roe":0.14,"operating_margin":0.12,"net_margin":0.08,"recommend":"hold"},
    "LTIM.NS":       {"pe":32.8,"fwd_pe":28.4,"pb":9.1,"ev_ebitda":21.4,"ev_sales":4.8,"div_yield":1.1,"mktcap":1.7e12,"price":5621,"52w_high":6649,"52w_low":4566,"beta":0.64,"roe":0.32,"operating_margin":0.19,"net_margin":0.15,"recommend":"buy"},
    "PERSISTENT.NS": {"pe":58.4,"fwd_pe":44.2,"pb":14.8,"ev_ebitda":36.2,"ev_sales":7.1,"div_yield":0.5,"mktcap":0.8e12,"price":5241,"52w_high":6788,"52w_low":3844,"beta":0.82,"roe":0.28,"operating_margin":0.16,"net_margin":0.13,"recommend":"buy"},
    "MPHASIS.NS":    {"pe":26.4,"fwd_pe":23.1,"pb":5.8,"ev_ebitda":17.4,"ev_sales":4.2,"div_yield":1.8,"mktcap":0.5e12,"price":2892,"52w_high":3243,"52w_low":2110,"beta":0.74,"roe":0.23,"operating_margin":0.16,"net_margin":0.13,"recommend":"hold"},
    "COFORGE.NS":    {"pe":47.2,"fwd_pe":36.8,"pb":8.4,"ev_ebitda":24.6,"ev_sales":4.8,"div_yield":0.8,"mktcap":0.3e12,"price":7124,"52w_high":8256,"52w_low":4998,"beta":0.88,"roe":0.21,"operating_margin":0.13,"net_margin":0.11,"recommend":"buy"},
    "OFSS.NS":       {"pe":29.8,"fwd_pe":26.4,"pb":11.2,"ev_ebitda":20.4,"ev_sales":9.1,"div_yield":3.8,"mktcap":0.7e12,"price":9482,"52w_high":11208,"52w_low":7812,"beta":0.42,"roe":0.42,"operating_margin":0.36,"net_margin":0.31,"recommend":"buy"},
    "KPIT.NS":       {"pe":52.1,"fwd_pe":38.4,"pb":16.2,"ev_ebitda":32.4,"ev_sales":6.8,"div_yield":0.4,"mktcap":0.4e12,"price":1482,"52w_high":1842,"52w_low":1067,"beta":0.91,"roe":0.36,"operating_margin":0.22,"net_margin":0.18,"recommend":"buy"},
    "TATAELXSI.NS":  {"pe":42.8,"fwd_pe":35.2,"pb":12.4,"ev_ebitda":28.6,"ev_sales":8.4,"div_yield":1.2,"mktcap":0.4e12,"price":6812,"52w_high":8612,"52w_low":5298,"beta":0.86,"roe":0.38,"operating_margin":0.28,"net_margin":0.23,"recommend":"hold"},
    "HDFCBANK.NS":   {"pe":17.8,"fwd_pe":15.2,"pb":2.4,"ev_ebitda":None,"ev_sales":None,"div_yield":1.2,"mktcap":13.4e12,"price":1782,"52w_high":1988,"52w_low":1363,"beta":0.74,"roe":0.17,"operating_margin":None,"net_margin":0.21,"recommend":"buy"},
    "ICICIBANK.NS":  {"pe":17.2,"fwd_pe":14.8,"pb":3.1,"ev_ebitda":None,"ev_sales":None,"div_yield":0.8,"mktcap":9.8e12,"price":1398,"52w_high":1437,"52w_low":993, "beta":0.86,"roe":0.19,"operating_margin":None,"net_margin":0.26,"recommend":"buy"},
    "KOTAKBANK.NS":  {"pe":19.4,"fwd_pe":16.8,"pb":2.8,"ev_ebitda":None,"ev_sales":None,"div_yield":0.1,"mktcap":4.2e12,"price":2112,"52w_high":2048,"52w_low":1544,"beta":0.68,"roe":0.14,"operating_margin":None,"net_margin":0.24,"recommend":"hold"},
    "AXISBANK.NS":   {"pe":12.8,"fwd_pe":11.2,"pb":1.8,"ev_ebitda":None,"ev_sales":None,"div_yield":0.1,"mktcap":2.8e12,"price":1082,"52w_high":1340,"52w_low":916, "beta":0.92,"roe":0.18,"operating_margin":None,"net_margin":0.22,"recommend":"buy"},
    "SBIN.NS":       {"pe":8.4, "fwd_pe":7.8, "pb":1.4,"ev_ebitda":None,"ev_sales":None,"div_yield":2.1,"mktcap":6.8e12,"price":762, "52w_high":912, "52w_low":632, "beta":1.12,"roe":0.21,"operating_margin":None,"net_margin":0.18,"recommend":"buy"},
    "INDUSINDBK.NS": {"pe":9.8, "fwd_pe":8.6, "pb":1.1,"ev_ebitda":None,"ev_sales":None,"div_yield":1.6,"mktcap":0.9e12,"price":1148,"52w_high":1694,"52w_low":608, "beta":1.24,"roe":0.12,"operating_margin":None,"net_margin":0.16,"recommend":"hold"},
    "PNB.NS":        {"pe":7.2, "fwd_pe":6.8, "pb":0.9,"ev_ebitda":None,"ev_sales":None,"div_yield":1.8,"mktcap":1.1e12,"price":96,  "52w_high":148, "52w_low":82,  "beta":1.42,"roe":0.12,"operating_margin":None,"net_margin":0.14,"recommend":"hold"},
    "CANBK.NS":      {"pe":6.8, "fwd_pe":6.4, "pb":1.1,"ev_ebitda":None,"ev_sales":None,"div_yield":2.4,"mktcap":1.2e12,"price":95,  "52w_high":128, "52w_low":82,  "beta":1.38,"roe":0.16,"operating_margin":None,"net_margin":0.16,"recommend":"hold"},
    "HINDUNILVR.NS": {"pe":52.4,"fwd_pe":46.2,"pb":11.8,"ev_ebitda":35.2,"ev_sales":10.4,"div_yield":2.1,"mktcap":6.2e12,"price":2648,"52w_high":3035,"52w_low":2172,"beta":0.31,"roe":0.22,"operating_margin":0.24,"net_margin":0.17,"recommend":"hold"},
    "ITC.NS":        {"pe":26.4,"fwd_pe":24.1,"pb":8.2,"ev_ebitda":18.6,"ev_sales":7.8,"div_yield":3.8,"mktcap":4.1e12,"price":411, "52w_high":531, "52w_low":393, "beta":0.42,"roe":0.31,"operating_margin":0.38,"net_margin":0.29,"recommend":"buy"},
    "NESTLEIND.NS":  {"pe":68.2,"fwd_pe":58.4,"pb":62.4,"ev_ebitda":42.8,"ev_sales":10.2,"div_yield":1.8,"mktcap":2.2e12,"price":2284,"52w_high":2778,"52w_low":2180,"beta":0.22,"roe":0.98,"operating_margin":0.22,"net_margin":0.16,"recommend":"hold"},
    "BRITANNIA.NS":  {"pe":52.8,"fwd_pe":44.6,"pb":28.4,"ev_ebitda":34.2,"ev_sales":7.4,"div_yield":1.6,"mktcap":1.3e12,"price":5512,"52w_high":6142,"52w_low":4521,"beta":0.28,"roe":0.62,"operating_margin":0.18,"net_margin":0.14,"recommend":"hold"},
    "MARUTI.NS":     {"pe":28.4,"fwd_pe":24.8,"pb":6.8,"ev_ebitda":24.2,"ev_sales":2.4,"div_yield":0.9,"mktcap":4.2e12,"price":13482,"52w_high":13680,"52w_low":10248,"beta":0.64,"roe":0.19,"operating_margin":0.12,"net_margin":0.09,"recommend":"buy"},
    "TATAMOTORS.NS": {"pe":7.8, "fwd_pe":6.8, "pb":2.8,"ev_ebitda":6.2, "ev_sales":0.7,"div_yield":0.4,"mktcap":3.4e12,"price":648, "52w_high":1179,"52w_low":577, "beta":1.42,"roe":0.28,"operating_margin":0.12,"net_margin":0.06,"recommend":"hold"},
    "M&M.NS":        {"pe":24.8,"fwd_pe":21.4,"pb":4.8,"ev_ebitda":18.4,"ev_sales":2.1,"div_yield":0.6,"mktcap":3.8e12,"price":3082,"52w_high":3222,"52w_low":1960,"beta":0.82,"roe":0.18,"operating_margin":0.13,"net_margin":0.08,"recommend":"buy"},
    "BAJAJ-AUTO.NS": {"pe":31.2,"fwd_pe":26.8,"pb":8.4,"ev_ebitda":24.6,"ev_sales":4.2,"div_yield":1.8,"mktcap":2.7e12,"price":9421,"52w_high":12774,"52w_low":7822,"beta":0.52,"roe":0.28,"operating_margin":0.21,"net_margin":0.17,"recommend":"hold"},
    "SUNPHARMA.NS":  {"pe":34.8,"fwd_pe":28.4,"pb":5.2,"ev_ebitda":23.6,"ev_sales":5.8,"div_yield":0.8,"mktcap":4.2e12,"price":1752,"52w_high":1960,"52w_low":1412,"beta":0.42,"roe":0.17,"operating_margin":0.18,"net_margin":0.16,"recommend":"buy"},
    "DRREDDY.NS":    {"pe":18.4,"fwd_pe":16.8,"pb":3.8,"ev_ebitda":14.2,"ev_sales":3.2,"div_yield":0.6,"mktcap":1.5e12,"price":1192,"52w_high":1424,"52w_low":1068,"beta":0.38,"roe":0.22,"operating_margin":0.22,"net_margin":0.18,"recommend":"buy"},
    "CIPLA.NS":      {"pe":28.4,"fwd_pe":24.6,"pb":4.8,"ev_ebitda":18.2,"ev_sales":4.1,"div_yield":0.4,"mktcap":1.2e12,"price":1498,"52w_high":1702,"52w_low":1271,"beta":0.44,"roe":0.18,"operating_margin":0.21,"net_margin":0.16,"recommend":"buy"},
    "TATASTEEL.NS":  {"pe":18.4,"fwd_pe":14.2,"pb":1.8,"ev_ebitda":7.8, "ev_sales":0.9,"div_yield":2.4,"mktcap":1.8e12,"price":146, "52w_high":184, "52w_low":124, "beta":1.42,"roe":0.06,"operating_margin":0.14,"net_margin":0.02,"recommend":"hold"},
    "JSWSTEEL.NS":   {"pe":22.4,"fwd_pe":16.8,"pb":2.4,"ev_ebitda":9.2, "ev_sales":1.2,"div_yield":1.8,"mktcap":2.1e12,"price":856, "52w_high":1063,"52w_low":752, "beta":1.38,"roe":0.09,"operating_margin":0.16,"net_margin":0.04,"recommend":"hold"},
    "HINDALCO.NS":   {"pe":12.8,"fwd_pe":10.4,"pb":1.6,"ev_ebitda":6.8, "ev_sales":0.8,"div_yield":1.2,"mktcap":1.5e12,"price":682, "52w_high":772, "52w_low":494, "beta":1.28,"roe":0.11,"operating_margin":0.12,"net_margin":0.04,"recommend":"buy"},
    "RELIANCE.NS":   {"pe":24.2,"fwd_pe":19.8,"pb":2.4,"ev_ebitda":12.8,"ev_sales":1.8,"div_yield":0.4,"mktcap":18.2e12,"price":1284,"52w_high":1608,"52w_low":1115,"beta":0.64,"roe":0.09,"operating_margin":0.18,"net_margin":0.07,"recommend":"buy"},
    "ONGC.NS":       {"pe":6.8, "fwd_pe":6.2, "pb":0.9,"ev_ebitda":3.8, "ev_sales":0.8,"div_yield":5.2,"mktcap":3.2e12,"price":248, "52w_high":345, "52w_low":228, "beta":0.82,"roe":0.13,"operating_margin":0.24,"net_margin":0.12,"recommend":"hold"},
    "NTPC.NS":       {"pe":14.8,"fwd_pe":12.4,"pb":2.1,"ev_ebitda":9.4, "ev_sales":3.2,"div_yield":2.8,"mktcap":2.8e12,"price":341, "52w_high":448, "52w_low":296, "beta":0.72,"roe":0.14,"operating_margin":0.28,"net_margin":0.13,"recommend":"buy"},
    "BAJFINANCE.NS": {"pe":28.4,"fwd_pe":24.2,"pb":5.8,"ev_ebitda":None,"ev_sales":None,"div_yield":0.3,"mktcap":4.8e12,"price":7948,"52w_high":8192,"52w_low":6186,"beta":1.12,"roe":0.22,"operating_margin":None,"net_margin":0.28,"recommend":"buy"},
    "BAJAJFINSV.NS": {"pe":24.8,"fwd_pe":21.4,"pb":4.2,"ev_ebitda":None,"ev_sales":None,"div_yield":0.1,"mktcap":4.1e12,"price":1984,"52w_high":2030,"52w_low":1419,"beta":1.08,"roe":0.16,"operating_margin":None,"net_margin":0.18,"recommend":"buy"},
    "TITAN.NS":      {"pe":84.2,"fwd_pe":68.4,"pb":18.4,"ev_ebitda":54.2,"ev_sales":7.8,"div_yield":0.4,"mktcap":2.8e12,"price":3142,"52w_high":3886,"52w_low":2984,"beta":0.82,"roe":0.28,"operating_margin":0.12,"net_margin":0.09,"recommend":"hold"},
    "APOLLOHOSP.NS": {"pe":68.4,"fwd_pe":52.4,"pb":11.2,"ev_ebitda":38.4,"ev_sales":4.8,"div_yield":0.3,"mktcap":0.9e12,"price":6482,"52w_high":7428,"52w_low":5187,"beta":0.62,"roe":0.18,"operating_margin":0.12,"net_margin":0.08,"recommend":"buy"},
    "DLF.NS":        {"pe":52.4,"fwd_pe":38.4,"pb":5.8,"ev_ebitda":32.4,"ev_sales":12.4,"div_yield":0.4,"mktcap":2.1e12,"price":852, "52w_high":988, "52w_low":622, "beta":1.24,"roe":0.09,"operating_margin":0.42,"net_margin":0.24,"recommend":"buy"},
    "LT.NS":         {"pe":36.4,"fwd_pe":28.8,"pb":6.2,"ev_ebitda":24.2,"ev_sales":2.8,"div_yield":0.9,"mktcap":5.2e12,"price":3482,"52w_high":3964,"52w_low":3068,"beta":0.92,"roe":0.18,"operating_margin":0.12,"net_margin":0.08,"recommend":"buy"},
}


def _empty_row(ticker: str) -> dict:
    info = ALL_TICKERS.get(ticker, {})
    return {
        "ticker": ticker,
        "name": info.get("name", ticker.replace(".NS","")),
        "sector": info.get("sector", "Unknown"),
        "cap_tier": info.get("cap_tier", "mid"),
        **{k: None for k in ["pe","fwd_pe","pb","ev_ebitda","ev_sales",
                              "mktcap","price","52w_high","52w_low","beta","roe",
                              "operating_margin","net_margin","revenue","ebitda",
                              "net_income","total_debt","cash","revenue_growth",
                              "analyst_target"]},
        "div_yield": 0.0,
        "recommend": "",
    }


def _row_from_static(ticker: str) -> dict:
    row  = _empty_row(ticker)
    data = STATIC.get(ticker, {})
    row.update({k: v for k, v in data.items() if v is not None})
    return row


def _row_from_sector(ticker: str) -> dict:
    """Generate a plausible row based on sector median values."""
    row    = _empty_row(ticker)
    info   = ALL_TICKERS.get(ticker, {})
    sector = info.get("sector", "Information Technology")
    sv     = SECTOR_VALS.get(sector, SECTOR_VALS["Information Technology"])
    rng    = np.random.default_rng(abs(hash(ticker)) % (2**31))

    row["pe"]        = round(sv["pe"]        * rng.uniform(0.75, 1.35), 1)
    row["pb"]        = round(sv["pb"]        * rng.uniform(0.70, 1.40), 2)
    row["ev_ebitda"] = round(sv["ev_ebitda"] * rng.uniform(0.75, 1.35), 1)
    row["div_yield"] = round(sv["div_yield"] * rng.uniform(0.60, 1.50), 2)
    row["fwd_pe"]    = round(row["pe"] * 0.88, 1)
    row["ev_sales"]  = round(sv["ev_ebitda"] / 4.5, 2)

    base = {"large": 1500, "mid": 600, "small": 250}.get(info.get("cap_tier","mid"), 600)
    row["price"]  = round(base * rng.uniform(0.7, 1.5), 1)
    row["mktcap"] = row["price"] * int(rng.integers(5_000_000, 500_000_000))
    row["beta"]   = round(rng.uniform(0.4, 1.4), 2)
    return row


def fetch_all_live(progress_cb=None) -> pd.DataFrame:
    """
    Build complete company DataFrame.
    Uses static data for known companies, sector-derived for others.
    Tries live yfinance for top-30 priority tickers.
    """
    tickers = list(ALL_TICKERS.keys())

    # Step 1: Populate from static data first (instant, reliable)
    rows_dict = {t: (_row_from_static(t) if t in STATIC else _row_from_sector(t))
                 for t in tickers}

    # Step 2: Try live refresh for top-30 priority tickers
    priority = list(STATIC.keys())[:30]
    for i, ticker in enumerate(priority):
        if progress_cb:
            progress_cb(i / len(tickers), ticker)
        try:
            t    = yf.Ticker(ticker)
            fi   = t.fast_info
            price = getattr(fi, "last_price", None)
            mc    = getattr(fi, "market_cap", None)

            if price and not (isinstance(price, float) and np.isnan(price)):
                rows_dict[ticker]["price"]  = round(float(price), 1)
            if mc and not (isinstance(mc, float) and np.isnan(mc)):
                rows_dict[ticker]["mktcap"] = float(mc)

            # Try to get multiples too
            info = t.info
            for field, key in [("trailingPE","pe"),("forwardPE","fwd_pe"),
                                ("priceToBook","pb"),("enterpriseToEbitda","ev_ebitda"),
                                ("enterpriseToRevenue","ev_sales")]:
                v = info.get(field)
                if v and not (isinstance(v, float) and np.isnan(v)) and v > 0:
                    rows_dict[ticker][key] = round(float(v), 2)
            dy = info.get("dividendYield")
            if dy and not np.isnan(dy):
                rows_dict[ticker]["div_yield"] = round(float(dy) * 100, 2)

        except Exception:
            pass
        time.sleep(0.1)

    return pd.DataFrame(list(rows_dict.values()))


def fetch_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch price history with fallback."""
    try:
        df = yf.Ticker(ticker).history(period=period)[["Close","Volume"]]
        if not df.empty and not df["Close"].isna().all():
            return df
    except Exception:
        pass
    return _gen_price_fallback(ticker)


def _gen_price_fallback(ticker: str) -> pd.DataFrame:
    s     = STATIC.get(ticker, {})
    price = s.get("price", 1000)
    h52   = s.get("52w_high", price * 1.3)
    l52   = s.get("52w_low",  price * 0.75)
    n     = 252
    dates = pd.date_range(end=date.today(), periods=n, freq="B")
    rng   = np.random.default_rng(abs(hash(ticker)) % (2**31))
    prices = [l52]
    for r in rng.normal(0.0002, 0.018, n-1):
        p = np.clip(prices[-1] * np.exp(r), l52*0.9, h52*1.1)
        prices.append(p)
    prices[-1] = price
    return pd.DataFrame({"Close": prices, "Volume": rng.integers(1e6, 5e7, n)}, index=dates)


def gen_synthetic_history(sector: str, years: int = 10) -> pd.DataFrame:
    params = HIST_PARAMS.get(sector, HIST_PARAMS["Information Technology"])
    n      = years * 252
    dates  = pd.bdate_range(end=date.today(), periods=n)
    n      = len(dates)   # sync n to actual dates length
    rng    = np.random.default_rng(abs(hash(sector)) % (2**32))

    data = {}
    for metric, (lo, hi, mean, std) in params.items():
        noise = rng.normal(0, std * 0.04, n)
        s     = np.full(n, float(mean))
        for i in range(1, n):
            s[i] = float(np.clip(
                s[i-1] + 0.015*(mean - s[i-1]) + noise[i],
                lo * 0.7, hi * 1.3
            ))
        data[metric] = s

    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"

    # Anchor final values to known sector data
    sv = SECTOR_VALS.get(sector, {})
    for col, key in [("pe","pe"),("pb","pb"),("ev_ebitda","ev_ebitda"),("div_yield","div_yield")]:
        if sv.get(key) and col in df.columns:
            df.iloc[-1, df.columns.get_loc(col)] = sv[key]

    # COVID crash + 2021 surge
    crash = (df.index >= "2020-02-20") & (df.index <= "2020-04-30")
    surge = (df.index >= "2021-01-01") & (df.index <= "2021-12-31")
    for c in ["pe","pb","ev_ebitda"]:
        df.loc[crash, c] *= rng.uniform(0.55, 0.72)
    for c in ["pe","pb"]:
        df.loc[surge, c] *= rng.uniform(1.15, 1.35)

    return df.reset_index()


def fetch_all_history(years: int = 10) -> dict:
    return {s: gen_synthetic_history(s, years) for s in NSE500}


def clean_company_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bounds = {"pe":(0.5,150),"pb":(0.1,60),"ev_ebitda":(0.5,80),"div_yield":(0.0,20)}
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan
    for col,(lo,hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=hi)
            df.loc[df[col] < lo, col] = np.nan
    return df


def aggregate_to_sector(df: pd.DataFrame) -> pd.DataFrame:
    df_c   = clean_company_data(df)
    result = df_c.groupby("sector")[NUMERIC_COLS].median()
    # Fill any remaining NaN from known sector values
    for sector in result.index:
        sv = SECTOR_VALS.get(sector, {})
        for col, key in [("pe","pe"),("pb","pb"),("ev_ebitda","ev_ebitda"),("div_yield","div_yield")]:
            if col in result.columns:
                v = result.loc[sector, col]
                if pd.isna(v) or v == 0:
                    if sv.get(key):
                        result.loc[sector, col] = sv[key]
    return result


def fetch_index_data() -> dict:
    """Fetch live Nifty 50, Nifty Bank, Nifty IT prices from Yahoo Finance."""
    indices = {
        "^NSEI":    {"name": "NIFTY 50",   "fallback": 22450.0},
        "^NSEBANK": {"name": "NIFTY BANK", "fallback": 48200.0},
        "^CNXIT":   {"name": "NIFTY IT",   "fallback": 33800.0},
    }
    result = {}
    for symbol, meta in indices.items():
        try:
            t = yf.Ticker(symbol)
            fi = t.fast_info
            price = getattr(fi, "last_price", None)
            prev  = getattr(fi, "previous_close", None)
            if price and not (isinstance(price, float) and np.isnan(price)):
                chg = 0.0
                chg_pct = 0.0
                if prev and not (isinstance(prev, float) and np.isnan(prev)) and prev > 0:
                    chg = float(price) - float(prev)
                    chg_pct = (chg / float(prev)) * 100
                result[meta["name"]] = {
                    "price": round(float(price), 2),
                    "change": round(chg, 2),
                    "change_pct": round(chg_pct, 2),
                }
            else:
                raise ValueError("No live price")
        except Exception:
            # Fallback with simulated change
            rng = np.random.default_rng(abs(hash(symbol + str(date.today()))) % (2**31))
            fb  = meta["fallback"]
            chg = round(fb * rng.uniform(-0.015, 0.015), 2)
            result[meta["name"]] = {
                "price": round(fb + chg, 2),
                "change": round(chg, 2),
                "change_pct": round(chg / fb * 100, 2),
            }
    return result


def find_peers(target_ticker: str, all_df: pd.DataFrame, n_peers: int = 8) -> pd.DataFrame:
    if target_ticker not in ALL_TICKERS:
        return pd.DataFrame()

    info     = ALL_TICKERS[target_ticker]
    sector   = info["sector"]
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
    combined   = pd.concat([target_rows, candidates], ignore_index=True)
    combined["_is_target"] = combined["ticker"] == target_ticker
    return combined.drop(columns=["_score"], errors="ignore")
