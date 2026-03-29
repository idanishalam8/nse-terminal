# analytics.py — All financial calculations

import numpy as np
import pandas as pd
from scipy import stats
from src.config import (
    NSE500, DEFAULT_WEIGHTS, SECTOR_WEIGHTS,
    ZONES, ZONE_COLORS, NUMERIC_COLS, CCA_METRICS,
)

INVERTED = {"div_yield"}

CCA_BOUNDS = {
    "pe":        (1, 150),
    "fwd_pe":    (1, 100),
    "pb":        (0.1, 60),
    "ev_ebitda": (1, 80),
    "ev_sales":  (0.1, 30),
    "div_yield": (0, 20),
}


# ── Heat Map Analytics ─────────────────────────────────────────────────────────

def _hist_arr(hist_df: pd.DataFrame, metric: str) -> np.ndarray:
    """Extract clean historical array for a metric."""
    for col in hist_df.columns:
        if col.lower() == metric or col.lower().replace("/","_") == metric:
            arr = pd.to_numeric(hist_df[col], errors="coerce").dropna().values
            return arr[arr > 0]
    return np.array([])


def percentile_rank(val, arr) -> float | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if len(arr) < 20:
        return None
    return round(stats.percentileofscore(arr, float(val), kind="rank"), 1)


def richness_pct(raw, metric) -> float | None:
    if raw is None:
        return None
    return round(100 - raw, 1) if metric in INVERTED else raw


def z_score(val, arr) -> float | None:
    if val is None or len(arr) < 20:
        return None
    mu, sigma = arr.mean(), arr.std()
    if sigma == 0:
        return 0.0
    return round((float(val) - mu) / sigma, 2)


def build_percentile_matrix(sector_df: pd.DataFrame, hist_dict: dict) -> pd.DataFrame:
    """Build 12×4 percentile matrix."""
    sectors = list(NSE500.keys())
    matrix = pd.DataFrame(index=sectors, columns=NUMERIC_COLS, dtype=float)
    for sector in sectors:
        if sector not in hist_dict:
            continue
        for metric in NUMERIC_COLS:
            arr = _hist_arr(hist_dict[sector], metric)
            try:
                val = float(sector_df.loc[sector, metric])
            except (KeyError, TypeError, ValueError):
                val = None
            matrix.loc[sector, metric] = richness_pct(percentile_rank(val, arr), metric)
    return matrix


def composite_score(pct_matrix: pd.DataFrame, sector: str) -> float | None:
    weights = SECTOR_WEIGHTS.get(sector, DEFAULT_WEIGHTS)
    tw = ts = 0.0
    for m, w in weights.items():
        v = pct_matrix.loc[sector, m] if sector in pct_matrix.index else None
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            ts += float(v) * w
            tw += w
    return round(ts / tw, 1) if tw > 0 else None


def build_richness_series(pct_matrix: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {s: composite_score(pct_matrix, s) for s in pct_matrix.index},
        name="richness"
    ).dropna().sort_values()


def interpret_score(score: float) -> tuple:
    for label, (lo, hi) in ZONES.items():
        if lo <= score < hi:
            return label, ZONE_COLORS[label]
    return "Fair", ZONE_COLORS["Fair"]


# ── CCA Analytics ──────────────────────────────────────────────────────────────

def clean_cca(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in CCA_BOUNDS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < lo, col] = np.nan
            df[col] = df[col].clip(upper=hi)
    return df


def build_comps_table(peers_df: pd.DataFrame) -> pd.DataFrame:
    """Build formatted comps table."""
    clean = clean_cca(peers_df)
    rows = []
    for _, row in clean.iterrows():
        mc = row.get("mktcap")
        entry = {
            "Company":    row.get("name", row.get("ticker", "")),
            "Ticker":     row.get("ticker", ""),
            "Mkt Cap":    f"₹{mc/1e7:,.0f} Cr" if mc else "N/A",
            "_mktcap":    mc or 0,
            "_is_target": bool(row.get("_is_target", False)),
        }
        for m, cfg in CCA_METRICS.items():
            v = row.get(m)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                entry[cfg["label"]] = None
                entry[f"_{m}"] = None
            else:
                entry[cfg["label"]] = round(float(v), 1)
                entry[f"_{m}"] = float(v)
        rows.append(entry)
    return pd.DataFrame(rows)


def build_premium_discount(comps_tbl: pd.DataFrame) -> pd.DataFrame:
    target = comps_tbl[comps_tbl["_is_target"] == True]
    peers  = comps_tbl[comps_tbl["_is_target"] == False]
    if target.empty:
        return pd.DataFrame()

    t_row = target.iloc[0]
    results = []
    for m, cfg in CCA_METRICS.items():
        raw = f"_{m}"
        tv  = t_row.get(raw)
        pv  = pd.to_numeric(peers[raw] if raw in peers.columns else pd.Series(), errors="coerce")
        pm  = float(pv.median()) if not pv.dropna().empty else None
        pmean = float(pv.mean()) if not pv.dropna().empty else None
        tv_f = float(tv) if tv and not (isinstance(tv, float) and np.isnan(tv)) else None

        prem = None
        if tv_f and pm and pm != 0:
            prem = (tv_f / pm - 1) * 100
            if not cfg["higher_is_expensive"]:
                prem = -prem

        pct = None
        if tv_f and not pv.dropna().empty:
            raw_pct = stats.percentileofscore(pv.dropna().values, tv_f, kind="rank")
            pct = round(100 - raw_pct if not cfg["higher_is_expensive"] else raw_pct, 1)

        curr_price = float(t_row.get("price") or 0)
        implied = None
        if tv_f and pm and tv_f > 0 and curr_price > 0:
            implied = round(curr_price * (pm / tv_f), 1)

        results.append({
            "Metric":           cfg["label"],
            "Target":           round(tv_f, 1) if tv_f else None,
            "Peer Median":      round(pm, 1) if pm else None,
            "Peer Mean":        round(pmean, 1) if pmean else None,
            "Premium/Disc %":   round(prem, 1) if prem else None,
            "Pct in Peers":     pct,
            "Implied Price":    f"₹{implied:,.0f}" if implied else "N/A",
            "_higher_exp":      cfg["higher_is_expensive"],
        })
    return pd.DataFrame(results)


def football_field(comps_tbl: pd.DataFrame) -> dict:
    peers = comps_tbl[comps_tbl.get("_is_target", pd.Series(False)) == False]
    target = comps_tbl[comps_tbl.get("_is_target", pd.Series(False)) == True]
    result = {}
    for m, cfg in CCA_METRICS.items():
        raw = f"_{m}"
        if raw not in peers.columns:
            continue
        arr = pd.to_numeric(peers[raw], errors="coerce").dropna()
        if arr.empty:
            continue
        tv = None
        if not target.empty and raw in target.columns:
            v = target.iloc[0].get(raw)
            if v:
                tv = float(v)
        result[m] = {
            "label":  cfg["label"],
            "min":    float(arr.min()),
            "p25":    float(arr.quantile(0.25)),
            "median": float(arr.median()),
            "p75":    float(arr.quantile(0.75)),
            "max":    float(arr.max()),
            "target": tv,
        }
    return result


def peer_stats(comps_tbl: pd.DataFrame) -> pd.DataFrame:
    peers = comps_tbl[comps_tbl.get("_is_target", pd.Series(False)) == False]
    rows = []
    for m, cfg in CCA_METRICS.items():
        raw = f"_{m}"
        if raw not in peers.columns:
            continue
        arr = pd.to_numeric(peers[raw], errors="coerce").dropna()
        if arr.empty:
            continue
        rows.append({
            "Metric":  cfg["label"],
            "N":       len(arr),
            "Mean":    round(arr.mean(), 1),
            "Median":  round(arr.median(), 1),
            "Min":     round(arr.min(), 1),
            "Max":     round(arr.max(), 1),
            "P25":     round(arr.quantile(0.25), 1),
            "P75":     round(arr.quantile(0.75), 1),
        })
    return pd.DataFrame(rows)
