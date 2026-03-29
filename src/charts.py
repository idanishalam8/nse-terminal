# charts.py — Bloomberg Terminal styled charts

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go
from datetime import date
from scipy import stats
from src.config import ZONES, ZONE_COLORS, CCA_METRICS, NUMERIC_COLS

METRIC_LABELS = {"pe":"P/E","pb":"P/BV","ev_ebitda":"EV/EBITDA","div_yield":"Div Yield %"}

# Bloomberg colors
BG    = "#000000"
BG2   = "#0a0a0a"
BG3   = "#111111"
ORANGE = "#ff6600"
GREEN  = "#00cc44"
RED    = "#ff3333"
AMBER  = "#ffaa00"
GRID   = "#1a1a1a"
TEXT   = "#cccccc"
MUTED  = "#555555"

HEAT_CMAP = mcolors.LinearSegmentedColormap.from_list("bb", [
    (0.00, "#003300"),
    (0.20, "#00cc44"),
    (0.50, "#1a1a1a"),
    (0.80, "#cc5500"),
    (1.00, "#ff0000"),
])


def _zone_color(score: float) -> str:
    for lbl, (lo, hi) in ZONES.items():
        if lo <= score < hi:
            return ZONE_COLORS[lbl]
    return ZONE_COLORS["Fair"]


def interpret_zone(score: float) -> tuple:
    for lbl, (lo, hi) in ZONES.items():
        if lo <= score < hi:
            return lbl, ZONE_COLORS[lbl]
    return "Fair", ZONE_COLORS["Fair"]


# ── HEAT MAP ──────────────────────────────────────────────────────────────────

def draw_heatmap(pct_matrix: pd.DataFrame, as_of: str = None) -> plt.Figure:
    display = pct_matrix[NUMERIC_COLS].copy()
    display.columns = [METRIC_LABELS.get(c, c) for c in display.columns]
    display.index = [
        s.replace("Information Technology", "INFO TECH")
         .replace("Capital Goods & Infra", "CAP GOODS")
         .replace("Energy & Oil Gas", "ENERGY")
         .replace("Financial Services", "FIN SVCS")
         .upper()
        for s in display.index
    ]

    annot = display.copy().astype(object)
    for r in display.index:
        for c in display.columns:
            v = display.loc[r, c]
            annot.loc[r, c] = f"{int(round(v))}" if not pd.isna(v) else "—"

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    sns.heatmap(
        display.astype(float),
        annot=annot, fmt="",
        cmap=HEAT_CMAP, vmin=0, vmax=100, center=50,
        linewidths=1.5, linecolor="#1a1a1a",
        annot_kws={"size": 13, "weight": "bold", "color": "white",
                   "fontfamily": "monospace"},
        cbar_kws={"shrink": 0.5, "pad": 0.02},
        ax=ax, square=False,
    )

    ax.set_xticklabels(ax.get_xticklabels(),
                       color=ORANGE, fontsize=11, fontweight="bold",
                       fontfamily="monospace")
    ax.set_yticklabels(ax.get_yticklabels(),
                       color=TEXT, fontsize=10, rotation=0,
                       fontfamily="monospace")
    ax.tick_params(colors=TEXT, left=False, bottom=False)

    cb = ax.collections[0].colorbar
    cb.set_label(
        "0 = CHEAPEST VS OWN HISTORY  ·  100 = MOST EXPENSIVE VS OWN HISTORY",
        color=MUTED, fontsize=8, fontfamily="monospace"
    )
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=MUTED)

    ds = as_of or date.today().strftime("%d %b %Y").upper()
    ax.set_title(
        f"NSE SECTOR VALUATION PERCENTILE MATRIX  ·  {ds}",
        color=ORANGE, fontsize=12, fontweight="bold",
        fontfamily="monospace", pad=14,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    return fig


# ── RANKING BAR ───────────────────────────────────────────────────────────────

def draw_ranking(richness: pd.Series) -> go.Figure:
    df = richness.reset_index()
    df.columns = ["Sector", "Score"]
    df["Short"] = (df["Sector"]
                   .str.replace("Information Technology", "INFO TECH")
                   .str.replace("Capital Goods & Infra", "CAP GOODS")
                   .str.replace("Energy & Oil Gas", "ENERGY")
                   .str.replace("Financial Services", "FIN SVCS")
                   .str.upper())
    df["Color"] = df["Score"].apply(_zone_color)
    df["Zone"]  = df["Score"].apply(lambda s: interpret_zone(s)[0])

    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["Score"]], y=[row["Short"]], orientation="h",
            marker_color=row["Color"], marker_line_width=0,
            text=f"{row['Score']:.0f}", textposition="outside",
            textfont=dict(size=11, color=TEXT, family="monospace"),
            hovertemplate=(f"<b>{row['Sector']}</b><br>"
                           f"Score: {row['Score']:.1f}/100<br>"
                           f"Zone: {row['Zone']}<extra></extra>"),
            showlegend=False,
        ))

    fig.add_vline(x=50, line_dash="dot",
                  line_color="rgba(255,102,0,0.3)", line_width=1,
                  annotation_text="FAIR VALUE",
                  annotation_font=dict(color=MUTED, size=9, family="monospace"))
    fig.add_vrect(x0=0, x1=35, fillcolor=GREEN, opacity=0.04, layer="below", line_width=0)
    fig.add_vrect(x0=65, x1=100, fillcolor=RED, opacity=0.04, layer="below", line_width=0)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        xaxis=dict(range=[0, 115], showgrid=True, gridcolor=GRID,
                   tickfont=dict(color=MUTED, family="monospace"),
                   title="RICHNESS SCORE",
                   title_font=dict(color=MUTED, size=10, family="monospace")),
        yaxis=dict(tickfont=dict(color=TEXT, family="monospace"), autorange="reversed"),
        height=440, margin=dict(l=10, r=70, t=20, b=30),
        font=dict(family="monospace"),
    )
    return fig


# ── HISTORY CHART ─────────────────────────────────────────────────────────────

def draw_history(sector: str, metric: str, hist_df: pd.DataFrame,
                 current_val, years: int = 10) -> go.Figure:
    dcol = "date" if "date" in hist_df.columns else hist_df.columns[0]
    df   = hist_df.copy()
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    cutoff   = pd.Timestamp.today() - pd.DateOffset(years=years)
    df       = df[df[dcol] >= cutoff]

    if metric not in df.columns:
        return go.Figure().update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG2,
            title=dict(text="NO DATA", font=dict(color=ORANGE, family="monospace")))

    y = pd.to_numeric(df[metric], errors="coerce")
    x = df[dcol]
    p25 = float(np.nanpercentile(y, 25))
    p50 = float(np.nanpercentile(y, 50))
    p75 = float(np.nanpercentile(y, 75))

    fig = go.Figure()
    # Fair value band
    fig.add_trace(go.Scatter(
        x=list(x) + list(x)[::-1],
        y=[p75]*len(x) + [p25]*len(x),
        fill="toself", fillcolor="rgba(0,204,68,0.06)",
        line=dict(width=0), name="25–75 PCT BAND",
        hoverinfo="skip",
    ))
    # Median
    fig.add_hline(y=p50, line_dash="dash",
                  line_color="rgba(0,204,68,0.3)", line_width=1)
    # Historical line
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name="HISTORICAL",
        line=dict(color=ORANGE, width=1.5),
        hovertemplate="%{x|%b %Y}: %{y:.1f}<extra></extra>",
    ))
    # Current value
    if current_val and not np.isnan(float(current_val)):
        pct = stats.percentileofscore(y.dropna().values, float(current_val), kind="rank")
        _, zc = interpret_zone(pct)
        fig.add_hline(
            y=float(current_val), line_dash="solid",
            line_color=zc, line_width=2,
            annotation_text=f"  TODAY: {float(current_val):.1f}x  ({pct:.0f}TH PCT)",
            annotation_position="top left",
            annotation_font=dict(color=zc, size=10, family="monospace"),
        )

    lbl = METRIC_LABELS.get(metric, metric)
    sec_short = sector.replace("Information Technology", "INFO TECH").upper()

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        title=dict(
            text=f"{sec_short}  ·  {lbl}  ·  {years}Y HISTORY",
            font=dict(color=ORANGE, size=11, family="monospace"),
        ),
        xaxis=dict(showgrid=True, gridcolor=GRID,
                   tickfont=dict(color=MUTED, family="monospace")),
        yaxis=dict(showgrid=True, gridcolor=GRID,
                   tickfont=dict(color=MUTED, family="monospace"),
                   title=lbl,
                   title_font=dict(color=MUTED, size=10, family="monospace")),
        legend=dict(font=dict(color=MUTED, family="monospace"),
                    bgcolor="rgba(0,0,0,0)"),
        height=320, margin=dict(l=10, r=20, t=45, b=25),
        hovermode="x unified",
        font=dict(family="monospace"),
    )
    return fig


# ── RADAR ─────────────────────────────────────────────────────────────────────

def draw_radar(pct_row: pd.Series, sector: str) -> go.Figure:
    metrics = NUMERIC_COLS
    labels  = [METRIC_LABELS.get(m, m) for m in metrics]
    vals    = [float(pct_row.get(m, 50) or 50) for m in metrics]
    vals   += vals[:1]
    labels += labels[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=labels, fill="toself",
        fillcolor="rgba(255,102,0,0.12)",
        line=dict(color=ORANGE, width=2),
        name=sector,
    ))
    fig.add_trace(go.Scatterpolar(
        r=[50]*len(labels), theta=labels, mode="lines",
        line=dict(color="rgba(255,102,0,0.2)", dash="dot", width=1),
        name="FAIR", hoverinfo="skip",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(color=MUTED, size=8, family="monospace"),
                            gridcolor=GRID),
            angularaxis=dict(tickfont=dict(color=TEXT, size=10, family="monospace"),
                             gridcolor=GRID),
            bgcolor=BG,
        ),
        paper_bgcolor=BG, showlegend=False,
        height=280, margin=dict(l=30, r=30, t=20, b=20),
        font=dict(family="monospace"),
    )
    return fig


# ── FOOTBALL FIELD ────────────────────────────────────────────────────────────

def draw_football_field(ff_data: dict, target_name: str) -> go.Figure:
    metrics = list(ff_data.keys())
    n = len(metrics)
    fig = go.Figure()

    for i, m in enumerate(metrics):
        d = ff_data[m]
        # Range
        fig.add_trace(go.Scatter(
            x=[d["min"], d["max"]], y=[i, i], mode="lines",
            line=dict(color=GRID, width=8),
            showlegend=False, hoverinfo="skip",
        ))
        # IQR
        fig.add_trace(go.Scatter(
            x=[d["p25"], d["p75"]], y=[i, i], mode="lines",
            line=dict(color=ORANGE, width=14),
            name="25–75TH PCT" if i == 0 else None,
            showlegend=(i == 0),
            hovertemplate=f"{d['label']}: P25={d['p25']:.1f} · P75={d['p75']:.1f}<extra></extra>",
        ))
        # Median
        fig.add_trace(go.Scatter(
            x=[d["median"]], y=[i], mode="markers",
            marker=dict(color="white", size=10, symbol="diamond"),
            name="PEER MEDIAN" if i == 0 else None, showlegend=(i == 0),
            hovertemplate=f"MEDIAN: {d['median']:.1f}<extra></extra>",
        ))
        # Target
        if d["target"] is not None:
            fig.add_trace(go.Scatter(
                x=[d["target"]], y=[i], mode="markers",
                marker=dict(color=GREEN, size=14, symbol="star"),
                name=target_name.upper() if i == 0 else None, showlegend=(i == 0),
                hovertemplate=f"TARGET: {d['target']:.1f}<extra></extra>",
            ))

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        yaxis=dict(tickvals=list(range(n)),
                   ticktext=[ff_data[m]["label"] for m in metrics],
                   tickfont=dict(color=TEXT, size=11, family="monospace"),
                   showgrid=False),
        xaxis=dict(showgrid=True, gridcolor=GRID,
                   tickfont=dict(color=MUTED, family="monospace"),
                   title="MULTIPLE (x)",
                   title_font=dict(color=MUTED, size=10, family="monospace")),
        legend=dict(font=dict(color=TEXT, family="monospace"),
                    bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15),
        height=max(320, n*60+100),
        margin=dict(l=10, r=20, t=20, b=60),
        font=dict(family="monospace"),
    )
    return fig


# ── PREMIUM/DISCOUNT BAR ──────────────────────────────────────────────────────

def draw_premium_discount(pd_tbl: pd.DataFrame, target_name: str) -> go.Figure:
    df = pd_tbl.dropna(subset=["Premium/Disc %"]).copy()
    if df.empty:
        return go.Figure().update_layout(paper_bgcolor=BG, plot_bgcolor=BG2)

    df["color"] = df.apply(lambda r:
        RED   if (r["Premium/Disc %"] > 0  and r["_higher_exp"]) else
        GREEN if (r["Premium/Disc %"] < 0  and r["_higher_exp"]) else
        AMBER, axis=1)

    fig = go.Figure(go.Bar(
        x=df["Metric"], y=df["Premium/Disc %"],
        marker_color=df["color"], marker_line_width=0,
        text=[f"{v:+.1f}%" for v in df["Premium/Disc %"]],
        textposition="outside",
        textfont=dict(color=TEXT, size=11, family="monospace"),
        hovertemplate="%{x}: %{y:+.1f}% VS PEER MEDIAN<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="rgba(255,102,0,0.3)", line_width=1)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        xaxis=dict(tickfont=dict(color=TEXT, family="monospace")),
        yaxis=dict(tickfont=dict(color=MUTED, family="monospace"),
                   ticksuffix="%",
                   title="% VS PEER MEDIAN",
                   title_font=dict(color=MUTED, size=10, family="monospace")),
        height=320, margin=dict(l=10, r=20, t=20, b=30),
        font=dict(family="monospace"),
    )
    return fig


# ── PRICE CHART ───────────────────────────────────────────────────────────────

def draw_price_chart(price_df: pd.DataFrame, name: str) -> go.Figure:
    if price_df.empty:
        return go.Figure().update_layout(paper_bgcolor=BG, plot_bgcolor=BG2)

    close = price_df["Close"]
    start = float(close.iloc[0])
    ret   = (float(close.iloc[-1]) - start) / start * 100
    color = GREEN if ret >= 0 else RED

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=close.index, y=close.values, mode="lines",
        line=dict(color=color, width=1.8),
        name=name.upper(),
        hovertemplate="%{x|%d %b %Y}: ₹%{y:,.1f}<extra></extra>",
    ))
    # Fill area
    fig.add_trace(go.Scatter(
        x=list(close.index) + list(close.index)[::-1],
        y=list(close.values) + [start]*len(close),
        fill="toself",
        fillcolor=f"rgba({'0,204,68' if ret>=0 else '255,51,51'},0.05)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        title=dict(
            text=f"{name.upper()}  ·  1Y PRICE  ·  {ret:+.1f}%",
            font=dict(color=color if abs(ret) > 1 else TEXT,
                      size=11, family="monospace"),
        ),
        xaxis=dict(showgrid=True, gridcolor=GRID,
                   tickfont=dict(color=MUTED, family="monospace")),
        yaxis=dict(showgrid=True, gridcolor=GRID,
                   tickfont=dict(color=MUTED, family="monospace"),
                   tickprefix="₹"),
        height=260, margin=dict(l=10, r=10, t=45, b=25),
        showlegend=False, font=dict(family="monospace"),
    )
    return fig
