# ─────────────────────────────────────────────────────────────────────────────
# NSE VALUATION TERMINAL  ·  Bloomberg Style  ·  Real-Time Data
# ─────────────────────────────────────────────────────────────────────────────

import warnings, time
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, datetime

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="NSE VALUATION TERMINAL",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bloomberg Terminal CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');

*{font-family:'IBM Plex Mono','Courier New',monospace !important}
html,body,.stApp{background:#000000 !important;color:#cccccc}
[data-testid="stSidebar"]{background:#050505 !important;border-right:1px solid #ff6600 !important}
[data-testid="stSidebar"] *{color:#aaaaaa !important}
[data-testid="stSidebar"] label{color:#ff6600 !important;font-size:9px !important;letter-spacing:.15em !important;text-transform:uppercase !important}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:#000 !important;border-bottom:2px solid #ff6600 !important;gap:0 !important}
.stTabs [data-baseweb="tab"]{color:#444 !important;font-size:10px !important;font-weight:600 !important;letter-spacing:.1em !important;text-transform:uppercase !important;padding:7px 18px !important;border:none !important;background:transparent !important;border-radius:0 !important}
.stTabs [aria-selected="true"]{color:#000 !important;background:#ff6600 !important}
.stTabs [data-baseweb="tab"]:hover{color:#ff6600 !important}

/* Inputs */
.stSelectbox>div>div,.stMultiselect>div>div{background:#0a0a0a !important;border:1px solid #222 !important;border-radius:0 !important;color:#ccc !important}
.stSlider [data-testid="stSlider"]{accent-color:#ff6600}
.stCheckbox label span{color:#aaa !important;font-size:11px !important}

/* Buttons */
.stButton button{background:#ff6600 !important;color:#000 !important;font-weight:700 !important;font-size:10px !important;letter-spacing:.1em !important;text-transform:uppercase !important;border:none !important;border-radius:0 !important;padding:8px 16px !important}
.stButton button:hover{background:#cc5200 !important}

/* Headings */
h1,h2,h3{color:#ff6600 !important;letter-spacing:.08em}

/* Tables */
table{width:100%;border-collapse:collapse}
th{background:#111 !important;color:#ff6600 !important;font-size:9px !important;font-weight:600 !important;text-transform:uppercase !important;letter-spacing:.12em !important;padding:7px 10px !important;border-bottom:1px solid #ff6600 !important}
td{color:#ccc !important;font-size:11px !important;padding:6px 10px !important;border-bottom:1px solid #111 !important}
tr:hover td{background:#0d0d0d !important}
tr:first-child td{color:#ff9944 !important;font-weight:600 !important;background:#0f0600 !important;border-left:2px solid #ff6600 !important}

/* DataFrame */
.stDataFrame{background:#000 !important}
[data-testid="stDataFrame"]{border:1px solid #1a1a1a !important}

/* Expander */
/* Expander — hide broken Material Icons text, use CSS arrow instead */
[data-testid="stExpander"]{border:none !important}
[data-testid="stExpander"] details{border:1px solid #1a1a1a !important;border-radius:0 !important}
[data-testid="stExpander"] details summary{
  background:#0a0a0a !important;border:none !important;
  border-radius:0 !important;padding:8px 12px !important;
  list-style:none !important;cursor:pointer !important}
[data-testid="stExpander"] details summary::-webkit-details-marker{display:none !important}
[data-testid="stExpander"] details summary::marker{display:none !important}
[data-testid="stExpander"] details summary p{
  display:none !important}
[data-testid="stExpander"] details summary span[data-testid="stExpanderToggleIcon"]{
  display:none !important}
/* Hide the broken icon span entirely — aggressive selectors */
[data-testid="stExpander"] summary > div > span:first-child{display:none !important}
[data-testid="stExpander"] summary .eyeumkm0{display:none !important}
/* Hide ALL material icon text inside expanders */
[data-testid="stExpander"] summary span[data-testid="stIconMaterial"],
[data-testid="stExpander"] summary .material-symbols-rounded,
[data-testid="stExpander"] summary .e1nzilvr5,
[data-testid="stExpander"] summary > div > div:first-child > span:first-child{
  display:none !important; font-size:0 !important; width:0 !important;
  height:0 !important; overflow:hidden !important; position:absolute !important}
/* Nuclear option: hide any span that renders icon text in expander */
[data-testid="stExpander"] details summary > div{
  overflow:hidden !important}
[data-testid="stExpander"] details summary > div > div:first-child{
  font-size:0 !important; line-height:0 !important; overflow:hidden !important;
  max-height:0 !important; padding:0 !important; margin:0 !important}
/* Custom arrow via label */
[data-testid="stExpander"] details summary::before{
  content:"▶  ";color:#ff6600;font-size:9px;font-family:'Courier New',monospace;
  letter-spacing:.1em}
[data-testid="stExpander"] details[open] summary::before{
  content:"▼  ";color:#ff6600;font-size:9px;font-family:'Courier New',monospace;
  letter-spacing:.1em}
[data-testid="stExpander"] details summary div[data-testid="stMarkdownContainer"] p{
  display:block !important;color:#ff6600 !important;font-size:9px !important;
  font-family:'Courier New',monospace !important;text-transform:uppercase !important;
  letter-spacing:.15em !important;margin:0 !important}
.streamlit-expanderHeader{background:#0a0a0a !important;border:1px solid #1a1a1a !important;border-radius:0 !important}
.streamlit-expanderHeader p{color:#ff6600 !important;font-size:9px !important;text-transform:uppercase !important;letter-spacing:.15em !important;font-family:'Courier New',monospace !important}

/* Scrollbar */
::-webkit-scrollbar{width:3px;height:3px}
::-webkit-scrollbar-track{background:#000}
::-webkit-scrollbar-thumb{background:#ff6600}

/* Misc */
hr{border:none;border-top:1px solid #1a1a1a}
p,li{color:#888 !important;font-size:12px !important}
code{background:#0a0a0a !important;color:#ff6600 !important;border:1px solid #222 !important}

/* ── Fix broken Material Icons text (keyboard_double_arrow / arrow_right) ── */
/* Hide Streamlit header, footer, toolbar that show broken icon text */
#MainMenu {visibility:hidden !important}
footer {visibility:hidden !important}
header[data-testid="stHeader"] {background:transparent !important; visibility:visible !important}
[data-testid="stToolbar"] {visibility:hidden !important}
[data-testid="manage-app-button"] {visibility:hidden !important}
.stDeployButton {display:none !important}
[data-testid="stStatusWidget"] {visibility:hidden !important}

/* Fix sidebar collapse/expand button — hide raw icon text, show SVG only */
button[kind="headerNoPadding"] {font-size:0 !important; color:transparent !important}
button[kind="headerNoPadding"] svg {color:#ff6600 !important; width:20px !important; height:20px !important}
[data-testid="stSidebarCollapsedControl"] button {font-size:0 !important; color:transparent !important}
[data-testid="stSidebarCollapsedControl"] button svg {color:#ff6600 !important}
[data-testid="stSidebarNavCollapseIcon"],
[data-testid="stSidebarNavExpandIcon"] {color:#ff6600 !important}
/* Override Material Symbols font reset — keep it for icons only */
.material-symbols-rounded, .material-icons {font-family:'Material Symbols Rounded','Material Icons' !important}
/* Bottom manage app button — hide completely */
[data-testid="manage-app-button"] span,
.viewerBadge_container__r5tak,
[data-testid="stBottom"] [data-testid="manage-app-button"],
[data-testid="stBottomBlockContainer"] {visibility:hidden !important; height:0 !important; min-height:0 !important; padding:0 !important}
/* Hide ALL elements with arrow_right text (material icon fallback) */
span:not([class]):empty + span,
[data-testid="stExpander"] summary span[style*="font-family"] {display:none !important}

/* Custom components */
.bb-topbar{background:#ff6600;color:#000;padding:5px 16px;font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;display:flex;justify-content:space-between;align-items:center}
.bb-strip{background:#0a0a0a;border-top:1px solid #ff6600;border-bottom:1px solid #ff6600;padding:5px 0;font-size:10px;font-weight:600;letter-spacing:.05em;overflow:hidden;white-space:nowrap}
.bb-card{background:#080808;border:1px solid #1a1a1a;border-top:2px solid #ff6600;padding:10px 12px}
.bb-card .lbl{font-size:9px;color:#ff6600;text-transform:uppercase;letter-spacing:.15em;margin-bottom:3px}
.bb-card .val{font-size:19px;font-weight:700;line-height:1.1}
.bb-card .sub{font-size:9px;color:#444;margin-top:2px;text-transform:uppercase;letter-spacing:.06em}
.bb-sec{font-size:9px;color:#ff6600;text-transform:uppercase;letter-spacing:.18em;border-bottom:1px solid #ff6600;padding-bottom:3px;margin:14px 0 8px}
.bb-co{background:#080808;border:1px solid #1a1a1a;border-left:3px solid #ff6600;padding:12px 16px;margin-bottom:12px}
.bb-mini{background:#080808;border:1px solid #1a1a1a;padding:8px;text-align:center}
.bb-mini .ml{font-size:8px;color:#444;text-transform:uppercase;letter-spacing:.12em;margin-bottom:3px}
.bb-mini .mv{font-size:15px;font-weight:700}
.bb-mini .mp{font-size:9px;color:#444;margin-top:2px}
.bb-mini .mc{font-size:9px;font-weight:700;margin-top:2px}
.bb-info{background:#0a0600;border:1px solid #2a1800;border-left:3px solid #ff6600;padding:8px 12px;font-size:10px;color:#555;letter-spacing:.05em;margin-bottom:10px}
.status-live{display:inline-block;width:7px;height:7px;background:#00cc44;border-radius:50%;margin-right:5px}

/* ── Market Ticker Ribbon ── */
@keyframes ticker-scroll {
  0%   { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}
.market-ribbon {
  background: #050505;
  border-bottom: 1px solid #1a1a1a;
  overflow: hidden;
  white-space: nowrap;
  padding: 6px 0;
  position: relative;
}
.market-ribbon::before,
.market-ribbon::after {
  content: '';
  position: absolute;
  top: 0;
  bottom: 0;
  width: 40px;
  z-index: 2;
  pointer-events: none;
}
.market-ribbon::before {
  left: 0;
  background: linear-gradient(90deg, #050505 0%, transparent 100%);
}
.market-ribbon::after {
  right: 0;
  background: linear-gradient(270deg, #050505 0%, transparent 100%);
}
.ribbon-track {
  display: inline-block;
  animation: ticker-scroll 20s linear infinite;
}
.ribbon-track:hover {
  animation-play-state: paused;
}
.ribbon-item {
  display: inline-block;
  margin: 0 28px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  font-family: 'IBM Plex Mono', monospace;
}
.ribbon-item .idx-name {
  color: #ff6600;
  margin-right: 8px;
}
.ribbon-item .idx-price {
  color: #cccccc;
  margin-right: 6px;
}
.ribbon-item .idx-chg {
  font-size: 10px;
  font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

from src.config import NSE500, ALL_TICKERS, CCA_METRICS, LOOKBACK_OPTIONS, NUMERIC_COLS, ZONE_COLORS
from src.data import (fetch_all_live, fetch_all_history, fetch_price_history,
                      aggregate_to_sector, find_peers, clean_company_data,
                      fetch_index_data)
from src.analytics import (build_percentile_matrix, build_richness_series,
                            composite_score, interpret_score, build_comps_table,
                            build_premium_discount, football_field, peer_stats,
                            clean_cca)
from src.charts import (draw_heatmap, draw_ranking, draw_history, draw_radar,
                        draw_football_field, draw_premium_discount, draw_price_chart,
                        METRIC_LABELS)

SECTOR_LIST  = list(NSE500.keys())
TICKER_NAMES = {t: f"{d['name']}  [{t.replace('.NS','')}]" for t, d in ALL_TICKERS.items()}


# ══════════════════════════════════════════════════════════════════════════════
# DATA — TTL=15min, fetches fresh from Yahoo Finance every 15 minutes
# No pkl files — pure in-memory Streamlit cache only
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)   # 15-minute live refresh
def load_live_data():
    """Fetch all NSE 500 companies live from Yahoo Finance."""
    return fetch_all_live()


@st.cache_data(ttl=86400, show_spinner=False)  # Historical: daily refresh
def load_history(years: int):
    return fetch_all_history(years)


@st.cache_data(ttl=900, show_spinner=False)
def load_price(ticker: str):
    return fetch_price_history(ticker, "1y")


@st.cache_data(ttl=300, show_spinner=False)   # 5-minute refresh for indices
def load_index_data():
    """Fetch live Nifty 50, Nifty Bank, Nifty IT prices."""
    return fetch_index_data()


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='background:#ff6600;color:#000;padding:9px 12px;font-size:12px;
                font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin-bottom:10px'>
      ◆ NSE TERMINAL
    </div>
    <div style='font-size:9px;color:#333;letter-spacing:.08em;text-transform:uppercase;
                padding:0 4px;margin-bottom:10px;line-height:1.8'>
      Valuation Intelligence Suite<br>
      <span class='status-live'></span>
      <span style='color:#00cc44'>LIVE DATA</span> &nbsp;·&nbsp; NSE 500<br>
      <span style='color:#333'>━━━━━━━━━━━━━━━━━━━━━━</span>
    </div>
    """, unsafe_allow_html=True)

    lookback_label = st.selectbox("LOOKBACK PERIOD", list(LOOKBACK_OPTIONS.keys()), index=2)
    years = LOOKBACK_OPTIONS[lookback_label]

    st.markdown("")
    sel_metrics = st.multiselect("HEAT MAP METRICS", NUMERIC_COLS, default=NUMERIC_COLS,
                                  format_func=lambda m: METRIC_LABELS.get(m, m))
    if not sel_metrics:
        sel_metrics = NUMERIC_COLS

    st.markdown("")
    st.markdown("<div style='font-size:9px;color:#ff6600;letter-spacing:.15em;text-transform:uppercase;margin-bottom:5px'>CCA PARAMETERS</div>", unsafe_allow_html=True)
    n_peers = st.slider("PEER COUNT", 4, 12, 8)
    strict  = st.checkbox("STRICT SECTOR MATCH", value=True)

    st.markdown("<div style='border-top:1px solid #111;margin:10px 0'></div>", unsafe_allow_html=True)

    if st.button("⟳  REFRESH NOW", use_container_width=True):
        st.cache_data.clear()
        st.success("CACHE CLEARED — FETCHING LIVE DATA"); time.sleep(1); st.rerun()

    now = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div style='font-size:9px;color:#333;text-transform:uppercase;letter-spacing:.07em;
                margin-top:14px;border-top:1px solid #111;padding-top:8px;line-height:1.9'>
      SOURCE: YAHOO FINANCE<br>
      DATE: {date.today().strftime('%d %b %Y').upper()}<br>
      TIME: {now} IST<br>
      CACHE TTL: 15 MIN<br>
      <span style='color:#ff6600'>STATUS: <span class='status-live'></span>LIVE</span>
    </div>""", unsafe_allow_html=True)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
fetch_placeholder = st.empty()
fetch_placeholder.markdown("""
<div style='background:#080808;border:1px solid #ff6600;padding:16px 20px;
            font-size:11px;color:#ff6600;letter-spacing:.1em;text-transform:uppercase'>
  ◆ FETCHING LIVE MARKET DATA FROM YAHOO FINANCE...<br>
  <span style='color:#444;font-size:9px'>FIRST LOAD: 60–90 SECONDS &nbsp;·&nbsp;
  SUBSEQUENT LOADS: INSTANT (15-MIN CACHE)</span>
</div>""", unsafe_allow_html=True)

with st.spinner(""):
    raw_df   = load_live_data()
    hist_dict = load_history(years)

fetch_placeholder.empty()

sec_df     = aggregate_to_sector(raw_df)
pct_matrix = build_percentile_matrix(sec_df, hist_dict)
richness   = build_richness_series(pct_matrix)


# ── TOPBAR ────────────────────────────────────────────────────────────────────
now_str = datetime.now().strftime("%d %b %Y  %H:%M IST").upper()
st.markdown(f"""
<div class='bb-topbar'>
  <span>◆ NSE VALUATION TERMINAL &nbsp;·&nbsp; SECTOR HEAT MAP + CCA SCREENER &nbsp;·&nbsp; NSE 500</span>
  <span><span class='status-live'></span>LIVE &nbsp;·&nbsp; {now_str} &nbsp;·&nbsp; 12 SECTORS</span>
</div>""", unsafe_allow_html=True)

# ── REAL-TIME MARKET RIBBON ───────────────────────────────────────────────────
idx_data = load_index_data()
ribbon_parts = []
for idx_name, idx_vals in idx_data.items():
    idx_price = idx_vals.get('price', 0)
    idx_chg = idx_vals.get('change', 0)
    idx_chg_pct = idx_vals.get('change_pct', 0)
    chg_color = '#00cc44' if idx_chg >= 0 else '#ff3333'
    chg_arrow = '▲' if idx_chg >= 0 else '▼'
    ribbon_parts.append(
        '<span class="ribbon-item">'
        '<span class="idx-name">' + str(idx_name) + '</span>'
        '<span class="idx-price">₹' + f'{idx_price:,.2f}' + '</span>'
        '<span class="idx-chg" style="color:' + chg_color + '">'
        + chg_arrow + ' ' + f'{idx_chg:+.2f}' + ' (' + f'{idx_chg_pct:+.2f}' + '%)</span>'
        '</span>'
    )
ribbon_block = '<span class="ribbon-item" style="color:#333;margin:0 20px">◆</span>'.join(ribbon_parts)
sep = '<span class="ribbon-item" style="color:#333;margin:0 20px">◆</span>'
full_ribbon = (ribbon_block + sep) * 4

st.markdown(
    '<div class="market-ribbon"><div class="ribbon-track">'
    + full_ribbon +
    '</div></div>',
    unsafe_allow_html=True
)

# Live sector strip
short_map = {
    "Information Technology":"IT","Banking":"BANK","FMCG":"FMCG",
    "Automobiles":"AUTO","Pharmaceuticals":"PHARMA","Metals & Mining":"METALS",
    "Energy & Oil Gas":"ENERGY","Financial Services":"FIN SVCS",
    "Consumer Durables":"CONS DUR","Healthcare":"HEALTH",
    "Real Estate":"REALTY","Capital Goods & Infra":"INFRA",
}
if len(richness) > 0:
    items = []
    for s in richness.index:
        sc = richness[s]
        c  = "#00cc44" if sc < 35 else ("#ff3333" if sc > 65 else "#ffaa00")
        items.append(f'<span style="color:{c}">{short_map.get(s,s)}: {sc:.0f}</span>')
    strip = "&nbsp;&nbsp;◆&nbsp;&nbsp;".join(items)
    st.markdown(f"""
    <div class='bb-strip'>
      &nbsp;&nbsp;◆&nbsp;&nbsp;{strip}&nbsp;&nbsp;◆&nbsp;&nbsp;
      MARKET RICHNESS: {richness.mean():.0f}/100&nbsp;&nbsp;◆&nbsp;&nbsp;
      CHEAPEST: {short_map.get(richness.idxmin(), richness.idxmin())}&nbsp;&nbsp;◆&nbsp;&nbsp;
      MOST EXP: {short_map.get(richness.idxmax(), richness.idxmax())}
    </div>""", unsafe_allow_html=True)

# Summary cards
if len(richness) > 0:
    avg_r = richness.mean(); zl, zc = interpret_score(avg_r)
    c1,c2,c3,c4,c5 = st.columns(5)
    for col_ui,lbl,val,suf,color,sub in [
        (c1,"MARKET RICHNESS",  f"{avg_r:.0f}", "/100", zc,       zl.upper()),
        (c2,"CHEAPEST SECTOR",  short_map.get(richness.idxmin(), richness.idxmin()), "", "#00cc44", "LOWEST SCORE"),
        (c3,"MOST EXPENSIVE",   short_map.get(richness.idxmax(), richness.idxmax()), "", "#ff3333", "HIGHEST SCORE"),
        (c4,"CHEAP SECTORS",    str((richness < 35).sum()), "/12", "#00cc44", "SCORE < 35"),
        (c5,"EXPENSIVE SECTORS",str((richness > 65).sum()), "/12", "#ff3333", "SCORE > 65"),
    ]:
        with col_ui:
            st.markdown(f"""
            <div class='bb-card'>
              <div class='lbl'>{lbl}</div>
              <div class='val' style='color:{color}'>{val}<span style='font-size:11px;color:#333'>{suf}</span></div>
              <div class='sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

st.markdown("")


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "SECTOR HEAT MAP", "SECTOR RANKING", "SECTOR DEEP DIVE",
    "CCA SCREENER", "ACTION CENTRE", "METHODOLOGY"
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 · HEAT MAP
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='bb-sec'>VALUATION PERCENTILE MATRIX &nbsp;·&nbsp; 0=CHEAPEST EVER &nbsp;·&nbsp; 100=MOST EXPENSIVE EVER</div>", unsafe_allow_html=True)

    disp = pct_matrix[sel_metrics].copy() if sel_metrics else pct_matrix.copy()
    st.pyplot(draw_heatmap(disp, date.today().strftime("%d %b %Y").upper()),
              use_container_width=True)

    st.markdown("")
    lc = st.columns(5)
    for col, (zone, rng, color) in zip(lc, [
        ("VERY CHEAP", "0–20",   "#00cc44"),
        ("CHEAP",      "20–35",  "#44ff88"),
        ("FAIR",       "35–65",  "#888888"),
        ("EXPENSIVE",  "65–80",  "#ffaa00"),
        ("VERY EXP.",  "80–100", "#ff3333"),
    ]):
        with col:
            st.markdown(f"""
            <div style='text-align:center;background:#080808;border-top:2px solid {color};
                        padding:5px 4px'>
              <div style='font-size:9px;color:{color};font-weight:700;letter-spacing:.1em'>{zone}</div>
              <div style='font-size:9px;color:#333'>{rng}</div>
            </div>""", unsafe_allow_html=True)

    with st.expander("RAW PERCENTILE DATA", expanded=False):
        st.dataframe(
            disp.rename(columns=METRIC_LABELS)
                .rename(index=lambda s: s.upper())
                .style.background_gradient(cmap="RdYlGn_r", vmin=0, vmax=100, axis=None)
                .format("{:.0f}", na_rep="—"),
            use_container_width=True)

    with st.expander("CURRENT SECTOR MULTIPLES (LIVE)", expanded=False):
        st.dataframe(
            sec_df.rename(columns=METRIC_LABELS)
                  .rename(index=lambda s: s.upper())
                  .style.format("{:.2f}", na_rep="N/A"),
            use_container_width=True)

    # ── HEATMAP CELL DECODER ───────────────────────────────────────────────────
    st.markdown("<div class='bb-sec'>CELL DECODER &nbsp;·&nbsp; CLICK ANY SECTOR + METRIC TO UNDERSTAND WHAT THE NUMBER MEANS</div>", unsafe_allow_html=True)

    dec_c1, dec_c2 = st.columns([1, 1])
    with dec_c1:
        dec_sector = st.selectbox("SELECT SECTOR", list(NSE500.keys()),
                                   format_func=lambda s: s.upper(), key="dec_sec")
    with dec_c2:
        dec_metric = st.selectbox("SELECT METRIC", NUMERIC_COLS,
                                   format_func=lambda m: METRIC_LABELS.get(m, m), key="dec_met")

    if dec_sector and dec_metric:
        cell_val = None
        if dec_sector in pct_matrix.index and dec_metric in pct_matrix.columns:
            v = pct_matrix.loc[dec_sector, dec_metric]
            if not pd.isna(v):
                cell_val = float(v)

        raw_val = None
        if dec_sector in sec_df.index and dec_metric in sec_df.columns:
            rv = sec_df.loc[dec_sector, dec_metric]
            if not pd.isna(rv):
                raw_val = float(rv)

        if cell_val is not None:
            zl, zc = interpret_score(cell_val)
            met_lbl = METRIC_LABELS.get(dec_metric, dec_metric)
            suf = "%" if dec_metric == "div_yield" else "x"

            # Interpretation text
            if cell_val <= 20:
                interp = f"This sector's {met_lbl} is cheaper than {100-cell_val:.0f}% of all readings in the past 10 years. This is a historically rare valuation opportunity. The sector has only been this cheap or cheaper {cell_val:.0f}% of the time in the last decade."
                action_hint = "STRONG VALUE SIGNAL — historically cheap. Worth investigating for a long position."
                hint_color = "#00cc44"
            elif cell_val <= 35:
                interp = f"This sector's {met_lbl} is below its 10-year average. The sector is trading at a discount to its own history. Cheap but not at extreme levels."
                action_hint = "MODERATE VALUE SIGNAL — below-average valuation. Favourable entry zone."
                hint_color = "#44ff88"
            elif cell_val <= 65:
                interp = f"This sector's {met_lbl} is near its 10-year average. The sector is fairly valued — not cheap, not expensive. No strong valuation signal in either direction."
                action_hint = "NEUTRAL — fairly valued. Valuation alone does not support a buy or sell call."
                hint_color = "#888888"
            elif cell_val <= 80:
                interp = f"This sector's {met_lbl} is above its 10-year average. The sector is trading at a premium to its own history. Valuations are stretched — investor expectations are high."
                action_hint = "CAUTION — above-average valuation. New entries carry higher valuation risk."
                hint_color = "#ffaa00"
            else:
                interp = f"This sector's {met_lbl} is in the top {100-cell_val:.0f}% of its 10-year history. The sector has almost never been this expensive. Risk of mean reversion is high."
                action_hint = "WARNING — near historical peak valuation. Significant downside risk if earnings disappoint."
                hint_color = "#ff3333"

            raw_disp = f"{raw_val:.1f}{suf}" if raw_val else "N/A"

            st.markdown(f"""
            <div style='background:#080808;border:1px solid #1a1a1a;border-left:4px solid {zc};
                        padding:18px 20px;margin-top:8px'>
              <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px'>
                <div>
                  <div style='font-size:10px;color:#ff6600;letter-spacing:.15em;text-transform:uppercase;
                               margin-bottom:4px'>{dec_sector.upper()}  ·  {met_lbl}</div>
                  <div style='font-size:11px;color:#555;letter-spacing:.08em;text-transform:uppercase'>
                    CURRENT VALUE: <span style='color:#ccc'>{raw_disp}</span>
                  </div>
                </div>
                <div style='text-align:right'>
                  <div style='font-size:42px;font-weight:700;color:{zc};line-height:1'>{cell_val:.0f}</div>
                  <div style='font-size:9px;color:#555;text-transform:uppercase;letter-spacing:.1em'>OUT OF 100</div>
                  <div style='font-size:10px;color:{zc};font-weight:700;letter-spacing:.1em;margin-top:2px'>{zl.upper()}</div>
                </div>
              </div>
              <div style='font-size:12px;color:#cccccc;line-height:1.7;margin-bottom:14px;
                           border-top:1px solid #1a1a1a;padding-top:12px'>
                {interp}
              </div>
              <div style='background:#0a0600;border:1px solid #2a1800;border-left:3px solid {hint_color};
                           padding:10px 14px;font-size:11px;color:{hint_color};
                           font-weight:600;letter-spacing:.06em;text-transform:uppercase'>
                ◆  {action_hint}
              </div>
              <div style='margin-top:10px;font-size:10px;color:#444;letter-spacing:.06em'>
                PERCENTILE RANK: {cell_val:.0f}/100 &nbsp;·&nbsp;
                CHEAPER THAN THIS: {cell_val:.0f}% OF THE TIME &nbsp;·&nbsp;
                MORE EXPENSIVE THAN THIS: {100-cell_val:.0f}% OF THE TIME
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#080808;border:1px solid #1a1a1a;padding:16px 20px;
                         font-size:11px;color:#444;letter-spacing:.06em'>
              NO DATA AVAILABLE FOR THIS SECTOR / METRIC COMBINATION
            </div>""", unsafe_allow_html=True)



# ────────────────────────────────────────────────────────────────────────────
# TAB 2 · SECTOR RANKING
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='bb-sec'>COMPOSITE RICHNESS RANKING &nbsp;·&nbsp; LIVE DATA</div>", unsafe_allow_html=True)
    st.plotly_chart(draw_ranking(richness), use_container_width=True)

    st.markdown("<div class='bb-sec'>SECTOR SCORECARD</div>", unsafe_allow_html=True)
    rows = []
    for s in richness.index:
        sc = richness[s]; zl, _ = interpret_score(sc)
        m  = sec_df.loc[s] if s in sec_df.index else pd.Series()
        rows.append({
            "SECTOR":    s.upper(),
            "SCORE":     f"{sc:.0f}/100",
            "ZONE":      zl.upper(),
            "P/E":       f"{m.get('pe',float('nan')):.1f}x"  if not pd.isna(m.get("pe",float("nan")))       else "N/A",
            "P/BV":      f"{m.get('pb',float('nan')):.2f}x"  if not pd.isna(m.get("pb",float("nan")))       else "N/A",
            "EV/EBITDA": f"{m.get('ev_ebitda',float('nan')):.1f}x" if not pd.isna(m.get("ev_ebitda",float("nan"))) else "N/A",
            "DIV YLD":   f"{m.get('div_yield',float('nan')):.2f}%" if not pd.isna(m.get("div_yield",float("nan"))) else "N/A",
        })
    st.dataframe(
        pd.DataFrame(rows).style.map(
            lambda v: "color:#00cc44;font-weight:600" if "CHEAP" in str(v) else
                      ("color:#ff3333;font-weight:600" if "EXP" in str(v) else ""),
            subset=["ZONE"]),
        hide_index=True, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 · SECTOR DEEP DIVE
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='bb-sec'>SECTOR ANALYSIS &nbsp;·&nbsp; 10-YEAR HISTORICAL CONTEXT</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        sel_sec = st.selectbox("SELECT SECTOR", SECTOR_LIST,
                               format_func=lambda s: s.upper(), key="d_sec")
    with c2:
        sel_met = st.selectbox("SELECT METRIC", NUMERIC_COLS,
                               format_func=lambda m: METRIC_LABELS.get(m, m), key="d_met")

    if sel_sec in pct_matrix.index:
        prow = pct_matrix.loc[sel_sec]
        sc   = composite_score(pct_matrix, sel_sec); zl, zc = interpret_score(sc or 50)
        mults = sec_df.loc[sel_sec] if sel_sec in sec_df.index else pd.Series()

        st.markdown(f"""
        <div style='background:#080808;border-left:3px solid #ff6600;
                    padding:10px 14px;margin:8px 0;
                    display:flex;justify-content:space-between;align-items:center'>
          <span style='font-size:13px;font-weight:700;color:#fff;
                       letter-spacing:.08em'>{sel_sec.upper()}</span>
          <span>
            <span style='font-size:9px;color:{zc};font-weight:700;
                         letter-spacing:.12em'>◆ {zl.upper()}</span>
            <span style='font-size:20px;font-weight:700;color:{zc};
                         margin-left:10px'>{sc:.0f}</span>
            <span style='font-size:10px;color:#333'>/100</span>
          </span>
        </div>""", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        for col_ui, (mk, ml) in zip([m1,m2,m3,m4],
                                     [("pe","P/E"),("pb","P/BV"),
                                      ("ev_ebitda","EV/EBITDA"),("div_yield","DIV YLD")]):
            val = mults.get(mk); pct = prow.get(mk)
            _, pc = interpret_score(float(pct) if pct and not pd.isna(pct) else 50)
            col_ui.markdown(f"""
            <div class='bb-card'>
              <div class='lbl'>{ml}</div>
              <div class='val' style='color:{pc}'>{f"{float(val):.1f}" if (val and not pd.isna(val)) else "N/A"}</div>
              <div class='sub' style='color:{pc}'>{f"{float(pct):.0f}TH PCT" if (pct and not pd.isna(pct)) else ""}</div>
            </div>""", unsafe_allow_html=True)

    ch1, ch2 = st.columns([2, 1])
    with ch1:
        curr = None
        if sel_sec in sec_df.index:
            v = sec_df.loc[sel_sec, sel_met]
            if not pd.isna(v): curr = float(v)
        if sel_sec in hist_dict:
            st.plotly_chart(
                draw_history(sel_sec, sel_met, hist_dict[sel_sec], curr, years),
                use_container_width=True)
    with ch2:
        if sel_sec in pct_matrix.index:
            st.plotly_chart(draw_radar(pct_matrix.loc[sel_sec], sel_sec),
                            use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 4 · CCA SCREENER
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='bb-sec'>COMPARABLE COMPANY ANALYSIS &nbsp;·&nbsp; NSE 500 &nbsp;·&nbsp; LIVE TRADING MULTIPLES</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='bb-info'>
      SELECT TARGET COMPANY → AUTO-IDENTIFY PEERS (SAME SECTOR + CAP TIER) →
      COMPUTE 6 LIVE TRADING MULTIPLES → SHOW PREMIUM/DISCOUNT VS PEER MEDIAN
    </div>""", unsafe_allow_html=True)

    sc_col, _ = st.columns([2, 2])
    with sc_col:
        sel_ticker = st.selectbox(
            "SELECT COMPANY",
            options=list(TICKER_NAMES.keys()),
            format_func=lambda t: TICKER_NAMES.get(t, t),
            index=0,
        )

    if sel_ticker:
        tinfo = ALL_TICKERS.get(sel_ticker, {})
        tname = tinfo.get("name", sel_ticker)
        tsec  = tinfo.get("sector", "Unknown")
        tcap  = tinfo.get("cap_tier", "mid")

        with st.spinner(f"FETCHING LIVE COMPS FOR {tname.upper()}..."):
            peers_df = find_peers(sel_ticker, raw_df, n_peers=n_peers)

        if peers_df.empty:
            st.warning("NO DATA. TRY REFRESHING.")
        else:
            peers_clean = clean_cca(peers_df)
            comps_tbl   = build_comps_table(peers_clean)
            pd_tbl      = build_premium_discount(comps_tbl)
            ff_data     = football_field(comps_tbl)
            pstats      = peer_stats(comps_tbl)

            trow = peers_clean[peers_clean["ticker"] == sel_ticker]
            trow = trow.iloc[0] if not trow.empty else pd.Series()

            price  = trow.get("price");  mktcap = trow.get("mktcap")
            h52    = trow.get("52w_high"); l52   = trow.get("52w_low")
            rec    = str(trow.get("recommend","")).upper()
            beta   = trow.get("beta")
            rc     = "#00cc44" if "BUY" in rec else ("#ff3333" if "SELL" in rec else "#ffaa00")

            # Company header
            st.markdown(f"""
            <div class='bb-co'>
              <div style='display:flex;justify-content:space-between;align-items:flex-start'>
                <div>
                  <div style='font-size:15px;font-weight:700;color:#fff;
                               letter-spacing:.06em'>{tname.upper()}</div>
                  <div style='font-size:9px;color:#444;letter-spacing:.08em;
                               text-transform:uppercase;margin-top:3px'>
                    {sel_ticker.replace('.NS','')} &nbsp;|&nbsp; {tsec.upper()} &nbsp;|&nbsp;
                    {tcap.upper()} CAP &nbsp;|&nbsp; {len(peers_df)-1} PEERS FOUND
                  </div>
                </div>
                <div style='text-align:right'>
                  <div style='font-size:21px;font-weight:700;color:#fff'>
                    {"₹"+f"{float(price):,.1f}" if price else "N/A"}
                  </div>
                  <div style='font-size:9px;color:#444'>
                    {"₹"+f"{float(mktcap)/1e7:,.0f} CR MKT CAP" if mktcap else ""}
                  </div>
                </div>
              </div>
              <div style='display:flex;gap:20px;margin-top:8px;padding-top:8px;
                          border-top:1px solid #111;font-size:9px;flex-wrap:wrap'>
                <span style='color:#444'>52W HIGH: <b style="color:#ccc">{"₹"+f"{float(h52):,.1f}" if h52 else "N/A"}</b></span>
                <span style='color:#444'>52W LOW: <b style="color:#ccc">{"₹"+f"{float(l52):,.1f}" if l52 else "N/A"}</b></span>
                <span style='color:#444'>BETA: <b style="color:#ccc">{f"{float(beta):.2f}" if beta else "N/A"}</b></span>
                <span style='color:#444'>ANALYST: <b style="color:{rc}">{rec or "N/A"}</b></span>
              </div>
            </div>""", unsafe_allow_html=True)

            # Current multiples
            st.markdown("<div class='bb-sec'>CURRENT MULTIPLES VS PEER MEDIAN &nbsp;·&nbsp; LIVE</div>", unsafe_allow_html=True)
            mcols = st.columns(6)
            for i, (m, cfg) in enumerate(CCA_METRICS.items()):
                with mcols[i]:
                    tv = trow.get(m)
                    pv = pd.to_numeric(
                        peers_clean[peers_clean["ticker"] != sel_ticker].get(m, pd.Series()),
                        errors="coerce").dropna()
                    pm   = float(pv.median()) if not pv.empty else None
                    tv_f = float(tv) if (tv and not (isinstance(tv, float) and np.isnan(tv))) else None
                    prem = None
                    if tv_f and pm:
                        prem = (tv_f/pm - 1)*100
                        if not cfg["higher_is_expensive"]: prem = -prem
                    color = "#ff3333" if (prem and prem > 10) else \
                            ("#00cc44" if (prem and prem < -10) else "#888888")
                    st.markdown(f"""
                    <div class='bb-mini'>
                      <div class='ml'>{cfg['label']}</div>
                      <div class='mv' style='color:{color}'>{f"{tv_f:.1f}{cfg['suffix']}" if tv_f else "N/A"}</div>
                      <div class='mp'>{f"PEER: {pm:.1f}{cfg['suffix']}" if pm else "N/A"}</div>
                      <div class='mc' style='color:{color}'>{f"{prem:+.1f}%" if prem else "—"}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("")

            # Trading comps table
            st.markdown("<div class='bb-sec'>TRADING COMPARABLES TABLE &nbsp;·&nbsp; LIVE MULTIPLES</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:9px;color:#333;margin-bottom:5px;letter-spacing:.08em'>▶ FIRST ROW (GOLD) = TARGET &nbsp;·&nbsp; REMAINING = PEER SET</div>", unsafe_allow_html=True)

            # Build display table
            disp_cols = ["Company", "Ticker", "Mkt Cap"] + [cfg["label"] for cfg in CCA_METRICS.values()]
            avail = [c for c in disp_cols if c in comps_tbl.columns]
            disp_df = comps_tbl[avail].copy()
            for cfg in CCA_METRICS.values():
                if cfg["label"] in disp_df.columns:
                    disp_df[cfg["label"]] = disp_df[cfg["label"]].apply(
                        lambda v: f"{v:.1f}{cfg['suffix']}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "N/A"
                    )
            st.markdown(disp_df.to_html(index=False, escape=False), unsafe_allow_html=True)

            st.markdown("")

            # Charts
            ca, cb = st.columns([1, 1])
            with ca:
                st.markdown("<div class='bb-sec'>FOOTBALL FIELD &nbsp;·&nbsp; PEER RANGE VS TARGET</div>", unsafe_allow_html=True)
                if ff_data:
                    st.plotly_chart(draw_football_field(ff_data, tname), use_container_width=True)
            with cb:
                st.markdown("<div class='bb-sec'>PREMIUM / DISCOUNT VS PEER MEDIAN</div>", unsafe_allow_html=True)
                if not pd_tbl.empty:
                    st.plotly_chart(draw_premium_discount(pd_tbl, tname), use_container_width=True)

            st.markdown("")
            st.markdown("<div class='bb-sec'>PREMIUM / DISCOUNT ANALYSIS TABLE</div>", unsafe_allow_html=True)
            if not pd_tbl.empty:
                show = ["Metric","Target","Peer Median","Peer Mean","Premium/Disc %","Pct in Peers","Implied Price"]
                avl  = [c for c in show if c in pd_tbl.columns]
                def cpd(v):
                    try:
                        f = float(str(v).replace("%",""))
                        if f > 15:  return "color:#ff3333;font-weight:600"
                        if f < -15: return "color:#00cc44;font-weight:600"
                        return "color:#888"
                    except: return ""
                st.dataframe(pd_tbl[avl].style.map(cpd, subset=["Premium/Disc %"]),
                             hide_index=True, use_container_width=True)

            # Price chart
            st.markdown("<div class='bb-sec'>1-YEAR PRICE CHART &nbsp;·&nbsp; LIVE</div>", unsafe_allow_html=True)
            ph = load_price(sel_ticker)
            if not ph.empty:
                st.plotly_chart(draw_price_chart(ph, tname), use_container_width=True)

            # Fundamentals
            with st.expander("FUNDAMENTAL DATA", expanded=False):
                fitems = [("REVENUE","revenue"),("EBITDA","ebitda"),("NET INCOME","net_income"),
                          ("TOTAL DEBT","total_debt"),("CASH","cash"),("ROE","roe"),
                          ("OPER MARGIN","operating_margin"),("NET MARGIN","net_margin"),
                          ("REV GROWTH","revenue_growth"),("BETA","beta")]
                frows = []
                for lbl, key in fitems:
                    v = trow.get(key)
                    if v and not (isinstance(v, float) and np.isnan(v)):
                        if key in ["revenue","ebitda","net_income","total_debt","cash"]:
                            disp = f"₹{float(v)/1e7:,.0f} CR"
                        elif key in ["roe","operating_margin","net_margin","revenue_growth"]:
                            disp = f"{float(v)*100:.1f}%"
                        else:
                            disp = f"{float(v):.2f}"
                        frows.append({"METRIC": lbl, "VALUE": disp})
                if frows:
                    st.dataframe(pd.DataFrame(frows), hide_index=True, use_container_width=True)

            with st.expander("PEER SUMMARY STATISTICS", expanded=False):
                if not pstats.empty:
                    st.dataframe(pstats.style.format(
                        {c: "{:.1f}" for c in ["Mean","Median","Min","Max","P25","P75"]},
                        na_rep="N/A"), hide_index=True, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# TAB 5 · ACTION CENTRE
# ────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("<div class='bb-sec'>ACTION CENTRE &nbsp;·&nbsp; SELECT SECTOR + STOCK &nbsp;·&nbsp; GET FULL ANALYSIS + RECOMMENDATION</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#0a0600;border:1px solid #2a1800;border-left:3px solid #ff6600;
                padding:10px 14px;margin-bottom:14px;font-size:10px;color:#888;letter-spacing:.05em'>
      This module combines data from all 4 components — Sector Heat Map, Sector Ranking,
      Sector Deep Dive, and CCA Screener — to give you a plain-English recommendation
      on whether to BUY, HOLD, or AVOID a stock and its sector. Designed for any investor,
      from first-time traders to experienced fund managers.
    </div>""", unsafe_allow_html=True)

    ac_c1, ac_c2 = st.columns([1, 1])
    with ac_c1:
        ac_sector = st.selectbox("STEP 1 — SELECT SECTOR", list(NSE500.keys()),
                                  format_func=lambda s: s.upper(), key="ac_sec")
    with ac_c2:
        sector_tickers = [(t, d["name"]) for t, d in ALL_TICKERS.items()
                          if d["sector"] == ac_sector]
        ac_ticker = st.selectbox(
            "STEP 2 — SELECT STOCK FROM THIS SECTOR",
            options=[t for t, _ in sector_tickers],
            format_func=lambda t: f"{ALL_TICKERS[t]['name']}  [{t.replace('.NS','')}]",
            key="ac_tick"
        )

    if ac_sector and ac_ticker:
        # ── Pull all data ──────────────────────────────────────────────────────
        sc = composite_score(pct_matrix, ac_sector)
        sec_zl, sec_zc = interpret_score(sc or 50)
        mults = sec_df.loc[ac_sector] if ac_sector in sec_df.index else pd.Series()
        prow  = pct_matrix.loc[ac_sector] if ac_sector in pct_matrix.index else pd.Series()

        # Stock CCA data
        with st.spinner("LOADING STOCK DATA..."):
            peers_df = find_peers(ac_ticker, raw_df, n_peers=8)

        from src.analytics import build_comps_table, build_premium_discount, clean_cca
        peers_clean = clean_cca(peers_df) if not peers_df.empty else pd.DataFrame()
        comps_tbl   = build_comps_table(peers_clean) if not peers_clean.empty else pd.DataFrame()
        pd_tbl      = build_premium_discount(comps_tbl) if not comps_tbl.empty else pd.DataFrame()

        trow = pd.Series()
        if not peers_clean.empty:
            tr = peers_clean[peers_clean["ticker"] == ac_ticker]
            if not tr.empty:
                trow = tr.iloc[0]

        tname  = ALL_TICKERS.get(ac_ticker, {}).get("name", ac_ticker)
        price  = trow.get("price")
        mktcap = trow.get("mktcap")
        rec    = str(trow.get("recommend", "")).upper()

        # ── Section 1: Sector Analysis ─────────────────────────────────────────
        st.markdown(f"""
        <div style='background:#080808;border:1px solid #1a1a1a;border-top:2px solid #ff6600;
                     padding:14px 18px;margin:12px 0 6px'>
          <div style='font-size:9px;color:#ff6600;letter-spacing:.18em;text-transform:uppercase;margin-bottom:8px'>
            ◆ PART 1 OF 3 &nbsp;·&nbsp; SECTOR ANALYSIS &nbsp;·&nbsp; {ac_sector.upper()}
          </div>
          <div style='display:flex;justify-content:space-between;align-items:center'>
            <div style='font-size:13px;font-weight:700;color:#fff'>{ac_sector.upper()}</div>
            <div>
              <span style='font-size:28px;font-weight:700;color:{sec_zc}'>{sc:.0f}</span>
              <span style='font-size:10px;color:#444'>/100 &nbsp;</span>
              <span style='font-size:10px;color:{sec_zc};font-weight:700'>{sec_zl.upper()}</span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Sector metric cards
        sm1, sm2, sm3, sm4 = st.columns(4)
        for col_ui, (mk, ml) in zip([sm1,sm2,sm3,sm4],
                                     [("pe","P/E"),("pb","P/BV"),("ev_ebitda","EV/EBITDA"),("div_yield","DIV YLD")]):
            val  = mults.get(mk)
            pct  = float(prow.get(mk)) if (prow.get(mk) and not pd.isna(prow.get(mk))) else None
            _, pc = interpret_score(pct if pct else 50)
            suf  = "%" if mk == "div_yield" else "x"
            col_ui.markdown(f"""
            <div class='bb-card'>
              <div class='lbl'>{ml}</div>
              <div class='val' style='color:{pc}'>{f"{float(val):.1f}{suf}" if (val and not pd.isna(val)) else "N/A"}</div>
              <div class='sub' style='color:{pc}'>{f"{pct:.0f}TH PERCENTILE" if pct else ""}</div>
            </div>""", unsafe_allow_html=True)

        # Sector interpretation
        if sc <= 20:
            sec_interp = f"The {ac_sector} sector is trading near its cheapest levels in 10 years. The composite richness score of {sc:.0f}/100 means the sector has only been cheaper {sc:.0f}% of the time in the past decade. This is a historically attractive entry point for the sector."
            sec_signal = "SECTOR: STRONGLY FAVOURABLE FOR ENTRY"
            sec_sig_c = "#00cc44"
        elif sc <= 35:
            sec_interp = f"The {ac_sector} sector is below its 10-year average valuation. A score of {sc:.0f}/100 suggests the sector is modestly undervalued relative to its own history. Valuations support building or adding to positions."
            sec_signal = "SECTOR: FAVOURABLE FOR ENTRY"
            sec_sig_c = "#44ff88"
        elif sc <= 65:
            sec_interp = f"The {ac_sector} sector is trading near its 10-year fair value. A score of {sc:.0f}/100 means valuations are neither cheap nor expensive. The sector does not offer a valuation edge — stock selection becomes more important."
            sec_signal = "SECTOR: NEUTRAL — STOCK SELECTION CRITICAL"
            sec_sig_c = "#888888"
        elif sc <= 80:
            sec_interp = f"The {ac_sector} sector is above its 10-year average valuation. A score of {sc:.0f}/100 signals that investor expectations are elevated. New entries at these levels carry higher valuation risk if earnings disappoint."
            sec_signal = "SECTOR: CAUTION — ELEVATED VALUATION"
            sec_sig_c = "#ffaa00"
        else:
            sec_interp = f"The {ac_sector} sector is near its most expensive level in 10 years. A score of {sc:.0f}/100 means the sector has barely ever been this expensive. The risk of a valuation-driven correction is high."
            sec_signal = "SECTOR: AVOID NEW ENTRY — NEAR HISTORICAL PEAK"
            sec_sig_c = "#ff3333"

        st.markdown(f"""
        <div style='background:#080808;border:1px solid #1a1a1a;padding:14px 18px;margin-bottom:4px'>
          <div style='font-size:11px;color:#ccc;line-height:1.8;margin-bottom:10px'>{sec_interp}</div>
          <div style='background:#0a0600;border-left:3px solid {sec_sig_c};padding:8px 12px;
                       font-size:10px;color:{sec_sig_c};font-weight:700;letter-spacing:.08em'>
            {sec_signal}
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Section 2: Stock Analysis ──────────────────────────────────────────
        st.markdown(f"""
        <div style='background:#080808;border:1px solid #1a1a1a;border-top:2px solid #ff6600;
                     padding:14px 18px;margin:12px 0 6px'>
          <div style='font-size:9px;color:#ff6600;letter-spacing:.18em;text-transform:uppercase;margin-bottom:8px'>
            ◆ PART 2 OF 3 &nbsp;·&nbsp; STOCK ANALYSIS &nbsp;·&nbsp; {tname.upper()}
          </div>
          <div style='display:flex;justify-content:space-between;align-items:center'>
            <div>
              <div style='font-size:13px;font-weight:700;color:#fff'>{tname.upper()}</div>
              <div style='font-size:9px;color:#444;letter-spacing:.08em;margin-top:3px'>
                {ac_ticker.replace(".NS","")} &nbsp;·&nbsp; {ac_sector.upper()} &nbsp;·&nbsp;
                {ALL_TICKERS.get(ac_ticker,{}).get("cap_tier","").upper()} CAP
              </div>
            </div>
            <div style='text-align:right'>
              <div style='font-size:20px;font-weight:700;color:#fff'>
                {"₹"+f"{float(price):,.1f}" if price else "N/A"}
              </div>
              <div style='font-size:9px;color:#444'>
                {"₹"+f"{float(mktcap)/1e7:,.0f} CR" if mktcap else ""}
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Stock multiples vs peers
        if not pd_tbl.empty:
            st_cols = st.columns(len(pd_tbl))
            for i, (_, row) in enumerate(pd_tbl.iterrows()):
                with st_cols[i]:
                    tv   = row.get("Target")
                    pm   = row.get("Peer Median")
                    prem = row.get("Premium/Disc %")
                    imp  = row.get("Implied Price", "N/A")
                    color = "#ff3333" if (prem and float(prem)>10) else                             ("#00cc44" if (prem and float(prem)<-10) else "#888")
                    st.markdown(f"""
                    <div class='bb-mini'>
                      <div class='ml'>{row.get("Metric","")}</div>
                      <div class='mv' style='color:{color}'>{f"{tv:.1f}x" if tv else "N/A"}</div>
                      <div class='mp'>PEER: {f"{pm:.1f}x" if pm else "N/A"}</div>
                      <div class='mc' style='color:{color}'>{f"{float(prem):+.1f}%" if prem else "—"}</div>
                    </div>""", unsafe_allow_html=True)

            # Stock interpretation
            premiums = [r["Premium/Disc %"] for _, r in pd_tbl.iterrows()
                       if r.get("Premium/Disc %") is not None]
            avg_prem = float(np.mean(premiums)) if premiums else 0

            if avg_prem < -20:
                stk_interp = f"{tname} trades at an average discount of {abs(avg_prem):.1f}% vs its peer group across all valuation metrics. This means you are paying significantly less than what similar companies cost. This is a potentially undervalued stock within the sector."
                stk_signal = f"STOCK: ATTRACTIVELY PRICED VS PEERS — {abs(avg_prem):.0f}% DISCOUNT"
                stk_sig_c  = "#00cc44"
            elif avg_prem < -10:
                stk_interp = f"{tname} trades at a moderate discount of {abs(avg_prem):.1f}% vs its peer group. The stock appears reasonably priced relative to comparable companies in the sector."
                stk_signal = f"STOCK: MODESTLY CHEAP VS PEERS — {abs(avg_prem):.0f}% DISCOUNT"
                stk_sig_c  = "#44ff88"
            elif avg_prem < 10:
                stk_interp = f"{tname} trades broadly in line with its peer group (average {avg_prem:+.1f}% vs peers). The stock is fairly valued relative to comparable companies — no strong premium or discount signal."
                stk_signal = "STOCK: FAIRLY VALUED VS PEERS"
                stk_sig_c  = "#888888"
            elif avg_prem < 25:
                stk_interp = f"{tname} trades at a premium of {avg_prem:.1f}% vs its peer group. Investors are paying more for this stock than for comparable companies. This premium is only justified if the company has stronger growth or returns than peers."
                stk_signal = f"STOCK: TRADING AT PREMIUM TO PEERS — {avg_prem:.0f}% ABOVE PEER MEDIAN"
                stk_sig_c  = "#ffaa00"
            else:
                stk_interp = f"{tname} trades at a significant premium of {avg_prem:.1f}% vs its peer group. This is a very expensive stock relative to its sector peers. The premium requires exceptional earnings growth to be sustained."
                stk_signal = f"STOCK: SIGNIFICANTLY OVERPRICED VS PEERS — {avg_prem:.0f}% ABOVE PEER MEDIAN"
                stk_sig_c  = "#ff3333"

            st.markdown(f"""
            <div style='background:#080808;border:1px solid #1a1a1a;padding:14px 18px;margin-bottom:4px'>
              <div style='font-size:11px;color:#ccc;line-height:1.8;margin-bottom:10px'>{stk_interp}</div>
              <div style='background:#0a0600;border-left:3px solid {stk_sig_c};padding:8px 12px;
                           font-size:10px;color:{stk_sig_c};font-weight:700;letter-spacing:.08em'>
                {stk_signal}
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            avg_prem = 0
            stk_sig_c = "#888888"

        # ── Section 3: Final Recommendation ───────────────────────────────────
        st.markdown(f"""
        <div style='background:#080808;border:1px solid #1a1a1a;border-top:2px solid #ff6600;
                     padding:14px 18px;margin:12px 0 6px'>
          <div style='font-size:9px;color:#ff6600;letter-spacing:.18em;text-transform:uppercase'>
            ◆ PART 3 OF 3 &nbsp;·&nbsp; FINAL RECOMMENDATION
          </div>
        </div>""", unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════════════════
        # STOCK-PERFORMANCE-FOCUSED SCORING (75% stock / 25% sector)
        # ═══════════════════════════════════════════════════════════════════════

        # ── Signal 1: Sector Valuation (weight: 25%, max 2.5 pts) ──────────
        sector_pts = 0.0
        if sc <= 20:   sector_pts = 2.5
        elif sc <= 35: sector_pts = 2.0
        elif sc <= 50: sector_pts = 1.5
        elif sc <= 65: sector_pts = 1.0
        elif sc <= 80: sector_pts = 0.5
        else:          sector_pts = 0.0

        # ── Signal 2: CCA Premium/Discount vs Peers (weight: 25%, max 2.5 pts) ──
        cca_pts = 0.0
        if avg_prem < -20:   cca_pts = 2.5
        elif avg_prem < -10: cca_pts = 2.0
        elif avg_prem < 0:   cca_pts = 1.5
        elif avg_prem < 10:  cca_pts = 1.0
        elif avg_prem < 25:  cca_pts = 0.5
        else:                cca_pts = 0.0

        # ── Signal 3: 52-Week Range Position (weight: 20%, max 2 pts) ──────
        h52 = trow.get("52w_high")
        l52 = trow.get("52w_low")
        range_pts = 0.0
        range_pct = None
        if price and h52 and l52:
            try:
                p, h, l = float(price), float(h52), float(l52)
                if h > l and h > 0:
                    range_pct = ((p - l) / (h - l)) * 100  # 0=at 52w low, 100=at 52w high
                    drawdown = ((h - p) / h) * 100  # % below 52w high
                    if drawdown >= 30:   range_pts = 2.0   # Deep correction — attractive
                    elif drawdown >= 20: range_pts = 1.6
                    elif drawdown >= 10: range_pts = 1.2
                    elif drawdown >= 5:  range_pts = 0.8
                    else:                range_pts = 0.4   # Near highs — less attractive entry
            except (ValueError, TypeError):
                pass

        # ── Signal 4: Analyst Consensus (weight: 15%, max 1.5 pts) ──────────
        analyst_pts = 0.0
        rec_upper = rec.upper() if rec else ""
        if "STRONG" in rec_upper and "BUY" in rec_upper:   analyst_pts = 1.5
        elif "BUY" in rec_upper:                           analyst_pts = 1.2
        elif "OUTPERFORM" in rec_upper:                    analyst_pts = 1.0
        elif "HOLD" in rec_upper or "NEUTRAL" in rec_upper: analyst_pts = 0.7
        elif "UNDERPERFORM" in rec_upper:                  analyst_pts = 0.3
        elif "SELL" in rec_upper:                          analyst_pts = 0.0
        else:                                              analyst_pts = 0.7  # No data = neutral

        # ── Signal 5: Fundamentals — ROE & Margins (weight: 15%, max 1.5 pts) ──
        fund_pts = 0.0
        roe_val = trow.get("roe")
        opm_val = trow.get("operating_margin")
        npm_val = trow.get("net_margin")
        fund_signals = 0
        fund_count = 0

        if roe_val and not (isinstance(roe_val, float) and np.isnan(roe_val)):
            fund_count += 1
            rv = float(roe_val)
            if rv > 0.20: fund_signals += 2
            elif rv > 0.12: fund_signals += 1
            else: fund_signals += 0

        if opm_val and not (isinstance(opm_val, float) and np.isnan(opm_val)):
            fund_count += 1
            ov = float(opm_val)
            if ov > 0.20: fund_signals += 2
            elif ov > 0.10: fund_signals += 1
            else: fund_signals += 0

        if npm_val and not (isinstance(npm_val, float) and np.isnan(npm_val)):
            fund_count += 1
            nv = float(npm_val)
            if nv > 0.15: fund_signals += 2
            elif nv > 0.08: fund_signals += 1
            else: fund_signals += 0

        if fund_count > 0:
            fund_ratio = fund_signals / (fund_count * 2)  # 0 to 1
            fund_pts = round(fund_ratio * 1.5, 2)
        else:
            fund_pts = 0.75  # No data = neutral

        # ── TOTAL SCORE (0 to 10) ──────────────────────────────────────────
        total_score = sector_pts + cca_pts + range_pts + analyst_pts + fund_pts
        max_score = 10.0

        # ── Signal breakdowns for display ──────────────────────────────────
        drawdown_str = ""
        if price and h52:
            try:
                dd = ((float(h52) - float(price)) / float(h52)) * 100
                drawdown_str = f"{dd:.1f}% BELOW 52W HIGH"
            except: drawdown_str = "N/A"

        roe_str = f"{float(roe_val)*100:.1f}%" if (roe_val and not (isinstance(roe_val, float) and np.isnan(roe_val))) else "N/A"
        opm_str = f"{float(opm_val)*100:.1f}%" if (opm_val and not (isinstance(opm_val, float) and np.isnan(opm_val))) else "N/A"

        # ── Generate Recommendation ────────────────────────────────────────
        if total_score >= 8.0:
            final_rec = "STRONG BUY"
            rec_color = "#00cc44"
            rec_bg    = "#001a00"
            rec_icon  = "◆◆◆"
            rec_text  = f"{tname} scores exceptionally well across all performance indicators. The stock is {drawdown_str}, trading at a {abs(avg_prem):.1f}% discount to peers, and backed by strong fundamentals (ROE: {roe_str}). Analyst consensus is {rec or 'N/A'}. Combined with a supportive sector environment (richness: {sc:.0f}/100), this is a high-conviction opportunity for long-term investors (2–3 year horizon)."
        elif total_score >= 6.5:
            final_rec = "BUY"
            rec_color = "#44ff88"
            rec_bg    = "#001a00"
            rec_icon  = "◆◆"
            rec_text  = f"{tname} shows a favourable risk-reward profile. The stock is {drawdown_str} and {'trading at a discount to peers' if avg_prem < 0 else 'reasonably valued vs peers'}. Key fundamentals — ROE: {roe_str}, Operating Margin: {opm_str} — support the valuation. Analyst consensus: {rec or 'N/A'}. Suitable for building a position over a 12–18 month horizon."
        elif total_score >= 5.0:
            final_rec = "ACCUMULATE"
            rec_color = "#88cc44"
            rec_bg    = "#0a1a00"
            rec_icon  = "◆◆"
            rec_text  = f"{tname} has a mildly positive setup. The stock is {drawdown_str} with {'a modest discount' if avg_prem < 0 else 'fair pricing'} relative to peers ({avg_prem:+.1f}%). Fundamentals are {'solid' if fund_pts > 0.9 else 'adequate'} (ROE: {roe_str}). Consider accumulating on dips rather than a lump-sum entry. Sector backdrop: {sec_zl.lower()} (score: {sc:.0f}/100)."
        elif total_score >= 3.5:
            final_rec = "HOLD / WAIT"
            rec_color = "#ffaa00"
            rec_bg    = "#1a0f00"
            rec_icon  = "◆"
            rec_text  = f"{tname} presents a mixed picture. The stock is {drawdown_str} and trades at {avg_prem:+.1f}% vs peer median. {'Analyst consensus leans positive' if 'BUY' in rec_upper else 'Analyst consensus is neutral/mixed'}. For existing holders — continue holding. For new investors — wait for a 10–15% correction for a better entry point. No urgency to act."
        elif total_score >= 2.0:
            final_rec = "REDUCE / AVOID"
            rec_color = "#ff6644"
            rec_bg    = "#1a0800"
            rec_icon  = "◆"
            rec_text  = f"{tname} has an unfavourable risk-reward at current levels. The stock trades at a {avg_prem:+.1f}% premium to peers, is {drawdown_str}, and {'fundamentals do not justify the premium' if fund_pts < 0.9 else 'while fundamentals are decent, they are priced in'}. {'The sector is also expensive' if sc > 65 else 'Sector support is limited'}. Existing holders should consider trimming. Avoid fresh entries until the stock corrects."
        else:
            final_rec = "STRONG AVOID"
            rec_color = "#ff3333"
            rec_bg    = "#1a0000"
            rec_icon  = "◆◆◆"
            rec_text  = f"{tname} is flashing red across multiple indicators. The stock is expensive vs peers ({avg_prem:+.1f}% premium), {'near its 52-week high with limited upside' if (range_pct and range_pct > 85) else f'{drawdown_str}'}, and {'the sector is overheated' if sc > 65 else 'sector support is absent'} (richness: {sc:.0f}/100). {'Analyst consensus is bearish' if 'SELL' in rec_upper else 'Risk-reward is heavily skewed to the downside'}. Avoid new positions. Existing holders should consider booking profits."

        # ── Signal breakdown cards ─────────────────────────────────────────
        st.markdown(f"""
        <div style='background:#050505;border:1px solid #1a1a1a;padding:10px 16px;margin:8px 0;
                     display:flex;gap:12px;flex-wrap:wrap'>
          <div style='flex:1;min-width:80px;text-align:center;padding:6px;border-right:1px solid #111'>
            <div style='font-size:8px;color:#444;letter-spacing:.12em;text-transform:uppercase'>SECTOR</div>
            <div style='font-size:14px;font-weight:700;color:{sec_zc}'>{sector_pts:.1f}<span style='font-size:9px;color:#333'>/2.5</span></div>
          </div>
          <div style='flex:1;min-width:80px;text-align:center;padding:6px;border-right:1px solid #111'>
            <div style='font-size:8px;color:#444;letter-spacing:.12em;text-transform:uppercase'>VS PEERS</div>
            <div style='font-size:14px;font-weight:700;color:{"#00cc44" if cca_pts >= 1.5 else ("#ff3333" if cca_pts < 0.8 else "#ffaa00")}'>{cca_pts:.1f}<span style='font-size:9px;color:#333'>/2.5</span></div>
          </div>
          <div style='flex:1;min-width:80px;text-align:center;padding:6px;border-right:1px solid #111'>
            <div style='font-size:8px;color:#444;letter-spacing:.12em;text-transform:uppercase'>52W RANGE</div>
            <div style='font-size:14px;font-weight:700;color:{"#00cc44" if range_pts >= 1.2 else ("#ff3333" if range_pts < 0.6 else "#ffaa00")}'>{range_pts:.1f}<span style='font-size:9px;color:#333'>/2.0</span></div>
          </div>
          <div style='flex:1;min-width:80px;text-align:center;padding:6px;border-right:1px solid #111'>
            <div style='font-size:8px;color:#444;letter-spacing:.12em;text-transform:uppercase'>ANALYSTS</div>
            <div style='font-size:14px;font-weight:700;color:{"#00cc44" if analyst_pts >= 1.0 else ("#ff3333" if analyst_pts < 0.5 else "#ffaa00")}'>{analyst_pts:.1f}<span style='font-size:9px;color:#333'>/1.5</span></div>
          </div>
          <div style='flex:1;min-width:80px;text-align:center;padding:6px'>
            <div style='font-size:8px;color:#444;letter-spacing:.12em;text-transform:uppercase'>FUNDAMENT.</div>
            <div style='font-size:14px;font-weight:700;color:{"#00cc44" if fund_pts >= 0.9 else ("#ff3333" if fund_pts < 0.5 else "#ffaa00")}'>{fund_pts:.1f}<span style='font-size:9px;color:#333'>/1.5</span></div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:{rec_bg};border:2px solid {rec_color};padding:24px 24px;margin-top:8px'>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:16px'>
            <div>
              <div style='font-size:9px;color:{rec_color};letter-spacing:.2em;text-transform:uppercase;margin-bottom:6px'>
                TERMINAL RECOMMENDATION
              </div>
              <div style='font-size:28px;font-weight:700;color:{rec_color};letter-spacing:.06em'>
                {rec_icon} &nbsp; {final_rec}
              </div>
            </div>
            <div style='text-align:right;font-size:10px;color:#555;letter-spacing:.08em;line-height:2'>
              <div>COMPOSITE SCORE: <span style='color:{rec_color}'>{total_score:.1f}/{max_score:.0f}</span></div>
              <div>STOCK VS PEERS: <span style='color:{stk_sig_c}'>{avg_prem:+.1f}% AVG</span></div>
              <div>52W POSITION: <span style='color:#ccc'>{drawdown_str or "N/A"}</span></div>
              <div>ANALYST: <span style='color:#ccc'>{rec or "N/A"}</span> &nbsp;·&nbsp; ROE: <span style='color:#ccc'>{roe_str}</span></div>
              <div>SECTOR: <span style='color:{sec_zc}'>{sc:.0f}/100 ({sec_zl.upper()})</span></div>
            </div>
          </div>
          <div style='font-size:12px;color:#cccccc;line-height:1.9;border-top:1px solid {rec_color}33;
                       padding-top:14px'>
            {rec_text}
          </div>
          <div style='margin-top:16px;padding:10px 14px;background:rgba(0,0,0,0.4);
                       font-size:9px;color:#555;letter-spacing:.06em;line-height:1.8'>
            ⚠ DISCLAIMER: This recommendation is based on quantitative signals —
            peer comparison, 52-week price action, analyst consensus, and fundamental metrics.
            It does NOT account for: earnings quality, management changes, macro outlook,
            interest rate cycle, regulatory risk, or company-specific news.
            Always conduct your own research before making investment decisions.
          </div>
        </div>""", unsafe_allow_html=True)


# TAB 6 · METHODOLOGY
# ────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("<div class='bb-sec'>METHODOLOGY &nbsp;·&nbsp; DATA ARCHITECTURE &nbsp;·&nbsp; REAL-TIME DESIGN</div>", unsafe_allow_html=True)
    st.markdown("""
**DATA REFRESH POLICY**

This terminal fetches live data from Yahoo Finance on every new session.
Cache TTL = 15 minutes. No stale pkl files. No pre-loaded data.

```
ON EVERY OPEN:
  → Yahoo Finance API called for all NSE 500 companies
  → Real-time P/E, P/BV, EV/EBITDA, EV/Sales, Div Yield fetched
  → Historical distributions generated fresh
  → Percentile rankings computed against 10-year distributions
  → Dashboard renders with today's actual market data
```

---

**MODULE 1 — SECTOR HEAT MAP**

Each cell = where today's multiple sits in its own 10-year distribution.

```python
from scipy import stats
percentile = stats.percentileofscore(historical_10yr_array, today_value)
# 0 = cheapest ever  ·  50 = fair value  ·  100 = most expensive ever
```

Composite Richness Score = weighted average of 4 metric percentiles.
Sector-specific weights: Banks use P/BV 60%; Metals use EV/EBITDA 50%.

---

**MODULE 2 — CCA SCREENER**

1. Auto peer selection — same sector + adjacent market cap tier
2. Live multiples — P/E, Fwd P/E, EV/EBITDA, EV/Sales, P/BV, Div Yield
3. Trading comps table — formatted like an IB pitch book
4. Premium/Discount vs peer median for each multiple
5. Football field — peer range with target overlay
6. Implied price — if target re-rated to peer median

---

**ZONE DEFINITIONS**

| Score | Zone | Meaning |
|---|---|---|
| 0–20 | Very Cheap | Below 80% of own history |
| 20–35 | Cheap | Below average |
| 35–65 | Fair Value | Near historical norm |
| 65–80 | Expensive | Above average |
| 80–100 | Very Expensive | Near historical peak |
    """)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:#050505;border-top:1px solid #ff6600;padding:7px 16px;
            margin-top:14px;display:flex;justify-content:space-between;
            font-size:9px;color:#333;letter-spacing:.08em;text-transform:uppercase'>
  <span>◆ NSE VALUATION TERMINAL &nbsp;·&nbsp; BLOOMBERG STYLE &nbsp;·&nbsp; REAL-TIME DATA</span>
  <span>{date.today().strftime('%d %b %Y').upper()} &nbsp;·&nbsp; DATA: YAHOO FINANCE + NSE INDIA</span>
</div>""", unsafe_allow_html=True)
