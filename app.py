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
.streamlit-expanderHeader{background:#0a0a0a !important;color:#ff6600 !important;font-size:9px !important;text-transform:uppercase !important;letter-spacing:.12em !important;border:1px solid #1a1a1a !important;border-radius:0 !important}

/* Scrollbar */
::-webkit-scrollbar{width:3px;height:3px}
::-webkit-scrollbar-track{background:#000}
::-webkit-scrollbar-thumb{background:#ff6600}

/* Misc */
hr{border:none;border-top:1px solid #1a1a1a}
p,li{color:#888 !important;font-size:12px !important}
code{background:#0a0a0a !important;color:#ff6600 !important;border:1px solid #222 !important}

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
</style>
""", unsafe_allow_html=True)

from src.config import NSE500, ALL_TICKERS, CCA_METRICS, LOOKBACK_OPTIONS, NUMERIC_COLS, ZONE_COLORS
from src.data import (fetch_all_live, fetch_all_history, fetch_price_history,
                      aggregate_to_sector, find_peers, clean_company_data)
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

# Live sector strip
if len(richness) > 0:
    items = []
    short_map = {
        "Information Technology":"IT","Banking":"BANK","FMCG":"FMCG",
        "Automobiles":"AUTO","Pharmaceuticals":"PHARMA","Metals & Mining":"METALS",
        "Energy & Oil Gas":"ENERGY","Financial Services":"FIN SVCS",
        "Consumer Durables":"CONS DUR","Healthcare":"HEALTH",
        "Real Estate":"REALTY","Capital Goods & Infra":"INFRA",
    }
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "SECTOR HEAT MAP", "SECTOR RANKING", "SECTOR DEEP DIVE", "CCA SCREENER", "METHODOLOGY"
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

    with st.expander("RAW PERCENTILE DATA"):
        st.dataframe(
            disp.rename(columns=METRIC_LABELS)
                .rename(index=lambda s: s.upper())
                .style.background_gradient(cmap="RdYlGn_r", vmin=0, vmax=100, axis=None)
                .format("{:.0f}", na_rep="—"),
            use_container_width=True)

    with st.expander("CURRENT SECTOR MULTIPLES (LIVE)"):
        st.dataframe(
            sec_df.rename(columns=METRIC_LABELS)
                  .rename(index=lambda s: s.upper())
                  .style.format("{:.2f}", na_rep="N/A"),
            use_container_width=True)


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
            with st.expander("FUNDAMENTAL DATA"):
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

            with st.expander("PEER SUMMARY STATISTICS"):
                if not pstats.empty:
                    st.dataframe(pstats.style.format(
                        {c: "{:.1f}" for c in ["Mean","Median","Min","Max","P25","P75"]},
                        na_rep="N/A"), hide_index=True, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 5 · METHODOLOGY
# ────────────────────────────────────────────────────────────────────────────
with tab5:
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
