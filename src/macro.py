# macro.py — Qualitative, Macro, and News Data for NSE Terminal
# Curated data reflecting April 2026 market conditions
# Update periodically to keep assessments current

import time
import logging
import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. MACRO DASHBOARD — India economic indicators (April 2026)
# ══════════════════════════════════════════════════════════════════════════════

MACRO_DATA = {
    "repo_rate":        5.25,
    "repo_trend":       "neutral",       # "cutting" / "neutral" / "hiking"
    "repo_change_last": "-25 bps (Feb 2026)",
    "inflation_cpi":    4.2,
    "inflation_trend":  "stable",        # "rising" / "stable" / "falling"
    "gdp_growth":       6.8,
    "gdp_trend":        "stable",
    "iip_growth":       5.4,             # Index of Industrial Production
    "usd_inr":          86.5,
    "currency_trend":   "weakening",     # "strengthening" / "stable" / "weakening"
    "crude_oil_usd":    78.4,
    "crude_trend":      "rising",
    "fii_flow_ytd_cr":  -12400,          # FII net flow YTD in crores
    "dii_flow_ytd_cr":  48200,           # DII net flow YTD in crores
    "india_vix":        14.2,
    "vix_level":        "low",           # "low" (<15) / "moderate" (15-25) / "high" (>25)
    "us_10yr_yield":    4.35,
    "india_10yr_yield": 6.82,
    "policy_stance":    "Neutral — RBI paused after 125 bps of cuts since Feb 2025. "
                        "Focus on managing liquidity and supporting growth while monitoring "
                        "global uncertainty, crude oil prices, and rupee stability.",
    "macro_outlook":    "India's macro environment is broadly supportive. GDP growth at 6.8% "
                        "remains robust. Inflation is within RBI's comfort zone (4.2%). "
                        "However, global headwinds (US tariffs, crude oil volatility, geopolitical "
                        "tensions) create uncertainty. FII outflows (-₹12,400 Cr YTD) reflect "
                        "global risk-off sentiment, but strong DII inflows (+₹48,200 Cr) provide "
                        "domestic support. The rate cycle has paused after aggressive cuts, "
                        "benefiting rate-sensitive sectors.",
    "last_updated":     "April 2026",
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUSINESS CYCLE — Where each sector sits in its cycle
# ══════════════════════════════════════════════════════════════════════════════

SECTOR_CYCLE = {
    "Information Technology": {
        "phase":       "Recovery",
        "phase_score": 62,
        "trend":       "improving",
        "color":       "#44ff88",
        "drivers": [
            "AI/GenAI spending driving new deal pipeline",
            "Cloud migration cycle entering second wave",
            "Margin expansion from pyramid restructuring",
            "Strong order book visibility for FY27",
        ],
        "risks": [
            "US recession fears damping discretionary IT spend",
            "H-1B visa policy uncertainty under new US administration",
            "Rupee depreciation partially offset by revenue hedge",
            "GenAI could cannibalize traditional services revenue",
        ],
        "outlook": "IT sector is recovering from the FY25 slowdown. Deal wins are accelerating, "
                   "particularly in AI/cloud transformation. Large-cap IT (TCS, Infosys, HCL) "
                   "showing margin resilience while mid-caps (Persistent, Coforge) offer higher growth. "
                   "Revenue growth expected to recover to 8-12% in FY27 from 3-5% in FY25.",
    },
    "Banking": {
        "phase":       "Expansion",
        "phase_score": 78,
        "trend":       "stable",
        "color":       "#00cc44",
        "drivers": [
            "Credit growth at 14-15%, well above nominal GDP",
            "Net Interest Margins healthy at 3.2-3.5%",
            "Asset quality (GNPA) at decade-low 2.8%",
            "Rate cuts supporting loan demand and treasury gains",
        ],
        "risks": [
            "NIM compression as rate cuts flow through",
            "Unsecured lending (personal loans, credit cards) stress",
            "Potential asset quality deterioration in SME segment",
            "Competitive pressure from fintech/NBFCs",
        ],
        "outlook": "Banking sector is in a strong expansion phase. Credit growth remains robust, "
                   "asset quality is at multi-year highs, and rate cuts are providing treasury gains. "
                   "Large private banks (HDFC, ICICI) are well-positioned. PSU banks offer value "
                   "but face longer-term structural challenges. Watch for NIM compression in H2.",
    },
    "FMCG": {
        "phase":       "Contraction",
        "phase_score": 35,
        "trend":       "stable",
        "color":       "#ffaa00",
        "drivers": [
            "Rural demand recovery after 2 years of weakness",
            "Price hikes flowing through after commodity correction",
            "Premiumization trend benefiting top brands",
            "Distribution expansion into tier-3/4 cities",
        ],
        "risks": [
            "Volume growth sluggish at 3-5% for most companies",
            "Intense competition from D2C brands and quick commerce",
            "Input cost volatility (palm oil, crude-linked)",
            "Valuations remain elevated despite earnings slowdown",
        ],
        "outlook": "FMCG sector is in a mild contraction. Volume growth has been sluggish for "
                   "6+ quarters. Rural recovery is underway but slow. Companies are leaning on "
                   "price hikes rather than volume. Valuations remain stretched (50-70x P/E). "
                   "HUL, ITC trading at more reasonable levels. Wait for volume recovery confirmation.",
    },
    "Automobiles": {
        "phase":       "Peak",
        "phase_score": 85,
        "trend":       "stable",
        "color":       "#ff3333",
        "drivers": [
            "SUV and EV segments driving industry growth",
            "Export markets (especially Africa, ASEAN) expanding",
            "Premiumization — average selling prices rising",
            "Government EV incentives supporting transition",
        ],
        "risks": [
            "Peak cycle concerns — PV sales growth slowing to 3-5%",
            "EV transition disrupting traditional ICE OEMs",
            "Commodity costs (steel, aluminium) remain elevated",
            "2-wheeler segment recovery has plateaued",
        ],
        "outlook": "Auto sector appears near peak of its cycle. SUV demand remains strong but "
                   "overall PV growth is moderating. Maruti, M&M, Tata Motors are key beneficiaries. "
                   "EV transition creates both opportunity (Tata Motors, Bajaj) and risk (Hero, Ashok Leyland). "
                   "Expect growth normalization to 5-8% from 15-20% of FY24-25.",
    },
    "Pharmaceuticals": {
        "phase":       "Expansion",
        "phase_score": 72,
        "trend":       "improving",
        "color":       "#00cc44",
        "drivers": [
            "US generics pricing stabilizing after years of erosion",
            "Biosimilar launches providing new growth runway",
            "India domestic market growing at 10-12% consistently",
            "Contract manufacturing (CDMO) orders accelerating",
        ],
        "risks": [
            "US FDA inspection intensity — warning letters impact",
            "Drug price control orders (DPCO) constraining margins",
            "Currency headwinds on US revenue translation",
            "R&D productivity concerns for specialty pipeline",
        ],
        "outlook": "Pharma is in expansion mode. US generic pricing has stabilized, biosimilars "
                   "are driving new revenue streams, and India domestic growth is strong. "
                   "Sun Pharma leads with specialty focus. Dr. Reddy's and Cipla offer value. "
                   "CDMO trend benefits Divi's Labs, Syngene. FDA risk remains the key wildcard.",
    },
    "Metals & Mining": {
        "phase":       "Contraction",
        "phase_score": 30,
        "trend":       "deteriorating",
        "color":       "#ff6644",
        "drivers": [
            "China infrastructure stimulus creating demand floor",
            "India infrastructure spending (roads, railways) supportive",
            "Green energy transition driving copper, aluminium demand",
            "Consolidation improving industry pricing discipline",
        ],
        "risks": [
            "Global demand slowdown — China property sector weak",
            "US tariffs disrupting global trade flows",
            "Overcapacity in steel — Chinese dumping risk",
            "Commodity prices volatile and trending down",
        ],
        "outlook": "Metals sector is in contraction. Global demand weakness (especially China "
                   "property) and trade war concerns are weighing on prices. Steel margins are under "
                   "pressure from cheap Chinese imports. Tata Steel, JSW Steel face headwinds. "
                   "Hindalco better positioned through Novelis (US recycling). Wait for price cycle to turn.",
    },
    "Energy & Oil Gas": {
        "phase":       "Recovery",
        "phase_score": 55,
        "trend":       "stable",
        "color":       "#ffaa00",
        "drivers": [
            "Energy transition — renewables capacity addition accelerating",
            "Government push for energy security (domestic production)",
            "Gas economy expansion — city gas distribution growth",
            "Power demand growing at 7-8% (data centers, EVs)",
        ],
        "risks": [
            "Crude oil price volatility (geopolitical premium)",
            "Government pricing intervention (fuel subsidies)",
            "Stranded asset risk for oil E&P companies long-term",
            "Green hydrogen scaling slower than expected",
        ],
        "outlook": "Energy sector is in early recovery. Power demand growth is structural (7-8%), "
                   "driven by data centers, EVs, and industrialization. Reliance pivoting to new energy. "
                   "NTPC leading conventional + solar capacity. ONGC is a value play on crude. "
                   "Tata Power and Adani Green positioned for renewable growth.",
    },
    "Financial Services": {
        "phase":       "Expansion",
        "phase_score": 70,
        "trend":       "improving",
        "color":       "#00cc44",
        "drivers": [
            "Insurance penetration rising — structural growth runway",
            "NBFC credit growth at 15-18% — underpenetrated markets",
            "Wealth management / mutual fund AUM growth 20%+",
            "Digital financial services adoption accelerating",
        ],
        "risks": [
            "RBI tightening regulations on NBFCs (risk weights, provisioning)",
            "Microfinance / small ticket lending asset quality stress",
            "Competition intensifying from banks entering NBFC territory",
            "Market-linked revenue (AMCs, brokers) volatile",
        ],
        "outlook": "Financial services is expanding. Bajaj Finance leads NBFC space. Insurance "
                   "(HDFC Life, SBI Life) offers structural growth. Cholamandalam, Shriram Finance "
                   "benefit from vehicle/MSME financing. Key risk is regulatory tightening — RBI has "
                   "been proactive in curbing unsecured lending excesses.",
    },
    "Consumer Durables": {
        "phase":       "Recovery",
        "phase_score": 58,
        "trend":       "improving",
        "color":       "#44ff88",
        "drivers": [
            "Summer demand driving AC, cooler sales",
            "Premiumization in appliances (smart, energy-efficient)",
            "Real estate recovery fueling demand for home appliances",
            "PLI scheme benefits for domestic manufacturing",
        ],
        "risks": [
            "Commodity cost pressure (copper, steel, plastics)",
            "Chinese imports (low-cost consumer electronics)",
            "Rural demand recovery still nascent",
            "Valuations elevated at 50-80x P/E for quality names",
        ],
        "outlook": "Consumer durables recovering from FY25 slowdown. Summer seasonality and "
                   "real estate recovery are positive catalysts. Titan (jewelry + watches) and Havells "
                   "(electrical) are sector leaders. Dixon Technologies benefits from electronics PLI. "
                   "Valuations remain premium — selective approach recommended.",
    },
    "Healthcare": {
        "phase":       "Expansion",
        "phase_score": 75,
        "trend":       "improving",
        "color":       "#00cc44",
        "drivers": [
            "Hospital occupancy rates at 68-72% — highest ever",
            "Medical tourism growing 15-20% annually",
            "Health insurance penetration driving footfall",
            "Diagnostics volume growth recovering post-COVID normalization",
        ],
        "risks": [
            "Regulatory risk — government price caps on procedures",
            "Capex-heavy model — new hospital break-even takes 3-5 years",
            "Doctor availability and talent retention challenges",
            "Valuation premium already factors in growth",
        ],
        "outlook": "Healthcare is in strong expansion. Hospital chains (Apollo, Max, Fortis) "
                   "are at record occupancy. Medical tourism and insurance penetration are structural "
                   "tailwinds. Diagnostics (Dr Lal PathLabs, Metropolis) recovering after post-COVID "
                   "normalization. Pure-play hospitals offer best growth visibility.",
    },
    "Real Estate": {
        "phase":       "Peak",
        "phase_score": 82,
        "trend":       "stable",
        "color":       "#ff3333",
        "drivers": [
            "Residential sales at decade-high — ₹4.5L Cr+ in FY26",
            "Premiumization — luxury segment growing 25-30%",
            "Low unsold inventory — supply-demand balanced",
            "Office space absorption strong (GCCs, IT, flex space)",
        ],
        "risks": [
            "Affordability stretched — price-to-income at concerning levels",
            "Interest rate sensitivity — rate hikes would hurt demand",
            "Regulatory changes (RERA compliance costs rising)",
            "Cyclical peak — historical pattern suggests moderation ahead",
        ],
        "outlook": "Real estate is near cyclical peak. Residential sales volumes are at decade "
                   "highs driven by premiumization. DLF, Godrej Properties, Prestige leading. "
                   "However, affordability is stretched and any rate reversal could dampen demand. "
                   "Institutional players (branded developers) will outperform unorganized market.",
    },
    "Capital Goods & Infra": {
        "phase":       "Expansion",
        "phase_score": 76,
        "trend":       "improving",
        "color":       "#00cc44",
        "drivers": [
            "Government capex push — ₹11.1L Cr allocation in FY27 budget",
            "Private capex revival — manufacturing PLI schemes",
            "Defence indigenization — order pipeline growing",
            "Data center and renewable energy infrastructure buildout",
        ],
        "risks": [
            "Government fiscal constraints could slow capex momentum",
            "Execution challenges — labor, land acquisition delays",
            "Working capital intensive — cash flow conversion lag",
            "Valuation premium — many stocks at 40-60x P/E",
        ],
        "outlook": "Capital goods and infrastructure is in robust expansion driven by government "
                   "capex and early signs of private capex revival. L&T is the bellwether — strong "
                   "order book provides multi-year visibility. ABB, Siemens benefit from automation trend. "
                   "Cement (UltraTech) benefits from infrastructure demand. Valuations are elevated — "
                   "growth is priced in.",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 3. INTEREST RATE SENSITIVITY — Per-sector impact assessment
# ══════════════════════════════════════════════════════════════════════════════

RATE_SENSITIVITY = {
    "Information Technology": {
        "sensitivity":  "LOW",
        "direction":    "neutral",
        "score":        50,
        "explanation":  "IT companies have minimal debt and strong cash flows. Rate changes "
                        "have limited direct impact. Indirect impact via US rate cycle affecting "
                        "client IT budgets. Rupee depreciation from rate differentials is mildly positive.",
    },
    "Banking": {
        "sensitivity":  "HIGH",
        "direction":    "positive",
        "score":        80,
        "explanation":  "Banks are the primary beneficiary of rate cuts — NIMs improve in the short term "
                        "as deposit repricing lags lending rate cuts. Treasury gains from bond price "
                        "appreciation. Rate hikes have the reverse negative effect. Current neutral "
                        "stance is broadly supportive.",
    },
    "FMCG": {
        "sensitivity":  "LOW",
        "direction":    "neutral",
        "score":        55,
        "explanation":  "FMCG companies have low leverage and stable cash flows. Rate changes "
                        "have minimal direct impact. Lower rates mildly boost consumer spending "
                        "power, supporting volume growth, but the effect is indirect and small.",
    },
    "Automobiles": {
        "sensitivity":  "MEDIUM",
        "direction":    "positive",
        "score":        65,
        "explanation":  "Auto loan rates directly affect vehicle demand. Rate cuts reduce EMIs, "
                        "making vehicles more affordable — especially impactful for 2-wheelers "
                        "and entry-level cars. Commercial vehicle financing costs also decrease. "
                        "Current rate cuts are supportive for the sector.",
    },
    "Pharmaceuticals": {
        "sensitivity":  "LOW",
        "direction":    "neutral",
        "score":        50,
        "explanation":  "Pharma companies have moderate leverage and defensive cash flows. "
                        "Rate changes have limited impact on business fundamentals. US rate "
                        "movements can affect dollar-denominated debt servicing costs.",
    },
    "Metals & Mining": {
        "sensitivity":  "MEDIUM",
        "direction":    "positive",
        "score":        60,
        "explanation":  "Metals companies are capital-intensive with significant debt. Rate cuts "
                        "reduce interest costs and improve profitability. Global rate environment "
                        "also affects commodity demand through economic activity channels.",
    },
    "Energy & Oil Gas": {
        "sensitivity":  "MEDIUM",
        "direction":    "positive",
        "score":        60,
        "explanation":  "Energy companies carry significant debt for capex-heavy projects. "
                        "Rate cuts reduce financing costs for new capacity. Power utilities "
                        "(NTPC, Power Grid) benefit from lower borrowing costs on regulated assets.",
    },
    "Financial Services": {
        "sensitivity":  "HIGH",
        "direction":    "positive",
        "score":        75,
        "explanation":  "NBFCs and insurance companies are highly sensitive to rates. Rate cuts "
                        "reduce borrowing costs faster than lending rate reductions, expanding spreads. "
                        "Insurance investment income benefits from capital gains on bond portfolios. "
                        "Housing finance companies (LIC HF, Can Fin) are direct beneficiaries.",
    },
    "Consumer Durables": {
        "sensitivity":  "MEDIUM",
        "direction":    "positive",
        "score":        65,
        "explanation":  "Consumer durables demand is partly financed through EMIs (ACs, refrigerators, "
                        "electronics). Rate cuts make financing more affordable. Real estate recovery "
                        "(rate-sensitive) also drives demand for home appliances and furnishings.",
    },
    "Healthcare": {
        "sensitivity":  "LOW",
        "direction":    "neutral",
        "score":        50,
        "explanation":  "Healthcare demand is largely inelastic to interest rates. Hospital chains "
                        "have moderate capex-driven debt, so rate cuts mildly reduce expansion costs. "
                        "Overall, rate environment has minimal impact on sector fundamentals.",
    },
    "Real Estate": {
        "sensitivity":  "HIGH",
        "direction":    "positive",
        "score":        85,
        "explanation":  "Real estate is the most rate-sensitive sector. Home loan rates directly "
                        "affect buyer affordability and demand. A 50 bps cut can increase home loan "
                        "eligibility by 5-7%. Developer borrowing costs also decrease. Current "
                        "rate cut cycle has been a major catalyst for residential sales.",
    },
    "Capital Goods & Infra": {
        "sensitivity":  "MEDIUM",
        "direction":    "positive",
        "score":        65,
        "explanation":  "Infrastructure companies carry project-level debt. Rate cuts reduce "
                        "project financing costs, improving IRR on new projects. Government "
                        "infra spending is less rate-sensitive, but private capex benefits from "
                        "lower borrowing costs. Cement companies benefit from construction demand.",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 4. REGULATORY RISK — Per-sector assessment
# ══════════════════════════════════════════════════════════════════════════════

REGULATORY_RISK = {
    "Information Technology": {
        "level": "LOW",
        "score": 20,
        "color": "#00cc44",
        "factors": [
            "Minimal domestic regulatory overhang",
            "US H-1B/immigration policy changes — periodic risk",
            "Data privacy regulations (DPDP Act) — manageable compliance cost",
        ],
        "recent": "DPDP Act implementation underway — compliance costs minor for large IT firms.",
    },
    "Banking": {
        "level": "MEDIUM",
        "score": 50,
        "color": "#ffaa00",
        "factors": [
            "RBI regulatory framework — frequent policy changes",
            "Capital adequacy requirements (Basel III+) tightening",
            "Unsecured lending risk weight increase (Nov 2023)",
            "Digital lending regulations — KYC/AML compliance",
        ],
        "recent": "RBI tightened risk weights on unsecured personal loans and NBFC lending. "
                  "New guidelines on project finance provisioning from April 2026.",
    },
    "FMCG": {
        "level": "LOW",
        "score": 25,
        "color": "#00cc44",
        "factors": [
            "GST rate changes — occasional category-level impact",
            "FSSAI regulations — labeling and quality compliance",
            "Plastic packaging regulations (EPR) — cost increase",
        ],
        "recent": "Extended Producer Responsibility (EPR) guidelines tightened — marginal cost impact.",
    },
    "Automobiles": {
        "level": "MEDIUM",
        "score": 45,
        "color": "#ffaa00",
        "factors": [
            "Emission norms (BS-VII planning) — significant R&D cost",
            "EV policy incentives and mandates — requires transition capex",
            "Safety regulations (airbags, crash test) — cost addition",
            "Scrappage policy — long-term positive for replacement demand",
        ],
        "recent": "BS-VII norms under discussion for 2028 implementation. FAME III EV subsidy "
                  "scheme extended with modified terms.",
    },
    "Pharmaceuticals": {
        "level": "HIGH",
        "score": 70,
        "color": "#ff3333",
        "factors": [
            "US FDA inspection and compliance — warning letters can halt production",
            "Drug Price Control Order (DPCO) — government price caps on essential medicines",
            "Clinical trial regulations — approval timelines uncertainty",
            "Patent challenges and litigation in US market",
        ],
        "recent": "DPCO expanded coverage to include more drugs. FDA increased inspection "
                  "frequency for Indian manufacturing facilities.",
    },
    "Metals & Mining": {
        "level": "MEDIUM",
        "score": 55,
        "color": "#ffaa00",
        "factors": [
            "Environmental clearance and mining lease regulations",
            "Export duty changes — government uses as price stabilization tool",
            "Anti-dumping duties on Chinese steel imports",
            "Carbon border adjustment mechanism (CBAM) — long-term risk",
        ],
        "recent": "Anti-dumping investigation on Chinese steel ongoing. "
                  "Draft CBAM framework released for comment.",
    },
    "Energy & Oil Gas": {
        "level": "HIGH",
        "score": 65,
        "color": "#ff3333",
        "factors": [
            "Government fuel pricing control — OMCs face margin pressure",
            "Environmental regulations on fossil fuel production",
            "Renewable energy mandates — RPO (Renewable Purchase Obligation)",
            "Gas pricing formula — administered vs market pricing debate",
        ],
        "recent": "Government retained fuel price controls ahead of state elections. "
                  "New RPO trajectory mandates 43% renewable by 2030.",
    },
    "Financial Services": {
        "level": "HIGH",
        "score": 65,
        "color": "#ff3333",
        "factors": [
            "RBI scale-based regulation for NBFCs — compliance cost",
            "Insurance regulatory changes (IRDAI reforms)",
            "Microfinance lending rate caps under discussion",
            "Digital lending and fintech regulation tightening",
        ],
        "recent": "RBI imposed restrictions on several NBFCs for governance lapses. "
                  "IRDAI proposing composite insurance license framework.",
    },
    "Consumer Durables": {
        "level": "LOW",
        "score": 25,
        "color": "#00cc44",
        "factors": [
            "BIS (Bureau of Indian Standards) — product quality certification",
            "Energy efficiency labeling requirements",
            "E-waste management regulations — compliance cost",
        ],
        "recent": "New energy efficiency norms for ACs and refrigerators — marginal cost impact.",
    },
    "Healthcare": {
        "level": "MEDIUM",
        "score": 50,
        "color": "#ffaa00",
        "factors": [
            "Government price caps on medical procedures (Ayushman Bharat rates)",
            "Clinical establishment licensing — state-level variation",
            "Medical device pricing regulations",
            "Doctor-to-bed ratio mandates",
        ],
        "recent": "Ayushman Bharat package rates revised upward by 10-15% — partial relief "
                  "for hospitals. New medical device regulations proposed.",
    },
    "Real Estate": {
        "level": "MEDIUM",
        "score": 50,
        "color": "#ffaa00",
        "factors": [
            "RERA compliance — ongoing regulatory burden for developers",
            "Land acquisition regulations — state-specific complexities",
            "Environmental and forest clearances — project delay risk",
            "Stamp duty changes — state governments use as fiscal tool",
        ],
        "recent": "Several states revised stamp duty rates. RERA enforcement strengthened "
                  "with more consumer complaints being resolved.",
    },
    "Capital Goods & Infra": {
        "level": "LOW",
        "score": 30,
        "color": "#00cc44",
        "factors": [
            "Government policy supportive — PLI schemes, infra focus",
            "Defence procurement regulations — offset requirements",
            "Environmental clearances for large projects",
            "Labour regulations — code on wages implementation",
        ],
        "recent": "Defence procurement policy updated with higher indigenization requirements. "
                  "New PLI scheme for capital goods equipment.",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 5. MANAGEMENT / PROMOTER DATA — for key companies
# ══════════════════════════════════════════════════════════════════════════════

PROMOTER_DATA = {
    "TCS.NS":        {"promoter_pct": 72.3, "trend": "stable",     "quality": "excellent", "score": 92},
    "INFY.NS":       {"promoter_pct": 14.8, "trend": "decreasing", "quality": "excellent", "score": 85},
    "WIPRO.NS":      {"promoter_pct": 72.9, "trend": "stable",     "quality": "good",      "score": 78},
    "HCLTECH.NS":    {"promoter_pct": 60.8, "trend": "stable",     "quality": "excellent", "score": 88},
    "TECHM.NS":      {"promoter_pct": 35.2, "trend": "stable",     "quality": "good",      "score": 72},
    "HDFCBANK.NS":   {"promoter_pct": 0.0,  "trend": "n/a",        "quality": "excellent", "score": 90},
    "ICICIBANK.NS":  {"promoter_pct": 0.0,  "trend": "n/a",        "quality": "excellent", "score": 88},
    "KOTAKBANK.NS":  {"promoter_pct": 25.9, "trend": "decreasing", "quality": "excellent", "score": 85},
    "AXISBANK.NS":   {"promoter_pct": 8.2,  "trend": "stable",     "quality": "good",      "score": 75},
    "SBIN.NS":       {"promoter_pct": 57.5, "trend": "stable",     "quality": "good",      "score": 72},
    "HINDUNILVR.NS": {"promoter_pct": 61.9, "trend": "stable",     "quality": "excellent", "score": 92},
    "ITC.NS":        {"promoter_pct": 0.0,  "trend": "n/a",        "quality": "good",      "score": 78},
    "NESTLEIND.NS":  {"promoter_pct": 62.8, "trend": "stable",     "quality": "excellent", "score": 90},
    "MARUTI.NS":     {"promoter_pct": 56.4, "trend": "stable",     "quality": "excellent", "score": 88},
    "TATAMOTORS.NS": {"promoter_pct": 46.4, "trend": "stable",     "quality": "good",      "score": 75},
    "M&M.NS":        {"promoter_pct": 19.4, "trend": "stable",     "quality": "excellent", "score": 85},
    "BAJAJ-AUTO.NS": {"promoter_pct": 54.6, "trend": "stable",     "quality": "excellent", "score": 88},
    "SUNPHARMA.NS":  {"promoter_pct": 54.5, "trend": "stable",     "quality": "good",      "score": 75},
    "DRREDDY.NS":    {"promoter_pct": 26.7, "trend": "stable",     "quality": "good",      "score": 78},
    "CIPLA.NS":      {"promoter_pct": 33.5, "trend": "stable",     "quality": "good",      "score": 78},
    "TATASTEEL.NS":  {"promoter_pct": 33.2, "trend": "stable",     "quality": "good",      "score": 72},
    "JSWSTEEL.NS":   {"promoter_pct": 44.8, "trend": "stable",     "quality": "good",      "score": 75},
    "HINDALCO.NS":   {"promoter_pct": 34.7, "trend": "stable",     "quality": "good",      "score": 75},
    "RELIANCE.NS":   {"promoter_pct": 50.3, "trend": "stable",     "quality": "excellent", "score": 90},
    "ONGC.NS":       {"promoter_pct": 58.9, "trend": "stable",     "quality": "average",   "score": 60},
    "NTPC.NS":       {"promoter_pct": 51.1, "trend": "stable",     "quality": "good",      "score": 72},
    "BAJFINANCE.NS": {"promoter_pct": 54.7, "trend": "stable",     "quality": "excellent", "score": 92},
    "BAJAJFINSV.NS": {"promoter_pct": 55.5, "trend": "stable",     "quality": "excellent", "score": 90},
    "TITAN.NS":      {"promoter_pct": 52.9, "trend": "stable",     "quality": "excellent", "score": 90},
    "APOLLOHOSP.NS": {"promoter_pct": 29.3, "trend": "stable",     "quality": "excellent", "score": 85},
    "DLF.NS":        {"promoter_pct": 74.1, "trend": "stable",     "quality": "good",      "score": 72},
    "LT.NS":         {"promoter_pct": 0.0,  "trend": "n/a",        "quality": "excellent", "score": 88},
    "INDUSINDBK.NS": {"promoter_pct": 16.5, "trend": "decreasing", "quality": "average",   "score": 55},
    "PNB.NS":        {"promoter_pct": 73.2, "trend": "stable",     "quality": "average",   "score": 58},
    "CANBK.NS":      {"promoter_pct": 62.9, "trend": "stable",     "quality": "average",   "score": 60},
    "BRITANNIA.NS":  {"promoter_pct": 50.6, "trend": "stable",     "quality": "excellent", "score": 88},
    "LTIM.NS":       {"promoter_pct": 68.6, "trend": "stable",     "quality": "good",      "score": 78},
    "PERSISTENT.NS": {"promoter_pct": 31.2, "trend": "stable",     "quality": "excellent", "score": 85},
    "COALINDIA.NS":  {"promoter_pct": 63.1, "trend": "stable",     "quality": "average",   "score": 60},
}


# ══════════════════════════════════════════════════════════════════════════════
# 6. EARNINGS QUALITY SCORING — computed from financial metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_earnings_quality(row: pd.Series) -> dict:
    """
    Compute earnings quality score from existing financial metrics.
    Returns dict with score (0-100), grade, and quality flags.
    """
    score = 50  # Start neutral
    flags = []

    # ROE assessment (weight: 30%)
    roe = row.get("roe")
    if roe and not (isinstance(roe, float) and np.isnan(roe)):
        rv = float(roe)
        if rv > 0.25:
            score += 15; flags.append(("HIGH ROE (>25%)", "#00cc44"))
        elif rv > 0.15:
            score += 8;  flags.append(("GOOD ROE (15-25%)", "#44ff88"))
        elif rv > 0.08:
            score += 0;  flags.append(("AVERAGE ROE (8-15%)", "#888"))
        else:
            score -= 10; flags.append(("LOW ROE (<8%)", "#ff3333"))

    # Operating margin (weight: 25%)
    opm = row.get("operating_margin")
    if opm and not (isinstance(opm, float) and np.isnan(opm)):
        om = float(opm)
        if om > 0.25:
            score += 12; flags.append(("STRONG MARGINS (>25%)", "#00cc44"))
        elif om > 0.15:
            score += 6;  flags.append(("HEALTHY MARGINS (15-25%)", "#44ff88"))
        elif om > 0.08:
            score += 0;  flags.append(("MODERATE MARGINS (8-15%)", "#888"))
        else:
            score -= 8;  flags.append(("THIN MARGINS (<8%)", "#ff3333"))

    # Net margin (weight: 15%)
    npm = row.get("net_margin")
    if npm and not (isinstance(npm, float) and np.isnan(npm)):
        nm = float(npm)
        if nm > 0.18:
            score += 8; flags.append(("HIGH NET MARGIN (>18%)", "#00cc44"))
        elif nm > 0.10:
            score += 4; flags.append(("GOOD NET MARGIN (10-18%)", "#44ff88"))
        elif nm > 0.05:
            score += 0; flags.append(("MODEST NET MARGIN (5-10%)", "#888"))
        else:
            score -= 5; flags.append(("LOW NET MARGIN (<5%)", "#ff6644"))

    # P/E vs Forward P/E — earnings growth signal (weight: 15%)
    pe = row.get("pe")
    fwd_pe = row.get("fwd_pe")
    if pe and fwd_pe and not np.isnan(float(pe)) and not np.isnan(float(fwd_pe)):
        pe_f, fwd_f = float(pe), float(fwd_pe)
        if fwd_f < pe_f * 0.85:
            score += 8; flags.append(("EARNINGS GROWTH EXPECTED (FWD P/E << TRAILING)", "#00cc44"))
        elif fwd_f < pe_f * 0.95:
            score += 3; flags.append(("MILD EARNINGS GROWTH (FWD P/E < TRAILING)", "#44ff88"))
        elif fwd_f > pe_f * 1.10:
            score -= 5; flags.append(("EARNINGS DECLINE EXPECTED (FWD P/E > TRAILING)", "#ff3333"))

    # Dividend yield — cash return signal (weight: 10%)
    dy = row.get("div_yield")
    if dy and not (isinstance(dy, float) and np.isnan(dy)):
        dv = float(dy)
        if dv > 3.0:
            score += 5; flags.append(("HIGH DIVIDEND (>3%)", "#00cc44"))
        elif dv > 1.5:
            score += 2; flags.append(("MODERATE DIVIDEND (1.5-3%)", "#44ff88"))
        elif dv < 0.3:
            score -= 2; flags.append(("MINIMAL/NO DIVIDEND", "#888"))

    # Beta — earnings volatility proxy (weight: 5%)
    beta = row.get("beta")
    if beta and not (isinstance(beta, float) and np.isnan(beta)):
        bv = float(beta)
        if bv < 0.6:
            score += 3; flags.append(("LOW VOLATILITY (BETA<0.6)", "#00cc44"))
        elif bv > 1.3:
            score -= 3; flags.append(("HIGH VOLATILITY (BETA>1.3)", "#ff6644"))

    # Clamp score
    score = max(0, min(100, score))

    # Grade
    if score >= 85:   grade = "A+"
    elif score >= 75:  grade = "A"
    elif score >= 65:  grade = "B+"
    elif score >= 55:  grade = "B"
    elif score >= 45:  grade = "C+"
    elif score >= 35:  grade = "C"
    elif score >= 25:  grade = "D"
    else:              grade = "F"

    # Color
    if score >= 70:    color = "#00cc44"
    elif score >= 50:  color = "#ffaa00"
    else:              color = "#ff3333"

    return {
        "score": score,
        "grade": grade,
        "color": color,
        "flags": flags,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. NEWS FETCHING — Google News RSS
# ══════════════════════════════════════════════════════════════════════════════

_NEWS_BASE = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

# Sector → search query mapping
SECTOR_NEWS_QUERIES = {
    "Information Technology": "India+IT+sector+Infosys+TCS+technology",
    "Banking":                "India+banking+sector+HDFC+ICICI+SBI+RBI",
    "FMCG":                   "India+FMCG+sector+HUL+ITC+consumer+goods",
    "Automobiles":            "India+automobile+sector+Maruti+Tata+Motors+auto",
    "Pharmaceuticals":        "India+pharma+sector+Sun+Pharma+Cipla+drug",
    "Metals & Mining":        "India+metals+mining+steel+Tata+Steel+JSW",
    "Energy & Oil Gas":       "India+energy+oil+gas+Reliance+ONGC+NTPC+power",
    "Financial Services":     "India+financial+services+NBFC+Bajaj+Finance+insurance",
    "Consumer Durables":      "India+consumer+durables+Titan+Havells+appliances",
    "Healthcare":             "India+healthcare+hospital+Apollo+Fortis+Max",
    "Real Estate":            "India+real+estate+property+DLF+Godrej+housing",
    "Capital Goods & Infra":  "India+infrastructure+capital+goods+L%26T+construction",
}


def _parse_rss_feed(xml_text: str, max_items: int = 5) -> list:
    """Parse Google News RSS XML into a list of news items."""
    items = []
    try:
        root = ET.fromstring(xml_text)
        channel = root.find("channel")
        if channel is None:
            return items

        for item in channel.findall("item")[:max_items]:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")
            source_el = item.find("source")
            source = source_el.text if source_el is not None else ""

            # Parse time
            time_ago = ""
            try:
                # Google News uses RFC 2822 format
                from email.utils import parsedate_to_datetime
                pub_dt = parsedate_to_datetime(pub_date)
                now = datetime.now(pub_dt.tzinfo) if pub_dt.tzinfo else datetime.now()
                delta = now - pub_dt
                if delta.days > 0:
                    time_ago = f"{delta.days}d ago"
                elif delta.seconds > 3600:
                    time_ago = f"{delta.seconds // 3600}h ago"
                else:
                    time_ago = f"{max(1, delta.seconds // 60)}m ago"
            except Exception:
                time_ago = pub_date[:16] if pub_date else ""

            items.append({
                "title": title,
                "link": link,
                "source": source,
                "time_ago": time_ago,
                "pub_date": pub_date,
            })
    except ET.ParseError as e:
        log.warning(f"RSS parse error: {e}")
    return items


def fetch_sector_news(sector: str, max_items: int = 5) -> list:
    """Fetch latest news for a sector from Google News RSS."""
    query = SECTOR_NEWS_QUERIES.get(sector, sector.replace(" ", "+"))
    url = _NEWS_BASE.format(query=query)
    try:
        resp = requests.get(url, timeout=8, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        if resp.status_code == 200:
            return _parse_rss_feed(resp.text, max_items)
    except Exception as e:
        log.warning(f"Sector news fetch failed for {sector}: {e}")
    return []


def fetch_company_news(company_name: str, ticker: str = "", max_items: int = 5) -> list:
    """Fetch latest news for a company from Google News RSS."""
    # Build query: company name + NSE context
    clean_name = company_name.replace("&", "%26").replace(" ", "+")
    nse_sym = ticker.replace(".NS", "")
    query = f"{clean_name}+{nse_sym}+NSE+stock"
    url = _NEWS_BASE.format(query=query)
    try:
        resp = requests.get(url, timeout=8, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        if resp.status_code == 200:
            return _parse_rss_feed(resp.text, max_items)
    except Exception as e:
        log.warning(f"Company news fetch failed for {company_name}: {e}")
    return []


def get_sector_macro_impact(sector: str) -> dict:
    """Get how current macro environment impacts a specific sector."""
    cycle = SECTOR_CYCLE.get(sector, {})
    rate = RATE_SENSITIVITY.get(sector, {})
    reg = REGULATORY_RISK.get(sector, {})

    # Compute overall macro impact score (0-100, higher = more favorable)
    impact_score = 50  # neutral
    # Rate environment impact
    if MACRO_DATA["repo_trend"] == "cutting":
        impact_score += rate.get("score", 50) * 0.15 - 7.5
    elif MACRO_DATA["repo_trend"] == "hiking":
        impact_score -= rate.get("score", 50) * 0.15 - 7.5

    # GDP growth impact
    if MACRO_DATA["gdp_growth"] > 7.0:
        impact_score += 5
    elif MACRO_DATA["gdp_growth"] < 6.0:
        impact_score -= 5

    # FII flow impact
    if MACRO_DATA["fii_flow_ytd_cr"] > 0:
        impact_score += 3
    elif MACRO_DATA["fii_flow_ytd_cr"] < -10000:
        impact_score -= 3

    # Regulatory drag
    reg_score = reg.get("score", 50)
    if reg_score > 60:
        impact_score -= 5
    elif reg_score < 30:
        impact_score += 3

    impact_score = max(0, min(100, impact_score))

    if impact_score >= 65:
        label = "SUPPORTIVE"
        color = "#00cc44"
    elif impact_score >= 45:
        label = "NEUTRAL"
        color = "#ffaa00"
    else:
        label = "HEADWIND"
        color = "#ff3333"

    return {
        "score": round(impact_score),
        "label": label,
        "color": color,
    }
