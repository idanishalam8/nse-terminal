# intelligence.py — Trading Intelligence Engine for NSE Valuation Terminal
import logging
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from src.global_data import get_global_market_state

log = logging.getLogger(__name__)

# List of all sectors in the terminal
SECTORS_LIST = [
    "Information Technology", "Banking", "FMCG", "Automobiles", "Pharmaceuticals",
    "Metals & Mining", "Energy & Oil Gas", "Financial Services", "Consumer Durables",
    "Healthcare", "Real Estate", "Capital Goods & Infra"
]

# Standard stock details for winners/losers mapping
STOCK_INFO = {
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC Ltd",
    "MARUTI.NS": "Maruti Suzuki",
    "TATAMOTORS.NS": "Tata Motors",
    "SUNPHARMA.NS": "Sun Pharma",
    "TATASTEEL.NS": "Tata Steel",
    "HINDALCO.NS": "Hindalco Industries",
    "RELIANCE.NS": "Reliance Industries",
    "ONGC.NS": "ONGC",
    "BAJFINANCE.NS": "Bajaj Finance",
    "MUTHOOTFIN.NS": "Muthoot Finance",
    "TITAN.NS": "Titan Company",
    "DLF.NS": "DLF Ltd",
    "LT.NS": "Larsen & Toubro",
    "VOLTAS.NS": "Voltas"
}

# Cross-asset structural correlation matrix data
CROSS_ASSET_CORRELATION = {
    "Factors": [
        "US Bond Yields (▲)", 
        "Crude Oil Prices (▲)", 
        "USD-INR Rate (Rupee ▼)", 
        "RBI Interest Rates (▲)",
        "Geopolitical Tension (▲)"
    ],
    "Commodities": [
        "Gold ▼ (yield drag), Crude ▲ (global macro)",
        "Spikes directly; spreads to gold/silver",
        "Gold ▲ (rupee hedge), Industrial metals ▼",
        "Neutral to negative (curbs liquidity)",
        "Gold ▲ (safe-haven), Crude ▲ (supply risk)"
    ],
    "Bonds": [
        "Spikes US yields; India yields ▲ (FII outflow)",
        "Spikes India yields ▲ (imported inflation)",
        "India yields ▲ (currency defence risk)",
        "India yields curve shifts up ▲",
        "US yields ▼ (safe flight), India yields ▲"
    ],
    "Equity Market": [
        "Negative (capital flight, valuation multiple compression)",
        "Negative (inflation surge, OMCs & paints compressed)",
        "Neutral (IT/Pharma gain, imports/capex compressed)",
        "Negative (rate-sensitives cool down, credit decelerates)",
        "Negative (risk-off, global selloffs)"
    ]
}

# Pre-defined Simulation Templates
EVENT_TEMPLATES = {
    "us_fed_cut": {
        "title": "US Fed Cuts Rates by 50 bps",
        "category": "Macroeconomics / Bond Market",
        "overview": "US Federal Reserve aggressively cuts interest rates. This reduces the global cost of capital, weakens the US Dollar Index (DXY), and triggers significant capital inflows (FIIs) into emerging markets like India. Bond yields decline globally.",
        "transmission": "Lower US rates → Yield-seeking capital moves to emerging markets → FII inflows boost Indian equities → Rupee appreciates → Domestic cost of capital drops as RBI finds room to cut rates.",
        "assets": {
            "Nifty 50": ("Strong Positive", "▲ +1.5% to +2.5%", "valuation multiples expand as FII flows return"),
            "Bank Nifty": ("Positive", "▲ +1.2% to +2.0%", "treasury gains on G-Sec portfolios and lower cost of funds"),
            "IT Nifty": ("Positive", "▲ +1.8% to +3.0%", "US clients expand tech budgets as cost of debt falls"),
            "Gold": ("Strong Positive", "▲ +2.0% to +3.5%", "inverse correlation with real rates triggers heavy buying"),
            "Crude Oil": ("Positive", "▲ +1.0% to +2.0%", "dollar depreciation makes commodities cheaper, boosting demand"),
            "USD-INR": ("Negative (Rupee Appreciates)", "▼ -0.8% to -1.5%", "heavy FII dollar inflows strengthen the Rupee")
        },
        "sectors": {
            "Information Technology": ("Positive", "Medium", "US discretionary IT spend recovers; valuation discount rates drop."),
            "Banking": ("Positive", "Medium", "Gains from G-sec portfolio mark-to-market; NIMs stable in short term."),
            "FMCG": ("Neutral", "Low", "Indirect consumer sentiment boost, but defensive profile leads to relative underperformance."),
            "Automobiles": ("Positive", "Medium", "Lower retail auto financing costs stimulate consumer sales volumes."),
            "Pharmaceuticals": ("Neutral", "Low", "Defensive play, minor margin headwind from Rupee appreciation."),
            "Metals & Mining": ("Positive", "Medium", "Dollar weakness supports global LME metal prices; boost to export realization."),
            "Energy & Oil Gas": ("Positive", "Low", "Lower borrowing costs benefit heavy capital projects in green energy."),
            "Financial Services": ("Positive", "High", "NBFCs benefit immediately as wholesale borrowing costs fall faster than lending rates."),
            "Consumer Durables": ("Positive", "Medium", "EMI affordability boosts sales of premium appliances and jewelry."),
            "Healthcare": ("Neutral", "Low", "Capex borrowing for hospitals becomes cheaper, minor sentiment boost."),
            "Real Estate": ("Positive", "High", "Mortgage rates decline, directly boosting residential sales and developer liquidity."),
            "Capital Goods & Infra": ("Positive", "Medium", "Reduces interest burden on high-leverage infrastructure projects.")
        },
        "winners": [
            ("DLF.NS", "Real estate leader, direct beneficiary of home loan rate cuts boosting bookings."),
            ("INFY.NS", "IT major with heavy US exposure; US client budget constraints ease."),
            ("MUTHOOTFIN.NS", "Wholesale financing costs fall; spreads widen on gold loan books.")
        ],
        "losers": [
            ("HINDUNILVR.NS", "FMCG defensive; underperforms as capital rotates into high-beta growth sectors.")
        ]
    },
    "us_cpi_spike": {
        "title": "US CPI Spikes Unexpectedly (Higher Inflation)",
        "category": "Macroeconomics / Inflation",
        "overview": "US inflation prints significantly above expectations. This forces the Federal Reserve to maintain a hawkish stance ('higher for longer') or hike rates. Spikes US treasury yields, strengthens the dollar, and triggers global risk-off.",
        "transmission": "Higher US inflation → US yields spike → FIIs pull funds from emerging markets to safe US treasuries → Rupee depreciates → RBI forced to keep domestic rates elevated, compressing market multiples.",
        "assets": {
            "Nifty 50": ("Negative", "▼ -1.5% to -2.5%", "capital flight and multiple contraction"),
            "Bank Nifty": ("Negative", "▼ -1.2% to -2.0%", "high cost of funds and treasury losses on yields spike"),
            "IT Nifty": ("Strong Negative", "▼ -2.5% to -4.0%", "spikes in US discount rates hit growth valuations"),
            "Gold": ("Negative / Neutral", "▼ -0.5% to +0.5%", "spiked yields drag gold, but inflation hedge limits downside"),
            "Crude Oil": ("Negative", "▼ -1.0% to -2.0%", "stronger dollar weighs on crude demand"),
            "USD-INR": ("Strong Positive (Rupee Depreciates)", "▲ +0.8% to +1.5%", "capital flight depreciates the Rupee")
        },
        "sectors": {
            "Information Technology": ("Negative", "High", "Growth stock valuations hit by high discount rates; clients cut IT spend."),
            "Banking": ("Negative", "Medium", "Spike in G-Sec yields causes treasury mark-to-market losses."),
            "FMCG": ("Neutral", "Low", "Defensive volumes hold up, but dollar strength raises packaging/input costs."),
            "Automobiles": ("Negative", "Medium", "Domestic interest rates stay high, dragging vehicle financing affordability."),
            "Pharmaceuticals": ("Positive", "Medium", "Defensive export earner, benefits from Rupee depreciation translating revenue."),
            "Metals & Mining": ("Negative", "Medium", "LME metal prices weaken under dollar strength and high interest rates."),
            "Energy & Oil Gas": ("Negative", "Medium", "Crude import bill rises on Rupee depreciation, hitting downstream OMCs."),
            "Financial Services": ("Negative", "High", "NBFCs face higher cost of funds, squeezing net interest margins."),
            "Consumer Durables": ("Negative", "Medium", "Input costs rise; discretionary demand cools under persistent inflation."),
            "Healthcare": ("Neutral", "Low", "Inelastic healthcare demand acts as a defensive buffer against inflation."),
            "Real Estate": ("Negative", "High", "Mortgage rates remain high, cooling home buyer transactions."),
            "Capital Goods & Infra": ("Negative", "Medium", "High borrowing cost drags down private capex and infrastructure execution.")
        },
        "winners": [
            ("SUNPHARMA.NS", "Defensive stock; export revenues benefit from Rupee depreciation."),
            ("DRREDDY.NS", "Large US generic sales footprint translates to higher rupee margins.")
        ],
        "losers": [
            ("BAJFINANCE.NS", "NBFC giant, hit by margin squeeze as cost of capital rises."),
            ("COFORGE.NS", "High-beta midcap IT; heavily impacted by global growth-multiple selloffs."),
            ("LT.NS", "Capex giant; highly sensitive to rising interest rates and project financing costs.")
        ]
    },
    "mid_east_war": {
        "title": "Geopolitical Conflict Escalation in Middle East",
        "category": "Geopolitics / Commodity Market",
        "overview": "Escalation in geopolitical tensions in the Middle East threatens crude oil shipping routes (Strait of Hormuz/Red Sea). Crude oil prices spike instantly, causing global risk-off, safe-haven gold buying, and rupee weakness.",
        "transmission": "Supply threat → Crude oil spikes → India's trade deficit widens → Rupee depreciates → Fuel prices drive imported inflation → RBI forced into hawkish stance → Equities face systemic selloff.",
        "assets": {
            "Nifty 50": ("Negative", "▼ -2.0% to -3.5%", "imported inflation and margin squeeze across manufacturing"),
            "Bank Nifty": ("Negative", "▼ -1.5% to -2.5%", "higher systemic inflation delays rate cuts, increasing NPA risks"),
            "IT Nifty": ("Neutral / Positive", "▲ +0.5% to +1.5%", "export buffer and Rupee depreciation benefit revenues"),
            "Gold": ("Strong Positive", "▲ +3.0% to +5.0%", "massive safe-haven buying during global war risk"),
            "Crude Oil": ("Strong Positive", "▲ +8.0% to +15.0%", "direct supply risk premium priced in"),
            "USD-INR": ("Positive (Rupee Depreciates)", "▲ +1.0% to +2.0%", "oil import dollar demand weakens the Rupee")
        },
        "sectors": {
            "Information Technology": ("Positive", "Low", "Export revenue translation offsets systemic domestic headwinds."),
            "Banking": ("Negative", "Medium", "Inflation dampens credit growth; G-Sec yields rise, causing losses."),
            "FMCG": ("Negative", "Medium", "Input costs (palm oil, crude derivatives) rise; rural demand squeezed."),
            "Automobiles": ("Negative", "High", "High fuel prices increase vehicle running cost, hurting consumer auto sales."),
            "Pharmaceuticals": ("Positive", "Low", "Defensive sector; dollar-priced international sales expand in rupee terms."),
            "Metals & Mining": ("Neutral", "Medium", "Spiking energy costs squeeze margins, but commodity pricing provides floor."),
            "Energy & Oil Gas": ("Mixed (Positive E&P / Negative OMCs)", "High", "Upstream drillers benefit; refiners/marketers face massive margin squeeze."),
            "Financial Services": ("Negative", "Medium", "Inflation cools retail consumption and credit card volumes."),
            "Consumer Durables": ("Negative", "Medium", "Input costs rise; consumer sentiment dampens, delaying purchases."),
            "Healthcare": ("Neutral", "Low", "Safe defensive haven with stable inelastic domestic occupancy."),
            "Real Estate": ("Negative", "High", "Rising steel/cement costs squeeze developer margins; sentiment turns cautious."),
            "Capital Goods & Infra": ("Negative", "High", "Fuel and bitumen price hikes hit road contractors; project costs escalate.")
        },
        "winners": [
            ("ONGC.NS", "Upstream crude producer; directly benefits from higher crude prices."),
            ("MUTHOOTFIN.NS", "Gold loan provider; gold price surge expands collateral values."),
            ("SUNPHARMA.NS", "Export pharma defensive; benefits from Rupee depreciation and risk-off.")
        ],
        "losers": [
            ("MARUTI.NS", "Auto manufacturer hit by high fuel prices and steel cost inflation."),
            ("BPCL.NS", "Oil marketing company squeezed as retail fuel price hikes are capped by govt."),
            ("VOLTAS.NS", "AC maker; margins squeezed by rising input commodity costs and cooling consumer spending.")
        ]
    },
    "rupee_slide": {
        "title": "Rupee Slides to 96 against USD",
        "category": "Currency / Geo-economics",
        "overview": "The Indian Rupee experiences a sharp depreciation, breaching 96 per US Dollar due to global dollar strength, trade deficits, and FII outflows. Export sectors gain, while importers suffer.",
        "transmission": "Weak Rupee → IT and Pharma companies report margin expansion → Oil import costs surge in rupee terms → Imported inflation rises → Foreign investors see lower dollar returns, slowing capital inflows.",
        "assets": {
            "Nifty 50": ("Neutral", "▼ -0.5% to +0.5%", "IT/Pharma gains offset by banking/oil import pain"),
            "Bank Nifty": ("Negative", "▼ -1.0% to -1.8%", "foreign investor selling and imported inflation pressure"),
            "IT Nifty": ("Strong Positive", "▲ +2.0% to +3.5%", "every 1% Rupee fall adds ~30-40 bps to IT EBIT margins"),
            "Gold": ("Positive", "▲ +1.5% to +2.5%", "domestic gold price spikes as it is imported in USD terms"),
            "Crude Oil": ("Neutral (in USD) / Spike in INR", "▲ +1.0% to +2.0% (in INR)", "landed cost of crude increases for refiners"),
            "USD-INR": ("Strong Positive", "▲ +1.5% to +2.5%", "direct currency pair movement")
        },
        "sectors": {
            "Information Technology": ("Positive", "High", "Massive beneficiary. High dollar revenues translate to expanded margins."),
            "Banking": ("Negative", "Medium", "FII outflow pressure; imported inflation reduces scope for domestic rate cuts."),
            "FMCG": ("Negative", "Low", "Imported raw materials (palm oil, chemicals) become more expensive."),
            "Automobiles": ("Negative", "Medium", "Imported components and raw material costs increase."),
            "Pharmaceuticals": ("Positive", "High", "Strong beneficiary. US/export sales margins expand in rupee terms."),
            "Metals & Mining": ("Positive", "Medium", "Domestic metal prices (benchmarked to global LME import prices) increase."),
            "Energy & Oil Gas": ("Negative", "Medium", "Downstream refiners face higher landed crude costs; pricing pressure."),
            "Financial Services": ("Negative", "Low", "FII selling pressure; NBFCs with dollar debt face higher servicing costs."),
            "Consumer Durables": ("Negative", "Medium", "Imported electronic components (compressors, panels) rise in cost."),
            "Healthcare": ("Neutral", "Low", "Marginal benefit for hospital medical tourism earnings."),
            "Real Estate": ("Neutral", "Low", "No direct impact, but imported cement/steel components rise in cost."),
            "Capital Goods & Infra": ("Negative", "Medium", "Contractors with imported machinery/fuel dependency face margin hits.")
        },
        "winners": [
            ("TCS.NS", "IT giant; highly efficient dollar revenue translation directly boosting EBIT."),
            ("SUNPHARMA.NS", "Pharma exporter; large US specialty portfolio margins expand."),
            ("HINDALCO.NS", "Metal major; domestic aluminum/copper realizations rise with landed import costs.")
        ],
        "losers": [
            ("IOC.NS", "Oil marketer; landed crude costs rise, compressing refining margins."),
            ("MARUTI.NS", "Auto major; import component costs rise, squeezing vehicle margins.")
        ]
    },
    "rbi_hike": {
        "title": "RBI Unexpectedly Hikes Repo Rate by 25 bps",
        "category": "Domestic Policy / Rate Cycle",
        "overview": "Reserve Bank of India unexpectedly hikes the policy repo rate by 25 basis points to curb domestic inflation and support the rupee. Domestic lending rates jump, cooling rate-sensitive demand.",
        "transmission": "RBI hikes repo rate → Banks hike MCLR and deposit rates → Home and auto loan EMIs rise → Consumer discretionary demand slows → Bond yields rise, causing treasury losses for banks.",
        "assets": {
            "Nifty 50": ("Negative", "▼ -1.0% to -1.8%", "discretionary consumption cools down; interest expense rises"),
            "Bank Nifty": ("Negative", "▼ -1.5% to -2.5%", "treasury losses on bond books and retail loan deceleration"),
            "IT Nifty": ("Neutral", "▼ -0.2% to +0.2%", "export focus makes IT immune to domestic rate hikes"),
            "Gold": ("Negative", "▼ -0.5% to -1.2%", "higher domestic rates reduce gold's investment appeal"),
            "Crude Oil": ("Neutral", "■ 0.0%", "unaffected by Indian domestic rate policy"),
            "USD-INR": ("Negative (Rupee Appreciates)", "▼ -0.4% to -0.8%", "higher yield differential attracts carry trade, supporting INR")
        },
        "sectors": {
            "Information Technology": ("Neutral", "Low", "Unaffected. Demand is globally driven; minimal rupee debt."),
            "Banking": ("Negative", "High", "Treasury mark-to-market losses on G-sec portfolios; loan growth cools."),
            "FMCG": ("Neutral", "Low", "Defensive; rural distributor credit becomes slightly more expensive."),
            "Automobiles": ("Negative", "High", "Auto financing costs rise, directly cooling purchase volumes."),
            "Pharmaceuticals": ("Neutral", "Low", "Defensive; export-led earnings are unaffected by domestic rates."),
            "Metals & Mining": ("Negative", "Low", "High-leverage metal firms face rising interest servicing costs."),
            "Energy & Oil Gas": ("Negative", "Low", "Increases cost of debt servicing on large capital expenditure projects."),
            "Financial Services": ("Negative", "High", "NBFC margins squeezed as short-term borrowing rates spike immediately."),
            "Consumer Durables": ("Negative", "Medium", "EMI financing schemes become more expensive, cooling appliance sales."),
            "Healthcare": ("Neutral", "Low", "Inelastic hospital occupancy remains stable; expansion debt cost rises."),
            "Real Estate": ("Negative", "High", "Most sensitive. Home loan EMIs rise, immediately dampening housing sales."),
            "Capital Goods & Infra": ("Negative", "Medium", "Leveraged developers face project IRR compression on higher interest rates.")
        },
        "winners": [
            ("INFY.NS", "IT export defensive; cash-rich with no debt, yield on cash reserves rises.")
        ],
        "losers": [
            ("DLF.NS", "Real estate giant; home loan hikes drag residential bookings momentum."),
            ("CHOLAFIN.NS", "NBFC hit by immediate spike in commercial paper and wholesale deposit costs."),
            ("TATAMOTORS.NS", "Auto major; retail demand cools under rising auto finance rates.")
        ]
    },
    "metal_shock": {
        "title": "Global Commodity Shock: Metals Spike",
        "category": "Commodities / Macroeconomics",
        "overview": "Supply bottlenecks and strong infrastructure stimulus from China trigger a major spike in global base metal prices (Steel, Aluminum, Copper). Benefits metal producers, but compresses manufacturing margins.",
        "transmission": "Global demand/supply shock → LME metal prices surge → Indian domestic metal prices rise → Metal manufacturers report massive margin expansions → Automobiles, capital goods, and durables face margin compression due to raw material costs.",
        "assets": {
            "Nifty 50": ("Neutral", "▼ -0.3% to +0.5%", "Metal stock gains offset by manufacturing/auto margin squeeze"),
            "Bank Nifty": ("Neutral", "▼ -0.2% to +0.2%", "Metal corporate loan books strengthen; auto loan growth slows"),
            "IT Nifty": ("Neutral", "■ 0.0%", "Unaffected by raw industrial commodity costs"),
            "Gold": ("Positive", "▲ +1.0% to +1.8%", "General commodity asset class inflation lift"),
            "Crude Oil": ("Positive", "▲ +1.2% to +2.2%", "Global economic stimulus expectations lift crude demand"),
            "USD-INR": ("Neutral / Weak positive", "▲ +0.2% to +0.5%", "Rising commodity import bills put marginal pressure on Rupee")
        },
        "sectors": {
            "Information Technology": ("Neutral", "Low", "No impact on software services business model."),
            "Banking": ("Positive", "Low", "Asset quality of heavily leveraged steel/mining companies improves."),
            "FMCG": ("Negative", "Low", "Packaging costs (aluminum foil, tin plates) rise, squeezing margins."),
            "Automobiles": ("Negative", "High", "Raw material costs (steel, aluminum) rise; margins compressed."),
            "Pharmaceuticals": ("Neutral", "Low", "Stable; packaging cost increases are minor overall."),
            "Metals & Mining": ("Strong Positive", "High", "Primary beneficiary. Price realizations spike; margins expand dramatically."),
            "Energy & Oil Gas": ("Neutral", "Low", "Minor cost increase for steel pipes used in pipeline capex."),
            "Financial Services": ("Neutral", "Low", "Corporate lenders benefit from stronger metal account books."),
            "Consumer Durables": ("Negative", "Medium", "Copper and steel price hikes raise manufacturing costs of ACs, cables."),
            "Healthcare": ("Neutral", "Low", "No material impact on diagnostics or hospital operational costs."),
            "Real Estate": ("Negative", "Medium", "Construction raw materials (rebar, structural steel) become expensive."),
            "Capital Goods & Infra": ("Negative", "Medium", "Fixed-price infrastructure contracts face severe margin erosion.")
        },
        "winners": [
            ("TATASTEEL.NS", "Steel major; direct expansion of EBITDA margins on higher hot-rolled coil prices."),
            ("HINDALCO.NS", "Aluminum and copper major; gains from soaring LME pricing realizations."),
            ("JSWSTEEL.NS", "Leading steel producer; benefits from surging domestic steel demand and prices.")
        ],
        "losers": [
            ("MARUTI.NS", "Auto manufacturer facing severe raw material margin pressure from steel costs."),
            ("VOLTAS.NS", "AC maker hit by soaring copper prices (critical for coils and motors)."),
            ("HAVELLS.NS", "Wire/cable manufacturer squeezed by surging copper input costs.")
        ]
    }
}

def analyze_news_headline(headline: str, category: str = "Macroeconomics") -> dict:
    """
    NLP Rule Engine that parses headlines and returns impact simulations.
    """
    h_lower = headline.lower()
    
    # Check templates matching
    if any(k in h_lower for k in ["fed cut", "powell cut", "rate cut", "fed cuts", "interest rate cut", "monetary easing"]):
        if not any(k in h_lower for k in ["rbi", "india", "das"]):
            return EVENT_TEMPLATES["us_fed_cut"].copy()
            
    if any(k in h_lower for k in ["inflation", "cpi", "cpi spike", "prices surge", "wholesale price", "price index"]):
        if "cut" not in h_lower and "drop" not in h_lower:
            return EVENT_TEMPLATES["us_cpi_spike"].copy()
            
    if any(k in h_lower for k in ["war", "geopolitical", "middle east", "red sea", "suez canal", "oil price spike", "iran", "attacks"]):
        return EVENT_TEMPLATES["mid_east_war"].copy()
        
    if any(k in h_lower for k in ["rupee", "inr", "depreciates", "slides against dollar", "slides to 96", "weakens against usd"]):
        return EVENT_TEMPLATES["rupee_slide"].copy()
        
    if any(k in h_lower for k in ["rbi repo", "rbi hike", "das hike", "repo rate hike", "repo rate hikes"]):
        return EVENT_TEMPLATES["rbi_hike"].copy()
        
    if any(k in h_lower for k in ["copper", "steel", "aluminum", "metal prices", "commodity shock", "lme metals"]):
        return EVENT_TEMPLATES["metal_shock"].copy()

    # Generic dynamic parser if no template matches
    is_positive = any(k in h_lower for k in ["rise", "boost", "surge", "growth", "stimulus", "cut", "easing", "bullish", "recovery", "positive"])
    is_negative = any(k in h_lower for k in ["drop", "fall", "slump", "tariff", "hike", "tighten", "escalate", "inflation", "war", "bearish", "negative"])
    
    direction = "Positive" if is_positive else ("Negative" if is_negative else "Neutral")
    dir_symbol = "▲" if direction == "Positive" else ("▼" if direction == "Negative" else "■")
    
    affected_sectors = []
    if any(k in h_lower for k in ["it", "tech", "software", "infosys", "tcs"]): affected_sectors.append("Information Technology")
    if any(k in h_lower for k in ["bank", "nbfc", "hdfc", "icici", "sbi"]): affected_sectors.append("Banking")
    if any(k in h_lower for k in ["retail", "consumer", "fmcg", "unilever", "itc"]): affected_sectors.append("FMCG")
    if any(k in h_lower for k in ["auto", "car", "vehicle", "suzuki", "tata motors"]): affected_sectors.append("Automobiles")
    if any(k in h_lower for k in ["pharma", "drug", "healthcare", "hospital"]): affected_sectors.append("Pharmaceuticals")
    if any(k in h_lower for k in ["steel", "iron", "copper", "aluminum", "metal", "mining"]): affected_sectors.append("Metals & Mining")
    if any(k in h_lower for k in ["oil", "crude", "gas", "power", "utility"]): affected_sectors.append("Energy & Oil Gas")
    if any(k in h_lower for k in ["real estate", "realty", "dlf", "property"]): affected_sectors.append("Real Estate")
    if any(k in h_lower for k in ["infra", "road", "cement", "construction", "l&t"]): affected_sectors.append("Capital Goods & Infra")
        
    if not affected_sectors:
        affected_sectors = ["Banking", "Information Technology", "Real Estate"]
        
    sector_impacts = {}
    for s in SECTORS_LIST:
        if s in affected_sectors:
            sector_impacts[s] = (direction, "High", f"Direct keyword association found for '{s}'. Expected sector-specific volatility.")
        else:
            sector_impacts[s] = ("Neutral", "Low", "No direct keyword correlation; secondary macroeconomic spillover only.")
            
    winners = []
    losers = []
    
    if direction == "Positive":
        winners.append(("HDFCBANK.NS", "HDFC Bank: Positive liquidity flows and index momentum support."))
        winners.append(("TCS.NS", "TCS: Benefits from positive macro sentiment and global growth expectations."))
        losers.append(("HINDUNILVR.NS", "HUL: Relative underperformance under market rotation to growth."))
    else:
        winners.append(("SUNPHARMA.NS", "Sun Pharma: Defensive stock profile acts as a capital shelter."))
        losers.append(("DLF.NS", "DLF: Sensitive rate/macro stock hit by potential demand cooling."))
        losers.append(("LT.NS", "L&T: Capital goods major affected by raw material price volatility."))

    return {
        "title": headline[:60] + "..." if len(headline) > 60 else headline,
        "category": f"Dynamic / {category}",
        "overview": f"Automated dynamic analysis for the headline: '{headline}'. Based on parsing, it indicates a {direction.lower()} shock to related asset classes.",
        "transmission": f"Transmission operates through the: {', '.join(affected_sectors)} sectors. News sentiment: {direction}.",
        "assets": {
            "Nifty 50": (direction, f"{dir_symbol} +/- 0.8% to 1.5%", "driven by news sentiment shift"),
            "Bank Nifty": (direction, f"{dir_symbol} +/- 1.0% to 1.8%", "systemic banking flow impact"),
            "IT Nifty": (direction, f"{dir_symbol} +/- 1.2% to 2.2%", "export and US client sentiment shift"),
            "Gold": ("Neutral" if direction == "Positive" else "Positive", "▲/▼ +/- 0.5% to 1.2%", "safe haven flows dynamic"),
            "Crude Oil": ("Neutral", "■ 0.0%", "global supply fundamentals dominate"),
            "USD-INR": ("Neutral", "■ 0.0%", "currency volatility depends on FII flows")
        },
        "sectors": sector_impacts,
        "winners": winners,
        "losers": losers
    }

def fetch_custom_theme_news(theme_query: str, max_items: int = 5) -> list:
    """Fetch and parse theme-specific Google News RSS."""
    from src.macro import _parse_rss_feed, _NEWS_BASE
    url = _NEWS_BASE.format(query=theme_query)
    try:
        resp = requests.get(url, timeout=8, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        if resp.status_code == 200:
            return _parse_rss_feed(resp.text, max_items)
    except Exception as e:
        log.warning(f"Theme news fetch failed for query {theme_query}: {e}")
    return []

def get_cross_asset_correlations() -> pd.DataFrame:
    """Return cross-asset correlation rules as a DataFrame."""
    return pd.DataFrame(CROSS_ASSET_CORRELATION)


def render_trading_intelligence_tab():
    """
    Renders the Bloomberg-style Trading Intelligence tab in app.py.
    """
    st.markdown("<div class='bb-sec'>GLOBAL & MACRO TRADING INTELLIGENCE &nbsp;·&nbsp; SOVEREIGN YIELD CURVES &nbsp;·&nbsp; CROSS-ASSET CORRELATIONS</div>", unsafe_allow_html=True)

    with st.spinner("Fetching global market data..."):
        state = get_global_market_state()

    # ── 1. GLOBAL KEY INDICATORS GRID ──────────────────────────────────────────
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1:
        st.markdown(f"""
        <div class='bb-card'>
          <div class='lbl'>US CPI INFLATION (YoY)</div>
          <div class='val' style='color:#ffaa00'>{state["us_cpi_yoy"]:.2f}%</div>
          <div class='sub'>SOURCE: FRED</div>
        </div>""", unsafe_allow_html=True)
    with mcol2:
        st.markdown(f"""
        <div class='bb-card'>
          <div class='lbl'>US FED FUNDS RATE</div>
          <div class='val' style='color:#ffaa00'>{state["us_interest_rate"]:.2f}%</div>
          <div class='sub'>FED TARGET POLICY</div>
        </div>""", unsafe_allow_html=True)
    with mcol3:
        st.markdown(f"""
        <div class='bb-card'>
          <div class='lbl'>INDIA CPI INFLATION (YoY)</div>
          <div class='val' style='color:#ffaa00'>{state["india_cpi_yoy"]:.2f}%</div>
          <div class='sub'>SOURCE: FRED/RBI</div>
        </div>""", unsafe_allow_html=True)
    with mcol4:
        st.markdown(f"""
        <div class='bb-card'>
          <div class='lbl'>RBI REPO INTEREST RATE</div>
          <div class='val' style='color:#ff6600'>{state["india_repo_rate"]:.2f}%</div>
          <div class='sub'>STANCE: NEUTRAL</div>
        </div>""", unsafe_allow_html=True)
    with mcol5:
        chg_val = state["usd_inr_chg"]
        c_color = "#00cc44" if chg_val <= 0 else "#ff3333" # Inverted for currency slide
        c_arrow = "▲" if chg_val > 0 else "▼"
        st.markdown(f"""
        <div class='bb-card'>
          <div class='lbl'>USD-INR EXCHANGE RATE</div>
          <div class='val' style='color:#cccccc'>₹{state["usd_inr"]:.2f}</div>
          <div class='sub' style='color:{c_color}'>{c_arrow} {chg_val:+.2f}% TODAY</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── 2. YIELD CURVES ────────────────────────────────────────────────────────
    curve_col1, curve_col2 = st.columns([1, 1])
    
    with curve_col1:
        st.markdown("<div style='font-size:11px;color:#ff6600;font-weight:700;margin-bottom:8px'>◆ US TREASURY YIELD CURVE</div>", unsafe_allow_html=True)
        us_x = ['3M', '5Y', '10Y', '30Y']
        us_y = [state['us_3m_yield'], state['us_5y_yield'], state['us_10y_yield'], state['us_30y_yield']]
        
        fig_us = go.Figure()
        fig_us.add_trace(go.Scatter(
            x=us_x, y=us_y, 
            mode='lines+markers', 
            line=dict(color='#ff6600', width=2),
            marker=dict(size=8, color='#ff6600'),
            hovertemplate="US %{x} Yield: <b>%{y:.2f}%</b><extra></extra>"
        ))
        
        # Determine curve shape
        if state['us_3m_yield'] > state['us_10y_yield']:
            shape = "INVERTED (Recession Indicator)"
            shape_color = "#ff3333"
        elif abs(state['us_3m_yield'] - state['us_10y_yield']) < 0.2:
            shape = "FLAT"
            shape_color = "#ffaa00"
        else:
            shape = "NORMAL / EXPANSIONARY"
            shape_color = "#00cc44"
            
        fig_us.update_layout(
            paper_bgcolor='#000000',
            plot_bgcolor='#050505',
            xaxis=dict(showgrid=True, gridcolor='#111111', tickfont=dict(color='#888888', family='monospace')),
            yaxis=dict(showgrid=True, gridcolor='#111111', tickfont=dict(color='#888888', family='monospace'), ticksuffix='%'),
            height=200,
            margin=dict(l=30, r=10, t=10, b=10),
            showlegend=False
        )
        st.plotly_chart(fig_us, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"""
        <div style='background:#050505;padding:8px 12px;border:1px solid #111;font-size:10px;line-height:1.6;color:#888'>
          <span style='color:#ff6600;font-weight:700'>CURVE SHAPE:</span> 
          <span style='color:{shape_color};font-weight:700'>{shape}</span><br>
          An inverted curve (short-term rate > long-term rate) historically signals that the Fed funds rate is restrictive, increasing US recession risks and causing FII outflows from emerging markets.
        </div>""", unsafe_allow_html=True)

    with curve_col2:
        st.markdown("<div style='font-size:11px;color:#00cc44;font-weight:700;margin-bottom:8px'>◆ INDIAN SOVEREIGN YIELD CURVE (G-SEC)</div>", unsafe_allow_html=True)
        in_x = ['3M', '5Y', '10Y']
        in_y = [state['india_3m_yield'], state['india_5y_yield'], state['india_10y_yield']]
        
        fig_in = go.Figure()
        fig_in.add_trace(go.Scatter(
            x=in_x, y=in_y, 
            mode='lines+markers', 
            line=dict(color='#00cc44', width=2),
            marker=dict(size=8, color='#00cc44'),
            hovertemplate="India %{x} Yield: <b>%{y:.2f}%</b><extra></extra>"
        ))
        
        fig_in.update_layout(
            paper_bgcolor='#000000',
            plot_bgcolor='#050505',
            xaxis=dict(showgrid=True, gridcolor='#111111', tickfont=dict(color='#888888', family='monospace')),
            yaxis=dict(showgrid=True, gridcolor='#111111', tickfont=dict(color='#888888', family='monospace'), ticksuffix='%'),
            height=200,
            margin=dict(l=30, r=10, t=10, b=10),
            showlegend=False
        )
        st.plotly_chart(fig_in, use_container_width=True, config={'displayModeBar': False})
        st.markdown("""
        <div style='background:#050505;padding:8px 12px;border:1px solid #111;font-size:10px;line-height:1.6;color:#888'>
          <span style='color:#00cc44;font-weight:700'>EQUITY IMPACT RATIONALE:</span><br>
          Higher G-Sec yields raise the risk-free rate. According to discounting models (DCF), this increases the cost of equity and compresses stock valuation multiples (especially high-PE growth stocks).
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── 3. COMMODITIES MARKET DASHBOARD ────────────────────────────────────────
    st.markdown("<div class='bb-sec'>COMMODITIES MARKET DASHBOARD &nbsp;·&nbsp; REAL-TIME REALIZATION RATES</div>", unsafe_allow_html=True)
    ccol1, ccol2, ccol3, ccol4, ccol5 = st.columns(5)
    
    commodities = [
        (ccol1, "CRUDE OIL (WTI)", state["crude_oil_wti"], state["crude_oil_wti_chg"], "$/bbl", 
         "India imports 85% of its crude. Spikes drive inflation, expand current account deficits, hurt paint/auto/aviation margins, but benefit oil producers like ONGC."),
        (ccol2, "BRENT CRUDE", state["crude_oil_brent"], state["crude_oil_brent_chg"], "$/bbl", 
         "Global benchmark for crude. Directly impacts downstream retail fuel margins of Indian oil marketing companies (OMCs) if prices are capped by government."),
        (ccol3, "GOLD", state["gold"], state["gold_chg"], "$/oz", 
         "Primary safe-haven asset. Gold spikes expand collateral values for gold-lending NBFCs (Muthoot) and boost retail jewelry inventory valuation (Titan)."),
        (ccol4, "SILVER", state["silver"], state["silver_chg"], "$/oz", 
         "Hybrid commodity (safe-haven and industrial). Spikes signal both monetary hedge and positive electronics/photovoltaic manufacturing demand."),
        (ccol5, "COPPER", state["copper"], state["copper_chg"], "$/lb", 
         "Primary industrial metal ('Dr. Copper'). Spikes indicate strong global industrial capex and GDP expansion, boosting realizations of mining companies (Hindalco).")
    ]
    
    for col, name, price, chg, unit, desc in commodities:
        with col:
            color = "#00cc44" if chg >= 0 else "#ff3333"
            arrow = "▲" if chg >= 0 else "▼"
            st.markdown(f"""
            <div style='background:#050505;border:1px solid #111;padding:10px 14px'>
              <div style='font-size:9px;color:#ff6600;letter-spacing:.1em;margin-bottom:4px'>{name}</div>
              <div style='font-size:16px;font-weight:700;color:#ccc;display:flex;justify-content:between;align-items:center'>
                <span>{price:,.2f}<span style='font-size:10px;color:#555;font-weight:400'> {unit}</span></span>
              </div>
              <div style='font-size:10px;color:{color};font-weight:700;margin-top:2px'>{arrow} {chg:+.2f}%</div>
            </div>""", unsafe_allow_html=True)
            with st.expander("ECONOMIC TRANSMISSION", expanded=False):
                st.markdown(f"<div style='font-size:10px;color:#888;line-height:1.5'>{desc}</div>", unsafe_allow_html=True)

    st.markdown("")

    # ── 4. EVENT SIMULATOR ─────────────────────────────────────────────────────
    st.markdown("<div class='bb-sec'>INTERACTIVE MACRO & GEOPOLITICAL EVENT SIMULATOR</div>", unsafe_allow_html=True)
    
    sim_col1, sim_col2 = st.columns([2, 3])
    
    with sim_col1:
        st.markdown("<div style='font-size:11px;color:#ff6600;font-weight:700;margin-bottom:8px'>◆ CHOOSE SIMULATION INPUT</div>", unsafe_allow_html=True)
        
        sim_template = st.selectbox(
            "SELECT EVENT TEMPLATE",
            options=["None (Select a Template)"] + [EVENT_TEMPLATES[k]["title"] for k in EVENT_TEMPLATES.keys()],
            key="sim_template"
        )
        
        custom_headline = st.text_input(
            "OR: ENTER CUSTOM NEWS HEADLINE / EVENT",
            placeholder="e.g. US tariff war escalates or RBI unexpectedly cuts repo rate...",
            key="custom_headline"
        )
        
        st.markdown("""
        <div style='background:#080808;border:1px solid #111;padding:12px;font-size:10px;color:#666;line-height:1.6;margin-top:10px'>
          💡 <b>How to use:</b> Select a pre-configured template (like Fed Cuts or Middle East escalation) or type a custom financial headline. The engine will parse the context, apply cross-asset correlation rules, and generate a simulated impact report for all sectors and stocks.
        </div>""", unsafe_allow_html=True)

    # Resolve active simulation data
    sim_data = None
    if custom_headline.strip():
        sim_data = analyze_news_headline(custom_headline)
    elif sim_template != "None (Select a Template)":
        # Find template key
        t_key = next((k for k, v in EVENT_TEMPLATES.items() if v["title"] == sim_template), None)
        if t_key:
            sim_data = EVENT_TEMPLATES[t_key]

    with sim_col2:
        if sim_data:
            st.markdown(f"<div style='font-size:12px;color:#ffaa00;font-weight:700;margin-bottom:2px'>SIMULATED EVENT: {sim_data['title']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:9px;color:#555;letter-spacing:.08em;margin-bottom:12px;text-transform:uppercase'>CATEGORY: {sim_data['category']}</div>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background:#080808;border-left:3px solid #ff6600;padding:10px 14px;margin-bottom:12px'>
              <div style='font-size:11px;font-weight:700;color:#ff6600;margin-bottom:4px'>EVENT OVERVIEW & TRANSMISSION CHANNEL</div>
              <div style='font-size:11px;color:#bbb;line-height:1.6'>{sim_data['overview']}</div>
              <div style='font-size:10px;color:#666;margin-top:6px;font-style:italic'><b>Transmission path:</b> {sim_data.get('transmission', '')}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#050505;border:1px dashed #222;height:220px;display:flex;align-items:center;justify-content:center;color:#444;font-size:11px;text-align:center'>
              NO EVENT SELECTED<br>Choose a template or type a headline on the left to start the simulation.
            </div>""", unsafe_allow_html=True)

    if sim_data:
        st.markdown("<div style='font-size:11px;color:#ff6600;font-weight:700;margin-top:14px;margin-bottom:10px'>◆ EVENT TRANSMISSION & VOLATILITY SHIFTS</div>", unsafe_allow_html=True)
        
        # 1. Asset Impact Grid
        acol1, acol2 = st.columns([2, 3])
        with acol1:
            st.markdown("<div style='font-size:10px;color:#888;font-weight:700;margin-bottom:6px'>EXPECTED ASSET VOLATILITY RANGES</div>", unsafe_allow_html=True)
            asset_rows = []
            for asset, (dir_val, range_val, logic) in sim_data["assets"].items():
                color = "#00cc44" if "Pos" in dir_val or "Apprec" in dir_val else ("#ff3333" if "Neg" in dir_val or "Deprec" in dir_val else "#888")
                asset_rows.append(f"""
                <tr>
                  <td style='font-weight:600'>{asset}</td>
                  <td style='color:{color};font-weight:700'>{dir_val}</td>
                  <td style='font-family:monospace;color:#ffaa00'>{range_val}</td>
                  <td style='font-size:10px;color:#777'>{logic}</td>
                </tr>
                """)
            st.markdown(f"""
            <table style='width:100%;font-size:11px'>
              <thead>
                <tr>
                  <th style='text-align:left'>ASSET CLASS</th>
                  <th style='text-align:left'>DIRECTION</th>
                  <th style='text-align:left'>VOLATILITY EXPECTED</th>
                  <th style='text-align:left'>MARKET LOGIC</th>
                </tr>
              </thead>
              <tbody>
                {"".join(asset_rows)}
              </tbody>
            </table>
            """, unsafe_allow_html=True)
            
        with acol2:
            st.markdown("<div style='font-size:10px;color:#888;font-weight:700;margin-bottom:6px'>SECTORAL IMPACT MATRIX</div>", unsafe_allow_html=True)
            sector_rows = []
            for sector, (dir_val, magnitude, explanation) in sim_data["sectors"].items():
                color = "#00cc44" if dir_val == "Positive" else ("#ff3333" if dir_val == "Negative" else "#888")
                m_color = "#ff3333" if magnitude == "High" else ("#ffaa00" if magnitude == "Medium" else "#555")
                sector_rows.append(f"""
                <tr>
                  <td style='font-weight:600'>{sector}</td>
                  <td style='color:{color};font-weight:700'>{dir_val}</td>
                  <td style='color:{m_color};font-weight:700'>{magnitude}</td>
                  <td style='font-size:10px;color:#777'>{explanation}</td>
                </tr>
                """)
            st.markdown(f"""
            <div style='max-height:240px;overflow-y:auto;border:1px solid #111;padding:2px'>
              <table style='width:100%;font-size:10px'>
                <thead>
                  <tr>
                    <th style='text-align:left'>SECTOR</th>
                    <th style='text-align:left'>DIRECTION</th>
                    <th style='text-align:left'>MAGNITUDE</th>
                    <th style='text-align:left'>ANALYSIS / RATIONALE</th>
                  </tr>
                </thead>
                <tbody>
                  {"".join(sector_rows)}
                </tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        # 2. Stock Level Winners & Losers
        wcol1, wcol2 = st.columns([1, 1])
        with wcol1:
            st.markdown("<div style='background:#050f05;border:1px solid #003300;border-left:4px solid #00cc44;padding:12px'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:11px;color:#00cc44;font-weight:700;margin-bottom:6px'>◆ SIMULATED WINNERS (BENEFICIARIES)</div>", unsafe_allow_html=True)
            for sym, reason in sim_data["winners"]:
                st.markdown(f"""
                <div style='margin-bottom:10px;font-size:11px;line-height:1.5'>
                  <span style='color:#00cc44;font-weight:700'>{sym}</span> 
                  <span style='color:#555;font-size:9px'>({STOCK_INFO.get(sym, sym)})</span><br>
                  <span style='color:#888'>{reason}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with wcol2:
            st.markdown("<div style='background:#120505;border:1px solid #330000;border-left:4px solid #ff3333;padding:12px'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:11px;color:#ff3333;font-weight:700;margin-bottom:6px'>◆ SIMULATED LOSERS (UNDER PERFORMERS)</div>", unsafe_allow_html=True)
            for sym, reason in sim_data["losers"]:
                st.markdown(f"""
                <div style='margin-bottom:10px;font-size:11px;line-height:1.5'>
                  <span style='color:#ff3333;font-weight:700'>{sym}</span> 
                  <span style='color:#555;font-size:9px'>({STOCK_INFO.get(sym, sym)})</span><br>
                  <span style='color:#888'>{reason}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    # ── 5. LIVE NEWS IMPACT FEED ───────────────────────────────────────────────
    st.markdown("<div class='bb-sec'>LIVE FINANCIAL NEWS FEED & IMPACT ANALYZER</div>", unsafe_allow_html=True)
    
    ntab1, ntab2, ntab3, ntab4, ntab5 = st.tabs([
        "GLOBAL MACRO", "GEOPOLITICS", "BOND MARKETS", "COMMODITIES", "DOMESTIC POLITICS"
    ])
    
    news_feeds = [
        (ntab1, "global macroeconomics inflation rate", "Macroeconomics"),
        (ntab2, "geopolitics trade war tariff military", "Geopolitics"),
        (ntab3, "sovereign bond yields treasury curve", "Bond Market"),
        (ntab4, "commodity prices crude oil gold copper", "Commodities"),
        (ntab5, "Indian government policy reform budget", "Domestic Politics")
    ]
    
    for tab_ui, query, category in news_feeds:
        with tab_ui:
            with st.spinner(f"Fetching {category.lower()} news..."):
                items = fetch_custom_theme_news(query, max_items=5)
                
            if items:
                for item in items:
                    title = item.get("title", "")
                    link = item.get("link", "")
                    source = item.get("source", "")
                    time_ago = item.get("time_ago", "")
                    
                    # Run keyword analysis
                    sentiment_data = analyze_news_headline(title, category)
                    nifty_dir, nifty_chg, _ = sentiment_data["assets"]["Nifty 50"]
                    
                    if "Pos" in nifty_dir:
                        tag_color = "background:#003300;color:#00cc44;border:1px solid #005500"
                        tag_lbl = "BULLISH"
                    elif "Neg" in nifty_dir:
                        tag_color = "background:#330000;color:#ff3333;border:1px solid #550000"
                        tag_lbl = "BEARISH"
                    else:
                        tag_color = "background:#111;color:#888;border:1px solid #222"
                        tag_lbl = "NEUTRAL"
                        
                    st.markdown(f"""
                    <div style='background:#050505;border:1px solid #111;padding:12px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:start;gap:20px'>
                      <div style='flex-grow:1'>
                        <div style='font-size:12px;font-weight:600'><a href='{link}' target='_blank' style='color:#ccc;text-decoration:none'>{title}</a></div>
                        <div style='font-size:9px;color:#555;margin-top:6px;letter-spacing:.05em'>
                          SOURCE: {source} &nbsp;·&nbsp; PUBLISHED: {time_ago} &nbsp;·&nbsp; {category.upper()}
                        </div>
                      </div>
                      <div style='flex-shrink:0;text-align:right'>
                        <span style='display:inline-block;font-size:9px;font-weight:700;letter-spacing:.1em;padding:4px 8px;{tag_color}'>
                          EST. IMPACT: {tag_lbl}
                        </span>
                      </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='font-size:11px;color:#555;padding:20px 0;text-align:center'>
                  No recent news articles found for this topic (within 5 days).
                </div>""", unsafe_allow_html=True)

    st.markdown("")
    
    # ── 6. STRUCTURAL CORRELATION TABLE ────────────────────────────────────────
    st.markdown("<div class='bb-sec'>CROSS-ASSET STRUTURAL CORRELATIONS REFERENCE</div>", unsafe_allow_html=True)
    corr_df = get_cross_asset_correlations()
    
    # Format table manually
    corr_rows = []
    for _, row in corr_df.iterrows():
        corr_rows.append(f"""
        <tr>
          <td style='color:#ff6600;font-weight:600'>{row["Factors"]}</td>
          <td>{row["Commodities"]}</td>
          <td style='font-family:monospace'>{row["Bonds"]}</td>
          <td>{row["Equity Market"]}</td>
        </tr>
        """)
        
    st.markdown(f"""
    <table style='width:100%;font-size:11px'>
      <thead>
        <tr>
          <th style='text-align:left;width:20%'>MACRO / GEOPOLITICAL FACTOR</th>
          <th style='text-align:left;width:30%'>IMPACT ON COMMODITIES</th>
          <th style='text-align:left;width:25%'>IMPACT ON SOVEREIGN BONDS</th>
          <th style='text-align:left;width:25%'>IMPACT ON EQUITY MARKETS</th>
        </tr>
      </thead>
      <tbody>
        {"".join(corr_rows)}
      </tbody>
    </table>
    """, unsafe_allow_html=True)
