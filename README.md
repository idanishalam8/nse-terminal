# NSE Valuation Terminal 📊

**Bloomberg Terminal-style equity research tool — Sector Heat Map + CCA Screener**

## Features
- Real-time data from Yahoo Finance (15-min refresh, no stale cache)
- 12 NSE sectors · 180+ NSE 500 companies
- Historical percentile ranking (10-year lookback)
- Comparable Company Analysis with 6 trading multiples
- Bloomberg Terminal UI — black, orange, IBM Plex Mono
- Never sleeps — GitHub Actions pings every 5 minutes

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
1. Push to public GitHub repository
2. Go to share.streamlit.io → New app → app.py → Deploy
3. Update keep-alive.yml with your live URL

## Project Structure
```
nse-terminal/
├── app.py              ← Streamlit dashboard
├── src/
│   ├── config.py       ← NSE 500 universe, sectors, weights
│   ├── data.py         ← Real-time data fetching (no pkl cache)
│   ├── analytics.py    ← Percentile ranking, CCA calculations
│   └── charts.py       ← Bloomberg-styled chart generation
├── .github/workflows/
│   └── keep-alive.yml  ← Pings every 5 minutes (never sleeps)
└── requirements.txt
```
