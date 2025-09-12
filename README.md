# Fear & Greed + Markets (Streamlit)

A lightweight Streamlit dashboard that shows the **Crypto Fear & Greed Index** alongside practical **market tools**: pivots, rates, earnings multiples, mega-cap tech monitors, and an assets **correlation heatmap**.

> Educational only — not investment advice.

---

## What’s inside

### Page 1 — Markets (prices, monthly pivots S3, rates, meetings)
- **Prices**: EURUSD, SPY, QQQ, ES (S&P futures), NQ (Nasdaq futures)
- **Monthly Pivot S3** (previous full month):
  - `P = (H + L + C) / 3`
  - `S3 = P − 2 × (H − L)`
  - Distance to S3 and status (above/below)
- **Mini chart** with S3 overlay
- **Rates (FRED)**: Fed funds (upper, `DFEDTARU`), ECB main refi (`ECBMRRFR`), US 10Y (`DGS10`), DE 10Y (`IRLTLT01DEM156N`)
- **Next meetings**: FOMC & ECB (light HTML scrape with a manual override if needed)

### Page 2 — Fear & Greed + P/E & Beta
- **Fear & Greed**: live score (value/100)
- **P/E (TTM)** and **Beta vs SPY** for: SPY, QQQ, **NVDA, AMD, MSFT, AAPL, AMZN, META, NFLX**
  - Beta computed from ~1y daily returns: `β = cov(asset, SPY) / var(SPY)`
  - P/E from Yahoo (`trailingPE`) or fallback `price / trailingEps`

### Page 3 — Mega-cap Tech cards
- **NVDA, AMD, MSFT, AAPL, AMZN, META, NFLX, GOOGL**  
- For each: **last price**, **d%**, **7d%**, **30d%** (robust to holidays/holes)

### Page 4 — Correlations between assets
- Custom universe (defaults include SPY/QQQ/mega-caps, EURUSD, ES, NQ, BTC)
- Choice of **frequency** (daily/weekly/monthly), **window** (e.g., 90), and **returns** (simple/log)
- Heatmap with per-cell Pearson correlations

---

## Data sources

- **Fear & Greed Index**: `https://api.alternative.me/fng/`
- **Yahoo Finance** via `yfinance` (e.g., `EURUSD=X`, `ES=F`, `NQ=F`, `NVDA`, …)
- **FRED** CSV (no key): `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>`
  - `DFEDTARU`, `ECBMRRFR`, `DGS10`, `IRLTLT01DEM156N`
- **Calendars**: FOMC official site & ECB calendar (simple HTML parsing)

---

## Quick start

```bash
# (optional) venv
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

# install
pip install -r requirements.txt
# or:
pip install streamlit pandas numpy requests matplotlib yfinance

# run
streamlit run app.py
