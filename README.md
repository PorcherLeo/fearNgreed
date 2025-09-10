# Fear & Greed + FRED (Live)

A lightweight Streamlit dashboard that combines the **Crypto Fear & Greed Index** (alternative.me) with robust **macro risk proxies from FRED** to visualize sentiment, rolling correlations, and a simple composite risk score.

---

## Features

- **Live panel**: current Fear & Greed value + a minimal gauge.
- **History**: last 90 days with optional 7-day EMA smoothing.
- **FRED indicators** (toggle in the sidebar):
  - **T10Y2Y** — 10Y–2Y Treasury slope (risk-on when steepening)
  - **BAMLH0A0HYM2** — High-Yield OAS (risk-off when widening)
  - **DTWEXBGS** — Broad U.S. Dollar index (risk-off when strengthening)
  - **DGS10** — 10Y U.S. Treasury yield (often risk-off when surging)
- **Rolling correlations** (60 days) between Fear & Greed and each selected indicator.
- **Composite score** (z-score blend): ↑ implies more “greed” (risk-on).
- **CSV export** of the Fear & Greed history.
- **Auto-refresh** (default: every 60s).

---

## Data Sources

- Fear & Greed Index: `https://api.alternative.me/fng/`
- FRED CSV: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>`

No API keys needed.

---

## Quick Start

```bash
# 1) (optional) create & activate a virtual env
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

# 2) install dependencies
pip install streamlit streamlit-autorefresh matplotlib pandas numpy requests

# 3) run
streamlit run app.py
