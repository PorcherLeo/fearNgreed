import time
import math
import json
import io
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

# ---------- Config ----------
st.set_page_config(page_title="Fear & Greed + VIX + Put/Call (Live + ML)", page_icon="📈", layout="wide")
API_LIVE = "https://api.alternative.me/fng/"  # live & recent crypto F&G
API_HIST = "https://api.alternative.me/fng/?limit=0&format=json"  # full history
# Daily BTC close (simple):
BTC_HIST = "https://api.coindesk.com/v1/bpi/historical/close.json?currency=USD"
# VIX from Stooq daily CSV (no key): ^VIX symbol
VIX_CSV = "https://stooq.com/q/d/l/?s=%5Evix&i=d"
# Cboe Equity Put/Call Ratio historical CSV (no key)
PCR_EQUITY_CSV = "https://cdn.cboe.com/api/global/us_indices/daily_prices/EQUITY_PUT_CALL_RATIO_HISTORICAL.csv"
# (Optional) Cboe Total Put/Call Ratio
PCR_TOTAL_CSV = "https://cdn.cboe.com/api/global/us_indices/daily_prices/TOTAL_PUT_CALL_RATIO_HISTORICAL.csv"

REFRESH_SECONDS = 60  # auto-refresh interval for the live panel

# ---------- Helpers ----------
@st.cache_data(ttl=300)
def fetch_live():
    r = requests.get(API_LIVE, timeout=15)
    r.raise_for_status()
    data = r.json()["data"][0]
    # normalize
    return {
        "value": int(data["value"]),
        "classification": data["value_classification"],
        "timestamp": datetime.fromtimestamp(int(data["timestamp"]), tz=timezone.utc),
    }

@st.cache_data(ttl=3600)
def fetch_history():
    r = requests.get(API_HIST, timeout=30)
    r.raise_for_status()
    raw = r.json()["data"]
    df = pd.DataFrame(raw)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df = df.rename(columns={"value": "fng", "value_classification": "label"})
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    return df

@st.cache_data(ttl=3600)
def fetch_btc_close():
    # daily close for the last ~2 years
    end = datetime.utcnow().date()
    start = end - timedelta(days=730)
    url = f"https://api.coindesk.com/v1/bpi/historical/close.json?start={start}&end={end}&currency=USD"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()["bpi"]
    df = pd.DataFrame({"date": pd.to_datetime(list(data.keys())).date, "close": list(data.values())})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data(ttl=3600)
def fetch_vix():
    # Stooq CSV columns: Date, Open, High, Low, Close, Volume
    r = requests.get(VIX_CSV, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    # Ensure expected columns
    df.columns = [c.lower() for c in df.columns]
    # some stooq files use 'data' for date depending on locale; handle both
    date_col = 'date' if 'date' in df.columns else ('data' if 'data' in df.columns else None)
    if not date_col or 'close' not in df.columns:
        raise ValueError("Unexpected VIX CSV schema from Stooq")
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    out = df.rename(columns={date_col: 'date', 'close': 'vix_close'})[['date', 'vix_close']]
    out = out.sort_values('date').reset_index(drop=True)
    return out

@st.cache_data(ttl=3600)
def fetch_pcr_equity():
    r = requests.get(PCR_EQUITY_CSV, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # Typical columns: DATE, OPEN, HIGH, LOW, CLOSE
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = 'date'
    close_col = 'close'
    if date_col not in df.columns or close_col not in df.columns:
        # handle alt schemas
        candidates = [c for c in df.columns if 'date' in c]
        if candidates:
            date_col = candidates[0]
        candidates = [c for c in df.columns if 'close' in c or 'value' in c]
        if candidates:
            close_col = candidates[0]
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
    out = df.rename(columns={date_col: 'date', close_col: 'pcr_equity'})[['date', 'pcr_equity']]
    out = out.sort_values('date').reset_index(drop=True)
    return out

@st.cache_data(ttl=3600)
def fetch_pcr_total():
    r = requests.get(PCR_TOTAL_CSV, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = 'date'
    close_col = 'close'
    if date_col not in df.columns or close_col not in df.columns:
        candidates = [c for c in df.columns if 'date' in c]
        if candidates:
            date_col = candidates[0]
        candidates = [c for c in df.columns if 'close' in c or 'value' in c]
        if candidates:
            close_col = candidates[0]
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
    out = df.rename(columns={date_col: 'date', close_col: 'pcr_total'})[['date', 'pcr_total']]
    out = out.sort_values('date').reset_index(drop=True)
    return out

# ---------- UI: Header ----------
st.title("📈 Fear & Greed — Live + VIX + Put/Call + ML")
st.caption("Sources: alternative.me (Crypto F&G), CoinDesk (BTC), Stooq (^VIX), Cboe (Equity/Total Put‑Call)")

# ---------- Live Panel ----------
col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    st.subheader("Live index")
    try:
        live = fetch_live()
        st.metric("Fear & Greed", f"{live['value']}", help=f"{live['classification']}\nUpdated: {live['timestamp'].strftime('%Y-%m-%d %H:%M UTC')} ")
        # simple gauge via matplotlib
        fig, ax = plt.subplots(figsize=(5, 2.6))
        ax.axis('off')
        val = live['value']
        # draw an arc 0..100
        t = np.linspace(-np.pi, 0, 300)
        ax.plot(np.cos(t), np.sin(t))
        # needle
        angle = -np.pi + (val/100)*np.pi
        ax.plot([0, np.cos(angle)], [0, np.sin(angle)], linewidth=3)
        ax.text(0, -0.2, f"{val}", ha='center', va='center', fontsize=18)
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load live index: {e}")

with col2:
    st.subheader("Last 90 days — F&G")
    try:
        hist = fetch_history()
        last = hist.tail(90)
        st.line_chart(last.set_index(pd.to_datetime(last["date"]))[["fng"]])
    except Exception as e:
        st.error(f"History error: {e}")

with col3:
    st.subheader("Auto‑refresh")
    st.write(f"Refreshing every {REFRESH_SECONDS}s…")
    st.button("Refresh now", on_click=lambda: st.cache_data.clear())
    st.caption("Tip: use Streamlit's rerun toolbar to refresh manually.")

st.divider()

# ---------- Market Sentiment Add‑ons: VIX & Put/Call ----------
colv, colp = st.columns(2)
with colv:
    st.subheader("VIX (volatility index)")
    try:
        vix = fetch_vix()
        last90 = vix.tail(90)
        st.line_chart(last90.set_index(pd.to_datetime(last90['date']))[["vix_close"]])
        st.caption("Source: Stooq (^VIX)")
    except Exception as e:
        st.error(f"VIX error: {e}")

with colp:
    st.subheader("Put/Call Ratio (Equity & Total)")
    try:
        pcr_eq = fetch_pcr_equity()
        pcr_tot = fetch_pcr_total()
        merged_pcr = pd.merge(pcr_eq, pcr_tot, on='date', how='outer').sort_values('date')
        last90p = merged_pcr.tail(90)
        st.line_chart(last90p.set_index(pd.to_datetime(last90p['date']))[["pcr_equity", "pcr_total"]])
        st.caption("Source: Cboe daily CSVs")
    except Exception as e:
        st.error(f"Put/Call error: {e}")

st.divider()

# ---------- ML: Predict next‑day BTC direction ----------
st.header("🤖 Simple ML: predict next‑day BTC direction")
st.write(
    "This quick demo trains a time‑series logistic regression to predict whether BTC goes **up** tomorrow (close-to-close) using Fear & Greed, **VIX**, and **Put/Call ratios**, plus simple momentum/volatility features."
)

with st.expander("Settings", expanded=False):
    lookback = st.slider("Lookback window (days) for rolling stats", 3, 60, 14)
    n_splits = st.slider("TimeSeries CV splits", 3, 10, 5)

@st.cache_data(ttl=3600)
def prepare_features(lookback: int = 14):
    fng = fetch_history()[["date", "fng"]]
    btc = fetch_btc_close()[["date", "close"]]
    vix = fetch_vix()[["date", "vix_close"]]
    pcr_eq = fetch_pcr_equity()[["date", "pcr_equity"]]
    pcr_tot = fetch_pcr_total()[["date", "pcr_total"]]

    df = btc.merge(fng, on="date", how="left") \
            .merge(vix, on="date", how="left") \
            .merge(pcr_eq, on="date", how="left") \
            .merge(pcr_tot, on="date", how="left")

    # forward fill to align calendars (crypto trades every day, VIX/Cboe are business days)
    df = df.sort_values('date')
    df[['fng','vix_close','pcr_equity','pcr_total']] = df[['fng','vix_close','pcr_equity','pcr_total']].fillna(method='ffill')

    # returns
    df["ret1"] = df["close"].pct_change()

    # rolling stats
    for col, base in [("fng","fng"),("vix_close","vix"),("pcr_equity","pcrEQ"),("pcr_total","pcrTOT")]:
        ma = f"{base}_ma"; sd = f"{base}_std"; z = f"{base}_z"
        df[ma] = df[col].rolling(lookback).mean()
        df[sd] = df[col].rolling(lookback).std()
        df[z] = (df[col] - df[ma]) / (df[sd] + 1e-6)

    # momentum/volatility features on BTC
    for w in [3, 7, 14, 30]:
        df[f"mom_{w}"] = df["close"].pct_change(w)
        df[f"vol_{w}"] = df["ret1"].rolling(w).std()

    # daily deltas on VIX/PCR
    df['vix_chg'] = df['vix_close'].pct_change()
    df['pcrEQ_chg'] = df['pcr_equity'].pct_change()
    df['pcrTOT_chg'] = df['pcr_total'].pct_change()

    # target: next-day up (1) or not (0)
    df["ret1_fwd"] = df["close"].pct_change().shift(-1)
    df["target"] = (df["ret1_fwd"] > 0).astype(int)

    df = df.dropna().reset_index(drop=True)

    features = [
        # F&G
        "fng", "fng_ma", "fng_std", "fng_z",
        # VIX
        "vix_close", "vix_ma", "vix_std", "vix_z", "vix_chg",
        # Put/Call (Equity & Total)
        "pcr_equity", "pcrEQ_ma", "pcrEQ_std", "pcrEQ_z", "pcrEQ_chg",
        "pcr_total", "pcrTOT_ma", "pcrTOT_std", "pcrTOT_z", "pcrTOT_chg",
        # BTC price dynamics
        "mom_3", "mom_7", "mom_14", "mom_30",
        "vol_3", "vol_7", "vol_14", "vol_30",
    ]

    # Some columns may not exist if lookback trims too much; ensure they exist
    features = [f for f in features if f in df.columns]

    X = df[features].values
    y = df["target"].values
    dates = pd.to_datetime(df["date"])
    return X, y, dates, df, features


def train_and_eval(lookback: int = 14, n_splits: int = 5):
    X, y, dates, df, features = prepare_features(lookback)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    proba = np.zeros_like(y, dtype=float)
    preds = np.zeros_like(y, dtype=int)

    for train_idx, test_idx in tscv.split(X):
        model = LogisticRegression(max_iter=500)
        model.fit(X[train_idx], y[train_idx])
        proba[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        preds[test_idx] = (proba[test_idx] >= 0.5).astype(int)

    valid = proba > 0
    auc = roc_auc_score(y[valid], proba[valid]) if valid.any() else float('nan')
    acc = accuracy_score(y[valid], preds[valid]) if valid.any() else float('nan')

    return proba, preds, y, dates, df, features, auc, acc

# Train button
colA, colB = st.columns([1, 2])
with colA:
    if st.button("Train / Re‑train model"):
        st.session_state["ml_run"] = True

ran = st.session_state.get("ml_run", False)

with colB:
    if ran:
        with st.spinner("Training model…"):
            proba, preds, y, dates, df, features, auc, acc = train_and_eval(lookback, n_splits)
        st.success(f"Done. CV AUC = {auc:.3f} | Accuracy = {acc:.3f}")

        # Plot probability over time
        fig2, ax2 = plt.subplots(figsize=(9, 3.2))
        ax2.plot(dates, proba, label="P(up)")
        ax2.plot(dates, y, linestyle=":", label="Actual up (0/1)")
        ax2.set_title("Predicted probability of next‑day BTC going up")
        ax2.legend()
        st.pyplot(fig2, use_container_width=True)

        # Show recent feature importances via absolute coefficients
        model = LogisticRegression(max_iter=500)
        X, y, dates, df, features = prepare_features(lookback)[:5]
        model.fit(X, y)
        coefs = pd.Series(np.abs(model.coef_[0]), index=features).sort_values(ascending=False)
        st.subheader("Feature importance (|coefficients|)")
        st.bar_chart(coefs)

        st.caption("Note: Educational demo. Not financial advice.")

st.divider()

# ---------- Data Inspector ----------
st.subheader("Data preview")
try:
    hist = fetch_history().tail(5)
    vixp = fetch_vix().tail(5)
    pcre = fetch_pcr_equity().tail(5)
    dfprev = hist.merge(vixp, on='date', how='outer').merge(pcre, on='date', how='outer').sort_values('date').tail(10)
    st.dataframe(dfprev, use_container_width=True)
except Exception as e:
    st.error(f"Data preview error: {e}")

# ---------- Footer ----------
st.caption(
    "Built with Streamlit • F&G: alternative.me • BTC: CoinDesk • VIX: Stooq • Put/Call: Cboe. "
    "This demo auto‑refreshes the live index every 60 seconds."
)
