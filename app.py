# -*- coding: utf-8 -*-
# Fear & Greed — Live

from datetime import datetime, timezone
import functools
import io
import time as _time
import requests

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---------- Config ----------
st.set_page_config(page_title="Fear & Greed + FRED (Live)", page_icon="📈", layout="wide")

API_LIVE = "https://api.alternative.me/fng/"
API_HIST = "https://api.alternative.me/fng/?limit=0&format=json"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

REFRESH_SECONDS = 60
EXTREME_FEAR, EXTREME_GREED = 25, 75


# ---------- Retry générique ----------
def retry(n=3, base_wait=1.5):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **k):
            for i in range(n):
                try:
                    return fn(*a, **k)
                except Exception:
                    if i == n - 1:
                        raise
                    _time.sleep(base_wait * (i + 1))
        return wrap
    return deco

# ---------- Helpers F&G ----------
@st.cache_data(ttl=300)
@retry()
def fetch_live():
    r = requests.get(API_LIVE, timeout=15)
    r.raise_for_status()
    data = r.json()["data"][0]
    return {
        "value": int(data["value"]),
        "classification": data["value_classification"],
        "timestamp": datetime.fromtimestamp(int(data["timestamp"]), tz=timezone.utc),
    }

@st.cache_data(ttl=3600)
@retry()
def fetch_history():
    r = requests.get(API_HIST, timeout=30)
    r.raise_for_status()
    raw = r.json()["data"]
    df = pd.DataFrame(raw)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df = df.rename(columns={"value": "fng", "value_classification": "label"})
    df = df.sort_values("timestamp").reset_index(drop=True)
    # date en datetime64[ns] (naive) normalisée à minuit
    df["date"] = df["timestamp"].dt.tz_convert(None).dt.normalize()
    return df

# ---------- UI: Header ----------
st.title("Fear & Greed — Live")
st.caption("Source: alternative.me (Fear & Greed Index)")

# ---------- Sidebar (paramètres utiles au panneau Live) ----------
with st.sidebar:
    st.header("Paramètres")
    win = st.slider("Fenêtre (jours) pour l'historique", 30, 365, 90, step=15)
    smooth = st.checkbox("Lissage F&G (EMA 7j)", True)

# ---------- Chargement F&G ----------
try:
    hist = fetch_history()
except Exception as e:
    st.error(f"Impossible de charger l'historique F&G : {e}")
    st.stop()

# ---------- Onglets ----------
tab_p1, tab_p2, tab_p3 = st.tabs(["Page1", "Page2", "Page3"])



# ---------- Helpers FRED ----------
@st.cache_data(ttl=3600)
@retry()
def fetch_fred_series(series_id: str, colname: str) -> pd.DataFrame:
    """
    Télécharge un CSV FRED via fredgraph.csv?id=<SERIE> et renvoie un DF ['date', colname].
    Détection robuste des noms de colonnes (DATE / Observation Date / etc.).
    """
    url = FRED_CSV.format(sid=series_id)

    sess = requests.Session()
    sess.headers.update({
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"),
        "Accept": "text/csv,application/csv;q=0.9,*/*;q=0.8",
        "Referer": "https://fred.stlouisfed.org/",
    })
    r = sess.get(url, timeout=30)
    r.raise_for_status()

    # Lecture souple
    df = pd.read_csv(io.StringIO(r.text), engine="python", sep=None)
    # Nettoie les noms
    df.columns = [str(c).strip() for c in df.columns]

    # ---- Trouver la colonne de date
    def is_date_col(name: str) -> bool:
        n = name.lower()
        return ("date" in n) or ("observation" in n and "date" in n)

    date_col = None
    for c in df.columns:
        if is_date_col(c):
            date_col = c
            break
    if date_col is None:
        # Fallback: première colonne
        date_col = df.columns[0]

    # ---- Trouver la colonne de valeur
    val_col = None
    if series_id in df.columns:
        val_col = series_id
    else:
        # Si pas de colonne exactement = series_id, on prend la 2ᵉ colonne non-date
        non_date_cols = [c for c in df.columns if c != date_col]
        if len(non_date_cols) == 0:
            raise ValueError(f"Aucune colonne de valeur trouvée dans FRED pour {series_id}. Colonnes={list(df.columns)}")
        val_col = non_date_cols[0]

    out = df[[date_col, val_col]].copy()
    out.columns = ["date", colname]

    # Dans FRED, les valeurs manquantes sont souvent le caractère '.'
    # On remplace **uniquement** les cellules exactement '.' par NaN
    out[colname] = out[colname].replace(".", np.nan)
    out[colname] = pd.to_numeric(out[colname], errors="coerce")

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=["date"]).sort_values("date")

    # Il arrive que toutes les valeurs soient NaN sur les périodes récentes, on ffil pour corrélations
    if out[colname].isna().all():
        # On laisse vide, la suite gèrera; sinon tu peux faire un st.warning ici.
        return out
    return out

# -------------------- PAGE 1 : Cours + Pivots S3 + Taux + Meetings --------------------
with tab_p1:
    st.subheader("Marchés — Cours & Pivots mensuels S3")

    # --------- Imports/guard pour yfinance ----------
    try:
        import yfinance as yf
        YF_OK = True
    except Exception:
        YF_OK = False
        st.warning("`yfinance` n’est pas installé. Installe : `pip install yfinance` pour afficher les cours temps différé.")

    TICKERS = {
        "EURUSD": "EURUSD=X",
        "SPY": "SPY",
        "QQQ": "QQQ",
        "ES": "ES=F",
        "NQ": "NQ=F",
    }

    @st.cache_data(ttl=300)
    def fetch_prices(tickers: dict) -> pd.DataFrame:
        if not YF_OK:
            return pd.DataFrame(columns=["symbol","ticker","price","change1d"])
        syms = list(tickers.values())
        # plus long pour couvrir week-ends/jours fériés
        df = yf.download(syms, period="15d", interval="1d", auto_adjust=False, progress=False, threads=True)
        out_rows = []
        for name, y in tickers.items():
            try:
                closes = df["Close"][y].dropna()
                if len(closes) == 0:
                    raise ValueError("no close")
                last_close = float(closes.iloc[-1])
                prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else np.nan
                chg = (last_close/prev_close - 1.0) if np.isfinite(prev_close) and prev_close else np.nan
                out_rows.append({"symbol": name, "ticker": y, "price": last_close, "change1d": chg})
            except Exception:
                out_rows.append({"symbol": name, "ticker": y, "price": np.nan, "change1d": np.nan})
        return pd.DataFrame(out_rows)

    @st.cache_data(ttl=1800)
    def monthly_pivot_s3(tickers: dict) -> pd.DataFrame:
        """
        PPM = (H+L+C)/3 sur le mois précédent (complet)
        S3  = P - 2*(H - L)
        """
        if not YF_OK:
            return pd.DataFrame(columns=["symbol","P","S3","month"])
        res = []
        for name, y in tickers.items():
            try:
                d = yf.download(y, period="500d", interval="1d", auto_adjust=False, progress=False, threads=False)
                d = d[["High","Low","Close"]].dropna()
                if d.empty:
                    res.append({"symbol": name, "P": np.nan, "S3": np.nan, "month": None})
                    continue
                m = d.resample("M").agg({"High":"max","Low":"min","Close":"last"}).dropna(how="any")
                # on veut le mois PRECEDENT (évite le mois en cours potentiellement incomplet)
                if len(m) < 2:
                    res.append({"symbol": name, "P": np.nan, "S3": np.nan, "month": None})
                    continue
                prev = m.iloc[-2]
                H, L, C = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
                P = (H + L + C) / 3.0
                S3 = P - 2.0*(H - L)
                res.append({"symbol": name, "P": P, "S3": S3, "month": m.index[-2].strftime("%Y-%m")})
            except Exception:
                res.append({"symbol": name, "P": np.nan, "S3": np.nan, "month": None})
        return pd.DataFrame(res)

    left, right = st.columns([1.5, 1])

    with left:
        if YF_OK:
            px = fetch_prices(TICKERS)
            piv = monthly_pivot_s3(TICKERS)
            dfm = px.merge(piv, on="symbol", how="left")
            dfm["dist_to_S3_%"] = (dfm["price"]/dfm["S3"] - 1.0)*100.0
            dfm["status_vs_S3"] = np.where(dfm["price"].notna() & dfm["S3"].notna(),
                                           np.where(dfm["price"] < dfm["S3"], "Sous S3", "Au-dessus S3"),
                                           "N/A")
            st.dataframe(
                dfm[["symbol","ticker","price","change1d","P","S3","dist_to_S3_%","status_vs_S3","month"]]
                   .set_index("symbol")
                   .style.format({
                       "price":"{:.4f}", "change1d":"{:+.2%}", "P":"{:.4f}", "S3":"{:.4f}", "dist_to_S3_%":"{:+.2f}"
                   }),
                use_container_width=True
            )
        else:
            st.info("Installe `yfinance` pour voir le tableau des cours et pivots.")

    with right:
        st.markdown("#### Mini‐graph avec S3 (mensuel)")
        chosen = st.selectbox("Instrument", list(TICKERS.keys()), index=0)
        if YF_OK:
            try:
                y = TICKERS[chosen]
                d = yf.download(y, period="6mo", interval="1d", auto_adjust=False, progress=False)
                if d.empty or d["Close"].dropna().empty:
                    st.warning("Pas assez de données récentes pour tracer ce graph.")
                else:
                    piv_single = monthly_pivot_s3({chosen: y})
                    s3_series = piv_single.loc[piv_single["symbol"] == chosen, "S3"].dropna()
                    s3_val = float(s3_series.iloc[-1]) if len(s3_series) else np.nan

                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(d.index, d["Close"], label=chosen)
                    if np.isfinite(s3_val):
                        ax.axhline(s3_val, linestyle="--", linewidth=1.5, label=f"S3 (mensuel): {s3_val:.4f}")
                    ax.set_title(f"{chosen} — 6 mois")
                    ax.legend()
                    st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Graph indisponible: {e}")

    st.markdown("---")
    st.subheader("Taux directeurs & taux longs")

    # --------- FRED rates (réutilise ta fetch_fred_series existante) ----------
    @st.cache_data(ttl=12*3600)
    def get_rates():
        out = {}
        try:
            fed = fetch_fred_series("DFEDTARU", "fed_upper")  # Fed target upper
            out["fed_rate"] = (fed["fed_upper"].dropna().iloc[-1], fed["date"].dropna().iloc[-1])
        except Exception as e:
            out["fed_rate"] = (np.nan, None)

        try:
            ecb = fetch_fred_series("ECBMRRFR", "ecb_mro")    # ECB main refi
            out["ecb_rate"] = (ecb["ecb_mro"].dropna().iloc[-1], ecb["date"].dropna().iloc[-1])
        except Exception as e:
            out["ecb_rate"] = (np.nan, None)

        try:
            us10 = fetch_fred_series("DGS10", "us10y")        # UST 10Y (daily)
            out["us10"] = (us10["us10y"].dropna().iloc[-1], us10["date"].dropna().iloc[-1])
        except Exception as e:
            out["us10"] = (np.nan, None)

        try:
            de10 = fetch_fred_series("IRLTLT01DEM156N", "de10y")  # DE 10Y (monthly)
            out["de10"] = (de10["de10y"].dropna().iloc[-1], de10["date"].dropna().iloc[-1])
        except Exception as e:
            out["de10"] = (np.nan, None)

        return out

    rates = get_rates()
    c1, c2, c3, c4 = st.columns(4)
    val, dt = rates["fed_rate"]; c1.metric("Fed funds (upper)", f"{val:.2f}%" if np.isfinite(val) else "N/A", help=f"FRED DFEDTARU\nDernière MàJ: {dt.date() if dt is not None else '—'}")
    val, dt = rates["ecb_rate"]; c2.metric("BCE (MRO)", f"{val:.2f}%" if np.isfinite(val) else "N/A", help=f"FRED ECBMRRFR\nDernière MàJ: {dt.date() if dt is not None else '—'}")
    val, dt = rates["us10"];     c3.metric("US 10Y", f"{val:.2f}%" if np.isfinite(val) else "N/A", help=f"FRED DGS10\nDernière MàJ: {dt.date() if dt is not None else '—'}")
    val, dt = rates["de10"];     c4.metric("DE 10Y", f"{val:.2f}%" if np.isfinite(val) else "N/A", help=f"FRED IRLTLT01DEM156N (mensuel)\nDernière MàJ: {dt.date() if dt is not None else '—'}")

    st.markdown("---")
    st.subheader("Prochaines réunions — Fed & BCE")

    # --------- Scrapers légers (sans dépendances) ----------
    import re
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td

    @st.cache_data(ttl=12*3600)
    def next_fomc_date():
        try:
            url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
            html = requests.get(url, timeout=15).text
            # Cherche des motifs "September 16-17, 2025" ou "Sep 16-17, 2025"
            months = "(January|February|March|April|May|June|July|August|September|October|November|December|" \
                     "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
            rx = re.compile(rf"{months}\s+(\d{{1,2}})(?:\s*-\s*\d{{1,2}})?\,\s*(\d{{4}})", re.IGNORECASE)
            candidates = []
            for m in rx.finditer(html):
                mon, day, year = m.group(1), m.group(2), m.group(3)
                try:
                    d = _dt.strptime(f"{mon} {day} {year}", "%B %d %Y")
                except ValueError:
                    try:
                        d = _dt.strptime(f"{mon} {day} {year}", "%b %d %Y")
                    except ValueError:
                        continue
                candidates.append(d.date())
            today = _dt.now().date()
            fut = sorted([d for d in set(candidates) if d >= today])
            return fut[0] if fut else None
        except Exception:
            return None

    @st.cache_data(ttl=12*3600)
    def next_ecb_date():
        try:
            url = "https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html"
            html = requests.get(url, timeout=15).text
            # Format type "11/09/2025: ... monetary policy meeting ..."
            rx = re.compile(r"(\d{2}/\d{2}/\d{4}).*monetary policy meeting", re.IGNORECASE)
            dates = []
            for m in rx.finditer(html):
                d = _dt.strptime(m.group(1), "%d/%m/%Y").date()
                dates.append(d)
            today = _dt.now().date()
            fut = sorted([d for d in set(dates) if d >= today])
            return fut[0] if fut else None
        except Exception:
            return None

    nf, ne = next_fomc_date(), next_ecb_date()
    colA, colB = st.columns(2)
    colA.info(f"**Prochaine FOMC** : {nf.strftime('%Y-%m-%d') if nf else 'N/A'}")
    colB.info(f"**Prochaine BCE** : {ne.strftime('%Y-%m-%d') if ne else 'N/A'}")

    # Overrides manuels (au cas où scraping indisponible)
    #with st.expander("Ajuster manuellement (si besoin)"):
        #ov_fed = st.text_input("Override date FOMC (YYYY-MM-DD)", value=(nf.strftime("%Y-%m-%d") if nf else ""))
        #ov_ecb = st.text_input("Override date BCE (YYYY-MM-DD)", value=(ne.strftime("%Y-%m-%d") if ne else ""))
        #if ov_fed:
            #colA.info(f"**Prochaine FOMC (override)** : {ov_fed}")
        #if ov_ecb:
            #colB.info(f"**Prochaine BCE (override)** : {ov_ecb}")


# -------------------- PAGE 2 : F&G (score/100) + P/E & Beta --------------------
with tab_p2:
    # ===== 1) Fear & Greed — juste le score/100 =====
    st.subheader("Index en direct")
    try:
        live = fetch_live()
        st.metric(
            "Fear & Greed",
            f"{live['value']}/100",
            help=f"{live['classification']}\nMàJ: {live['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}"
        )
    except Exception as e:
        st.error(f"Échec du live index : {e}")

    st.markdown("---")

    # ===== 2) P/E & Beta — SPY, QQQ, Mega-cap Tech =====
    st.subheader("P/E & Beta — SPY, QQQ, Mega-cap Tech")

    # Import yfinance (optionnel, avec garde)
    try:
        import yfinance as yf
        YF_OK = True
    except Exception:
        YF_OK = False
        st.warning("`yfinance` n’est pas installé. Installe : `pip install yfinance` pour afficher P/E & Beta.")

    TICKERS = ["SPY", "QQQ", "NVDA", "AMD", "MSFT", "AAPL", "AMZN", "META", "NFLX"]
    DISPLAY = {
        "SPY":"SPY (S&P 500 ETF)",
        "QQQ":"QQQ (Nasdaq-100 ETF)",
        "NVDA":"NVIDIA (NVDA)",
        "AMD":"AMD (AMD)",
        "MSFT":"Microsoft (MSFT)",
        "AAPL":"Apple (AAPL)",
        "AMZN":"Amazon (AMZN)",
        "META":"Meta (META)",
        "NFLX":"Netflix (NFLX)",
    }

    if YF_OK:
        @st.cache_data(ttl=900)
        def get_prices_1y(tickers):
            # Close sur 1 an (couvre les jours fériés/week-end)
            df = yf.download(tickers, period="1y", interval="1d", auto_adjust=False, progress=False, threads=True)
            # Harmonise en DataFrame de closes simples
            if isinstance(df.columns, pd.MultiIndex):
                closes = df["Close"].copy()
            else:
                closes = df[["Close"]].rename(columns={"Close": tickers[0]})
            return closes

        @st.cache_data(ttl=3600)
        def get_pe(ticker: str, fallback_price: float | None = None) -> float | None:
            """
            Essaie: trailingPE ; sinon calcule Price / trailingEps (si dispo).
            """
            try:
                tk = yf.Ticker(ticker)
                info = tk.get_info()
                pe = info.get("trailingPE", None)
                if pe is None:
                    eps = info.get("trailingEps", None)
                    if eps and eps != 0:
                        if np.isfinite(fallback_price):
                            price = float(fallback_price)
                        else:
                            h = tk.history(period="5d")
                            price = float(h["Close"].dropna().iloc[-1]) if not h.empty else np.nan
                        if np.isfinite(price):
                            pe = price / float(eps)
                return float(pe) if pe is not None and np.isfinite(pe) else None
            except Exception:
                return None

        @st.cache_data(ttl=900)
        def compute_betas(closes: pd.DataFrame, bench: str = "SPY") -> dict:
            """
            Beta via régression simple: cov(stock, bench)/var(bench) sur ~1 an (quotidien).
            """
            out = {t: np.nan for t in closes.columns}
            if bench not in closes.columns:
                return out
            rets = closes.pct_change().dropna(how="all")
            b = rets[bench].dropna()
            if b.var(ddof=0) == 0 or b.empty:
                return out
            for t in closes.columns:
                s = rets[t].dropna()
                j = b.index.intersection(s.index)
                if len(j) < 30:
                    continue
                cov = np.cov(s.loc[j], b.loc[j])[0,1]
                varb = np.var(b.loc[j])
                beta = cov/varb if varb != 0 else np.nan
                out[t] = beta
            # Par convention, beta(SPY)=1.0
            if bench in out:
                out[bench] = 1.0
            return out

        # --- Récupération des données ---
        closes = get_prices_1y(TICKERS)
        # Dernier prix (pour calcul PE fallback)
        last_px = {}
        for t in TICKERS:
            try:
                last_px[t] = float(closes[t].dropna().iloc[-1])
            except Exception:
                last_px[t] = np.nan

        betas = compute_betas(closes, bench="SPY")

        rows = []
        for t in TICKERS:
            pe = get_pe(t, fallback_price=last_px.get(t, np.nan))
            rows.append({
                "Ticker": t,
                "Name": DISPLAY.get(t, t),
                "Last": last_px.get(t, np.nan),
                "P/E (TTM)": pe if pe is not None else np.nan,
                "Beta (vs SPY)": betas.get(t, np.nan),
            })

        df = pd.DataFrame(rows)
        # Tri du plus petit beta au plus grand
        df = df.sort_values("Beta (vs SPY)", na_position="last").reset_index(drop=True)

        # Affichage
        st.dataframe(
            df[["Name","Ticker","Last","P/E (TTM)","Beta (vs SPY)"]]
              .style.format({
                  "Last": lambda v: "" if pd.isna(v) else (f"{v:.2f}" if v>=10 else f"{v:.4f}"),
                  "P/E (TTM)": lambda v: "" if pd.isna(v) else f"{v:.1f}",
                  "Beta (vs SPY)": lambda v: "" if pd.isna(v) else f"{v:.2f}",
              }),
            use_container_width=True,
            height=420
        )

        st.caption("Notes: P/E = trailing (TTM) si dispo ; Beta calculé sur ~1 an de retours quotidiens vs SPY (fallback: info Yahoo).")
    else:
        st.info("Ajoute `yfinance` à requirements.txt puis redeploie pour activer cette section.")


# -------------------- PAGE 3 : Mega-cap Tech — Cours & var J / 7j / 30j --------------------
with tab_p3:
    st.subheader("Mega-cap Tech — Cours & variations (jour • 7j • 30j)")

    # Tickers + noms
    TICKERS = ["NVDA", "AMD", "MSFT", "AAPL", "AMZN", "META", "NFLX", "GOOGL"]
    NAMES = {
        "NVDA": "NVIDIA", "AMD": "AMD", "MSFT": "Microsoft", "AAPL": "Apple",
        "AMZN": "Amazon", "META": "Meta", "NFLX": "Netflix", "GOOGL": "Alphabet (A)"
    }

    # yfinance
    try:
        import yfinance as yf
        YF_OK = True
    except Exception:
        YF_OK = False
        st.warning("`yfinance` n’est pas installé. Fais: `pip install yfinance`.")

    if YF_OK:
        @st.cache_data(ttl=300)
        def fetch_closes(symbols):
            df = yf.download(symbols, period="2y", interval="1d", auto_adjust=False,
                             progress=False, threads=True)
            return df["Close"].copy() if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": symbols[0]})

        def pct_change_daily(s: pd.Series) -> float:
            s = s.dropna()
            if len(s) < 2: return np.nan
            prev, last = float(s.iloc[-2]), float(s.iloc[-1])
            if not (np.isfinite(prev) and prev != 0 and np.isfinite(last)): return np.nan
            return last/prev - 1.0

        def pct_change_since_days(s: pd.Series, days: int) -> float:
            s = s.dropna()
            if s.empty: return np.nan
            last_date = s.index.max()
            target = last_date - pd.Timedelta(days=days)
            idx = s.index.searchsorted(target, side="right") - 1
            if idx < 0: return np.nan
            ref, last = float(s.iloc[idx]), float(s.iloc[-1])
            if not (np.isfinite(ref) and ref != 0 and np.isfinite(last)): return np.nan
            return last/ref - 1.0

        closes = fetch_closes(TICKERS)

        # Mise en page: 4 cartes par ligne
        cols = st.columns(4)
        for i, t in enumerate(TICKERS):
            with cols[i % 4]:
                s = closes[t].dropna() if t in closes.columns else pd.Series(dtype=float)
                last = float(s.iloc[-1]) if len(s) else np.nan
                d = pct_change_daily(s)
                w = pct_change_since_days(s, 7)
                m = pct_change_since_days(s, 30)

                # Carte : prix + delta jour
                price_txt = "N/A" if not np.isfinite(last) else (f"{last:.2f}" if last >= 10 else f"{last:.4f}")
                delta_txt = None if not np.isfinite(d) else f"{d:+.2%}"
                st.metric(f"{NAMES.get(t, t)} ({t})", price_txt, delta=delta_txt)

                # Ligne de variations 7j / 30j
                def fmt(x): return "N/A" if not np.isfinite(x) else f"{x:+.2%}"
                st.caption(f"7j: {fmt(w)} • 30j: {fmt(m)}")



# ---------- Footer ----------
st.divider()
