# -*- coding: utf-8 -*-
# Fear & Greed + Indicateurs FRED (T10Y2Y, HY OAS, USD Broad, 10Y) — sans VIX / Put-Call / ML

from datetime import datetime, timezone
import functools
import time as _time
import io
import requests

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import streamlit as st

# ---------- Config ----------
st.set_page_config(page_title="Fear & Greed + FRED (Live)", page_icon="📈", layout="wide")

API_LIVE = "https://api.alternative.me/fng/"
API_HIST = "https://api.alternative.me/fng/?limit=0&format=json"

# FRED CSV direct (pas de clé) : https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIE>
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

REFRESH_SECONDS = 60
EXTREME_FEAR, EXTREME_GREED = 25, 75

# ---------- Auto-refresh ----------
def enable_autorefresh(seconds: int):
    st.components.v1.html(
        f"<script>setTimeout(function(){{window.location.reload()}}, {int(seconds*1000)});</script>",
        height=0
    )

# Active l'auto-refresh
enable_autorefresh(REFRESH_SECONDS)

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
    df["date"] = df["timestamp"].dt.tz_convert(None).dt.normalize()
    return df

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


# Indicateurs alternatifs (FRED)
def fetch_t10y2y():   # Courbe 10Y-2Y (spread)
    return fetch_fred_series("T10Y2Y", "yc_spread")
def fetch_hy_oas():   # High Yield OAS (%)
    return fetch_fred_series("BAMLH0A0HYM2", "hy_oas")
def fetch_usd_broad():  # USD Broad Index (2006=100)
    return fetch_fred_series("DTWEXBGS", "usd")
def fetch_dgs10():    # Taux 10 ans (%)
    return fetch_fred_series("DGS10", "ust10")

def zscore(s: pd.Series):
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd

# ---------- UI: Header ----------
st.title("Fear & Greed — Live (avec indicateurs FRED)")
st.caption("Sources: alternative.me (Crypto F&G) • FRED (T10Y2Y, HY OAS, USD Broad, DGS10)")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Paramètres")
    win = st.slider("Fenêtre (jours)", 30, 365, 180, step=15)
    smooth = st.checkbox("Lissage F&G (EMA 7j)", True)
    # nouveaux indicateurs (remplacent VIX/Put-Call)
    show_yc = st.checkbox("Courbe T10Y2Y (10Y–2Y)", True)
    show_hy = st.checkbox("HY OAS (BAMLH0A0HYM2)", True)
    show_usd = st.checkbox("USD Broad (DTWEXBGS)", True)
    show_u10 = st.checkbox("Taux 10 ans (DGS10)", False)
    st.markdown("---")
    st.write(f"🔁 Auto-refresh : {REFRESH_SECONDS}s")

# ---------- Charge F&G ----------
try:
    hist = fetch_history()
except Exception as e:
    st.error(f"Impossible de charger l'historique F&G : {e}")
    st.stop()

# ---------- Onglets ----------
tab_live, tab_hist, tab_corr = st.tabs(["Live", "Historique", "Corrélations & Composite"])

# ---------- Live ----------
with tab_live:
    col1, col2, col3 = st.columns([1.2, 1, 1])
    with col1:
        st.subheader("Index en direct")
        try:
            live = fetch_live()
            st.metric("Fear & Greed", f"{live['value']}",
                      help=f"{live['classification']}\nMàJ: {live['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}")
            # mini jauge
            fig, ax = plt.subplots(figsize=(5, 2.6))
            ax.axis("off")
            val = live["value"]
            t = np.linspace(-np.pi, 0, 300)
            ax.plot(np.cos(t), np.sin(t))
            angle = -np.pi + (val / 100) * np.pi
            ax.plot([0, np.cos(angle)], [0, np.sin(angle)], linewidth=3)
            ax.text(0, -0.2, f"{val}", ha="center", va="center", fontsize=18)
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Échec du live index : {e}")

    with col2:
        st.subheader("Derniers 90 jours — F&G")
        try:
            last = hist.tail(90).copy()
            if smooth:
                last["fng"] = last["fng"].ewm(span=7, adjust=False).mean()
            st.line_chart(last.set_index(pd.to_datetime(last["date"]))[["fng"]])
        except Exception as e:
            st.error(f"Erreur d'historique : {e}")

    with col3:
        st.subheader("Actions")
        st.download_button("⬇️ Export F&G (CSV)", data=hist.to_csv(index=False),
                           file_name="fng_history.csv", mime="text/csv")
        latest = int(hist.iloc[-1]["fng"])
        if latest <= EXTREME_FEAR:
            st.success("🟢 Zone **Extreme Fear** (contrarian bullish, hist.)")
        elif latest >= EXTREME_GREED:
            st.error("🔴 Zone **Extreme Greed** (risque de prise de bénéfices)")
        try:
            crossed = (hist["fng"].iloc[-2] > EXTREME_FEAR) and (latest <= EXTREME_FEAR)
            if crossed:
                st.toast("F&G vient d’entrer en Extreme Fear", icon="✅")
        except Exception:
            pass

# ---------- Historique (z-scores, fenêtre) ----------
with tab_hist:
    st.subheader("Vue normalisée (z-score) — indicateurs FRED")
    base = hist[["date", "fng"]].copy()

    # Ajoute les indicateurs choisis
    def merge_try(df, fetch_fn, name):
        try:
            fetched = fetch_fn()
            # Harmonise les types de date
            fetched["date"] = pd.to_datetime(fetched["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
            return df.merge(fetched, on="date", how="outer")
        except Exception as e:
            st.warning(f"{name} indisponible : {e}")
            return df

    if show_yc:  base = merge_try(base, fetch_t10y2y, "T10Y2Y")
    if show_hy:  base = merge_try(base, fetch_hy_oas, "HY OAS")
    if show_usd: base = merge_try(base, fetch_usd_broad, "USD Broad")
    if show_u10: base = merge_try(base, fetch_dgs10, "DGS10")

    base = base.sort_values("date").set_index("date").ffill().dropna(how="all")

    # z-scores
    z = pd.DataFrame(index=base.index)
    for c in [c for c in ["fng", "yc_spread", "hy_oas", "usd", "ust10"] if c in base.columns]:
        z[f"{c}_z"] = zscore(base[c])

    if len(z) == 0:
        st.info("Pas assez de données pour afficher la vue normalisée.")
    else:
        st.line_chart(z.iloc[-win:])

# ---------- Corrélations & Composite ----------
with tab_corr:
    st.subheader("Corrélations roulantes (60 j) & Score composite")

    # Reconstruit z (indépendant des autres onglets)
    base2 = hist[["date", "fng"]].copy()
    if show_yc:  base2 = merge_try(base2, fetch_t10y2y, "T10Y2Y")
    if show_hy:  base2 = merge_try(base2, fetch_hy_oas, "HY OAS")
    if show_usd: base2 = merge_try(base2, fetch_usd_broad, "USD Broad")
    if show_u10: base2 = merge_try(base2, fetch_dgs10, "DGS10")
    base2 = base2.sort_values("date").set_index("date").ffill().dropna(how="all")

    z2 = pd.DataFrame(index=base2.index)
    if "fng" in base2:      z2["fng_z"] = zscore(base2["fng"])
    if "yc_spread" in base2: z2["yc_z"]  = zscore(base2["yc_spread"])
    if "hy_oas" in base2:    z2["hy_z"]  = zscore(base2["hy_oas"])
    if "usd" in base2:       z2["usd_z"] = zscore(base2["usd"])
    if "ust10" in base2:     z2["u10_z"] = zscore(base2["ust10"])

    # Corrélations roulantes 60 j
    try:
        for name in ["yc_z", "hy_z", "usd_z", "u10_z"]:
            if name in z2.columns:
                corr = z2[["fng_z", name]].rolling(60).corr().dropna()
                st.write(f"Corrélation roulante F&G vs {name.replace('_z','').upper()} (60 j)")
                st.line_chart(corr.xs("fng_z", level=1))
    except Exception as e:
        st.warning(f"Corrélations non disponibles : {e}")

    # Composite (↑ = plus 'greed'):
    # + fng, + yc_spread (courbe + pentue = risk-on), - hy_oas (spread élevé = risk-off),
    # - usd (USD fort = risk-off), - ust10 (taux long qui s'envole = souvent risk-off)
    parts = []
    if "fng_z" in z2:   parts.append(z2["fng_z"])
    if "yc_z" in z2:    parts.append(z2["yc_z"])
    if "hy_z" in z2:    parts.append(-z2["hy_z"])
    if "usd_z" in z2:   parts.append(-z2["usd_z"])
    if "u10_z" in z2:   parts.append(-z2["u10_z"])
    if parts:
        comp = pd.concat(parts, axis=1).mean(axis=1)
        st.metric("Composite (F&G + FRED, z)", f"{comp.iloc[-1]:.2f}")
        st.line_chart(comp.iloc[-win:].to_frame("risk_score"))
    else:
        st.info("Sélectionnez au moins un indicateur FRED pour calculer le composite.")

# ---------- Footer ----------
st.divider()
