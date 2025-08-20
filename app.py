
import streamlit as st
import pandas as pd
import numpy as np

from config import *
from data import fetch_ohlcv
from indicators import compute_indicators
from waves import label_waves
from model import train_predict
from utils import tradingview_link, resample_to_weekly

st.set_page_config(page_title="NIFTY500 Waves & ML", layout="wide")
st.title("📈 NIFTY 500 — Impulsive/Corrective + Wave (Heuristics) + ML")

st.sidebar.header("Settings")
period = st.sidebar.selectbox("History", ["1y", "2y", "5y"], index=1)

st.sidebar.markdown("### Tickers")
default_tickers = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ITC.NS"]
user_text = st.sidebar.text_area("Paste tickers (.NS) separated by commas", value=",".join(default_tickers))
tickers = [t.strip() for t in user_text.split(",") if t.strip()]

st.sidebar.info("Upload a CSV with a column **Ticker** for full NIFTY500 list (optional).")
file = st.sidebar.file_uploader("nifty500.csv", type=["csv"])
if file is not None:
    try:
        df_csv = pd.read_csv(file)
        if "Ticker" in df_csv.columns:
            tickers = df_csv["Ticker"].dropna().astype(str).tolist()
    except Exception as e:
        st.sidebar.error(f"CSV read error: {e}")

if st.sidebar.button("Run Scan"):
    rows = []
    progress = st.progress(0.0)
    for i, t in enumerate(tickers, start=1):
        try:
            df = fetch_ohlcv(t, period=period, interval="1d")
            idf = compute_indicators(df)
            clf, acc, proba = train_predict(idf)

            w_daily = label_waves(idf["Close"])
            df_w = resample_to_weekly(idf[["Open","High","Low","Close","Volume"]])
            w_week = label_waves(df_w["Close"])

            price = float(idf["Close"].iloc[-1])
            tv = tradingview_link(t)

            rows.append({
                "Ticker": t,
                "Price": round(price, 2),
                "TradingView": f'<a href="{tv}" target="_blank">Chart</a>',
                "Prob_Buy": None if proba is None else round((proba.get("p_buy") or 0.0)*100, 1),
                "Prob_Sell": None if proba is None else round((proba.get("p_sell") or 0.0)*100, 1),
                "ML_Recommendation": None if proba is None else proba.get("rec"),
                "Daily_Wave": None if w_daily is None else w_daily.get("wave"),
                "Daily_State": None if w_daily is None else w_daily.get("state"),
                "Weekly_Wave": None if w_week is None else w_week.get("wave"),
                "Weekly_State": None if w_week is None else w_week.get("state"),
                "RSI_Divergence": idf["rsi_divergence"].iloc[-1],
                "TripleSMA_Signal": idf["triple_sma"].iloc[-1],
                "Support": float(idf["support"].iloc[-1]) if pd.notna(idf["support"].iloc[-1]) else None,
                "Resistance": float(idf["resistance"].iloc[-1]) if pd.notna(idf["resistance"].iloc[-1]) else None,
                "Trend": idf["trend"].iloc[-1]
            })
        except Exception as e:
            rows.append({"Ticker": t, "ML_Recommendation": f"ERR: {e}"})
        progress.progress(i/len(tickers))

    ml_df = pd.DataFrame(rows)
    if not ml_df.empty:
        cols = ["Ticker","Price","TradingView","Prob_Buy","Prob_Sell","ML_Recommendation",
            "Daily_Wave","Daily_State","Weekly_Wave","Weekly_State",
            "RSI_Divergence","TripleSMA_Signal","Support","Resistance","Trend"]
        ml_df = ml_df.reindex(columns=cols)

        st.markdown("### Results")

    # Convert TradingView HTML link to Markdown clickable link
        ml_df["TradingView"] = ml_df["TradingView"].str.replace( r'<a href="(.*?)" target="_blank">Chart</a>',r'[\g<0>](\1)', regex=True  )

    # Show interactive table with sorting
        st.dataframe(
            ml_df,
            column_config={
            "TradingView": st.column_config.LinkColumn("TradingView", display_text="Chart")
        },
        use_container_width=True   )

    # Download option remains same
        st.download_button("Download CSV", data=ml_df.to_csv(index=False), file_name="scan_results.csv", mime="text/csv")
    
  else:
        st.warning("No results.")
else:
    st.write("➡️ Configure settings and click **Run Scan**.")
