
# NIFTY 500 ML — Impulsive/Corrective + Wave Heuristics (Streamlit)

This project fetches data from **yfinance** for a list of NIFTY500 tickers, computes indicators (RSI divergence, triple SMA, support/resistance),
runs a **RandomForest** model to estimate **Buy/Sell probabilities**, and heuristically tags **Elliott-like waves** (impulsive 1–5, corrective A–C) on **daily** and **weekly** timeframes.

> Wave labeling is **heuristic** (rule-based, via a simple ZigZag). Treat as an educational starting point.

## Quick Start (Local)
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push these files to a GitHub repo.
2. Create app with `file=app.py`.
