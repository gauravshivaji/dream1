
import pandas as pd
import numpy as np

def tradingview_link(ticker: str) -> str:
    base = "https://in.tradingview.com/chart/?symbol=NSE%3A"
    sym = ticker.replace(".NS", "")
    return f"{base}{sym}"

def safe_pct_change(x):
    return x.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    rule = "W-FRI"
    o = df["Open"].resample(rule).first()
    h = df["High"].resample(rule).max()
    l = df["Low"].resample(rule).min()
    c = df["Close"].resample(rule).last()
    v = df["Volume"].resample(rule).sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    out.dropna(inplace=True)
    return out

def last_or_none(series):
    try:
        return series.dropna().iloc[-1]
    except Exception:
        return None
