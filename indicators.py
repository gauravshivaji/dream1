
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator

from config import RSI_PERIOD, DIVERGENCE_LOOKBACK, FAST_SMA, MID_SMA, SLOW_SMA

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[f"sma_{FAST_SMA}"] = out["Close"].rolling(FAST_SMA).mean()
    out[f"sma_{MID_SMA}"]  = out["Close"].rolling(MID_SMA).mean()
    out[f"sma_{SLOW_SMA}"] = out["Close"].rolling(SLOW_SMA).mean()

    rsi = RSIIndicator(close=out["Close"], window=RSI_PERIOD).rsi()
    out["rsi"] = rsi

    def triple_state(row):
        a, b, c = row[f"sma_{FAST_SMA}"], row[f"sma_{MID_SMA}"], row[f"sma_{SLOW_SMA}"]
        if pd.isna(a) or pd.isna(b) or pd.isna(c):
            return None
        if a > b > c:
            return "Bullish"
        if a < b < c:
            return "Bearish"
        return "Mixed"
    out["triple_sma"] = out.apply(triple_state, axis=1)

    lookback = 20
    out["support"] = out["Low"].rolling(lookback).min()
    out["resistance"] = out["High"].rolling(lookback).max()

    out["trend"] = np.where(out["Close"] > out[f"sma_{SLOW_SMA}"], "Up", "Down")

    out["rsi_divergence"] = None
    w = DIVERGENCE_LOOKBACK

    def swing_high(s):
        mid = s.index[w//2]
        return s[mid] == s.max()

    def swing_low(s):
        mid = s.index[w//2]
        return s[mid] == s.min()

    price_high = out["High"].rolling(w, center=True).apply(lambda s: 1.0 if swing_high(s) else 0.0, raw=False)
    price_low  = out["Low"].rolling(w, center=True).apply(lambda s: 1.0 if swing_low(s) else 0.0, raw=False)
    rsi_high   = out["rsi"].rolling(w, center=True).apply(lambda s: 1.0 if s.iloc[w//2] == s.max() else 0.0, raw=False)
    rsi_low    = out["rsi"].rolling(w, center=True).apply(lambda s: 1.0 if s.iloc[w//2] == s.min() else 0.0, raw=False)

    def label_divergence(idx):
        ph_idx = list(out.index[(price_high == 1.0) & (out.index <= idx)])
        pl_idx = list(out.index[(price_low == 1.0) & (out.index <= idx)])
        rh_idx = list(out.index[(rsi_high == 1.0) & (out.index <= idx)])
        rl_idx = list(out.index[(rsi_low == 1.0) & (out.index <= idx)])

        div = None
        if len(ph_idx) >= 2 and len(rh_idx) >= 2:
            p1, p2 = out.loc[ph_idx[-2], "High"], out.loc[ph_idx[-1], "High"]
            r1, r2 = out.loc[rh_idx[-2], "rsi"], out.loc[rh_idx[-1], "rsi"]
            if p2 > p1 and r2 < r1:
                div = "Bearish"
        if len(pl_idx) >= 2 and len(rl_idx) >= 2:
            p1, p2 = out.loc[pl_idx[-2], "Low"], out.loc[pl_idx[-1], "Low"]
            r1, r2 = out.loc[rl_idx[-2], "rsi"], out.loc[rl_idx[-1], "rsi"]
            if p2 < p1 and r2 > r1:
                div = "Bullish"
        return div

    out["rsi_divergence"] = [label_divergence(ts) for ts in out.index]
    return out
