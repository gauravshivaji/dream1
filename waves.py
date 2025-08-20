
import pandas as pd
import numpy as np
from config import ZIGZAG_PCT

def _zigzag(close: pd.Series, pct: float) -> pd.DataFrame:
    if close.empty:
        return pd.DataFrame(columns=["type", "price"])
    pivots = []
    last_pivot_idx = close.index[0]
    last_pivot_price = close.iloc[0]
    uptrend = None

    for i in range(1, len(close)):
        price = close.iloc[i]
        if uptrend is None:
            uptrend = price > last_pivot_price
            continue
        if uptrend and price < last_pivot_price * (1 - pct/100.0):
            pivots.append((close.index[i-1], "H", close.iloc[i-1]))
            last_pivot_idx = close.index[i-1]
            last_pivot_price = close.iloc[i-1]
            uptrend = False
        elif (not uptrend) and price > last_pivot_price * (1 + pct/100.0):
            pivots.append((close.index[i-1], "L", close.iloc[i-1]))
            last_pivot_idx = close.index[i-1]
            last_pivot_price = close.iloc[i-1]
            uptrend = True
        else:
            if uptrend and price > last_pivot_price:
                last_pivot_price = price
                last_pivot_idx = close.index[i]
            elif (not uptrend) and price < last_pivot_price:
                last_pivot_price = price
                last_pivot_idx = close.index[i]
    pivots.append((last_pivot_idx, "H" if uptrend else "L", last_pivot_price))
    piv = pd.DataFrame(pivots, columns=["ts", "type", "price"]).set_index("ts")
    return piv

def label_waves(close: pd.Series) -> dict:
    piv = _zigzag(close, ZIGZAG_PCT)
    if len(piv) < 4:
        return {"state": None, "wave": None}
    recent_close = close.iloc[-1]
    prev_close = close.iloc[-20] if len(close) > 20 else close.iloc[0]
    uptrend = recent_close > prev_close
    if uptrend:
        num_legs = len(piv)
        wave_num = (num_legs % 5) or 5
        return {"state": "Impulsive", "wave": str(wave_num)}
    else:
        abc = ["A","B","C"]
        wave = abc[(len(piv)-1) % 3]
        return {"state": "Corrective", "wave": wave}
