# model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from config import LABEL_HORIZON, BUY_THRESHOLD, SELL_THRESHOLD, TEST_SIZE, RANDOM_STATE, N_ESTIMATORS, MAX_DEPTH

FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_20d",
    "rsi", "sma_gap_fast_mid", "sma_gap_mid_slow", "dist_to_slow_sma",
    "vol_norm"
]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sma_cols = [c for c in out.columns if c.startswith("sma_")]
    out["ret_1d"] = out["Close"].pct_change(1)
    out["ret_5d"] = out["Close"].pct_change(5)
    out["ret_20d"] = out["Close"].pct_change(20)

    if len(sma_cols) >= 3:
        out["sma_gap_fast_mid"] = (out[sma_cols[0]] - out[sma_cols[1]]) / out["Close"]
        out["sma_gap_mid_slow"] = (out[sma_cols[1]] - out[sma_cols[2]]) / out["Close"]
        out["dist_to_slow_sma"] = (out["Close"] - out[sma_cols[2]]) / out["Close"]
    else:
        out["sma_gap_fast_mid"] = np.nan
        out["sma_gap_mid_slow"] = np.nan
        out["dist_to_slow_sma"] = np.nan

    out["vol_norm"] = (out["Volume"] - out["Volume"].rolling(20).mean()) / out["Volume"].rolling(20).std()
    # avoid infs from zero std
    out["vol_norm"] = out["vol_norm"].replace([np.inf, -np.inf], np.nan)
    return out

def make_labels(df: pd.DataFrame) -> pd.Series:
    fwd = df["Close"].shift(-LABEL_HORIZON) / df["Close"] - 1.0
    y = pd.Series(index=df.index, dtype="float64")
    y[fwd >= BUY_THRESHOLD] = 1
    y[fwd <= SELL_THRESHOLD] = -1
    y[(fwd < BUY_THRESHOLD) & (fwd > SELL_THRESHOLD)] = 0
    # drop unlabeled tail instead of casting NaN -> int
    y = y.iloc[:-LABEL_HORIZON].dropna()
    return y.astype(int)

def train_predict(df: pd.DataFrame):
    # build features first, then align to non-NaN rows
    feats = make_features(df)[FEATURE_COLUMNS]
    feats = feats.dropna()

    # labels computed on the same df; subset to feat index, then drop any NaNs
    y_all = make_labels(df)
    y = y_all.reindex(feats.index).dropna()
    X = feats.reindex(y.index)

    if len(X) < 200 or y.nunique() < 2:
        return None, None, None

    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=(y != 0).astype(int)
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # predict probabilities for the latest available row with a label-able index
    X_last = X.iloc[[-1]]
    try:
        proba = clf.predict_proba(X_last).flatten()
        class_to_idx = {c: i for i, c in enumerate(clf.classes_)}
        p_sell = float(proba[class_to_idx.get(-1, 0)])
        p_hold = float(proba[class_to_idx.get(0, 1)])
        p_buy  = float(proba[class_to_idx.get(1, 2)])
    except Exception:
        p_sell = p_hold = p_buy = None

    rec = "Sell" if (p_buy or 0) > max(p_sell or 0, p_hold or 0) else \
          ("Buy" if (p_sell or 0) > max(p_buy or 0, p_hold or 0) else "Hold")

    return clf, acc, {"p_buy": p_buy, "p_sell": p_sell, "p_hold": p_hold, "rec": rec}
