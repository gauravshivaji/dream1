# model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import (
    LABEL_HORIZON,
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    TEST_SIZE,
    RANDOM_STATE,
    N_ESTIMATORS,
    MAX_DEPTH,
)

# Features we will use
FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "rsi",
    "sma_gap_fast_mid",
    "sma_gap_mid_slow",
    "dist_to_slow_sma",
    "vol_norm",
]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical features from OHLCV data."""
    out = df.copy()

    sma_cols = [c for c in out.columns if c.startswith("sma_")]

    # returns
    out["ret_1d"] = out["Close"].pct_change(1)
    out["ret_5d"] = out["Close"].pct_change(5)
    out["ret_20d"] = out["Close"].pct_change(20)

    # SMA-based features
    if len(sma_cols) >= 3:
        out["sma_gap_fast_mid"] = (out[sma_cols[0]] - out[sma_cols[1]]) / out["Close"]
        out["sma_gap_mid_slow"] = (out[sma_cols[1]] - out[sma_cols[2]]) / out["Close"]
        out["dist_to_slow_sma"] = (out["Close"] - out[sma_cols[2]]) / out["Close"]
    else:
        out["sma_gap_fast_mid"] = np.nan
        out["sma_gap_mid_slow"] = np.nan
        out["dist_to_slow_sma"] = np.nan

    # Volume normalization (z-score)
    out["vol_norm"] = (
        (out["Volume"] - out["Volume"].rolling(20).mean())
        / out["Volume"].rolling(20).std()
    )
    out["vol_norm"] = out["vol_norm"].replace([np.inf, -np.inf], np.nan)

    return out


def make_labels(df: pd.DataFrame) -> pd.Series:
    """Label data into Buy(1), Hold(0), Sell(-1) based on forward returns."""
    fwd = df["Close"].shift(-LABEL_HORIZON) / df["Close"] - 1.0
    y = pd.Series(index=df.index, dtype="float64")

    y[fwd >= BUY_THRESHOLD] = 1
    y[fwd <= SELL_THRESHOLD] = -1
    y[(fwd < BUY_THRESHOLD) & (fwd > SELL_THRESHOLD)] = 0

    # drop tail (NaN because of shift)
    y = y.iloc[:-LABEL_HORIZON].dropna()

    return y.astype(int)


def train_predict(df: pd.DataFrame):
    """Train model, compute accuracy, and return predictions."""
    feats = make_features(df)[FEATURE_COLUMNS].dropna()
    labels = make_labels(df)

    y = labels.reindex(feats.index).dropna()
    X = feats.reindex(y.index)

    # Not enough samples or only 1 class
    if len(X) < 200 or y.nunique() < 2:
        return None, None, None

    # Improved RandomForest
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Predict probabilities for last row
    X_last = X.iloc[[-1]]
    try:
        proba = clf.predict_proba(X_last).flatten()
        class_to_idx = {c: i for i, c in enumerate(clf.classes_)}
        p_sell = float(proba[class_to_idx.get(-1, 0)])
        p_hold = float(proba[class_to_idx.get(0, 1)])
        p_buy = float(proba[class_to_idx.get(1, 2)])
    except Exception:
        p_sell = p_hold = p_buy = None

    # Recommendation logic
    rec = (
        "Buy" if (p_buy or 0) > max(p_sell or 0, p_hold or 0)
        else "Sell" if (p_sell or 0) > max(p_buy or 0, p_hold or 0)
        else "Hold"
    )

    return clf, acc, {"p_buy": p_buy, "p_sell": p_sell, "p_hold": p_hold, "rec": rec}
