# model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from config import LABEL_HORIZON, BUY_THRESHOLD, SELL_THRESHOLD, TEST_SIZE, RANDOM_STATE, N_ESTIMATORS, MAX_DEPTH

FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_20d",
    "rsi", "sma_gap_fast_mid", "sma_gap_mid_slow", "dist_to_slow_sma",
    "vol_norm", "macd", "bollinger_width", "momentum"
]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Returns
    out["ret_1d"] = out["Close"].pct_change(1)
    out["ret_5d"] = out["Close"].pct_change(5)
    out["ret_20d"] = out["Close"].pct_change(20)

    # SMA gaps
    sma_cols = [c for c in out.columns if c.startswith("sma_")]
    if len(sma_cols) >= 3:
        out["sma_gap_fast_mid"] = (out[sma_cols[0]] - out[sma_cols[1]]) / out["Close"]
        out["sma_gap_mid_slow"] = (out[sma_cols[1]] - out[sma_cols[2]]) / out["Close"]
        out["dist_to_slow_sma"] = (out["Close"] - out[sma_cols[2]]) / out["Close"]
    else:
        out["sma_gap_fast_mid"] = np.nan
        out["sma_gap_mid_slow"] = np.nan
        out["dist_to_slow_sma"] = np.nan

    # Normalized Volume
    out["vol_norm"] = (out["Volume"] - out["Volume"].rolling(20).mean()) / out["Volume"].rolling(20).std()
    out["vol_norm"] = out["vol_norm"].replace([np.inf, -np.inf], np.nan)

    # MACD
    exp1 = out["Close"].ewm(span=12, adjust=False).mean()
    exp2 = out["Close"].ewm(span=26, adjust=False).mean()
    out["macd"] = exp1 - exp2

    # Bollinger Bands Width
    rolling_mean = out["Close"].rolling(window=20).mean()
    rolling_std = out["Close"].rolling(window=20).std()
    out["bollinger_width"] = (rolling_std * 2) / rolling_mean

    # Momentum
    out["momentum"] = out["Close"] / out["Close"].shift(10) - 1

    return out

def make_labels(df: pd.DataFrame) -> pd.Series:
    fwd = df["Close"].shift(-LABEL_HORIZON) / df["Close"] - 1.0
    y = pd.Series(index=df.index, dtype="float64")
    y[fwd >= BUY_THRESHOLD] = 1
    y[fwd <= SELL_THRESHOLD] = -1
    y[(fwd < BUY_THRESHOLD) & (fwd > SELL_THRESHOLD)] = 0
    y = y.iloc[:-LABEL_HORIZON].dropna()
    return y.astype(int)

def train_predict(df: pd.DataFrame):
    feats = make_features(df)[FEATURE_COLUMNS]
    feats = feats.dropna()

    y_all = make_labels(df)
    y = y_all.reindex(feats.index).dropna()
    X = feats.reindex(y.index)

    if len(X) < 100 or y.nunique() < 2:  # require min data + >1 class
        return None, None, None

    # Handle class imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )
    class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

    # Train-Test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError:
        return None, None, None

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS * 2,
        max_depth=MAX_DEPTH if MAX_DEPTH else None,
        random_state=RANDOM_STATE,
        class_weight=class_weight_dict,
        max_features="sqrt",
        min_samples_split=5,
        min_samples_leaf=3,
        bootstrap=True
    )

    # Cross-validation (only if enough samples)
    cv_acc = None
    if len(X) >= 5:
        try:
            cv_scores = cross_val_score(clf, X, y, cv=min(5, len(y)), scoring="accuracy")
            cv_acc = np.nanmean(cv_scores)
        except Exception:
            pass

    # Train & test
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    # Final accuracy
    if cv_acc is not None and not np.isnan(cv_acc):
        acc = (0.7 * cv_acc) + (0.3 * test_acc)
    else:
        acc = test_acc

    # Prediction for last row
    X_last = X.iloc[[-1]]
    try:
        proba = clf.predict_proba(X_last).flatten()
        class_to_idx = {c: i for i, c in enumerate(clf.classes_)}
        p_sell = float(proba[class_to_idx.get(-1, 0)])
        p_hold = float(proba[class_to_idx.get(0, 1)])
        p_buy  = float(proba[class_to_idx.get(1, 2)])
    except Exception:
        p_sell = p_hold = p_buy = None

    rec = "Buy" if (p_buy or 0) > max(p_sell or 0, p_hold or 0) else \
          ("Sell" if (p_sell or 0) > max(p_buy or 0, p_hold or 0) else "Hold")

    return clf, round(acc, 4), {"p_buy": p_buy, "p_sell": p_sell, "p_hold": p_hold, "rec": rec}
