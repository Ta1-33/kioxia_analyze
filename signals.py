import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def rule_based_signal(df: pd.DataFrame) -> pd.Series:
    """Returns +1 (buy), -1 (sell), 0 (hold) per day based on rules."""
    sig = pd.Series(0, index=df.index)

    # Golden/Dead cross: MA5 vs MA25
    golden = (df["MA5"] > df["MA25"]) & (df["MA5"].shift(1) <= df["MA25"].shift(1))
    dead = (df["MA5"] < df["MA25"]) & (df["MA5"].shift(1) >= df["MA25"].shift(1))

    # RSI extremes
    rsi_buy = df["RSI"] < 30
    rsi_sell = df["RSI"] > 70

    # MACD crossover
    macd_buy = (df["MACD"] > df["MACD_signal"]) & (df["MACD"].shift(1) <= df["MACD_signal"].shift(1))
    macd_sell = (df["MACD"] < df["MACD_signal"]) & (df["MACD"].shift(1) >= df["MACD_signal"].shift(1))

    # Price vs Bollinger
    bb_buy = df["Close"] < df["BB_lower"]
    bb_sell = df["Close"] > df["BB_upper"]

    # Score: each condition votes
    score = (
        golden.astype(int) * 2
        - dead.astype(int) * 2
        + rsi_buy.astype(int)
        - rsi_sell.astype(int)
        + macd_buy.astype(int)
        - macd_sell.astype(int)
        + bb_buy.astype(int)
        - bb_sell.astype(int)
    )

    sig[score >= 2] = 1
    sig[score <= -2] = -1
    return sig


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    feats["rsi"] = df["RSI"]
    feats["macd_diff"] = df["MACD"] - df["MACD_signal"]
    feats["bb_pos"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-9)
    feats["ma5_25"] = df["MA5"] / df["MA25"] - 1
    feats["ma25_75"] = df["MA25"] / df["MA75"] - 1
    feats["vol_ratio"] = df["Volume"] / df["Vol_MA20"]
    for lag in [1, 2, 3, 5]:
        feats[f"ret_{lag}"] = df["Return"].shift(lag)
    feats["rsi_lag1"] = df["RSI"].shift(1)
    feats["macd_hist"] = df["MACD_hist"]
    return feats


def ml_signal(df: pd.DataFrame):
    """Train on historical data, return per-day prediction (+1/0/-1) and OOS accuracy."""
    feats = _build_features(df)

    # Label: +1 if next 5-day return > +2%, -1 if < -2%, else 0
    fwd_ret = df["Close"].pct_change(5).shift(-5)
    labels = pd.cut(fwd_ret, bins=[-np.inf, -0.02, 0.02, np.inf], labels=[-1, 0, 1]).astype(float)

    data = pd.concat([feats, labels.rename("label")], axis=1).dropna()
    X = data.drop(columns="label")
    y = data["label"].astype(int)

    # Walk-forward: train on first 70%, evaluate rest
    split = int(len(X) * 0.7)
    if split < 50:
        return pd.Series(0, index=df.index), None, 0.0

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X.iloc[:split])
    X_te = scaler.transform(X.iloc[split:])

    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_tr, y.iloc[:split])

    acc = model.score(X_te, y.iloc[split:])

    # Predict all available rows
    X_all = scaler.transform(X)
    preds = pd.Series(model.predict(X_all), index=data.index)
    full = preds.reindex(df.index).fillna(0).astype(int)
    return full, model, acc


def predict_next_day(df: pd.DataFrame):
    """Predict next day's price using RandomForestRegressor.
    Returns dict with pred_price, pred_high, pred_low, pred_return, pred_std, or None if insufficient data.
    """
    feats = _build_features(df)
    fwd_ret = df["Close"].pct_change(1).shift(-1)

    data = pd.concat([feats, fwd_ret.rename("fwd")], axis=1).dropna()
    X = data.drop(columns="fwd")
    y = data["fwd"]

    split = int(len(X) * 0.7)
    if split < 30:
        return None

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X.iloc[:split])

    model = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42)
    model.fit(X_tr, y.iloc[:split])

    latest_feats = feats.iloc[[-1]]
    if latest_feats.isnull().any(axis=1).iloc[0]:
        return None

    X_pred = scaler.transform(latest_feats)

    # Use individual tree predictions for confidence interval
    tree_preds = np.array([tree.predict(X_pred)[0] for tree in model.estimators_])
    pred_return = float(tree_preds.mean())
    pred_std = float(tree_preds.std())

    current_price = float(df["Close"].iloc[-1])
    return {
        "pred_return": pred_return,
        "pred_price": current_price * (1 + pred_return),
        "pred_high": current_price * (1 + pred_return + pred_std),
        "pred_low": current_price * (1 + pred_return - pred_std),
        "pred_std": pred_std,
        "current_price": current_price,
    }


def backtest_next_day(df: pd.DataFrame, retrain_interval: int = 21):
    """Walk-forward backtest for next-day price prediction.
    Retrains every `retrain_interval` days. Returns a DataFrame with
    columns: actual_price, pred_price, actual_return, pred_return, correct_direction.
    """
    feats = _build_features(df)
    fwd_ret = df["Close"].pct_change(1).shift(-1)
    data = pd.concat([feats, fwd_ret.rename("fwd")], axis=1).dropna()

    min_train = int(len(data) * 0.6)
    if min_train < 30:
        return pd.DataFrame()

    records = []
    model = None
    scaler = None

    for i in range(min_train, len(data) - 1):
        # Retrain periodically
        if model is None or (i - min_train) % retrain_interval == 0:
            X_tr = data.drop(columns="fwd").iloc[:i]
            y_tr = data["fwd"].iloc[:i]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_tr)
            model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_scaled, y_tr)

        X_pred = scaler.transform(data.drop(columns="fwd").iloc[[i]])
        pred_ret = float(model.predict(X_pred)[0])

        date = data.index[i]
        actual_ret = float(data["fwd"].iloc[i])
        actual_price = float(df.loc[date, "Close"])
        pred_price = actual_price * (1 + pred_ret)
        next_actual_price = float(df["Close"].iloc[df.index.get_loc(date) + 1])

        records.append({
            "date": date,
            "actual_price": next_actual_price,
            "pred_price": pred_price,
            "actual_return": actual_ret * 100,
            "pred_return": pred_ret * 100,
            "correct_direction": (pred_ret > 0) == (actual_ret > 0),
        })

    result = pd.DataFrame(records).set_index("date")
    return result


def combined_signal(df: pd.DataFrame):
    """Returns rule, ml, combined signals and ML accuracy."""
    rule = rule_based_signal(df)
    ml, _, acc = ml_signal(df)

    # Combine: agree → strong signal, else weight
    score = rule + ml
    combined = pd.Series(0, index=df.index)
    combined[score >= 2] = 1
    combined[score <= -2] = -1
    # Single vote still counts if both methods produce something
    combined[(score == 1) & (ml == 1)] = 1
    combined[(score == -1) & (ml == -1)] = -1

    return rule, ml, combined, acc
