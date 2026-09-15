"""
regime_model.py
----------------
Step 9 of the spec: an AI component that predicts market regime
(Bull / Bear / High Volatility / Low Volatility) from features like
volatility, volume, RSI, and trend, and can be used to switch EWMA
parameter sets adaptively.

Label design history (read before trusting any accuracy number here):

v1: labeled each row using the SAME trailing trend/volatility values fed
    to the model as input features. Circular -- the "model" reconstructed
    a threshold rule from data handed to it for free, hence ~100% accuracy.

v2: fixed the circularity by labeling from returns STRICTLY AFTER the
    feature date, evaluated with walk-forward + a purge gap. BUT the
    Bull/Bear/HighVol/LowVol threshold (the median of forward volatility)
    was computed over the WHOLE dataset before splitting -- so the
    definition of "high vs low volatility" for a 2016 test fold had
    already been informed by volatility data through 2025. A purge gap
    stops the MODEL from training on test rows; it does nothing to stop
    a globally-computed THRESHOLD from leaking future information into
    how test rows get labeled in the first place.

v3 (here): the volatility threshold is now computed PER FOLD, from the
    training portion of that fold only, and applied to label both that
    fold's train and test rows. A test fold's labels no longer depend on
    any data outside (train ∪ purge-gap ∪ that fold's own test window).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


REGIME_PARAMS = {
    "Bull": {"fast_span": 20, "slow_span": 50},
    "Bear": {"fast_span": 10, "slow_span": 30},
    "HighVol": {"fast_span": 5, "slow_span": 30},
    "LowVol": {"fast_span": 30, "slow_span": 100},
}

FEATURE_COLS = ["volatility", "volume_z", "rsi", "trend"]


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Standard RSI (Wilder-style, simplified with rolling mean)."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def build_regime_features(df: pd.DataFrame, vol_window: int = 20, trend_window: int = 20) -> pd.DataFrame:
    """
    Build the model's INPUT features: trailing (backward-looking only)
    volatility, volume z-score, RSI, and trend, each using only data
    available up to and including the current row.
    """
    out = df.copy()
    if "returns" not in out.columns:
        out["returns"] = out["Close"].pct_change()

    out["volatility"] = out["returns"].rolling(vol_window).std() * np.sqrt(252)
    out["volume_z"] = (out["Volume"] - out["Volume"].rolling(vol_window).mean()) / out["Volume"].rolling(vol_window).std()
    out["rsi"] = compute_rsi(out["Close"])
    out["trend"] = out["Close"].pct_change(trend_window)

    return out


def compute_forward_stats(df: pd.DataFrame, horizon: int = 20) -> tuple[pd.Series, pd.Series]:
    """
    Forward-looking volatility and trend, built ONLY from returns
    strictly AFTER each row (t+1 .. t+horizon) -- never from the same
    trailing window used as input features. Returns (fwd_vol, fwd_trend);
    labeling still needs a volatility THRESHOLD, deliberately kept
    separate here (see label_from_forward_stats) so that threshold can
    be computed per-fold instead of globally.
    """
    close = df["Close"]
    returns = close.pct_change()

    future_returns = pd.concat([returns.shift(-k) for k in range(1, horizon + 1)], axis=1)
    fwd_vol = future_returns.std(axis=1) * np.sqrt(252)
    fwd_trend = close.shift(-horizon) / close - 1.0
    return fwd_vol, fwd_trend


def label_from_forward_stats(fwd_vol: pd.Series, fwd_trend: pd.Series, vol_threshold: float) -> pd.Series:
    """
    Apply the Bull/Bear/HighVol/LowVol rule using an EXTERNALLY SUPPLIED
    volatility threshold, rather than computing the threshold from the
    same data being labeled. Callers responsible for out-of-sample
    integrity (e.g. train_regime_classifier) pass a threshold computed
    only from training-fold data.
    """
    labels = pd.Series(index=fwd_vol.index, dtype="object")
    bull = (fwd_trend > 0) & (fwd_vol <= vol_threshold)
    highvol = (fwd_trend > 0) & (fwd_vol > vol_threshold)
    lowvol = (fwd_trend <= 0) & (fwd_vol <= vol_threshold)
    bear = (fwd_trend <= 0) & (fwd_vol > vol_threshold)
    labels[bull] = "Bull"
    labels[highvol] = "HighVol"
    labels[lowvol] = "LowVol"
    labels[bear] = "Bear"
    return labels


def label_future_regimes(df: pd.DataFrame, horizon: int = 20) -> pd.Series:
    """
    Convenience, SINGLE-SHOT version of the forward-looking label, using
    a threshold computed from the WHOLE series. Fine for a quick, casual
    look at what the labels look like -- NOT used by train_regime_classifier
    (which uses a fold-local threshold instead, see module docstring) and
    should not be used to report an accuracy number.
    """
    fwd_vol, fwd_trend = compute_forward_stats(df, horizon=horizon)
    return label_from_forward_stats(fwd_vol, fwd_trend, vol_threshold=fwd_vol.median())


def train_regime_classifier(
    df: pd.DataFrame,
    horizon: int = 20,
    n_splits: int = 5,
    gap: int | None = None,
    random_state: int = 42,
):
    """
    Train and evaluate the regime classifier using walk-forward folds
    with a purge gap AND a per-fold label threshold (see module
    docstring for why the threshold has to be fold-local too).

    Also reports a MAJORITY-CLASS baseline accuracy per fold alongside
    the model's accuracy: comparing to a flat 1/n_classes chance level
    is only valid if the classes are balanced, and forward-looking
    regime labels usually aren't (e.g. more "Bull" rows than "LowVol"
    ones in an uptrending series). If the model can't beat "always
    predict the fold's most common training-set class", it has no
    demonstrated skill regardless of how its accuracy compares to 25%.

    Returns (model, feature_cols, report_str). `model` is refit on the
    LAST fold's training set, for use in predict_current_regime().
    """
    from validation import walk_forward_splits

    gap = gap if gap is not None else horizon

    feats = build_regime_features(df)
    fwd_vol, fwd_trend = compute_forward_stats(feats, horizon=horizon)

    valid = feats[FEATURE_COLS].notna().all(axis=1) & fwd_vol.notna() & fwd_trend.notna()
    X = feats.loc[valid, FEATURE_COLS]
    fwd_vol, fwd_trend = fwd_vol[valid], fwd_trend[valid]

    splits = walk_forward_splits(len(X), n_splits=n_splits, gap=gap, min_train=max(200, 5 * horizon))

    accuracies, majority_accuracies = [], []
    last_report = "(no folds evaluated -- not enough data)"
    last_model = None
    class_counts_overall = pd.Series(dtype=int)

    for train_idx, test_idx in splits:
        if len(train_idx) < 50 or len(test_idx) < 10:
            continue

        # Fold-local threshold: computed ONLY from this fold's training
        # rows, then applied to label both train and test rows. Test
        # labels never depend on data outside (train U gap U test).
        vol_threshold = fwd_vol.iloc[train_idx].median()
        y = label_from_forward_stats(fwd_vol, fwd_trend, vol_threshold)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=random_state)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, preds))

        majority_class = y_train.value_counts().idxmax()
        majority_preds = pd.Series(majority_class, index=y_test.index)
        majority_accuracies.append(accuracy_score(y_test, majority_preds))

        last_report = classification_report(y_test, preds, zero_division=0)
        last_model = model
        class_counts_overall = y.value_counts()

    n_classes = len(class_counts_overall) if len(class_counts_overall) else 4
    chance = 1.0 / n_classes

    if accuracies:
        class_dist_str = ", ".join(f"{k}={v} ({v / class_counts_overall.sum():.0%})"
                                    for k, v in class_counts_overall.items())
        header = (
            f"Walk-forward accuracy across {len(accuracies)} folds "
            f"(forward horizon={horizon}d, purge gap={gap}d, PER-FOLD label threshold):\n"
            f"  model accuracy:    mean = {np.mean(accuracies):.1%}   std = {np.std(accuracies):.1%}\n"
            f"  majority-class baseline: mean = {np.mean(majority_accuracies):.1%}   "
            f"std = {np.std(majority_accuracies):.1%}\n"
            f"  model - majority baseline: {np.mean(accuracies) - np.mean(majority_accuracies):+.1%}\n"
            f"  per-fold model accuracy = {[f'{a:.1%}' for a in accuracies]}\n"
            f"  per-fold majority baseline = {[f'{a:.1%}' for a in majority_accuracies]}\n"
            f"  (naive 1/{n_classes}-class chance level = {chance:.1%}; class distribution "
            f"(last fold's full label set): {class_dist_str})\n\n"
            f"Classification report for the most recent fold:\n"
        )
    else:
        header = "No valid walk-forward folds (not enough data for the requested horizon/gap/n_splits).\n"

    report = header + last_report
    return last_model, FEATURE_COLS, report


def predict_current_regime(model, feature_cols: list[str], latest_row: pd.Series) -> str:
    """
    Predict the regime expected over the NEXT `horizon` days, using only
    today's trailing (already-observable) features.
    """
    X = latest_row[feature_cols].to_frame().T
    return model.predict(X)[0]


def adaptive_ewma_params(regime: str) -> dict:
    """Look up the EWMA fast/slow spans recommended for a given regime."""
    return REGIME_PARAMS.get(regime, {"fast_span": 20, "slow_span": 50})


def backtest_regime_adaptive_strategy(
    prices: pd.DataFrame,
    horizon: int = 20,
    n_splits: int = 5,
    gap: int | None = None,
    starting_capital: float = 10_000.0,
    flat_on_sell: bool = True,
    random_state: int = 42,
) -> dict:
    """
    Actually backtest the "suggested EWMA params" idea, instead of just
    displaying a forecast and leaving it at that.

    For each walk-forward fold: train the classifier on the fold's
    training data (fold-safe, as in train_regime_classifier), then for
    every `horizon`-day chunk of the TEST period, predict the regime
    using only features available at the start of that chunk, pick
    EWMA params via adaptive_ewma_params(), and use that chunk's slice
    of a precomputed signal for those params. Splicing together every
    fold's test-period chunks gives one continuous out-of-sample
    "adaptive" equity curve, which is compared against a STATIC
    (fast=20, slow=50) baseline over the exact same combined
    out-of-sample period -- an apples-to-apples test of whether
    switching params on the model's forecasts actually helped.

    Returns a dict with 'adaptive_summary', 'static_summary', and the
    spliced 'adaptive_bt' / 'static_bt' DataFrames (test-period rows only).
    """
    from ewma import add_returns, add_ewma
    from signals import generate_ewma_crossover_signals
    from backtester import run_backtest
    from metrics import summary
    from validation import walk_forward_splits

    gap = gap if gap is not None else horizon
    base = add_returns(prices)

    feats_full = build_regime_features(base)
    fwd_vol, fwd_trend = compute_forward_stats(feats_full, horizon=horizon)
    valid = feats_full[FEATURE_COLS].notna().all(axis=1) & fwd_vol.notna() & fwd_trend.notna()

    # Precompute one signal series per distinct (fast, slow) combo used
    # across all regimes, so per-chunk selection is just a lookup.
    unique_params = {(p["fast_span"], p["slow_span"]) for p in REGIME_PARAMS.values()}
    unique_params.add((20, 50))  # static baseline combo
    signal_cache = {}
    for fast, slow in unique_params:
        feats = add_ewma(base, fast_span=fast, slow_span=slow)
        signaled = generate_ewma_crossover_signals(feats, flat_on_sell=flat_on_sell)
        signal_cache[(fast, slow)] = signaled["signal"]

    n = len(base)
    splits = walk_forward_splits(n, n_splits=n_splits, gap=gap, min_train=max(200, 5 * horizon))

    adaptive_signal = pd.Series(0, index=base.index, dtype=float)
    test_mask = pd.Series(False, index=base.index)

    X_full = feats_full[FEATURE_COLS]

    for train_idx, test_idx in splits:
        train_idx_valid = [i for i in train_idx if valid.iloc[i]]
        if len(train_idx_valid) < 50 or len(test_idx) < horizon:
            continue

        vol_threshold = fwd_vol.iloc[train_idx_valid].median()
        y_train = label_from_forward_stats(fwd_vol.iloc[train_idx_valid], fwd_trend.iloc[train_idx_valid], vol_threshold)
        X_train = X_full.iloc[train_idx_valid]

        model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=random_state)
        model.fit(X_train, y_train)

        test_dates = base.index[test_idx]
        test_mask.loc[test_dates] = True

        # Walk the test period in horizon-sized chunks; forecast once per
        # chunk using only the feature row at the chunk's start (causal).
        for chunk_start in range(0, len(test_idx), horizon):
            chunk_idx = test_idx[chunk_start: chunk_start + horizon]
            if len(chunk_idx) == 0:
                continue
            forecast_row_idx = chunk_idx[0]
            if not valid.iloc[forecast_row_idx]:
                chosen = (20, 50)
            else:
                x_row = X_full.iloc[[forecast_row_idx]]
                regime = model.predict(x_row)[0]
                params = adaptive_ewma_params(regime)
                chosen = (params["fast_span"], params["slow_span"])

            chunk_dates = base.index[chunk_idx]
            adaptive_signal.loc[chunk_dates] = signal_cache[chosen].loc[chunk_dates]

    if not test_mask.any():
        return {"adaptive_summary": {}, "static_summary": {}, "adaptive_bt": pd.DataFrame(), "static_bt": pd.DataFrame()}

    # Backtest the spliced adaptive signal and the static baseline, both
    # restricted to the SAME combined out-of-sample rows for a fair
    # comparison, but each keeps full history before it for the EWMA/
    # execution lag to be computed correctly, then we slice metrics to
    # the test rows only.
    adaptive_full = base.copy()
    adaptive_full["signal"] = adaptive_signal
    adaptive_bt_full = run_backtest(adaptive_full, starting_capital=starting_capital)

    static_full = base.copy()
    static_full["signal"] = signal_cache[(20, 50)]
    static_bt_full = run_backtest(static_full, starting_capital=starting_capital)

    adaptive_bt = adaptive_bt_full.loc[test_mask]
    static_bt = static_bt_full.loc[test_mask]

    # Re-base portfolio value to starting_capital at the start of the
    # (non-contiguous) test-only slice so CAGR/Sharpe reflect only the
    # out-of-sample period, not fold gaps.
    def _rebased_summary(bt_slice):
        rebased = bt_slice.copy()
        rebased["portfolio_value"] = starting_capital * (1 + rebased["strategy_return"].fillna(0)).cumprod()
        return summary(rebased)

    return {
        "adaptive_summary": _rebased_summary(adaptive_bt),
        "static_summary": _rebased_summary(static_bt),
        "adaptive_bt": adaptive_bt,
        "static_bt": static_bt,
    }


if __name__ == "__main__":
    from data_loader import load_price_data

    prices = load_price_data("AAPL", start="2015-01-01", end="2025-01-01")
    model, cols, report = train_regime_classifier(prices)
    print(report)

    feats = build_regime_features(prices).dropna(subset=cols)
    latest_regime = predict_current_regime(model, cols, feats.iloc[-1])
    print(f"\nForecast regime for the next 20 trading days: {latest_regime}")
    print(f"Suggested EWMA params: {adaptive_ewma_params(latest_regime)}")

    print("\n=== Does actually USING the regime forecast help? (out-of-sample) ===")
    result = backtest_regime_adaptive_strategy(prices)
    import pandas as pd
    print(pd.DataFrame({"Adaptive (regime-switching)": result["adaptive_summary"],
                         "Static (fixed 20/50)": result["static_summary"]}).T)
