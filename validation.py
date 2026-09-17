"""
validation.py
--------------
Out-of-sample validation utilities. This module exists to close the two
biggest methodological gaps in a naive backtest:

  1. Parameter search picking a "winner" in-sample and reporting only
     that in-sample number, with no check on whether it holds up on
     data the search never saw (classic overfitting).
  2. Any ML component being evaluated on data whose rolling feature
     windows overlap with its own training data, which inflates
     reported accuracy without the model actually predicting anything.

It also includes a block-bootstrap hypothesis test that quantifies how
much sampling noise there is in a single-path "beat X% of random
traders" result -- one draw of market history has real variance, and
that variance should be reported, not treated as fact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def walk_forward_splits(n: int, n_splits: int = 5, gap: int = 0, min_train: int = 100):
    """
    Generate (train_idx, test_idx) index arrays for expanding-window
    walk-forward validation.

    `gap` is a purge window of excluded observations between the end of
    train and the start of test. This matters whenever features or
    labels are built from rolling/rolling-forward windows: without a
    gap, the last few training rows and the first few test rows can
    share overlapping window data, silently leaking information across
    the split and inflating out-of-sample results.
    """
    fold_size = (n - min_train - gap) // n_splits
    if fold_size <= 0:
        raise ValueError(
            f"Not enough data for {n_splits} walk-forward folds with min_train={min_train}, "
            f"gap={gap}, n={n}. Reduce n_splits/min_train/gap or supply more data."
        )
    splits = []
    for i in range(n_splits):
        train_end = min_train + i * fold_size
        test_start = train_end + gap
        test_end = test_start + fold_size if i < n_splits - 1 else n
        if test_start >= n:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, min(test_end, n))
        if len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def walk_forward_strategy_validation(
    prices: pd.DataFrame,
    fast_spans=(5, 10, 20, 30, 50),
    slow_spans=(50, 100, 200),
    n_splits: int = 5,
    gap: int = 200,
    starting_capital: float = 10_000.0,
    flat_on_sell: bool = True,
    stop_loss_grid=None,
    target_vol_grid=None,
) -> pd.DataFrame:
    """
    True walk-forward test of the EWMA parameter search:

      for each fold:
        1. Search parameter combinations using ONLY the training window.
        2. Take the combo ranked #1 by in-sample Sharpe.
        3. Backtest that exact, already-fixed combo on the held-out test
           window it was never optimized on.

    stop_loss_grid, target_vol_grid: OPTIONAL. If either is given, the
    in-fold search becomes a JOINT search over (fast, slow, stop_loss,
    target_vol) instead of just (fast, slow) -- e.g.
    stop_loss_grid=[None, 0.05, 0.08, 0.12]. This exists specifically to
    check COMPOUND overfitting: tuning several axes together against the
    same training data (as a real user experimenting with this platform
    would) and then testing the jointly-chosen combo out-of-sample, so a
    spuriously good combination found by searching multiple dimensions
    at once still has to survive a genuine held-out test -- something no
    single-axis safeguard (grid search alone, or walk-forward on spans
    alone) checks by itself.

    Returns one row per fold with both the in-sample metrics (what the
    search reported while choosing the combo) and the true out-of-sample
    metrics (what that same combo actually did on unseen data), so the
    in-sample/out-of-sample gap -- the signature of overfitting -- is
    visible directly instead of buried in one aggregate, cherry-pickable
    number.
    """
    from ewma import add_returns, add_ewma
    from signals import generate_ewma_crossover_signals
    from backtester import run_backtest
    from metrics import summary
    from optimizer import grid_search
    import itertools

    base = add_returns(prices)
    n = len(base)
    min_train = max(slow_spans) + 50
    splits = walk_forward_splits(n, n_splits=n_splits, gap=gap, min_train=min_train)

    joint_search = stop_loss_grid is not None or target_vol_grid is not None
    sl_options = stop_loss_grid if stop_loss_grid is not None else [None]
    tv_options = target_vol_grid if target_vol_grid is not None else [None]

    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
        train_df = base.iloc[train_idx]
        test_df = base.iloc[test_idx]

        if not joint_search:
            grid = grid_search(
                train_df, fast_spans=fast_spans, slow_spans=slow_spans,
                starting_capital=starting_capital, flat_on_sell=flat_on_sell,
            )
            if grid.empty:
                continue
            best = grid.iloc[0]
            fast, slow = int(best["fast_span"]), int(best["slow_span"])
            sl, tv = None, None
            in_sample_sharpe, in_sample_cagr = best["Sharpe"], best["CAGR"]
        else:
            # Joint in-sample search over (fast, slow, stop_loss, target_vol).
            best_row = None
            best_score = float("-inf")
            for fast, slow in itertools.product(fast_spans, slow_spans):
                if fast >= slow:
                    continue
                train_feats = add_ewma(train_df, fast_span=fast, slow_span=slow)
                train_signaled = generate_ewma_crossover_signals(train_feats, flat_on_sell=flat_on_sell)
                for sl_opt, tv_opt in itertools.product(sl_options, tv_options):
                    bt = run_backtest(
                        train_signaled, starting_capital=starting_capital,
                        stop_loss_pct=sl_opt, target_vol=tv_opt,
                    )
                    stats = summary(bt)
                    score = stats["Sharpe"] if not np.isnan(stats["Sharpe"]) else float("-inf")
                    if score > best_score:
                        best_score = score
                        best_row = {
                            "fast": fast, "slow": slow, "sl": sl_opt, "tv": tv_opt,
                            "Sharpe": stats["Sharpe"], "CAGR": stats["CAGR"],
                        }
            if best_row is None:
                continue
            fast, slow, sl, tv = best_row["fast"], best_row["slow"], best_row["sl"], best_row["tv"]
            in_sample_sharpe, in_sample_cagr = best_row["Sharpe"], best_row["CAGR"]

        test_feats = add_ewma(test_df, fast_span=fast, slow_span=slow)
        test_signaled = generate_ewma_crossover_signals(test_feats, flat_on_sell=flat_on_sell)
        test_bt = run_backtest(
            test_signaled, starting_capital=starting_capital,
            stop_loss_pct=sl, target_vol=tv,
        )
        oos_stats = summary(test_bt)

        rows.append({
            "fold": fold_i,
            "train_start": train_df.index[0].date(), "train_end": train_df.index[-1].date(),
            "test_start": test_df.index[0].date(), "test_end": test_df.index[-1].date(),
            "chosen_fast": fast, "chosen_slow": slow,
            "chosen_stop_loss": sl, "chosen_target_vol": tv,
            "in_sample_Sharpe": in_sample_sharpe, "in_sample_CAGR": in_sample_cagr,
            "oos_Sharpe": oos_stats["Sharpe"], "oos_CAGR": oos_stats["CAGR"],
            "oos_MaxDrawdown": oos_stats["MaxDrawdown"],
        })

    return pd.DataFrame(rows)


def block_bootstrap_returns(returns: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Resample a returns series via the circular moving-block bootstrap.
    Resampling in contiguous blocks (rather than individual iid days)
    preserves short-run autocorrelation and volatility clustering that
    a naive iid resample would destroy, which matters because EWMA
    signals are themselves driven by that short-run structure.
    """
    n = len(returns)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    pieces = [np.take(returns, np.arange(s, s + block_size) % n) for s in starts]
    return np.concatenate(pieces)[:n]


def bootstrap_hypothesis_test(
    prices: pd.DataFrame,
    fast_span: int = 20,
    slow_span: int = 50,
    starting_capital: float = 10_000.0,
    n_bootstrap: int = 100,
    n_random_traders: int = 300,
    block_size: int = 20,
    flat_on_sell: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Quantify the sampling noise in the "strategy beat X% of random
    traders" result from a single historical path.

    For each of `n_bootstrap` draws:
      1. Block-bootstrap-resample the historical Open-to-Open returns
         (preserves volatility clustering).
      2. Reconstruct a synthetic price path from the resampled returns.
      3. Recompute the EWMA signal FRESH on that synthetic path (the
         decision rule reacts to the new path, it isn't just replayed).
      4. Backtest with the same realistic next-open execution model.
      5. Compare against a fresh batch of random-trader simulations on
         the SAME synthetic path.

    Returns one percentile-vs-random-traders value per draw, so the full
    distribution -- not just one number -- can be reported.
    """
    from ewma import add_returns, add_ewma
    from signals import generate_ewma_crossover_signals
    from backtester import run_backtest, run_random_baseline

    rng = np.random.default_rng(seed)
    open_returns = prices["Open"].pct_change().dropna().values
    first_open = float(prices["Open"].iloc[0])
    dates = prices.index[: len(open_returns) + 1]

    percentiles = []
    for _ in range(n_bootstrap):
        resampled = block_bootstrap_returns(open_returns, block_size, rng)
        synth_open = first_open * np.cumprod(np.concatenate([[1.0], 1 + resampled]))
        # Close == Open in this reconstruction: the bootstrap only needs a
        # single self-consistent price series to drive the EWMA signal and
        # the next-open execution model, not a full realistic OHLC shape.
        synth_df = pd.DataFrame({"Open": synth_open, "Close": synth_open}, index=dates)

        feats = add_ewma(add_returns(synth_df), fast_span=fast_span, slow_span=slow_span)
        signaled = generate_ewma_crossover_signals(feats, flat_on_sell=flat_on_sell)
        bt = run_backtest(signaled, starting_capital=starting_capital)
        strategy_final = bt["portfolio_value"].iloc[-1]

        baseline = run_random_baseline(
            signaled, starting_capital=starting_capital,
            n_simulations=n_random_traders, seed=int(rng.integers(0, 2**31 - 1)),
        )
        final_values = baseline.iloc[-1]
        pct = (final_values < strategy_final).mean() * 100
        percentiles.append(pct)

    return pd.DataFrame({"bootstrap_draw": range(len(percentiles)), "percentile_vs_random": percentiles})


def cost_sensitivity_analysis(
    prices: pd.DataFrame,
    fast_spans=(5, 10, 20, 30, 50),
    slow_span: int = 100,
    cost_bps_grid=(0, 5, 10, 20, 40, 80),
    starting_capital: float = 10_000.0,
    flat_on_sell: bool = True,
    n_periods: int = 3,
) -> pd.DataFrame:
    """
    Directly demonstrates a real dynamic a single "28 trades over 10
    years, 0 bps default cost" backtest can't show: SMALLER EWMA spans
    trade more often, and more frequent trading is proportionally more
    exposed to transaction costs.

    `n_periods`: splits `prices` into this many contiguous,
    non-overlapping sub-periods (e.g. 3 roughly-equal multi-year chunks)
    and reports the cost/Sharpe grid separately for EACH sub-period, not
    just once on the full sample. This exists because the specific
    degradation ratio (e.g. "fast spans degrade ~3x faster") is a
    mechanical, fairly robust relationship, but a single full-sample
    number can still mask period-to-period variation -- splitting it out
    lets you check whether the ratio holds up consistently or was itself
    somewhat sample-dependent. Use groupby("fast_span")[["Sharpe"]] on
    the result, or see the __main__ block below, for an aggregate view.
    """
    from ewma import add_returns, add_ewma
    from signals import generate_ewma_crossover_signals, trade_log
    from backtester import run_backtest
    from metrics import summary

    base = add_returns(prices)
    n = len(base)
    bounds = np.linspace(0, n, n_periods + 1).astype(int)

    rows = []
    for period_i in range(n_periods):
        period_df = base.iloc[bounds[period_i]: bounds[period_i + 1]]
        if len(period_df) < slow_span + 100:
            continue

        for fast in fast_spans:
            if fast >= slow_span:
                continue
            feats = add_ewma(period_df, fast_span=fast, slow_span=slow_span)
            signaled = generate_ewma_crossover_signals(feats, flat_on_sell=flat_on_sell)
            n_trades = len(trade_log(signaled))

            for cost_bps in cost_bps_grid:
                bt = run_backtest(
                    signaled, starting_capital=starting_capital,
                    transaction_cost_bps=cost_bps, slippage_bps=0,
                )
                stats = summary(bt)
                rows.append({
                    "period": period_i + 1,
                    "period_start": period_df.index[0].date(), "period_end": period_df.index[-1].date(),
                    "fast_span": fast, "slow_span": slow_span, "cost_bps": cost_bps,
                    "num_trades": n_trades, "CAGR": stats["CAGR"], "Sharpe": stats["Sharpe"],
                    "FinalValue": stats["FinalValue"],
                })

    return pd.DataFrame(rows)


def hyperparameter_robustness_check(
    prices: pd.DataFrame,
    fast_span: int = 20,
    slow_span: int = 50,
    starting_capital: float = 10_000.0,
    flat_on_sell: bool = True,
    gap_options=(50, 100, 200),
    n_splits_options=(3, 5, 8),
    block_size_options=(10, 20, 40),
    n_bootstrap: int = 30,
) -> dict:
    """
    None of the walk-forward gap, fold count, or bootstrap block size
    used elsewhere in this project were tuned or sensitivity-tested --
    they're reasonable defaults, not validated choices. This reruns the
    walk-forward out-of-sample Sharpe across several `gap`/`n_splits`
    combinations, and the bootstrap hypothesis test across several
    `block_size` choices, so you can see whether the headline conclusions
    (e.g. "out-of-sample Sharpe collapses", "the single-run percentile is
    noisy") are STABLE across reasonable hyperparameter choices, or
    whether they were themselves sensitive to the specific numbers used.

    Returns {'walk_forward_grid': DataFrame, 'bootstrap_grid': DataFrame}.
    Deliberately kept small (n_bootstrap defaults to 30, not 100+) since
    this reruns several full validation passes -- increase for a more
    precise (but much slower) check.
    """
    from ewma import add_returns, add_ewma
    from signals import generate_ewma_crossover_signals

    wf_rows = []
    for gap in gap_options:
        for n_splits in n_splits_options:
            try:
                wf = walk_forward_strategy_validation(
                    prices, n_splits=n_splits, gap=gap,
                    starting_capital=starting_capital, flat_on_sell=flat_on_sell,
                )
            except ValueError:
                continue
            if wf.empty:
                continue
            wf_rows.append({
                "gap": gap, "n_splits": n_splits,
                "mean_oos_Sharpe": wf["oos_Sharpe"].mean(),
                "std_oos_Sharpe": wf["oos_Sharpe"].std(),
                "mean_in_sample_Sharpe": wf["in_sample_Sharpe"].mean(),
                "n_folds_evaluated": len(wf),
            })
    wf_grid = pd.DataFrame(wf_rows)

    boot_rows = []
    base = add_returns(prices)
    feats = add_ewma(base, fast_span=fast_span, slow_span=slow_span)
    signaled = generate_ewma_crossover_signals(feats, flat_on_sell=flat_on_sell)
    for block_size in block_size_options:
        boot = bootstrap_hypothesis_test(
            prices, fast_span=fast_span, slow_span=slow_span, starting_capital=starting_capital,
            n_bootstrap=n_bootstrap, n_random_traders=100, block_size=block_size, flat_on_sell=flat_on_sell,
        )
        pct = boot["percentile_vs_random"]
        boot_rows.append({
            "block_size": block_size, "mean_percentile": pct.mean(), "std_percentile": pct.std(),
            "pct_draws_above_50": (pct > 50).mean(),
        })
    boot_grid = pd.DataFrame(boot_rows)

    return {"walk_forward_grid": wf_grid, "bootstrap_grid": boot_grid}


if __name__ == "__main__":
    from data_loader import load_price_data

    prices = load_price_data("AAPL", start="2015-01-01", end="2025-01-01")

    print("=== Walk-forward strategy validation ===")
    wf = walk_forward_strategy_validation(prices, n_splits=4, gap=100)
    pd.set_option("display.width", 160)
    print(wf)
    if not wf.empty:
        print(f"\nMean in-sample Sharpe:  {wf['in_sample_Sharpe'].mean():.3f}")
        print(f"Mean out-of-sample Sharpe: {wf['oos_Sharpe'].mean():.3f}")

    print("\n=== Walk-forward strategy validation: COMPOUND search (fast/slow + stop-loss + vol-target) ===")
    wf_joint = walk_forward_strategy_validation(
        prices, n_splits=4, gap=100,
        fast_spans=(10, 20, 30), slow_spans=(50, 100),
        stop_loss_grid=[None, 0.08], target_vol_grid=[None, 0.15],
    )
    print(wf_joint[["fold", "chosen_fast", "chosen_slow", "chosen_stop_loss", "chosen_target_vol",
                     "in_sample_Sharpe", "oos_Sharpe"]])
    if not wf_joint.empty:
        print(f"\nJoint search: mean in-sample Sharpe {wf_joint['in_sample_Sharpe'].mean():.3f} "
              f"vs mean OOS Sharpe {wf_joint['oos_Sharpe'].mean():.3f} -- compare this gap to the "
              f"single-axis search above; a bigger gap here would mean compound tuning overfits more.")

    print("\n=== Bootstrap hypothesis test (20 draws, quick demo) ===")
    boot = bootstrap_hypothesis_test(prices, n_bootstrap=20, n_random_traders=100)
    print(boot["percentile_vs_random"].describe())

    print("\n=== Cost sensitivity across 3 sub-periods: does the degradation ratio hold up? ===")
    cost_sens = cost_sensitivity_analysis(prices, fast_spans=(5, 20, 50), cost_bps_grid=(0, 20, 80), n_periods=3)
    for period in sorted(cost_sens["period"].unique()):
        sub = cost_sens[cost_sens["period"] == period]
        print(f"\n-- Period {period} ({sub['period_start'].iloc[0]} to {sub['period_end'].iloc[0]}) --")
        print(sub.pivot(index="fast_span", columns="cost_bps", values="Sharpe"))

    print("\n=== Hyperparameter robustness check (small demo grid) ===")
    robustness = hyperparameter_robustness_check(
        prices, gap_options=(100, 200), n_splits_options=(3, 5),
        block_size_options=(10, 20), n_bootstrap=15,
    )
    print("\nWalk-forward OOS Sharpe across gap/n_splits choices:")
    print(robustness["walk_forward_grid"])
    print("\nBootstrap percentile across block_size choices:")
    print(robustness["bootstrap_grid"])
