"""
dashboard/app.py
-----------------
Step 10 of the spec: the final Streamlit dashboard.

Run with:
    streamlit run dashboard/app.py

Lets the user pick an asset, date range, and fast/slow EWMA spans, then
runs the full pipeline (data -> EWMA -> signals -> backtest -> metrics)
and displays results with charts, a metrics table, a trade log, and
optional deeper-validation panels: execution-model comparison, a
block-bootstrap hypothesis test (with a full distribution, not one
number), walk-forward out-of-sample validation, parameter grid search,
and the ML regime forecaster.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data_loader import load_price_data
from ewma import build_feature_frame
from signals import generate_ewma_crossover_signals, trade_log
from backtester import run_backtest, run_random_baseline, compare_execution_models
from metrics import summary, summary_table, cagr, excess_cagr, sharpe_ratio, max_drawdown, annualized_volatility
from visualize import plot_price_and_ewma, plot_signals, plot_portfolio_growth, plot_drawdown
from optimizer import grid_search
from validation import walk_forward_strategy_validation, bootstrap_hypothesis_test, cost_sensitivity_analysis
from regime_model import train_regime_classifier, build_regime_features, predict_current_regime, adaptive_ewma_params
from multi_asset import run_cross_sectional_validation, run_scenario_validation, DEFAULT_BASKET, MARKET_SCENARIOS


st.set_page_config(page_title="EWMA Strategy Research Platform", layout="wide")

st.title("📈 AI-Powered EWMA Strategy Research Platform")
st.caption(
    "Research question: *Can an EWMA crossover strategy generate risk-adjusted "
    "returns?*  H0: EWMA crossover does not outperform random trading.  "
    "H1: it generates positive risk-adjusted returns."
)

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Asset (ticker)", value="AAPL")
    start = st.date_input("Start date", value=pd.to_datetime("2015-01-01"))
    end = st.date_input("End date", value=pd.to_datetime("2025-01-01"))

    st.subheader("EWMA Parameters")
    fast_span = st.slider("Fast EWMA span", min_value=2, max_value=60, value=20)
    slow_span = st.slider("Slow EWMA span", min_value=10, max_value=250, value=50)

    starting_capital = st.number_input("Starting capital ($)", min_value=1000, value=10_000, step=1000)
    flat_on_sell = st.checkbox(
        "Long-only (go flat on sell, don't short)", value=True,
        help="If unchecked, a sell signal takes a short position (long/short symmetric)."
    )
    transaction_cost_bps = st.slider("Transaction cost (bps per trade)", 0, 50, 0)
    slippage_bps = st.slider("Slippage (bps per trade)", 0, 50, 5)

    st.subheader("Risk management (optional)")
    use_dynamic_costs = st.checkbox(
        "Use dynamic cost model instead of flat bps", value=False,
        help="Replaces the flat cost sliders above with a cost that scales "
             "with realized volatility and trade size vs. average dollar "
             "volume — see README 'Transaction costs' section."
    )
    use_stop_loss = st.checkbox("Enable stop-loss", value=False)
    stop_loss_pct = st.slider("Stop-loss (%)", 2, 30, 8, disabled=not use_stop_loss) / 100.0 if use_stop_loss else None
    use_vol_target = st.checkbox("Enable volatility-target position sizing", value=False)
    target_vol = st.slider("Target annualized volatility (%)", 5, 40, 15, disabled=not use_vol_target) / 100.0 if use_vol_target else None

    run_button = st.button("🚀 Run Strategy", type="primary", use_container_width=True)
    st.divider()
    st.caption("Deeper validation (slower, run after the basic result looks interesting):")
    show_execution_comparison = st.checkbox(
        "Compare realistic vs. naive execution timing", value=False,
        help="Shows the cost of assuming you can trade at the exact close "
             "price the signal was computed from, vs. the realistic model "
             "that fills at the next session's open."
    )
    show_baseline = st.checkbox("Bootstrap hypothesis test vs. random trading (H0)", value=False,
                                 help="Runs the H0 comparison across many resampled histories instead "
                                      "of just the one you happened to load, to show how much sampling "
                                      "noise a single-path result actually has.")
    show_walkforward = st.checkbox("Walk-forward out-of-sample validation", value=False,
                                    help="Picks parameters using only a training window each fold, then "
                                         "tests on data that window never saw — the real test of whether "
                                         "a strategy generalizes, vs. just fitting the past.")
    show_optimizer = st.checkbox("Run parameter grid search (in-sample only)", value=False)
    show_regime = st.checkbox("ML market regime forecast", value=False,
                               help="Forecasts the regime over the NEXT 20 trading days from today's "
                                    "features, evaluated with walk-forward folds and a purge gap.")
    show_cost_sensitivity = st.checkbox("Transaction cost sensitivity", value=False,
                                         help="Shows how much faster higher-frequency (smaller fast span) "
                                              "variants lose their edge as costs rise, vs. lower-frequency ones.")
    show_multi_asset = st.checkbox("Cross-sectional & cross-regime validation", value=False,
                                    help="Runs the SAME strategy across a basket of different tickers and "
                                         "several synthetic market regimes (bull/bear/choppy/high-vol/low-vol) "
                                         "to check whether results generalize beyond one ticker, one decade.")

if fast_span >= slow_span:
    st.sidebar.warning("Fast span should be smaller than slow span.")

# ---------------------------------------------------------------- Main
if run_button or "last_run" in st.session_state:
    if run_button:
        with st.spinner(f"Loading data for {ticker}..."):
            prices = load_price_data(ticker, start=str(start), end=str(end), use_cache=True)
            feats = build_feature_frame(prices, fast_span=fast_span, slow_span=slow_span)
            signaled = generate_ewma_crossover_signals(feats, flat_on_sell=flat_on_sell)
            bt = run_backtest(
                signaled,
                starting_capital=starting_capital,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
                use_dynamic_costs=use_dynamic_costs,
                stop_loss_pct=stop_loss_pct,
                target_vol=target_vol,
            )
            trades = trade_log(signaled)
        st.session_state["last_run"] = dict(
            ticker=ticker, prices=prices, feats=feats, signaled=signaled, bt=bt, trades=trades,
            starting_capital=starting_capital, fast_span=fast_span, slow_span=slow_span,
            flat_on_sell=flat_on_sell,
        )

    state = st.session_state["last_run"]
    bt, trades, signaled = state["bt"], state["trades"], state["signaled"]

    # ---- Metrics row
    stats = summary(bt, trades)
    bh_stats = {
        "CAGR": cagr(bt["buy_hold_value"]),
        "ExcessCAGR": excess_cagr(bt["buy_hold_value"]),
        "Sharpe": sharpe_ratio(bt["returns"]),
        "MaxDrawdown": max_drawdown(bt["buy_hold_value"]),
        "AnnualVolatility": annualized_volatility(bt["returns"]),
        "FinalValue": bt["buy_hold_value"].iloc[-1],
    }

    st.subheader(f"Results: {state['ticker']}")
    st.caption("Sharpe uses an approximate historical risk-free rate by year (not a flat 0%) — see README.")
    active_rm = []
    if use_dynamic_costs:
        active_rm.append("dynamic transaction costs")
    if stop_loss_pct:
        active_rm.append(f"{stop_loss_pct:.0%} stop-loss")
    if target_vol:
        active_rm.append(f"{target_vol:.0%} vol-target sizing")
    if active_rm:
        st.caption(f"Risk management active: {', '.join(active_rm)}.")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{stats['CAGR']:.1%}", f"{(stats['CAGR'] - bh_stats['CAGR']):+.1%} vs B&H")
    c2.metric("Sharpe", f"{stats['Sharpe']:.2f}", f"{(stats['Sharpe'] - bh_stats['Sharpe']):+.2f} vs B&H")
    c3.metric("Max Drawdown", f"{stats['MaxDrawdown']:.1%}")
    c4.metric("Trades", f"{stats.get('NumTrades', 0)}")
    c5.metric("Win Rate", f"{stats.get('WinRate', float('nan')):.1%}" if not np.isnan(stats.get('WinRate', np.nan)) else "N/A")

    # ---- Charts
    tab1, tab2, tab3, tab4 = st.tabs(["Price & EWMA", "Buy/Sell Signals", "Portfolio Growth", "Drawdown"])
    with tab1:
        st.pyplot(plot_price_and_ewma(bt, title=f"{state['ticker']}: Price & EWMA"))
    with tab2:
        st.pyplot(plot_signals(bt, title=f"{state['ticker']}: Buy/Sell Signals"))
    with tab3:
        st.pyplot(plot_portfolio_growth(bt))
    with tab4:
        st.pyplot(plot_drawdown(bt))

    # ---- Strategy vs Buy & Hold table
    st.subheader("Strategy vs. Buy & Hold")
    st.dataframe(
        summary_table(("EWMA Strategy", stats), ("Buy & Hold", bh_stats)).style.format("{:.4f}"),
        use_container_width=True,
    )

    # ---- Trade log
    with st.expander(f"Trade log ({len(trades)} round-trip trades)"):
        if trades.empty:
            st.write("No completed round-trip trades in this period.")
        else:
            st.dataframe(trades.style.format({"entry_price": "{:.2f}", "exit_price": "{:.2f}", "return_pct": "{:.2%}"}),
                         use_container_width=True)

    # ---- Execution realism comparison
    if show_execution_comparison:
        st.subheader("Execution Timing: Realistic vs. Naive")
        st.caption(
            "The **naive** model assumes you can trade at the exact close price "
            "the signal was computed from — not achievable in real trading. "
            "The **realistic** model fills at the next session's open instead."
        )
        with st.spinner("Running both execution models..."):
            exec_comparison = compare_execution_models(
                signaled, starting_capital=state["starting_capital"],
                transaction_cost_bps=transaction_cost_bps, slippage_bps=slippage_bps,
            )
        st.dataframe(exec_comparison.style.format({
            "CAGR": "{:.2%}", "ExcessCAGR": "{:.2%}", "Sharpe": "{:.2f}", "MaxDrawdown": "{:.2%}",
            "AnnualVolatility": "{:.2%}", "FinalValue": "{:.2f}",
        }), use_container_width=True)

    # ---- Bootstrap hypothesis test (H0)
    if show_baseline:
        st.subheader("Hypothesis Test: EWMA vs. Random Trading (H0)")
        st.caption(
            "Instead of comparing against random traders on just the one historical "
            "path you loaded, this resamples that history many times (a block "
            "bootstrap, which preserves volatility clustering), recomputes the EWMA "
            "signal fresh on each resampled path, and records where the strategy "
            "lands vs. a fresh batch of random traders each time. The spread across "
            "draws IS the sampling noise a single-run result would hide."
        )
        n_bootstrap = st.slider("Number of bootstrap resamples", 20, 200, 60, step=20)
        with st.spinner(f"Running {n_bootstrap} bootstrap resamples (this takes a bit)..."):
            boot = bootstrap_hypothesis_test(
                state["prices"], fast_span=state["fast_span"], slow_span=state["slow_span"],
                starting_capital=state["starting_capital"], n_bootstrap=n_bootstrap,
                n_random_traders=150, flat_on_sell=state["flat_on_sell"],
            )
        pct = boot["percentile_vs_random"]

        colA, colB = st.columns([2, 1])
        with colA:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(pct, bins=20, color="lightsteelblue", edgecolor="steelblue")
            ax.axvline(50, color="gray", linestyle="--", linewidth=1, label="50th percentile (coin flip vs. random)")
            ax.axvline(pct.mean(), color="tab:blue", linewidth=2, label=f"Mean = {pct.mean():.0f}th percentile")
            ax.set_xlabel("Percentile vs. random traders, per bootstrap draw")
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of Outcomes Across {n_bootstrap} Bootstrap Resamples")
            ax.legend()
            st.pyplot(fig)
        with colB:
            st.metric("Mean percentile vs. random", f"{pct.mean():.0f}th")
            st.metric("Std. dev. across draws", f"{pct.std():.0f} pts")
            st.metric("5th–95th percentile range", f"{pct.quantile(0.05):.0f}–{pct.quantile(0.95):.0f}")
            frac_beat_half = (pct > 50).mean()
            st.write(f"Beat the median random trader in **{frac_beat_half:.0%}** of resamples.")
        st.caption(
            "A wide spread here means the single-path result reported elsewhere in "
            "this dashboard (or in the project report) could easily have looked very "
            "different with a slightly different market history — treat any one-shot "
            "percentile with real skepticism."
        )

    # ---- Walk-forward validation
    if show_walkforward:
        st.subheader("Walk-Forward Out-of-Sample Validation")
        st.caption(
            "Each fold picks the best (fast, slow) EWMA combo using ONLY a training "
            "window, then tests that exact, already-fixed combo on a later window it "
            "never saw. Compare the in-sample and out-of-sample columns — a big gap "
            "is the signature of overfitting, and it's common."
        )
        n_folds = st.slider("Number of walk-forward folds", 3, 8, 5)
        with st.spinner("Running walk-forward validation..."):
            wf = walk_forward_strategy_validation(
                state["prices"], n_splits=n_folds, gap=100,
                starting_capital=state["starting_capital"], flat_on_sell=state["flat_on_sell"],
            )
        if wf.empty:
            st.warning("Not enough data for this many walk-forward folds — try fewer folds or a longer date range.")
        else:
            st.dataframe(wf.style.format({
                "in_sample_Sharpe": "{:.2f}", "in_sample_CAGR": "{:.2%}",
                "oos_Sharpe": "{:.2f}", "oos_CAGR": "{:.2%}", "oos_MaxDrawdown": "{:.2%}",
            }), use_container_width=True)
            mean_is, mean_oos = wf["in_sample_Sharpe"].mean(), wf["oos_Sharpe"].mean()
            colX, colY = st.columns(2)
            colX.metric("Mean in-sample Sharpe", f"{mean_is:.2f}")
            colY.metric("Mean out-of-sample Sharpe", f"{mean_oos:.2f}", f"{mean_oos - mean_is:+.2f} vs in-sample")
            if mean_oos < mean_is - 0.15:
                st.warning(
                    "Out-of-sample Sharpe is meaningfully lower than in-sample — the "
                    "parameter search is likely overfitting to the training windows."
                )

    # ---- Parameter grid search
    if show_optimizer:
        st.subheader("Parameter Grid Search (In-Sample Only)")
        st.caption(
            "⚠️ This ranks parameter combinations on the SAME data used to evaluate "
            "them — the classic overfitting setup. Use the walk-forward panel above "
            "to see whether the top combo here actually holds up out-of-sample."
        )
        with st.spinner("Testing EWMA span combinations..."):
            grid = grid_search(state["prices"], starting_capital=state["starting_capital"], flat_on_sell=flat_on_sell)
        st.dataframe(grid.style.format({
            "CAGR": "{:.2%}", "ExcessCAGR": "{:.2%}", "Sharpe": "{:.2f}", "MaxDrawdown": "{:.2%}",
            "AnnualVolatility": "{:.2%}", "FinalValue": "{:.2f}",
        }), use_container_width=True)
        best = grid.iloc[0]
        st.info(
            f"Best in-sample combination by Sharpe: fast={int(best['fast_span'])}, "
            f"slow={int(best['slow_span'])} (Sharpe={best['Sharpe']:.2f}) — "
            f"check the walk-forward panel before trusting this."
        )

    # ---- ML regime forecast
    if show_regime:
        st.subheader("ML Market Regime Forecast")
        st.caption(
            "Forecasts the regime (Bull / Bear / HighVol / LowVol) over the NEXT 20 "
            "trading days from today's trailing features. Labels are built from "
            "strictly future returns (never the same window used as input), and "
            "accuracy is measured with walk-forward folds and a purge gap — so this "
            "number reflects genuine forecasting skill, not label leakage. Compare "
            "it to the chance level shown below; regime forecasting is genuinely hard."
        )
        with st.spinner("Training and walk-forward-validating the regime classifier..."):
            model, cols, report = train_regime_classifier(state["prices"])
        st.code(report, language=None)

        regime_feats = build_regime_features(state["prices"]).dropna(subset=cols)
        if model is not None and not regime_feats.empty:
            current_regime = predict_current_regime(model, cols, regime_feats.iloc[-1])
            suggested = adaptive_ewma_params(current_regime)
            colP, colQ = st.columns(2)
            colP.metric("Forecast regime (next 20 trading days)", current_regime)
            colQ.metric("Suggested adaptive EWMA spans", f"{suggested['fast_span']} / {suggested['slow_span']}")

    # ---- Transaction cost sensitivity
    if show_cost_sensitivity:
        st.subheader("Transaction Cost Sensitivity")
        st.caption(
            "A flat 0-bps default (or a fixed 28-trades-over-10-years example) hides a real "
            "dynamic: smaller EWMA spans trade more often, so higher-frequency variants lose "
            "their edge to costs much faster. This backtests several fast spans across a range "
            "of flat cost levels, split across 3 non-overlapping sub-periods, so the degradation "
            "curve is visible AND you can see whether it holds up consistently across different "
            "multi-year windows or was itself somewhat period-dependent."
        )
        with st.spinner("Running cost sensitivity grid across sub-periods..."):
            cost_sens = cost_sensitivity_analysis(
                state["prices"], fast_spans=(5, 10, 20, 30, 50), slow_span=max(100, state["slow_span"]),
                cost_bps_grid=(0, 5, 10, 20, 40, 80), starting_capital=state["starting_capital"],
                flat_on_sell=state["flat_on_sell"], n_periods=3,
            )
        agg = cost_sens.groupby(["fast_span", "cost_bps"], as_index=False).agg(
            Sharpe=("Sharpe", "mean"), Sharpe_std=("Sharpe", "std"), num_trades=("num_trades", "sum"),
        )
        pivot_sharpe = agg.pivot(index="fast_span", columns="cost_bps", values="Sharpe")
        fig, ax = plt.subplots(figsize=(9, 5))
        for fast_span_val in pivot_sharpe.index:
            trades_n = agg[agg["fast_span"] == fast_span_val]["num_trades"].iloc[0]
            ax.plot(pivot_sharpe.columns, pivot_sharpe.loc[fast_span_val],
                    marker="o", label=f"fast={fast_span_val} ({int(trades_n)} total trades)")
        ax.set_xlabel("Transaction cost (bps per trade)")
        ax.set_ylabel("Sharpe ratio (mean across 3 sub-periods)")
        ax.set_title("Sharpe Degradation vs. Cost, by Trading Frequency")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        with st.expander("Per-period breakdown (does the degradation pattern hold up consistently?)"):
            for period in sorted(cost_sens["period"].unique()):
                sub = cost_sens[cost_sens["period"] == period]
                p_start, p_end = sub["period_start"].iloc[0], sub["period_end"].iloc[0]
                st.markdown(f"**Period {period}** ({p_start} to {p_end})")
                st.dataframe(sub.pivot(index="fast_span", columns="cost_bps", values="Sharpe").style.format("{:.2f}"),
                             use_container_width=True)

    # ---- Cross-sectional & cross-regime validation
    if show_multi_asset:
        st.subheader("Cross-Sectional & Cross-Regime Validation")
        st.caption(
            "Everything else on this page is one ticker over one historical window. This runs "
            "the SAME strategy across a basket of different tickers and several synthetic market "
            "regimes to check whether results generalize."
        )
        st.markdown(f"**Cross-sectional** (basket: {', '.join(DEFAULT_BASKET)})")
        st.caption(
            "⚠️ This uses a hard-coded, CURRENT list of tickers — subject to survivorship bias "
            "(delisted/bankrupt/acquired companies are invisible from a list built today). "
            "Illustrative only; see README."
        )
        import warnings as _warnings
        with st.spinner("Running cross-sectional validation..."):
            with _warnings.catch_warnings(record=True):
                _warnings.simplefilter("always")
                cross = run_cross_sectional_validation(
                    start=str(start), end=str(end), fast_span=state["fast_span"], slow_span=state["slow_span"],
                    starting_capital=state["starting_capital"], flat_on_sell=state["flat_on_sell"],
                )
        display_cols = [c for c in ["CAGR", "Sharpe", "MaxDrawdown", "NumTrades"] if c in cross.columns]
        st.dataframe(cross[display_cols].style.format({"CAGR": "{:.2%}", "Sharpe": "{:.2f}", "MaxDrawdown": "{:.2%}"}),
                     use_container_width=True)

        st.markdown("**Cross-regime** (synthetic market scenarios, multiple seeds per scenario)")
        n_seeds = st.slider("Random seeds per scenario", 3, 15, 5,
                             help="Each scenario is only one draw of simulated history per seed — "
                                  "more seeds give a more honest picture of the variance, at the "
                                  "cost of more compute.")
        with st.spinner(f"Running cross-regime validation ({n_seeds} seeds/scenario)..."):
            scenarios = run_scenario_validation(
                start=str(start), end=str(end), fast_span=state["fast_span"], slow_span=state["slow_span"],
                starting_capital=state["starting_capital"], flat_on_sell=state["flat_on_sell"],
                n_seeds=n_seeds,
            )
        fig2, ax2 = plt.subplots(figsize=(9, 4.5))
        colors = ["tab:green" if s > 0 else "tab:red" for s in scenarios["Sharpe_mean"]]
        ax2.bar(scenarios.index, scenarios["Sharpe_mean"], yerr=scenarios["Sharpe_std"],
                color=colors, capsize=4, alpha=0.85)
        ax2.axhline(0, color="gray", linewidth=0.8)
        ax2.set_ylabel("Sharpe ratio (mean ± std across seeds)")
        ax2.set_title(f"Strategy Sharpe Ratio Across Synthetic Market Regimes ({n_seeds} seeds each)")
        plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
        st.pyplot(fig2)
        st.caption(
            "The error bars are the point: a scenario whose error bar crosses zero could easily "
            "have shown the opposite sign with a different random draw — a single-seed bar chart "
            "would have hidden that entirely."
        )
        st.dataframe(scenarios.style.format({
            "annual_drift": "{:.0%}", "annual_vol": "{:.0%}", "Sharpe_mean": "{:.2f}", "Sharpe_std": "{:.2f}",
            "CAGR_mean": "{:.2%}", "CAGR_std": "{:.2%}", "MaxDrawdown_mean": "{:.2%}",
            "pct_seeds_positive_Sharpe": "{:.0%}",
        }), use_container_width=True)
        n_reliably_positive = (scenarios["Sharpe_mean"] - scenarios["Sharpe_std"] > 0).sum()
        st.caption(f"Mean Sharpe positive AND more than one std above zero in "
                   f"{n_reliably_positive} of {len(scenarios)} tested regimes. "
                   f"A strategy that only reliably works in one regime is a regime bet, not an edge.")

    # ---- Does the regime forecast actually help? (real backtest, not just a display)
    if show_regime:
        st.subheader("Does Using the Regime Forecast Actually Help?")
        st.caption(
            "The forecast above is just a prediction. This actually backtests a strategy that "
            "switches EWMA parameters based on the walk-forward-trained model's out-of-sample "
            "forecasts, and compares it to a static (fixed 20/50) baseline over the identical "
            "out-of-sample periods — an honest test rather than a decorative 'suggested params' card."
        )
        with st.spinner("Backtesting the regime-adaptive strategy out-of-sample..."):
            from regime_model import backtest_regime_adaptive_strategy
            adaptive_result = backtest_regime_adaptive_strategy(
                state["prices"], starting_capital=state["starting_capital"], flat_on_sell=state["flat_on_sell"],
            )
        if adaptive_result["adaptive_summary"]:
            comp = pd.DataFrame({
                "Adaptive (regime-switching)": adaptive_result["adaptive_summary"],
                "Static (fixed 20/50)": adaptive_result["static_summary"],
            }).T
            st.dataframe(comp[["CAGR", "Sharpe", "MaxDrawdown", "FinalValue"]].style.format({
                "CAGR": "{:.2%}", "Sharpe": "{:.2f}", "MaxDrawdown": "{:.2%}", "FinalValue": "{:.2f}",
            }), use_container_width=True)
            adaptive_sharpe = adaptive_result["adaptive_summary"]["Sharpe"]
            static_sharpe = adaptive_result["static_summary"]["Sharpe"]
            if adaptive_sharpe > static_sharpe:
                st.info(f"Adaptive beat static by {adaptive_sharpe - static_sharpe:+.2f} Sharpe out-of-sample "
                        f"in this run — worth a closer look, but not proof given the classifier's near-chance accuracy.")
            else:
                st.warning(f"Adaptive did NOT beat static out-of-sample ({adaptive_sharpe:.2f} vs {static_sharpe:.2f} "
                           f"Sharpe) — consistent with the classifier's accuracy being close to its majority-class baseline.")
        else:
            st.warning("Not enough data for a walk-forward adaptive backtest with the current date range.")

else:
    st.info("Set your parameters in the sidebar and click **Run Strategy** to begin.")
    st.caption(
        "⚠️ Data note: this app tries Yahoo Finance first via `yfinance`. If it's "
        "unreachable (offline, blocked network), it automatically falls back to a "
        "synthetic price series so the app still works end-to-end."
    )
