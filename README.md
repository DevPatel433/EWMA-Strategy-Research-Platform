# AI-Powered EWMA Strategy Research Platform

> **Changelog:**
> - Fixed a reproducibility bug in `src/data_loader.py` —
>   the synthetic-data fallback used to seed its RNG with Python's
>   built-in `hash(ticker)`, which is randomized per process and produced
>   a *different* fake price history on every run. It now seeds with
>   `hashlib.sha256(ticker, start, end)`, which is stable across runs,
>   processes, and machines. See "Notes on data" below for details.
> - Fixed an unrealistic trade-execution assumption in `src/backtester.py`
>   and `src/signals.py`'s `trade_log()`. The backtest used to apply a
>   signal computed from `Close[t]` to the same day's close-to-close
>   return — implicitly assuming you could trade at the exact close price
>   the signal was calculated from, which isn't achievable in real
>   trading. `run_backtest()` now defaults to a realistic **next-open**
>   execution model: the signal is filled at the *following* session's
>   `Open` and held to the next open after that. The old behavior is
>   still available as `execution_model="same_close"` for side-by-side
>   comparison via the new `compare_execution_models()` helper. See
>   "Execution realism" below for details.
> - Fixed label leakage in the ML regime classifier (`src/regime_model.py`).
>   It used to label each row from the *same* trailing trend/volatility
>   window it was also given as an input feature — the model was
>   reconstructing a rule from data handed to it for free, hence the old
>   ~100% "accuracy". Labels are now built from returns **strictly after**
>   the feature date, evaluated with walk-forward folds plus a purge gap.
>   **Update:** a second, subtler leak was later found and fixed too — the
>   Bull/Bear/HighVol/LowVol volatility THRESHOLD was computed globally
>   over the whole dataset before splitting, so even purged folds were
>   labeled using a threshold informed by future data. It's now computed
>   per-fold from training data only. Honest accuracy is ~26%, and the
>   model beats a **majority-class baseline** (also newly added, since
>   1/n_classes chance is only valid for balanced classes) by only ~+1.8
>   points — see "Out-of-sample validation" below.
> - Added `src/validation.py`: walk-forward out-of-sample testing for the
>   parameter grid search (`walk_forward_strategy_validation`) and a
>   block-bootstrap version of the H0 hypothesis test
>   (`bootstrap_hypothesis_test`) that reports a full distribution instead
>   of one single-path percentile. Both are wired into the dashboard and
>   the notebook. See "Out-of-sample validation" below.
> - Fixed `sharpe_ratio()` defaulting to a flat 0% risk-free rate. It now
>   defaults to an approximate historical rate by calendar year (still
>   overridable with a flat float or a real daily-rate series). See
>   "Risk-free rate" below.
> - Added realistic friction and risk controls to `run_backtest()` via
>   `src/risk_management.py`: a **dynamic transaction cost model**
>   (scales with realized volatility and trade size vs. average dollar
>   volume, instead of one flat bps number), **volatility-target position
>   sizing** (instead of always being 100% in or 100% out), and a
>   **stop-loss overlay**. Also added `validation.cost_sensitivity_analysis()`,
>   which shows that higher-frequency (smaller EWMA span) variants lose
>   their edge to costs several times faster than lower-frequency ones —
>   a dynamic a flat 0-bps default hides completely. See "Transaction
>   costs & risk management" below.
> - Added `src/multi_asset.py`: runs the strategy across a basket of
>   tickers (cross-sectional) and across several synthetic market regimes
>   with deliberately different drift/volatility (cross-regime), since a
>   single ticker over a single decade says little about whether results
>   generalize. The cross-sectional basket is a hard-coded, CURRENT
>   ticker list, so it's subject to survivorship bias -- the code now
>   raises an explicit `UserWarning` every time that default basket is
>   used, rather than silently proceeding. See "Generalization" below.
> - `data_loader.load_price_data()` now runs all data (live or synthetic,
>   cached or fresh) through a new `clean_ohlcv()` step: drops duplicate/
>   invalid rows, and flags (without silently deleting) extreme single-day
>   moves that often indicate an unadjusted stock split or a bad tick, plus
>   long gaps between consecutive rows. Previously the only cleaning was
>   an unenforced `dropna()` left to the caller. See "Data quality" below.
>
> **Second round of fixes**, after auditing the first round's own limitations:
> - Stop-loss now uses `risk_management.apply_realized_stop_loss()`, which
>   operates on the already-lagged/realized position with its own short
>   (1-bar) exit lag, decoupled from the strategy's slower multi-bar entry
>   lag. The old `apply_stop_loss()` coupled stop-outs to the full entry
>   lag, so a "stop-loss" was no faster than an ordinary signal change —
>   defeating the point of having one. Old function kept, marked deprecated.
> - `multi_asset.run_cross_sectional_validation()`'s offline fallback used
>   to give every ticker in the basket the SAME synthetic statistical
>   regime — meaningful-looking Sharpe dispersion that was mostly sampling
>   noise, not real cross-sectional heterogeneity. `TICKER_PROFILES` now
>   gives each fallback ticker its own rough drift/vol profile.
> - `multi_asset.run_scenario_validation()` used to run each market regime
>   on exactly one random seed — a single-path result with the same
>   fragility the bootstrap test exists to catch elsewhere. It now runs
>   `n_seeds` draws per scenario (default 10) and reports mean ± std, not
>   a point estimate.
> - `regime_model.py` now reports a **majority-class baseline** accuracy
>   alongside the model's accuracy (see above), and adds
>   `backtest_regime_adaptive_strategy()`, which actually backtests a
>   regime-switching strategy out-of-sample and compares it to a static
>   baseline — the "suggested EWMA params" card was previously decorative
>   (a forecast with no evidence acting on it helps).
> - `validation.walk_forward_strategy_validation()` gained optional
>   `stop_loss_grid`/`target_vol_grid` for a **joint, compound search**
>   across multiple axes at once (not just fast/slow spans), to check
>   compound overfitting that single-axis safeguards can't catch on
>   their own.
> - Added `validation.hyperparameter_robustness_check()`, which reruns
>   walk-forward validation and the bootstrap test across several
>   gap/fold-count/block-size choices, since none of those were
>   previously validated as robust choices themselves.
> - `validation.cost_sensitivity_analysis()` now splits data into
>   multiple sub-periods (default 3) instead of reporting one full-sample
>   degradation curve, so the "higher frequency degrades faster" finding
>   can be checked for consistency across different multi-year windows.
> - `data_loader.clean_ohlcv()`'s extreme-move threshold was lowered
>   (40% → 25%) and a separate `detect_possible_splits()` pattern-match
>   was added, since some real split ratios (e.g. 5-for-4, a ~20% move)
>   fall under any reasonable pure-magnitude threshold.
> - The dynamic transaction cost model's constants remain admittedly
>   illustrative, not calibrated to real quoted spreads/impact for any
>   venue — flagged explicitly rather than presented as precise.

A complete, beginner → intermediate quant research environment for testing
EWMA (Exponentially Weighted Moving Average) crossover trading strategies:
data collection → cleaning → EWMA engine → signal generation → backtesting
→ performance analytics → visualization → an interactive dashboard, plus
a parameter optimizer and a machine-learning market-regime classifier.

## Research question

> Can an EWMA crossover strategy generate risk-adjusted returns?

- **H0:** EWMA crossover does not outperform random trading.
- **H1:** EWMA crossover generates positive risk-adjusted returns.

The dashboard includes a built-in "H0 test" that compares the strategy's
final portfolio value against a distribution of 200 random long/flat
traders over the same period.

## Project structure

```
EWMA-Trading-Research-Platform/
│
├── data/                     # cached price CSVs (auto-created)
├── notebooks/
│   └── research.ipynb        # step-by-step exploratory notebook
├── src/
│   ├── data_loader.py         # Step 1: Yahoo Finance download + synthetic fallback
│   ├── ewma.py                 # Step 3: EWMA / returns / volatility features
│   ├── signals.py              # Step 4: crossover signals + trade log
│   ├── backtester.py           # Step 5: vectorized backtest + random baseline
│   ├── metrics.py               # Step 6: CAGR, Sharpe, drawdown, win rate
│   ├── optimizer.py             # Step 8: EWMA span grid search (+ optional VectorBT)
│   ├── regime_model.py          # Step 9: ML market-regime forecaster (forward-looking labels)
│   ├── validation.py            # Walk-forward validation + bootstrap hypothesis test + cost sensitivity
│   ├── risk_management.py       # Dynamic costs, vol-target sizing, stop-loss overlay
│   ├── multi_asset.py           # Cross-sectional + cross-regime generalization checks
│   └── visualize.py             # Step 7: price/EWMA, signals, portfolio, drawdown charts
├── dashboard/
│   └── app.py                 # Step 10: Streamlit dashboard
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick start (Python)

```python
import sys; sys.path.insert(0, "src")

from data_loader import load_price_data
from ewma import build_feature_frame
from signals import generate_ewma_crossover_signals, trade_log
from backtester import run_backtest
from metrics import summary

prices = load_price_data("AAPL", start="2015-01-01", end="2025-01-01")
feats = build_feature_frame(prices, fast_span=20, slow_span=50)
signaled = generate_ewma_crossover_signals(feats)
bt = run_backtest(signaled, starting_capital=10_000)

print(summary(bt, trade_log(signaled)))
```

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

Pick a ticker, date range, and fast/slow EWMA spans in the sidebar, then
click **Run Strategy**. Optional checkboxes run a parameter grid search
and the random-trading H0 comparison.

## Run the research notebook

```bash
jupyter notebook notebooks/research.ipynb
```

## Notes on data

- `load_price_data()` tries **Yahoo Finance** (`yfinance`) first.
- If Yahoo Finance is unreachable (offline, blocked network, rate-limited —
  this happens in some sandboxed/CI environments), it **automatically
  falls back to a synthetic geometric-Brownian-motion price series** so
  every module keeps working end-to-end. You'll see a console message
  when this happens. On a normal internet connection this fallback is
  never triggered.
- Downloaded/generated data is cached to `data/<ticker>_<start>_<end>.csv`
  so repeated runs don't re-hit the network.
- **The synthetic fallback is fully deterministic across runs and
  machines.** The random seed is derived from `(ticker, start, end)`
  via `hashlib.sha256`, not Python's built-in `hash()`. This matters:
  `hash()` on strings is randomized per-process by default (PEP 456
  hash-seed salting), so an earlier version of this function that
  seeded with `hash(ticker)` silently produced a *different* synthetic
  price history on every run — a genuine reproducibility bug, now
  fixed. You can verify this yourself:
  ```bash
  PYTHONHASHSEED=1   python3 src/data_loader.py   # note the printed checksum
  PYTHONHASHSEED=999 python3 src/data_loader.py   # checksum is identical
  ```

## Execution realism

`run_backtest()` supports two execution models via `execution_model=`:

- **`"next_open"` (default, realistic).** A signal computed from `Close[t]`
  is only known after the market closes that day. The earliest price you
  could actually trade at is the *next* session's `Open[t+1]`. A position
  entered there is held until the following open, so its return is
  measured open-to-open, two bars after the signal that decided it:
  ```
  strategy_return[t] = signal[t-2] * (Open[t] - Open[t-1]) / Open[t-1]
  ```
- **`"same_close"` (naive, comparison only).** The original model: applies
  the signal to the same-day close-to-close return with a 1-bar lag. This
  avoids using *future* prices, but still implicitly assumes you can trade
  at the exact close price the signal was computed from — not achievable
  in live trading. Kept only so its impact can be measured, not for
  evaluating a strategy you intend to trade.

Use `compare_execution_models(df)` to see both side by side. The dashboard
has a "Compare realistic vs. naive execution timing" checkbox that shows
the same comparison interactively. There's also a `slippage_bps` parameter
(separate from `transaction_cost_bps`) representing the gap between the
quoted open price and what you'd likely be filled at.

`signals.trade_log()` was updated to match: by default it logs fills at
`Open`, two bars after the crossover bar (matching the `next_open` model),
instead of at the same-bar `Close`.

This is still an approximation — it assumes your whole order fills exactly
at the open print with no market impact, which is reasonable for a liquid
large-cap name in modest size but not for illiquid assets or large orders.
Modeling that properly would need intraday order-book data, which is out
of scope for this daily-bar research platform.

## Out-of-sample validation

Three separate things in this project used to be reported in-sample only,
with no check on whether the result held up on data they weren't built or
tuned on. `src/validation.py` and the fixed `regime_model.py` address all
three; the dashboard's "Deeper validation" checkboxes and the notebook
both exercise them.

**1. Parameter grid search overfitting.** Ranking EWMA span combinations
by Sharpe ratio on the same data used to evaluate them is a classic
overfitting setup — with 14+ combinations tested, something will look
good by chance even with no real edge. `walk_forward_strategy_validation()`
fixes this properly: for each fold, it runs the grid search on a training
window only, takes the #1 combo, and backtests that already-fixed combo on
a later window it never saw. On the AAPL case study this reliably shows a
large in-sample vs. out-of-sample gap (e.g. mean in-sample Sharpe ~0.2
vs. mean out-of-sample Sharpe around -0.3 to -0.5 depending on the run) —
the textbook signature of a search fitting noise, not finding a real edge.

**2. The single-path random-trading hypothesis test.** Comparing the
strategy to random traders on the one historical path you loaded doesn't
tell you how *reliable* that comparison is. `bootstrap_hypothesis_test()`
block-bootstrap-resamples the historical returns many times (preserving
volatility clustering), reconstructs a synthetic price path from each
resample, recomputes the EWMA signal fresh on that path, and records the
strategy's percentile vs. a fresh batch of random traders each time. On
the AAPL case study this reliably produces a wide distribution (mean
around the 40th-50th percentile, std ~25-30 points, ranging from near 0
to near 100 across 100 draws) — meaning a single run's percentile (e.g.
"beat 34% of random traders") is close to meaningless on its own; only
the distribution is informative.

**3. ML regime classifier label leakage.** See the changelog above and
`regime_model.py`'s module docstring — labels are now strictly
forward-looking, evaluation uses walk-forward folds with a purge gap, AND
(after a second audit found a subtler version of the same problem) the
Bull/Bear/HighVol/LowVol threshold is now computed per-fold from training
data only, not globally. Honest accuracy on the AAPL case study comes out
around 26%, beating a **majority-class baseline** (also newly reported,
since comparing to a flat 1/n_classes chance level silently assumes
balanced classes) by only about +1.8 percentage points — i.e. close to no
demonstrated skill with the current feature set. `backtest_regime_adaptive_strategy()`
confirms this the direct way: actually switching EWMA params on the
model's forecasts, out-of-sample, produced essentially no improvement
over a static baseline in testing.

**4. Compound overfitting across multiple axes.** Each check above covers
one axis (spans, or the random-trading comparison, or the classifier) in
isolation. `walk_forward_strategy_validation(..., stop_loss_grid=..., target_vol_grid=...)`
jointly searches spans AND stop-loss AND vol-target together in-sample,
then tests the jointly-chosen combo out-of-sample — and on the AAPL case
study this compound search shows an even LARGER in-sample/out-of-sample
gap than searching spans alone, confirming that tuning more knobs
together against the same data overfits more, not less.

**5. Are the validation hyperparameters themselves robust?** `gap`,
`n_splits`, and bootstrap `block_size` were reasonable defaults, not
validated choices. `hyperparameter_robustness_check()` reruns the
walk-forward and bootstrap checks across several nearby choices for each,
so you can see whether the headline conclusions are stable or were
themselves sensitive to the specific numbers picked.

**General takeaway:** all of these fixes point the same direction — the
strategy and model, as currently built, don't show a convincingly real
edge once evaluated properly, and that conclusion holds up even when
checked from several different angles (single-axis and compound search,
one seed and many seeds, one sub-period and several). That's a legitimate
research finding, not a bug in the tooling; it's exactly what rigorous
validation is supposed to surface before anyone risks capital on results
that only looked good in-sample.

## Risk-free rate

`sharpe_ratio()` and `summary()` used to default to a flat 0% risk-free
rate, which silently inflates Sharpe ratios during any period where
short-term rates were meaningfully positive -- and 2015-2025 ranged from
roughly 0% to roughly 5%. They now default to `historical_risk_free_rate()`,
a coarse but non-zero approximation of the annual rate by calendar year.
Pass an explicit float (e.g. `risk_free_rate=0.0` to reproduce the old
behavior) or a real daily-rate `pd.Series` (e.g. from FRED's `DGS3MO`) for
more rigorous work. `summary()` also now reports `ExcessCAGR` (CAGR minus
the average risk-free rate over the period) alongside plain CAGR.

## Transaction costs & risk management

A `transaction_cost_bps=0` default plus a small example of "28 trades over
10 years" together hide two real problems: costs that don't respond to
market conditions at all, and a strategy that's always either fully
invested or fully out with no other risk control. `src/risk_management.py`
and `run_backtest()`'s new parameters address both:

- **Dynamic transaction costs** (`use_dynamic_costs=True`): replaces the
  flat bps fee with `dynamic_transaction_cost_bps()`, which combines a
  bid-ask-spread proxy that widens with realized volatility and a
  square-root market-impact cost based on trade size vs. trailing average
  dollar volume. It's a stylized, illustrative model (not calibrated to
  any specific broker/venue), but it captures the right *qualitative*
  behavior -- costs rise in choppy markets and for less liquid names --
  which a flat number cannot represent at all.
- **Volatility-target position sizing** (`target_vol=0.15`, etc.): scales
  exposure via `compute_vol_target_scale()` so the position's *expected*
  volatility stays near a target, instead of always being 100% in or 100%
  out. This tends to reduce max drawdown; on the AAPL case study it also
  changed the Sharpe ratio (not always for the better -- see the notebook's
  risk-management comparison table for the actual trade-offs, not just the
  claim that sizing "helps").
- **Stop-loss overlay** (`stop_loss_pct=0.08`, etc.): `apply_realized_stop_loss()`
  forces the position flat starting one bar after price falls more than
  the given fraction below the *realized* entry fill price. Deliberately
  uses only a 1-bar exit lag, decoupled from the strategy's own slower
  multi-bar entry lag -- an earlier version coupled stop-loss exits to
  that same entry lag, meaning a "stop-loss" was no faster than an
  ordinary signal change. See `risk_management.py`'s module docstring.
- **Cost sensitivity** (`validation.cost_sensitivity_analysis()`): backtests
  several EWMA fast spans (a proxy for trading frequency) across a range of
  flat cost levels. On the AAPL case study, the highest-frequency variant
  tested (fast=5, ~42 trades) degrades roughly 3x faster than the
  lowest-frequency variant (fast=50, ~14 trades) as costs rise from 0 to 80
  bps -- exactly the dynamic a single "28 trades, 0 bps" example can't show.

None of these make the strategy "safe" -- they replace a few silently
optimistic defaults with explicit, tunable assumptions you can inspect and
argue with.

## Generalization: does this hold up beyond one ticker, one decade?

Every other result in this project is for one ticker (AAPL) over one
historical window (2015-2025). `src/multi_asset.py` adds two checks:

- **Cross-sectional** (`run_cross_sectional_validation()`): runs the
  identical strategy across a basket of different tickers
  (`DEFAULT_BASKET`). Offline, each ticker's synthetic fallback now uses
  a ticker-specific drift/vol profile (`TICKER_PROFILES`) instead of one
  shared default -- an earlier version gave every ticker the same
  statistical regime, so its Sharpe "dispersion" was mostly sampling
  noise rather than real heterogeneity. With that fix, Sharpe ranged from
  about -0.28 (GME-like high-vol profile) to +0.54 (JPM-like profile)
  across 8 tickers -- dispersion now tied to genuinely different
  simulated characteristics, though still not real historical data.

  ⚠️ **Survivorship bias**: `DEFAULT_BASKET` is a hard-coded, CURRENT
  list of tickers. Companies that were delisted, went bankrupt, or were
  acquired during the backtest period are invisible from a list built
  today, which makes cross-sectional results look systematically better
  than a point-in-time-correct universe would. `run_cross_sectional_validation()`
  raises a `UserWarning` every time it's called with the default basket,
  specifically so this doesn't get overlooked. There's no good fix for
  this without a licensed point-in-time constituents feed (e.g. a
  historical S&P 500 membership file); this project doesn't attempt to
  fabricate one -- pass your own point-in-time-correct `tickers=` list
  for anything you intend to trust.

- **Cross-regime** (`run_scenario_validation()`): runs the identical
  strategy against five synthetic market regimes with deliberately
  different drift/volatility (bull, bear, choppy sideways, high-vol
  crypto-like, low-vol grind) -- illustrative stress tests, not historical
  replays. Each scenario now runs across `n_seeds` random draws (default
  10) rather than one -- an earlier version used a single seed per
  scenario, which is exactly the single-path fragility the bootstrap test
  elsewhere in this project exists to catch. With multiple seeds, several
  scenarios show a Sharpe mean within one standard deviation of zero
  (e.g. `high_vol_crypto_like` at roughly -0.02 ± 0.29) -- meaning a
  single-seed version of this check could easily have shown either sign
  by chance. Only regimes where the mean stays reliably above zero even
  after subtracting one std should be read as "this strategy tends to
  work here."

## Data quality

`load_price_data()` used to only select the expected OHLCV columns; any
actual cleaning (`dropna()`, sorting) was left to the caller and easy to
skip. It now runs every load -- live, synthetic, or from cache -- through
`clean_ohlcv()`, which:

- Drops duplicate index entries and missing OHLC values.
- Drops rows with non-positive prices or internally inconsistent bars
  (e.g. `High < Low`), which indicate corrupted data rather than real
  prices.
- **Flags** (prints a warning with the specific dates, does NOT silently
  drop) single-day moves beyond 25% (lowered from an earlier 40% -- a 40%
  threshold misses common split ratios like 3-for-2, ~33%) -- these often
  indicate an unadjusted stock split or a bad tick, but genuine extreme
  moves do happen, so the decision to exclude them is left to you.
- **Flags separately**, via `detect_possible_splits()`, single-day price
  RATIOS close to a common split factor (2x, 3x, 1.5x, 4:1, 3:2, 5:4,
  ...), specifically because some real splits (e.g. 5-for-4, only a ~20%
  move) fall under any reasonable pure-magnitude threshold and would
  otherwise slip through unflagged.
- **Flags** gaps of more than 7 calendar days between consecutive rows,
  which can indicate a trading halt, delisting, or a feed outage.

This is not a substitute for cross-referencing multiple data vendors --
it catches the most common free-feed failure modes, not everything that
could be wrong with the data.

## Extending this project

- **Step 8 (VectorBT):** `optimizer.optimize_with_vectorbt()` is a stub
  that uses VectorBT if you `pip install vectorbt`, for much larger
  parameter sweeps than the pure-pandas `grid_search()`.
- **Step 9 (ML regime detection):** `regime_model.py` forecasts the
  regime over the next `horizon` trading days using walk-forward-validated
  labels built from strictly future returns. Swap in real regime labels
  (VIX thresholds, NBER recession dates, hand-labeled regimes) and/or
  richer features for a stronger model — the current honest accuracy
  (~25-30%) suggests the current feature set alone has little edge.
- **Transaction costs & slippage:** see "Transaction costs & risk
  management" above for the flat, dynamic, and cost-sensitivity options.
- **Long-only vs. long/short:** `generate_ewma_crossover_signals(..., flat_on_sell=True)`
  switches between going flat on a sell signal (long-only) vs. shorting.
- **Real risk-free data:** swap `historical_risk_free_rate()`'s hardcoded
  table for a real daily Treasury-yield series for more rigorous Sharpe
  ratios.
- **Point-in-time universes:** swap `multi_asset.DEFAULT_BASKET` for a
  real point-in-time constituents feed to remove the survivorship-bias
  caveat on cross-sectional results.

## Caveats

This is a research/education tool, not investment advice. Even with the
fixes in this README's changelog, backtests still carry residual
limitations worth naming plainly:

- The dynamic cost model is illustrative and not calibrated to any
  specific broker or venue.
- Position sizing and stop-losses are implemented at the single-strategy
  level, not as true portfolio-level risk management (no cross-asset
  correlation, no capital allocation across multiple simultaneous
  strategies).
- Cross-sectional validation uses a hard-coded current ticker list and is
  subject to survivorship bias (see "Generalization" above) unless you
  supply a point-in-time-correct universe yourself.
- Cross-regime validation uses synthetic, illustrative market scenarios,
  not real historical regimes.
- Data cleaning catches common free-feed failure modes, not everything
  that could be wrong with a vendor's data (see "Data quality" above).

Always validate on out-of-sample / walk-forward periods, across multiple
assets and regimes, and with realistic costs before trusting any result.
