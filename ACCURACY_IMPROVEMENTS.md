# Accuracy & Win-Rate Improvement Report

This document summarizes the audit findings and the corrective changes made to
the AI trading bot's accuracy-critical paths. Changes span both the TypeScript
live-trading stack and the Python ML training pipeline.

The guiding principle throughout is the conventions of *Advances in Financial
Machine Learning* (López de Prado): no look-ahead, purged/embargoed splits,
sample-uniqueness-aware weighting, probability calibration, and meta-decision
gates over raw model outputs.

---

## 1. Critical bugs eliminated

### 1.1 Silent training on random data (CATASTROPHIC)

`train_tcn.py` would silently generate a 2,000-row random dataset and write it
to `btc_training_dataset.csv` whenever the real dataset was missing. Any model
produced that way had **zero predictive signal** but was indistinguishable on
disk from a genuinely trained one.

**Fix:** the dummy-data branch now `raise FileNotFoundError(...)` with
instructions to run `build_dataset.py` and `tbm_labeler.py`. There is no longer
any code path that produces a synthetic-trained model.

### 1.2 Feature-list mismatch (silent zero-row training)

`train_tcn.py` referenced features `close_fd`, `rsi_14`, `MACD_12_26_9`, and
`atr_14` — none of which were emitted by the data pipeline (the pipeline writes
`close_fd_04` and does not compute RSI/MACD/ATR at all). The result was a
silent empty training set.

**Fix:** added `_ensure_features()` that, if any of the four expected columns
are missing, computes them from raw OHLCV using:
- `close_fd_04` — fractional differentiation fallback (or alias of pipeline output).
- `rsi_14` — Wilder's smoothing (not simple mean).
- `MACD_12_26_9` — EMA(12)–EMA(26) MACD histogram with EMA(9) signal.
- `atr_14` — Wilder ATR over true range.
Raises `RuntimeError` if zero usable rows remain after dropna.

### 1.3 Future-data leakage via `bfill()`

`build_dataset.py`, `build_global_dataset.py`, and `build_institutional_dataset.py`
all called `.ffill().bfill()` on `funding_rate` and `open_interest`. The
backfill leaks **future** values into past rows.

**Fix:** changed to `.ffill().fillna(0.0)` (forward-fill only; zero is the
neutral prior at the start of history).

### 1.4 Direct target leakage in `train_model.py`

`target_return_4h` (the regression target = `close.shift(-4)`) was being passed
in as a feature, alongside `barrier_hit_time` (timestamp at which the barrier
was hit — a future-looking column).

**Fix:** `DROP_COLS` extended to
`['timestamp', 'close_fd', 'barrier_hit_time', 'target_return_4h', 'barrier_hit_label']`
with explanatory comments.

### 1.5 Same leakage in `train_tbm_model.py` / `train_tbm_model_v2.py`

Both omitted `barrier_hit_time` from `DROP_COLS`.

**Fix:** added `barrier_hit_time` and `close_fd` to `DROP_COLS` in both.

### 1.6 Train/inference scaler mismatch

`train_global_model.py` saves a `StandardScaler` to `global_scaler.pkl`, but
`api.py` ignored it and applied a rolling-30 z-score at inference time.
Different distribution → systematically biased probabilities.

**Fix:** `api.py` now loads `global_scaler.pkl` if present and applies it; falls
back to the legacy rolling z-score only if the file is missing (logs which
mode is active).

### 1.7 Hardcoded uncalibrated decision threshold

`api.py` hardcoded `0.65` against raw, uncalibrated XGBoost probabilities.

**Fix:** loads `tbm_xgboost_model_v2_calibrated.pkl` (isotonic-calibrated
wrapper) when available, and reads the threshold from
`tbm_xgboost_model_v2_threshold.json` (validation-F1-optimal). Response now
returns `threshold` and `calibrated` fields for transparency.

### 1.8 Asymmetric / close-only triple-barrier labeling

`tbm_labeler.py` used 1.5% upper / -1.0% lower thresholds (asymmetric, biases
the dataset positive) and only checked `close` prices, missing intra-bar barrier
touches.

**Fix:** full rewrite — symmetric ±1.5%, uses `high`/`low` for intra-bar
detection, conservative SL-first ordering when both barriers are touched in
the same bar, and exposes `barrier_hit_time` for downstream
sample-uniqueness weighting.

---

## 2. Pipeline upgrades for accuracy

### 2.1 Embargoed sequential splits

`train_tbm_model_v2.py` and the new v2-style `train_institutional_model.py` now
insert a 1% embargo gap between train/val and val/test splits. This prevents
overlapping triple-barrier labels at the split boundary from leaking
information across folds (López de Prado §7.4).

### 2.2 Probability calibration

Both `train_tbm_model_v2.py` and `train_institutional_model.py` now wrap the
fitted booster in `CalibratedClassifierCV(method='isotonic', cv='prefit')`,
fit on the embargoed validation slice. Downstream confidence gating, Kelly
sizing, and threshold optimization all assume calibrated probabilities — this
is the prerequisite that makes the rest of the stack honest.

### 2.3 Threshold optimization

A `find_best_threshold` helper scans 0.30–0.80 in 51 steps and selects the F1-
maximizing threshold on the validation set. The chosen threshold is persisted
to `*_threshold.json` and consumed by `api.py`.

### 2.4 Per-trade artifact set

Each trainer now produces a coherent triple:
- `*.json` / `*.pkl` — raw booster (for inspection / re-calibration).
- `*_calibrated.pkl` — calibrated wrapper (what inference should use).
- `*_threshold.json` — operating point + dataset metadata.

---

## 3. TypeScript live-trading fixes (accuracy & win-rate)

### 3.1 Indicators (`lib/technical-indicators.ts`)

Full rewrite. Key fixes:
- **Wilder smoothing** for RSI, ADX, ATR (was simple-mean, which under-weights
  recent moves and lags reversals).
- **Streaming MACD** (state-preserving EMAs, not naive recompute on each bar).
- **Time-shifted Ichimoku** (Senkou A/B shifted +26, Chikou shifted -26 — the
  prior implementation read the "future" line at the current bar).
- **Bollinger Band squeeze** measured as a percentile of BB-width history,
  enabling true volatility-regime detection.
- **`prev` field** on every indicator output, so cross detection in
  `trading-strategies.ts` no longer needs ad-hoc state.
- **Regime classifier** (trend / range / squeeze / breakout) from ADX + BB
  squeeze + slope.

### 3.2 Strategies (`lib/trading-strategies.ts`)

- Replaced state-equality checks with **event-based crosses** (uses `prev` ↔
  `current` transitions). Previously a strategy fired every bar a condition
  held, instead of only at the transition — this dominated the false-positive
  rate.
- Added **ADX gate** (≥20 by default) and **volume gate** (volume ≥ MA20) to
  filter weak/illiquid setups.
- New **`MultiSignalStrategy`** with weighted voting across all sub-strategies;
  a trade fires only when weighted score crosses a configurable threshold.

### 3.3 Strategy engine (`lib/strategy-engine.ts`)

- Fixed a **time-exit unit bug** that compared milliseconds to bar counts.
- Implemented **Kelly-fraction position sizing** (fractional Kelly capped).
- Implemented **PAVA isotonic calibration** for in-engine probability
  rectification (matches the Python pipeline so the gate is consistent).
- **Parametric Sharpe annualization** with proper bars-per-year scaling.

### 3.4 Risk management (`lib/risk-management.ts`)

- New `setRiskRewardATR(...)` helper: stop-loss at `entry ± k_sl·ATR`, take-
  profit at the symmetric R-multiple. Replaces hardcoded percentage stops.

### 3.5 Backtester (`lib/backtester.ts`)

- **Next-bar fill** for all signals (eliminates same-bar look-ahead).
- **Intra-bar SL/TP** check using `high`/`low` (was close-only).
- **SL-first conservative ordering** when both barriers touched in the same bar.
- Fees + slippage applied on every fill.
- **Force-close at end** so final equity is realized, not unrealized.

### 3.6 AI config engine (`lib/ai-config-engine.ts`)

- **Volume-node SL snapping** now requires the snap distance to be at least
  `max(1×ATR, 0.8%)` and adds a 0.25-ATR buffer. Prevents the previous
  behavior of placing SL inside typical bar noise (the dominant cause of
  premature stop-outs).
- **MULTI_SIGNAL fallback footgun** fixed: only fall back to second-best when
  the top-3 are within 5 points AND none of them is itself MULTI_SIGNAL
  (otherwise the system would oscillate between MULTI_SIGNAL and a
  near-identical sub-strategy).
- **`runPaperBacktest`** rewritten for next-bar fills, intra-bar SL/TP/trail
  with conservative SL-first ordering, and proper fee/slippage accounting.

### 3.7 Trading engine (`lib/trading-engine.ts`)

- **Confidence gate**: trades require calibrated probability ≥ configured
  threshold.
- **ADX gate**: trades require ADX ≥ configured floor.
- **One eval per closed candle** (was 5-second polling, which double-counted
  signals and inflated trade count).
- **ATR sizing** via `setRiskRewardATR`.
- **Max-hold-bars time exit**.
- **Websocket-failure REST fallback** so a transient WS drop no longer freezes
  decisions.
- **Agentic orchestrator wired in**:
  - Vetoes entries when `decision.confidence < 0.5`.
  - Skips entries when regime-mismatch confidence ≥ 0.75.
  - Scales `riskMultiplier` and `tpMultiplier` per trade.
  - Calls `recordTradeResult` on every close to feed the learning loop.

### 3.8 ML models (`lib/ml-models.ts`)

- `predictDirection` now accepts an optional calibrator and applies it to the
  ensemble output.
- New `fitCalibrator` (PAVA on a held-out validation set).
- New `optimizeThreshold` that scans 0.50–0.75 and picks the maximizer of
  `Sharpe + 0.1·winRate`, gated by a minimum-trade-count floor.

### 3.9 Walk-forward analyzer (`lib/walk-forward-analyzer.ts`)

- Full rewrite. `TAKER_FEE_PCT` and `SLIPPAGE_PCT` now applied via a single
  `recordTrade` helper across **every** simulator (all simulators were
  previously fee-free, which inflated reported edge).
- Per-trade returns + Sharpe now tracked.
- New `rollingWindows` + `runRollingWalkForward` implementing proper N-fold WFA
  with mean-OOS-Sharpe minus `std/2` stability penalty.
- Legacy `runAnalysis` now ranks by **in-sample Sharpe**, not OOS-PnL — this
  removes the data-snooping bias that came from selecting on OOS performance.
- `BacktestResult` extended with `inSampleSharpe`, `outOfSampleSharpe`,
  `foldSharpes`; `WFAReport.foldsUsed` added.

---

## 4. Why each change moves the win rate

| Change | Mechanism by which win rate / Sharpe improves |
|---|---|
| Calibrated probabilities | Confidence gate is honest; high-conf trades actually have high realized win rate. |
| Embargoed splits | Reported validation skill is closer to live skill — fewer false-positive deployments. |
| Wilder indicators | Match every textbook/exchange — fewer regime misclassifications. |
| Event-based crosses | Removes streak of duplicate signals on every bar inside a regime; cuts false positives drastically. |
| ADX + volume gates | Filters chop and illiquid bars where edge is statistically zero. |
| Intra-bar SL/TP | Stops trigger when they actually would have — eliminates fictitious "saved" trades. |
| Fee + slippage in WFA | Strategy ranking now reflects net edge, not gross. |
| In-sample Sharpe ranking | Eliminates OOS-snooping → lower deployment regret. |
| Volume-node SL buffer | Stops sit beyond bar-noise → fewer premature stop-outs. |
| ATR sizing | Position size adapts to volatility regime → smoother equity curve, higher risk-adjusted return. |
| Agentic orchestrator wired | Regime/confidence vetoes block the worst setups; learning loop adapts to drift. |
| Symmetric TBM labels | Removes positive-class bias from the training set → calibrated direction. |
| Forward-fill only on funding/OI | Removes future leak that artificially inflated training accuracy. |
| Refusing to train on dummy data | Eliminates the only failure mode where a deployed model has zero predictive signal. |

---

## 5. Operating-procedure notes

- After any pipeline change, regenerate artifacts in this order:
  1. `python build_dataset.py` (or `build_institutional_dataset.py`)
  2. `python tbm_labeler.py`
  3. `python train_global_model.py` → produces `global_scaler.pkl`.
  4. `python train_tbm_model_v2.py` → produces calibrated booster + threshold.
  5. (Optional) `python train_institutional_model.py` for the institutional set.
- `api.py` will refuse to silently fall back. Watch its startup logs for which
  model, scaler, calibrator, and threshold it loaded — all four lines should
  read "loaded" in production.
- For walk-forward studies, prefer `runRollingWalkForward` over the legacy
  `runAnalysis`; report mean-OOS-Sharpe ± std across folds.

---

## 6. Remaining recommended work (not yet implemented)

These were considered but are gated on explicit go-ahead because they touch
training cost or model topology:

1. **Purged k-fold CV with sample-uniqueness weights** (López de Prado §4.5).
   Now that `barrier_hit_time` is exposed by `tbm_labeler.py`, the inputs are
   ready; only the CV loop is missing.
2. **Meta-labeling layer** — train a second classifier on the primary model's
   outputs to decide whether to act. Requires a primary-model-trade log.
3. **De-duplicate `tbm.py` vs `tbm_labeler.py`** — two TBM implementations
   exist; consolidate to the rewritten `tbm_labeler.py`.
4. **Adaptive thresholds** that tighten in chop and loosen in trend — currently
   one global threshold per model.

---
