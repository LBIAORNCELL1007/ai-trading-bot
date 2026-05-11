"""
One-Shot Pipeline Orchestrator + Before/After Comparison (multi-symbol).

Default configuration (set by user choice):
    Symbols   : BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT
    Timeframe : 1h
    History   : 365 days

This single script runs the full training pipeline twice — once on the
LEGACY single-split code path and once on the NEW purged-k-fold /
meta-labeling / regime-threshold path — then prints a side-by-side metrics
table so you can see, on your own data, how much the new stack moves
the needle.

Pipeline graph (multi-symbol)
-----------------------------

    build_global_dataset.py          → global_alpha_dataset.csv
        (TBM labels + fracdiff + barrier_hit_time, all in one step)
    │
    ├── (LEGACY)  train_tbm_model_v2.py --data global_alpha_dataset.csv
    │                                → tbm_xgboost_model_v2_threshold.json
    │                                  { method: "single_split", val_f1, val_brier }
    │                                  (stashed as *.legacy.json by this script)
    │
    └── (NEW)     train_tbm_model_v2.py --purged-kfold --data global_alpha_dataset.csv
                                     → tbm_xgboost_model_v2_threshold.json
                                       { method: "purged_kfold...", oof_f1, oof_brier }
                                       tbm_xgboost_model_v2_oof.csv
                  train_meta_labeler.py --data global_alpha_dataset.csv
                                     → meta_xgboost_model_threshold.json
                                       { oof_f1, oof_brier }
                  tune_regime_thresholds.py --data global_alpha_dataset.csv
                                     → tbm_xgboost_model_v2_regime_thresholds.json
                                       { thresholds[], global_threshold }

Usage
-----
    python run_full_pipeline.py
    python run_full_pipeline.py --refresh-data
    python run_full_pipeline.py --skip-data
    python run_full_pipeline.py --symbols BTCUSDT,ETHUSDT --days 730
    python run_full_pipeline.py --regime-feature open_interest --regime-bins 5

Notes
-----
* Long-running steps are skipped on re-runs unless you pass --refresh-data
  or remove ``global_alpha_dataset.csv``.
* Each step's stdout is streamed live AND captured to ``pipeline_logs/``.
* Exit code is non-zero if any required step fails.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
import universe as universe_lib


ROOT = Path(__file__).parent.resolve()
LOG_DIR = ROOT / "pipeline_logs"
LOG_DIR.mkdir(exist_ok=True)

# Artifact paths.  We use the multi-symbol global dataset throughout —
# train_tbm_model_v2.py et al accept a --data flag so they read from this
# file instead of the legacy single-symbol fracdiff_alpha_dataset.csv.
GLOBAL_CSV = ROOT / "global_alpha_dataset.csv"
PRIMARY_THRESHOLD_JSON = ROOT / "tbm_xgboost_model_v2_threshold.json"
PRIMARY_OOF_CSV = ROOT / "tbm_xgboost_model_v2_oof.csv"
META_THRESHOLD_JSON = ROOT / "meta_xgboost_model_threshold.json"
REGIME_THRESHOLD_JSON = ROOT / "tbm_xgboost_model_v2_regime_thresholds.json"

# Where we stash the LEGACY threshold JSON before overwriting it with the NEW run.
LEGACY_THRESHOLD_JSON = ROOT / "tbm_xgboost_model_v2_threshold.legacy.json"


class StepFailed(RuntimeError):
    pass


def run_step(name: str, cmd: list[str], log_file: Path) -> None:
    """
    Run a subprocess, streaming stdout to console AND a per-step log file.
    Raises StepFailed on non-zero exit.
    """
    print(f"\n{'=' * 78}")
    print(f">  {name}")
    print(f"   $ {' '.join(cmd)}")
    print(f"   log: {log_file.relative_to(ROOT)}")
    print(f"{'=' * 78}")

    t0 = time.time()
    with log_file.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        rc = proc.wait()
    dt = time.time() - t0
    print(f"\n   -> exit={rc}  elapsed={dt:.1f}s")
    if rc != 0:
        raise StepFailed(f"Step '{name}' failed with exit code {rc}. See {log_file}.")


def maybe_skip(name: str, target: Path, force: bool) -> bool:
    if force or not target.exists():
        return False
    print(
        f"\n[skip] {name} -- output already exists at {target.name} "
        f"(use --refresh-data to force)."
    )
    return True


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not parse {path.name}: {e}")
        return None


def fmt_num(v, digits: int = 4) -> str:
    if v is None:
        return "--"
    if isinstance(v, (int, float)):
        return f"{v:.{digits}f}"
    return str(v)


def print_comparison_table(
    legacy: Optional[dict],
    new: Optional[dict],
    meta: Optional[dict],
    regime: Optional[dict],
) -> None:
    print("\n" + "=" * 78)
    print("  BEFORE  ->  AFTER  comparison  (numbers come from the JSON sidecars)")
    print("=" * 78)

    def row(label, before, after, note=""):
        print(f"  {label:<32} {str(before):<14} {str(after):<14} {note}")

    print()
    print(f"  {'METRIC':<32} {'BEFORE':<14} {'AFTER':<14} NOTE")
    print(f"  {'-' * 32} {'-' * 14} {'-' * 14} {'-' * 30}")

    legacy_method = (legacy or {}).get("method", "single_split")
    new_method = (new or {}).get("method", "--")
    row("primary training method", legacy_method, new_method)

    legacy_f1 = (legacy or {}).get("val_f1")
    new_f1 = (new or {}).get("oof_f1")
    row(
        "primary F1 (val->OOF)",
        fmt_num(legacy_f1),
        fmt_num(new_f1),
        "AFTER is OOF (honest)" if new_f1 is not None else "",
    )

    legacy_brier = (legacy or {}).get("val_brier")
    new_brier = (new or {}).get("oof_brier")
    row("primary Brier (lower=better)", fmt_num(legacy_brier), fmt_num(new_brier))

    legacy_th = (legacy or {}).get("threshold")
    new_th = (new or {}).get("threshold")
    row("primary threshold", fmt_num(legacy_th, 3), fmt_num(new_th, 3))

    n_splits = (new or {}).get("n_splits", "--")
    embargo = (new or {}).get("embargo_pct", "--")
    row("CV folds / embargo", "1 / 0%", f"{n_splits} / {embargo}")

    # ─── New diagnostics: the numbers that actually decide live performance ────
    print()
    print(f"  {'PRIMARY HEALTH-CHECK':<32} {'BEFORE':<14} {'AFTER':<14} NOTE")
    print(f"  {'-' * 32} {'-' * 14} {'-' * 14} {'-' * 30}")
    legacy_epnl = (legacy or {}).get("val_expected_pnl_per_trade")
    new_epnl = (new or {}).get("oof_expected_pnl_per_trade")
    row(
        "expected PnL / trade (post-fee)",
        f"{legacy_epnl * 100:.3f}%" if legacy_epnl is not None else "--",
        f"{new_epnl * 100:.3f}%" if new_epnl is not None else "--",
        "negative = bot would lose money",
    )
    legacy_wr = (legacy or {}).get("val_win_rate")
    new_wr = (new or {}).get("oof_win_rate")
    row(
        "win rate at threshold",
        fmt_num(legacy_wr, 3),
        fmt_num(new_wr, 3),
        ">0.50 + fee/edge buffer needed",
    )
    legacy_tr = (legacy or {}).get("val_trade_rate")
    new_tr = (new or {}).get("oof_trade_rate")
    row(
        "trade rate (acts/bar)",
        fmt_num(legacy_tr, 4),
        fmt_num(new_tr, 4),
        "<0.001 -> threshold too tight",
    )
    row(
        "boundary_warning",
        str((legacy or {}).get("boundary_warning", "--")),
        str((new or {}).get("boundary_warning", "--")),
        "True = optimum at grid edge",
    )
    row(
        "fold_unstable",
        "n/a",
        str((new or {}).get("fold_unstable", "--")),
        "True = best_iter inconsistent",
    )

    print()
    print(f"  {'META-LABELER (act-only)':<32} {'BEFORE':<14} {'AFTER':<14}")
    print(f"  {'-' * 32} {'-' * 14} {'-' * 14}")
    row("meta F1 (OOF)", "n/a", fmt_num((meta or {}).get("oof_f1")))
    row("meta Brier (OOF)", "n/a", fmt_num((meta or {}).get("oof_brier")))
    row("meta threshold", "n/a", fmt_num((meta or {}).get("threshold"), 3))
    row(
        "primary->meta chain",
        "off",
        "ON" if meta else "off",
        "removes low-conviction trades only",
    )

    print()
    print(f"  {'REGIME-CONDITIONAL THRESHOLDS':<32} {'BEFORE':<14} {'AFTER':<14}")
    print(f"  {'-' * 32} {'-' * 14} {'-' * 14}")
    if regime:
        feat = regime.get("feature", "--")
        ths = regime.get("thresholds", [])
        glob = regime.get("global_threshold")
        row("regime feature", "--", str(feat))
        row("# bins", "1", str(len(ths)))
        row("global F1 (OOF)", "--", fmt_num(regime.get("global_f1")))
        if ths:
            spread = max(ths) - min(ths)
            row(
                "per-bin threshold spread",
                "0",
                f"{spread:.3f}",
                "0 = no regime adaptation",
            )
        row("global threshold (sanity)", fmt_num(legacy_th, 3), fmt_num(glob, 3))
    else:
        row("regime feature", "--", "n/a")
        row("# bins", "1", "n/a")

    print()
    print("-" * 78)
    print("  How to read this:")
    print("    * BEFORE used a single 70/15/15 split with the standard label leakage")
    print("      profile.  Its 'val_f1' was inflated by overlapping TBM horizons at")
    print("      the train/val boundary.")
    print("    * AFTER uses purged k-fold with sample-uniqueness weights.  OOF")
    print("      metrics are an honest estimate of LIVE performance -- typically")
    print("      LOWER than the leaky 'val_f1' but they actually predict reality.")
    print("    * Meta-labeler can ONLY remove trades, never add them.  Its OOF F1")
    print("      reports precision on the act-set; expect a precision lift at a")
    print("      moderate recall cost -- exactly the trade-off a sniper strategy")
    print("      wants.")
    print("    * Regime thresholds spread away from the global threshold whenever")
    print("      precision varies systematically across bins of the chosen feature.")
    print("      Spread > 0.05 means regime adaptation found real heterogeneity.")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full multi-symbol training pipeline (legacy + new) and "
            "print a before/after comparison."
        )
    )
    
    universe_lib.add_universe_args(parser)

    parser.add_argument(
        "--days",
        type=int,
        default=1095,
        help="Days of 1h history to fetch per symbol (default: 1095 = 3y).",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Re-fetch data even if global_alpha_dataset.csv already exists.",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Do not run build_global_dataset.py.  Requires the CSV to exist.",
    )
    parser.add_argument(
        "--skip-meta",
        action="store_true",
        help="Skip the meta-labeling stage.",
    )
    parser.add_argument(
        "--skip-regime",
        action="store_true",
        help="Skip the regime-threshold stage.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="K for purged k-fold (default 5).",
    )
    parser.add_argument(
        "--regime-feature",
        type=str,
        default="volume",
        help="Feature column to bin by for regime-conditional thresholds.",
    )
    parser.add_argument(
        "--regime-bins",
        type=int,
        default=4,
        help="Number of quantile bins for regime thresholds.",
    )
    args = parser.parse_args()

    py = sys.executable

    # Resolve universe and join into comma-separated string for subprocess flags
    selected_universe = universe_lib.resolve_universe(args)
    universe_str = ",".join(selected_universe)

    # ── 1. build_global_dataset.py ─────────────────────────────────────────
    if not args.skip_data and not maybe_skip(
        "build_global_dataset.py", GLOBAL_CSV, args.refresh_data
    ):
        run_step(
            "1. build_global_dataset.py",
            [
                py,
                "build_global_dataset.py",
                "--days",
                str(args.days),
                "--universe",
                "manual",
                "--symbols",
                universe_str,
                "--output",
                str(GLOBAL_CSV.name),
            ],
            LOG_DIR / "01_build_global_dataset.log",
        )
    if not GLOBAL_CSV.exists():
        raise StepFailed(f"Required artifact missing after step 1: {GLOBAL_CSV}")

    # ── 2a. LEGACY train (single 70/15/15 split) ───────────────────────────
    run_step(
        "2a. train_tbm_model_v2.py  (LEGACY single-split)",
        [py, "train_tbm_model_v2.py", "--data", str(GLOBAL_CSV.name)],
        LOG_DIR / "02a_train_v2_legacy.log",
    )
    if PRIMARY_THRESHOLD_JSON.exists():
        shutil.copy2(PRIMARY_THRESHOLD_JSON, LEGACY_THRESHOLD_JSON)
        print(f"   stashed legacy threshold -> {LEGACY_THRESHOLD_JSON.name}")
    legacy_data = load_json(LEGACY_THRESHOLD_JSON)

    # ── 2b. NEW train (purged k-fold + uniqueness weights) ─────────────────
    run_step(
        "2b. train_tbm_model_v2.py --purged-kfold (NEW)",
        [
            py,
            "train_tbm_model_v2.py",
            "--purged-kfold",
            "--n-splits",
            str(args.n_splits),
            "--data",
            str(GLOBAL_CSV.name),
        ],
        LOG_DIR / "02b_train_v2_purged.log",
    )
    new_data = load_json(PRIMARY_THRESHOLD_JSON)
    if not PRIMARY_OOF_CSV.exists():
        print(
            f"[WARN] {PRIMARY_OOF_CSV.name} missing -- meta + regime steps will be skipped."
        )
        args.skip_meta = True
        args.skip_regime = True

    # ── 3. train_meta_labeler.py ───────────────────────────────────────────
    meta_data = None
    if not args.skip_meta:
        try:
            run_step(
                "3. train_meta_labeler.py",
                [
                    py,
                    "train_meta_labeler.py",
                    "--n-splits",
                    str(args.n_splits),
                    "--data",
                    str(GLOBAL_CSV.name),
                ],
                LOG_DIR / "03_train_meta.log",
            )
            meta_data = load_json(META_THRESHOLD_JSON)
        except StepFailed as e:
            print(f"[WARN] {e}  Continuing without meta-labeler.")

    # ── 4. tune_regime_thresholds.py ───────────────────────────────────────
    regime_data = None
    if not args.skip_regime:
        try:
            run_step(
                "4. tune_regime_thresholds.py",
                [
                    py,
                    "tune_regime_thresholds.py",
                    "--feature",
                    args.regime_feature,
                    "--n-bins",
                    str(args.regime_bins),
                    "--data",
                    str(GLOBAL_CSV.name),
                ],
                LOG_DIR / "04_tune_regime.log",
            )
            regime_data = load_json(REGIME_THRESHOLD_JSON)
        except StepFailed as e:
            print(f"[WARN] {e}  Continuing without regime thresholds.")

    # ── 5. Comparison table ────────────────────────────────────────────────
    print_comparison_table(legacy_data, new_data, meta_data, regime_data)
    print(
        "\nFull per-step logs are in "
        f"{LOG_DIR.relative_to(ROOT)}/  (one file per step)."
    )


if __name__ == "__main__":
    try:
        main()
    except StepFailed as e:
        print(f"\n[ABORT] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[ABORT] interrupted by user")
        sys.exit(130)
