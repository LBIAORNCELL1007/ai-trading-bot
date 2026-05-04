"""
Regime-Conditional Threshold Tuner.

A single global decision threshold is suboptimal: model precision typically
varies systematically across volatility regimes (chop vs. breakout) and
across crowding regimes (low-OI vs. high-OI).  Holding one threshold over
all regimes leaves money on the table on the high-precision regimes and
generates avoidable losses on the low-precision regimes.

This script consumes the out-of-fold (OOF) predictions written by
``train_tbm_model_v2.py --purged-kfold`` and the calibrator trained there,
buckets the OOF rows by a chosen regime feature, and finds the F1-optimal
threshold per bucket.  The bucket edges and per-bucket thresholds are
persisted as ``tbm_xgboost_model_v2_regime_thresholds.json`` which
``api.py`` automatically loads.

Defaults to the ``volume`` feature with 4 quantile bins; both can be
overridden from the CLI.

Usage
-----
    python tune_regime_thresholds.py
    python tune_regime_thresholds.py --feature open_interest --n-bins 5
"""

from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Import the wrapper class so pickle can resolve it on load.
# The calibrator artifact is an instance of this class when produced by
# the purged-kfold path of train_tbm_model_v2.py.
try:
    from train_tbm_model_v2 import IsotonicCalibratorWrapper  # noqa: F401
except ImportError:
    pass


DATA_FILE = "fracdiff_alpha_dataset.csv"
OOF_FILE = "tbm_xgboost_model_v2_oof.csv"
CALIBRATOR_PATH = "tbm_xgboost_model_v2_calibrated.pkl"
OUTPUT_PATH = "tbm_xgboost_model_v2_regime_thresholds.json"


def find_best_threshold(y_true, probas, grid=None):
    # Wider grid so degenerate "always trade" / "never trade" optima surface.
    if grid is None:
        grid = np.arange(0.20, 0.851, 0.01)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (probas >= t).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = float(f), float(t)
    return best_t, best_f1


def main():
    parser = argparse.ArgumentParser(
        description="Compute regime-conditional decision thresholds from OOF predictions."
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="volume",
        help="Column from fracdiff_alpha_dataset.csv used to define regimes "
        "(default: volume).  Must be a numeric column also computed at inference time.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=4,
        help="Number of quantile bins (default: 4 -> quartiles).",
    )
    parser.add_argument(
        "--min-bin-size",
        type=int,
        default=200,
        help="Minimum rows required per bin to fit a per-regime threshold "
        "(default: 200).  Bins below this size fall back to the global threshold.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DATA_FILE,
        help=f"Input CSV (default: {DATA_FILE}).",
    )
    args = parser.parse_args()

    print(f"Loading dataset from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Loading OOF predictions from {OOF_FILE}...")
    oof = pd.read_csv(OOF_FILE)
    if len(oof) != len(df):
        raise ValueError(
            f"OOF length ({len(oof)}) does not match dataset length ({len(df)}). "
            "Re-run train_tbm_model_v2.py --purged-kfold to regenerate."
        )
    if args.feature not in df.columns:
        raise ValueError(
            f"Feature '{args.feature}' not in dataset columns. "
            f"Available: {list(df.columns)}"
        )

    # Calibrate the OOF probabilities the same way the live API does.
    print(f"Loading calibrator from {CALIBRATOR_PATH}...")
    with open(CALIBRATOR_PATH, "rb") as f:
        calib = pickle.load(f)

    # The calibrator wrapper expects a feature DataFrame; for the OOF path we
    # instead apply the underlying isotonic transform directly to the OOF
    # probabilities (those are already the model output).
    if hasattr(calib, "iso"):
        # IsotonicCalibratorWrapper from train_tbm_model_v2.py
        p_cal = np.clip(calib.iso.transform(oof["oof_proba"].values), 0.0, 1.0)
    elif hasattr(calib, "calibrated_classifiers_"):
        # sklearn CalibratedClassifierCV — reach in to re-apply just the
        # isotonic stage.  All inner classifiers share the same base model
        # in cv='prefit' mode, so any of them works.
        cc = calib.calibrated_classifiers_[0]
        p_cal = np.clip(cc.calibrators[0].transform(oof["oof_proba"].values), 0.0, 1.0)
    else:
        print("[WARN] Unknown calibrator type; using uncalibrated OOF probabilities.")
        p_cal = oof["oof_proba"].values

    y = df["tbm_label"].astype(int).values
    feat = df[args.feature].values

    # Global threshold (sanity baseline).
    global_t, global_f1 = find_best_threshold(y, p_cal)
    print(f"Global F1-optimal threshold: {global_t:.3f} (F1={global_f1:.4f})")

    # Quantile bin edges (interior cuts only — len = n_bins - 1).
    qs = np.linspace(0.0, 1.0, args.n_bins + 1)[1:-1]
    edges = [float(np.quantile(feat, q)) for q in qs]
    print(f"Bin edges on '{args.feature}' (quantiles {qs.tolist()}): {edges}")

    # Assign each row to a bin index in [0, n_bins-1].
    bin_idx = np.zeros(len(feat), dtype=int)
    for e in edges:
        bin_idx += (feat >= e).astype(int)

    per_bin_thresholds = []
    per_bin_diagnostics = []
    for b in range(args.n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        n_pos = int(y[mask].sum())
        if n_b < args.min_bin_size or n_pos < 10 or n_pos == n_b:
            print(
                f"  bin {b}: n={n_b} positives={n_pos}  -> falling back to global "
                f"({global_t:.3f})"
            )
            per_bin_thresholds.append(global_t)
            per_bin_diagnostics.append(
                {"bin": b, "n": n_b, "positives": n_pos, "f1": None, "fallback": True}
            )
            continue
        t_b, f1_b = find_best_threshold(y[mask], p_cal[mask])
        per_bin_thresholds.append(t_b)
        per_bin_diagnostics.append(
            {"bin": b, "n": n_b, "positives": n_pos, "f1": f1_b, "fallback": False}
        )
        print(
            f"  bin {b}: n={n_b} positives={n_pos}  -> threshold={t_b:.3f} "
            f"(F1={f1_b:.4f})"
        )

    payload = {
        "feature": args.feature,
        "bins": edges,
        "thresholds": per_bin_thresholds,
        "global_threshold": global_t,
        "global_f1": global_f1,
        "n_bins": args.n_bins,
        "min_bin_size": args.min_bin_size,
        "diagnostics": per_bin_diagnostics,
        "note": (
            "API selects threshold by finding the bin whose right-edge "
            "interval contains the live row's value of `feature`. "
            "If `feature` is absent at inference time the global threshold is used."
        ),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved regime-conditional thresholds to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
