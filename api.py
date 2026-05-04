from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import numpy as np
import requests
import time
import os
import json
import pickle

# Import calibrator wrapper classes so pickle can resolve them on load.
# These are produced by the purged-kfold paths of the training scripts.
try:
    from train_tbm_model_v2 import (
        IsotonicCalibratorWrapper as _PrimaryCalibratorWrapper,  # noqa: F401
    )
except ImportError:
    pass
try:
    from train_meta_labeler import (
        IsotonicCalibratorWrapper as _MetaCalibratorWrapper,  # noqa: F401
    )
except ImportError:
    pass

"""
Institutional FastAPI Backend: Universal Blindness Model
This microservice fetches live Binance Futures data, performs dynamic 
Rolling Z-Score normalization, and executes a sniper-threshold prediction.
"""

app = FastAPI(title="AI Trading Bot Institutional API")

# Configure CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for debugging connection issues
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Setup & Model Loading
model = xgb.XGBClassifier()
MODEL_PATH = "institutional_xgboost_model.json"

# Optional companion artifacts (created by upgraded train_*_model.py):
#   global_scaler.pkl                        — sklearn scaler fit on TRAINING data
#   tbm_xgboost_model_v2_calibrated.pkl      — sklearn CalibratedClassifierCV
#   tbm_xgboost_model_v2_threshold.json      — F1-optimised threshold
SCALER_PATH = "global_scaler.pkl"
CALIBRATOR_PATH = "tbm_xgboost_model_v2_calibrated.pkl"
THRESHOLD_PATH = "tbm_xgboost_model_v2_threshold.json"

# Optional meta-labeler (López de Prado §3.6).  When all three artifacts are
# present, the API chains primary → meta: a trade is acted upon ONLY if the
# primary signals AND the meta-labeler also agrees.  Meta never *adds* trades,
# only *removes* low-conviction ones.
META_CALIBRATOR_PATH = "meta_xgboost_model_calibrated.pkl"
META_THRESHOLD_PATH = "meta_xgboost_model_threshold.json"

# Optional regime-conditional thresholds.  When present, the API picks the
# threshold corresponding to the live row's regime instead of the global one.
REGIME_THRESHOLD_PATH = "tbm_xgboost_model_v2_regime_thresholds.json"

calibrator = None
training_scaler = None
meta_calibrator = None
meta_threshold = None
meta_feature_order = None
regime_thresholds = None  # dict: {"feature": str, "bins": [...], "thresholds": [...]}
PREDICT_THRESHOLD = 0.65  # legacy default; overridden below if sidecar exists

try:
    model.load_model(MODEL_PATH)
    print(f"[SUCCESS] Loaded institutional model: {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    # Fallback to older model if institutional is missing
    try:
        model.load_model("global_tbm_xgboost_model.json")
        print("[FALLBACK] Loaded global_tbm_xgboost_model.json")
    except:
        pass

# Load training-time scaler if available — without this, training z-scores
# (computed over the WHOLE training history) and inference z-scores
# (rolling 30-bar window below) come from different distributions, which
# silently degrades accuracy.  This was a major train/inference mismatch.
if os.path.exists(SCALER_PATH):
    try:
        with open(SCALER_PATH, "rb") as f:
            training_scaler = pickle.load(f)
        print(
            f"[SUCCESS] Loaded training scaler: {SCALER_PATH} — will use for normalization (train/inference parity)."
        )
    except Exception as e:
        print(
            f"[WARN] Could not load {SCALER_PATH}: {e}.  Falling back to rolling 30-bar z-score."
        )

# Load calibrated classifier if available — applies isotonic regression
# to raw XGBoost probabilities so that a threshold of 0.6 means
# "right 60% of the time" empirically.
if os.path.exists(CALIBRATOR_PATH):
    try:
        with open(CALIBRATOR_PATH, "rb") as f:
            calibrator = pickle.load(f)
        print(f"[SUCCESS] Loaded calibrator: {CALIBRATOR_PATH}")
    except Exception as e:
        print(
            f"[WARN] Could not load {CALIBRATOR_PATH}: {e}.  Using raw model probabilities."
        )

# Load tuned threshold (F1-optimal) if available — replaces the hardcoded 0.65.
if os.path.exists(THRESHOLD_PATH):
    try:
        with open(THRESHOLD_PATH) as f:
            tdata = json.load(f)
            PREDICT_THRESHOLD = float(tdata.get("threshold", 0.65))
        print(f"[SUCCESS] Tuned threshold: {PREDICT_THRESHOLD:.3f}")
    except Exception as e:
        print(f"[WARN] Could not load {THRESHOLD_PATH}: {e}.  Using default 0.65.")

# Load meta-labeler artifacts.  Both calibrator and threshold must be present.
if os.path.exists(META_CALIBRATOR_PATH) and os.path.exists(META_THRESHOLD_PATH):
    try:
        with open(META_CALIBRATOR_PATH, "rb") as f:
            meta_calibrator = pickle.load(f)
        with open(META_THRESHOLD_PATH) as f:
            mdata = json.load(f)
        meta_threshold = float(mdata["threshold"])
        meta_feature_order = mdata.get("feature_order")
        print(
            f"[SUCCESS] Loaded meta-labeler: threshold={meta_threshold:.3f}, "
            f"features={len(meta_feature_order) if meta_feature_order else '?'}"
        )
    except Exception as e:
        print(f"[WARN] Could not load meta-labeler: {e}. Disabling meta layer.")
        meta_calibrator = None
        meta_threshold = None

# Load regime-conditional thresholds (optional).
if os.path.exists(REGIME_THRESHOLD_PATH):
    try:
        with open(REGIME_THRESHOLD_PATH) as f:
            regime_thresholds = json.load(f)
        print(
            f"[SUCCESS] Loaded regime-conditional thresholds on "
            f"'{regime_thresholds.get('feature')}' with "
            f"{len(regime_thresholds.get('thresholds', []))} bins."
        )
    except Exception as e:
        print(f"[WARN] Could not load {REGIME_THRESHOLD_PATH}: {e}.")
        regime_thresholds = None


def _resolve_threshold(live_row: pd.DataFrame) -> tuple[float, str]:
    """Return (threshold, label) using the regime map if loaded, else global."""
    if regime_thresholds is None:
        return PREDICT_THRESHOLD, "global"
    feat = regime_thresholds.get("feature")
    bins = regime_thresholds.get("bins") or []
    ths = regime_thresholds.get("thresholds") or []
    if not feat or feat not in live_row.columns or len(ths) == 0:
        return PREDICT_THRESHOLD, "global-fallback"
    val = float(live_row[feat].iloc[-1])
    # bins are right-edges of each region; len(bins) == len(ths)-1.
    idx = 0
    for i, edge in enumerate(bins):
        if val >= edge:
            idx = i + 1
    idx = max(0, min(idx, len(ths) - 1))
    return float(ths[idx]), f"regime[{idx}]@{feat}={val:.3f}"


class PredictRequest(BaseModel):
    symbol: str


def _fallback_zscore_last_row(df: pd.DataFrame) -> None:
    """
    Legacy live-z-score normalization (used only when no training_scaler is
    available).  Z-scores the LAST row of df in-place, using stats from the
    most recent 30 bars.  The mismatch between this rolling-30 distribution
    and the training-time global distribution is a known accuracy degrader
    — train models with `train_global_model.py` so a `global_scaler.pkl`
    is available, then this function will not be called.
    """
    window = df.iloc[-30:]
    for col in ("volume", "open_interest", "funding_rate"):
        mean = window[col].mean()
        std = window[col].std()
        df.at[df.index[-1], col] = (df.iloc[-1][col] - mean) / (
            std if std and std != 0 else 1
        )


def get_weights_ffd(d, thres=1e-4):
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1])


def apply_frac_diff_live(series, d, thres=1e-4):
    """
    Applies Fractional Differencing to a live series.
    Uses convolution for efficiency and returns the last (current) value.
    """
    w = get_weights_ffd(d, thres)
    width = len(w)
    if len(series) < width:
        # If not enough data, use truncated weights for a rough estimate
        # or return NaN. To ensure production stability, we fetch more rows in fetcher.
        return np.nan

    vals = series.values
    output = np.convolve(vals, w, mode="valid")
    return float(output[-1])


# 2. The Live Fetch
def fetch_binance_data(symbol):
    """
    Fetches 500 hours of data with timestamp-based joining to ensure
    all features (OHLCV, OI, Funding) are perfectly aligned and non-NaN.
    """
    try:
        limit = 500
        # 1. OHLCV
        ohlcv_url = "https://fapi.binance.com/fapi/v1/klines"
        res_ohlcv = requests.get(
            ohlcv_url, params={"symbol": symbol, "interval": "1h", "limit": limit}
        )
        res_ohlcv.raise_for_status()
        df = pd.DataFrame(
            res_ohlcv.json(),
            columns=[
                "ts",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "ct",
                "qv",
                "nt",
                "tb",
                "tq",
                "i",
            ],
        )
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        # 2. Open Interest
        oi_url = "https://fapi.binance.com/futures/data/openInterestHist"
        res_oi = requests.get(
            oi_url, params={"symbol": symbol, "period": "1h", "limit": limit}
        )
        res_oi.raise_for_status()
        df_oi = pd.DataFrame(res_oi.json())
        if not df_oi.empty:
            df_oi["timestamp"] = pd.to_datetime(df_oi["timestamp"], unit="ms")
            df_oi.set_index("timestamp", inplace=True)
            df_oi["open_interest"] = df_oi["sumOpenInterest"].astype(float)
            df = df.join(df_oi[["open_interest"]], how="left")

        # 3. Funding Rate
        fr_url = "https://fapi.binance.com/fapi/v1/fundingRate"
        res_fr = requests.get(fr_url, params={"symbol": symbol, "limit": limit})
        res_fr.raise_for_status()
        df_fr = pd.DataFrame(res_fr.json())
        if not df_fr.empty:
            df_fr["timestamp"] = pd.to_datetime(df_fr["fundingTime"], unit="ms")
            df_fr.set_index("timestamp", inplace=True)
            df_fr["funding_rate"] = df_fr["fundingRate"].astype(float)
            df = df.join(df_fr[["funding_rate"]], how="left")

        # Final Clean: Forward fill gaps (like 8h funding) and zero-fill remaining
        df["open_interest"] = df["open_interest"].ffill().fillna(0)
        df["funding_rate"] = df["funding_rate"].ffill().fillna(0)

        return df
    except Exception as e:
        print(f"Data Fetch Error: {e}")
        return None


@app.post("/api/predict")
async def predict(request: PredictRequest):
    symbol = request.symbol.upper()
    print(f"\n--- Live Prediction Request: {symbol} ---")

    # Fetch data
    df = fetch_binance_data(symbol)
    if df is None or df.empty:
        raise HTTPException(
            status_code=500, detail="Failed to fetch data from Binance."
        )

    try:
        # 3. Dynamic Rolling Normalization & Feature Engineering
        # Calculate changes/rejections BEFORE normalization
        df["oi_change_1h"] = df["open_interest"].pct_change()
        df["volume_change_1h"] = df["volume"].pct_change()
        df["buying_rejection"] = df["high"] - df["close"]
        df["selling_rejection"] = df["close"] - df["low"]

        # Stationary Close (FracDiff d=0.4)
        w = get_weights_ffd(d=0.4, thres=1e-4)
        width = len(w)
        vals = df["close"].values

        # Safety Check: Ensure data is longer than the weights window
        if len(vals) >= width:
            fd_values = np.convolve(vals, w, mode="valid")
            full_fd_series = np.full(len(df), np.nan)
            full_fd_series[width - 1 :] = fd_values
            df["close_fd_04"] = full_fd_series
        else:
            print(
                f"Warning: Data length ({len(vals)}) is less than FracDiff window ({width})"
            )
            df["close_fd_04"] = 0.0  # Fallback

        # 3. Dynamic Rolling Normalization (CRITICAL)
        # If a training-time scaler is loaded, use it for *exact parity* with
        # how the model was trained.  Otherwise fall back to the live rolling
        # 30-bar z-score (legacy path — the train/inference distribution
        # mismatch this creates is documented as a known accuracy degrader).
        if training_scaler is not None:
            try:
                cols_to_scale = ["volume", "open_interest", "funding_rate"]
                # Scaler may be fit on more cols; pad with what we have.
                if hasattr(training_scaler, "feature_names_in_"):
                    cols_to_scale = [
                        c for c in training_scaler.feature_names_in_ if c in df.columns
                    ]
                scaled = training_scaler.transform(df[cols_to_scale])
                for k, c in enumerate(cols_to_scale):
                    df.loc[df.index[-1], c] = scaled[-1, k]
            except Exception as e:
                print(
                    f"[WARN] training_scaler.transform failed ({e}), falling back to rolling z-score"
                )
                _fallback_zscore_last_row(df)
        else:
            _fallback_zscore_last_row(df)

        # 4. Model Execution
        # Isolate the final formatted row
        feature_cols = [
            "volume",
            "open_interest",
            "funding_rate",
            "oi_change_1h",
            "volume_change_1h",
            "buying_rejection",
            "selling_rejection",
            "close_fd_04",
        ]

        live_row = df[feature_cols].iloc[-1:].copy()

        # Verify no NaNs in the prediction row
        if live_row.isnull().values.any():
            print(f"Warning: NaNs detected in live row: \n{live_row}")
            live_row.fillna(0, inplace=True)

        print(f"Live Normalized Row:\n{live_row.to_dict('records')[0]}")

        # Predict Probabilities — calibrated if available
        if calibrator is not None:
            probs = calibrator.predict_proba(live_row)
            confidence_kind = "calibrated"
        else:
            probs = model.predict_proba(live_row)
            confidence_kind = "raw"
        raw_confidence = float(probs[0][1])

        # 5. Sniper threshold — F1-optimal from training (regime-conditional
        # if a regime sidecar is loaded), otherwise legacy 0.65.  Compares the
        # **calibrated** probability against the **calibrated** threshold so
        # the decision boundary is meaningful.
        active_threshold, threshold_source = _resolve_threshold(live_row)
        primary_action = 1 if raw_confidence >= active_threshold else 0

        # 6. Meta-labeling overlay (López §3.6).  Meta only RUNS if primary
        #    signalled act, and only converts a primary 1 → 0; it can never
        #    flip a 0 → 1.  The final `action` is therefore strictly safer.
        meta_confidence = None
        if (
            primary_action == 1
            and meta_calibrator is not None
            and meta_threshold is not None
        ):
            try:
                meta_row = live_row.copy()
                meta_row["primary_proba"] = raw_confidence
                if meta_feature_order is not None:
                    # Reorder / select to match training-time column order.
                    missing = [
                        c for c in meta_feature_order if c not in meta_row.columns
                    ]
                    for c in missing:
                        meta_row[c] = 0.0
                    meta_row = meta_row[meta_feature_order]
                meta_proba = float(meta_calibrator.predict_proba(meta_row)[0][1])
                meta_confidence = meta_proba
                if meta_proba < meta_threshold:
                    final_action = 0
                    print(
                        f"Meta-labeler vetoed primary signal "
                        f"(meta={meta_proba:.4f} < {meta_threshold:.3f})"
                    )
                else:
                    final_action = 1
            except Exception as e:
                print(
                    f"[WARN] Meta-labeler failed at inference: {e}. Acting on primary."
                )
                final_action = primary_action
        else:
            final_action = primary_action

        print(
            f"{confidence_kind.title()} Confidence (Win): {raw_confidence:.4f}  "
            f"(threshold={active_threshold:.3f}, src={threshold_source})"
        )
        print(f"Final Action Decision: {final_action}")

        # 7. Return payload
        return {
            "symbol": symbol,
            "action": final_action,
            "primary_action": primary_action,
            "confidence": raw_confidence,
            "threshold": active_threshold,
            "threshold_source": threshold_source,
            "calibrated": calibrator is not None,
            "meta_confidence": meta_confidence,
            "meta_threshold": meta_threshold,
            "meta_active": meta_calibrator is not None and meta_threshold is not None,
        }

    except Exception as e:
        print(f"Prediction Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
