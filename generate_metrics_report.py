import pandas as pd
import numpy as np
import json
from sklearn.metrics import f1_score, brier_score_loss, classification_report

def generate_metrics_report():
    print("📊 Generating P1 Research Metrics Report...")
    
    # 1. Load OOF predictions
    oof_df = pd.read_csv("tbm_xgboost_model_v2_oof.csv")
    with open("tbm_xgboost_model_v2_threshold.json", "r") as f:
        thr_data = json.load(f)
    
    threshold = thr_data["threshold"]
    y_true = oof_df["y"]
    y_proba = oof_df["oof_proba"]
    y_pred = (y_proba >= threshold).astype(int)
    
    # 2. Overall Metrics
    f1 = f1_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_proba)
    
    # Acted trades metrics
    acted_mask = y_pred == 1
    num_trades = int(acted_mask.sum())
    trade_rate = num_trades / len(y_true)
    win_rate = float(y_true[acted_mask].mean()) if num_trades > 0 else 0.0
    
    # PnL Estimation (from threshold data)
    expected_pnl = thr_data["oof_expected_pnl_per_trade"]
    
    print("\n--- Out-of-Sample (OOF) Performance ---")
    print(f"F1 Score:         {f1:.4f}")
    print(f"Brier Score:      {brier:.4f} (lower is better calibration)")
    print(f"Trade Rate:       {trade_rate:.2%} ({num_trades} trades)")
    print(f"Win Rate:         {win_rate:.2%}")
    print(f"Expected PnL:     {expected_pnl:.4%}/trade (incl. fees)")
    
    # 3. Per-Symbol Analysis
    # We need to re-load the dataset to get symbols
    # Note: Using the processed dataset generated in previous step
    data_df = pd.read_parquet("data/processed/p1_research_dataset.parquet")
    # Align OOF with data_df (they should match in size after dropping tail)
    # The trainer drops 24 tail rows per symbol.
    
    symbol_metrics = []
    # Simple alignment for report (approximation if exact index was lost during train-drop)
    # In practice, we'd save (timestamp, symbol, proba) in the OOF file.
    # For now, we'll just report the global OOF vs the per-symbol raw counts.
    
    print("\n--- Per-Symbol Dataset Balance ---")
    for sym in data_df["symbol"].unique():
        sym_data = data_df[data_df["symbol"] == sym]
        pos_rate = sym_data["tbm_label"].mean()
        print(f"{sym:10}: {len(sym_data):5} rows | {pos_rate:.2%} positive labels")

    # 4. Calibration Quality
    print("\n--- Calibration Check ---")
    # Group by proba bins
    bins = np.linspace(0, 1, 11)
    oof_df["bin"] = pd.cut(y_proba, bins)
    cal_table = oof_df.groupby("bin")["y"].agg(["mean", "count"])
    print(cal_table)

    print("\n✨ Metrics Report Complete.")

if __name__ == "__main__":
    generate_metrics_report()
