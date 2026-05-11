"""
Research Validation Pipeline for AI Trading Bot.
Produces ablation study, walk-forward simulation, visualizations, and research-grade markdown report.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, brier_score_loss, confusion_matrix, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import json
import os
import argparse
from datetime import datetime, timezone
import time
import matplotlib.pyplot as plt
import seaborn as sns

from build_global_dataset import BinanceDataLoader, FeatureEngineer, DataQualityAuditor
from tbm_labeler import apply_triple_barrier

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_theme(style="whitegrid")

def compute_drawdown(equity_series):
    peak = equity_series.expanding(min_periods=1).max()
    dd = (equity_series - peak) / peak
    return dd

def get_drawdown_table(equity_curve):
    if equity_curve.empty:
        return pd.DataFrame(columns=["Start Date", "End Date", "Max Drawdown (%)"])
        
    peak = equity_curve["equity"].expanding(min_periods=1).max()
    dd = (equity_curve["equity"] - peak) / peak
    equity_curve["drawdown"] = dd
    
    # Find max drawdowns
    is_dd = dd < 0
    starts = []
    ends = []
    max_dds = []
    
    in_dd = False
    start_idx = None
    max_dd = 0
    
    for idx, val in dd.items():
        timestamp = equity_curve.loc[idx, "timestamp"]
        if val < 0 and not in_dd:
            in_dd = True
            start_idx = timestamp
            max_dd = val
        elif val < 0 and in_dd:
            if val < max_dd:
                max_dd = val
        elif val == 0 and in_dd:
            in_dd = False
            starts.append(start_idx)
            ends.append(timestamp)
            max_dds.append(max_dd)
            
    if in_dd:
        starts.append(start_idx)
        ends.append(equity_curve.iloc[-1]["timestamp"])
        max_dds.append(max_dd)
        
    dd_df = pd.DataFrame({
        "Start Date": starts,
        "End Date": ends,
        "Max Drawdown (%)": [x * 100 for x in max_dds]
    })
    return dd_df.sort_values("Max Drawdown (%)").head(5)

def create_visualizations(results, taken_trades, dd_table):
    print("Generating Visualizations...")
    
    # 1. Equity Curve
    plt.figure(figsize=(12, 6))
    if not taken_trades.empty:
        plt.plot(taken_trades["timestamp"], taken_trades["equity"], label="Account Equity", color="#2ecc71", linewidth=2)
        plt.fill_between(taken_trades["timestamp"], 10000, taken_trades["equity"], where=(taken_trades["equity"] >= 10000), color="#2ecc71", alpha=0.3)
        plt.fill_between(taken_trades["timestamp"], 10000, taken_trades["equity"], where=(taken_trades["equity"] < 10000), color="#e74c3c", alpha=0.3)
    plt.axhline(y=10000, color='black', linestyle='--', alpha=0.5)
    plt.title("Out-of-Sample Cumulative Performance (Full Pipeline)", fontsize=14, fontweight='bold')
    plt.xlabel("Timeline", fontsize=12)
    plt.ylabel("Equity (USDT)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("equity_curve.png", dpi=150)
    plt.close()

    # 2. Drawdown Plot
    plt.figure(figsize=(12, 4))
    if not taken_trades.empty:
        dd_series = (taken_trades["equity"] - taken_trades["equity"].expanding().max()) / taken_trades["equity"].expanding().max() * 100
        plt.fill_between(taken_trades["timestamp"], 0, dd_series, color="#e74c3c", alpha=0.6)
    plt.title("Portfolio Drawdown Profile (%)", fontsize=14, fontweight='bold')
    plt.ylabel("Drawdown %", fontsize=12)
    plt.ylim(None, 0)
    plt.tight_layout()
    plt.savefig("drawdown_profile.png", dpi=150)
    plt.close()

    # 3. Ablation Study Comparison
    #
    # Keep financial and classification metrics on separate panels.  The
    # previous dual-axis chart mixed negative expected PnL bars with a
    # compressed accuracy line, which made the ablation look visually better
    # than the economics actually were.  For a research paper, the first read
    # should be: "all configurations are still negative after friction."
    ablation_data = []
    # Ensure correct order
    order = ["1_Baseline", "2_Plus_FFD", "3_Plus_TBM", "4_Plus_Funding", "5_Plus_Calibration", "6_Plus_Risk_Filters"]
    for conf in order:
        if conf in results:
            m = results[conf]
            ablation_data.append({
                "Config": conf.replace("_", " "),
                "Accuracy": m["Accuracy"],
                "F1 Score": m["F1"],
                "Exp PnL": m["Expected PnL/Trade"] * 100 # Convert to %
            })
    ablation_df = pd.DataFrame(ablation_data)

    fig, ax_pnl = plt.subplots(figsize=(11, 6.2))
    colors = ["#d95f5f" if v < 0 else "#2ca25f" for v in ablation_df["Exp PnL"]]
    bars = ax_pnl.barh(
        ablation_df["Config"],
        ablation_df["Exp PnL"],
        color=colors,
        alpha=0.86,
        edgecolor="#333333",
        linewidth=0.7,
    )
    ax_pnl.axvline(0, color="#222222", linewidth=1.2)
    ax_pnl.set_xlabel("Expected PnL / Trade (%)", fontsize=11, fontweight="bold")
    ax_pnl.set_ylabel("Pipeline Configuration", fontsize=11, fontweight="bold")
    ax_pnl.set_title(
        "Ablation Study: Expected PnL After Friction\n"
        "All configurations include 9 bps friction; bars left of zero lose money per acted trade",
        fontsize=14,
        fontweight="bold",
    )
    ax_pnl.grid(axis="x", alpha=0.35)

    y_min = min(ablation_df["Exp PnL"].min() * 1.25, -0.02)
    y_max = max(ablation_df["Exp PnL"].max() * 0.25, 0.02)
    ax_pnl.set_xlim(y_min, y_max)
    for bar, value in zip(bars, ablation_df["Exp PnL"]):
        ax_pnl.annotate(
            f"{value:+.3f}%",
            xy=(value, bar.get_y() + bar.get_height() / 2),
            xytext=(-8 if value < 0 else 8, 0),
            textcoords="offset points",
            ha="right" if value < 0 else "left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#222222",
        )

    best_idx = ablation_df["Exp PnL"].idxmax()
    worst_idx = ablation_df["Exp PnL"].idxmin()
    ax_pnl.annotate(
        "Least negative",
        xy=(ablation_df.loc[best_idx, "Exp PnL"], best_idx),
        xytext=(y_max * 0.75, best_idx),
        arrowprops={"arrowstyle": "->", "color": "#2ca25f"},
        ha="left",
        va="center",
        fontsize=9,
        color="#2ca25f",
        fontweight="bold",
    )
    ax_pnl.annotate(
        "Worst",
        xy=(ablation_df.loc[worst_idx, "Exp PnL"], worst_idx),
        xytext=(y_min * 0.92, worst_idx),
        arrowprops={"arrowstyle": "->", "color": "#b2182b"},
        ha="right",
        va="center",
        fontsize=9,
        color="#b2182b",
        fontweight="bold",
    )
    ax_pnl.invert_yaxis()
    fig.tight_layout()
    fig.savefig("ablation_pnl.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, ax_cls = plt.subplots(figsize=(12, 5.2))
    x_coords = np.arange(len(ablation_df))
    ax_cls.plot(
        x_coords,
        ablation_df["Accuracy"],
        color="#1f77b4",
        marker="o",
        linewidth=2.4,
        label="Accuracy",
    )
    ax_cls.plot(
        x_coords,
        ablation_df["F1 Score"],
        color="#ff7f0e",
        marker="s",
        linewidth=2.4,
        label="F1 Score",
    )
    ax_cls.axhline(0.50, color="#777777", linestyle="--", linewidth=1, alpha=0.8)
    ax_cls.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax_cls.set_xlabel("Pipeline Configuration", fontsize=11, fontweight="bold")
    ax_cls.set_ylim(0.45, 0.58)
    ax_cls.set_xticks(x_coords)
    ax_cls.set_xticklabels(ablation_df["Config"], rotation=15, ha="right")
    ax_cls.grid(axis="y", alpha=0.35)
    ax_cls.legend(loc="upper right", frameon=True)

    ax_cls.set_title(
        "Ablation Study: Classification Metrics",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig("ablation_classification.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # 4. Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    cm = np.array(results["6_Plus_Risk_Filters"]["Confusion Matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred Loss", "Pred Win"],
                yticklabels=["Actual Loss", "Actual Win"])
    plt.title("Confusion Matrix: Optimized Pipeline", fontsize=14, fontweight='bold')
    plt.ylabel("Ground Truth")
    plt.xlabel("Model Prediction")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

def run_ablation_study(df):
    print("Running Ablation Study...")
    
    df = df.sort_index()
    
    # Feature sets
    tech_feats = [
        "volume", "volume_change_1h", "buying_rejection", "selling_rejection", 
        "realized_vol_24h", "rsi_14", "volume_zscore_24h"
    ]
    ffd_feat = ["close_fd_04"]
    funding_feats = [
        "funding_rate", "funding_change_8h", "funding_zscore_7d", "funding_sign_streak"
    ]
    
    configs = {
        "1_Baseline": {"features": tech_feats, "label": "fixed_24h_label"},
        "2_Plus_FFD": {"features": tech_feats + ffd_feat, "label": "fixed_24h_label"},
        "3_Plus_TBM": {"features": tech_feats + ffd_feat, "label": "tbm_label"},
        "4_Plus_Funding": {"features": tech_feats + ffd_feat + funding_feats, "label": "tbm_label"},
        "5_Plus_Calibration": {"features": tech_feats + ffd_feat + funding_feats, "label": "tbm_label", "calibrate": True},
        "6_Plus_Risk_Filters": {"features": tech_feats + ffd_feat + funding_feats, "label": "tbm_label", "calibrate": True, "risk_filters": True}
    }
    
    # Define rolling window splits
    total_len = len(df)
    split1 = int(total_len * 0.5)
    split2 = int(total_len * 0.66)
    split3 = int(total_len * 0.83)
    
    folds = [
        (df.iloc[:split1], df.iloc[split1:split2]),
        (df.iloc[:split2], df.iloc[split2:split3]),
        (df.iloc[:split3], df.iloc[split3:])
    ]
    
    results = {}
    all_preds_config6 = []
    
    for conf_name, conf in configs.items():
        print(f"  Evaluating {conf_name}...")
        feats = conf["features"]
        label_col = conf["label"]
        
        y_true_all = []
        y_pred_all = []
        y_proba_all = []
        pnl_all = []
        
        for train_df, test_df in folds:
            # Embargo
            train_df = train_df.iloc[:-24]
            
            X_train = train_df[feats]
            y_train = train_df[label_col].astype(int)
            
            X_test = test_df[feats]
            # ALWAYS evaluate against tbm_label for consistent metric comparison
            y_test = test_df["tbm_label"].astype(int)
            
            model = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            if conf.get("calibrate"):
                calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
                calibrator.fit(X_train, y_train) 
                y_proba = calibrator.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.predict_proba(X_test)[:, 1]
                
            # Config 6 uses a tighter threshold as a "Risk Filter"
            threshold = 0.55 if conf.get("risk_filters") else 0.5
            y_pred = (y_proba >= threshold).astype(int)
            
            y_true_all.extend(y_test.values)
            y_pred_all.extend(y_pred)
            y_proba_all.extend(y_proba)
            
            # ALWAYS use tbm_return for PnL benchmark to ensure apples-to-apples
            test_returns = test_df["tbm_return"].values
            
            for i in range(len(y_pred)):
                if y_pred[i] == 1:
                    # Apply friction to ALL configs to show realistic institutional edge
                    gross_ret = test_returns[i]
                    net_ret = gross_ret - 0.0009
                    pnl_all.append(net_ret)
                    
                    if conf_name == "6_Plus_Risk_Filters":
                        all_preds_config6.append({
                            "timestamp": test_df.index[i],
                            "symbol": test_df["symbol"].iloc[i],
                            "proba": y_proba[i],
                            "label": y_test.iloc[i],
                            "net_return": net_ret
                        })
                else:
                    pnl_all.append(0.0)

        y_t = np.array(y_true_all)
        y_p = np.array(y_pred_all)
        y_prob = np.array(y_proba_all)
        pnl = np.array(pnl_all)
        
        f1 = f1_score(y_t, y_p, zero_division=0)
        brier = brier_score_loss(y_t, y_prob)
        acc = accuracy_score(y_t, y_p)
        cm = confusion_matrix(y_t, y_p).tolist()
        
        trades = np.sum(y_p)
        # Expected PnL per trade when the model says "BUY"
        exp_pnl = np.mean(pnl[y_p == 1]) if trades > 0 else 0.0
        wr = np.sum(pnl[y_p == 1] > 0) / trades if trades > 0 else 0.0
        
        results[conf_name] = {
            "Accuracy": acc,
            "F1": f1,
            "Brier": brier,
            "Trades": int(trades),
            "Win Rate": wr,
            "Expected PnL/Trade": exp_pnl,
            "Confusion Matrix": cm
        }
        
    return results, pd.DataFrame(all_preds_config6)


def generate_markdown(results, dd_table):
    md = "# 🛡️ Institutional Research Validation Report\n\n"
    md += "## 📈 Executive Summary\n"
    md += "This report details the rolling walk-forward validation of the AI Trading Bot. We evaluate out-of-sample performance through an **ablation study**, systematically adding layers of institutional realism to isolate the impact of each component (Fractional Differencing, Triple Barrier Method, Microstructure Features, Calibration, and Realistic Risk Filters).\n\n"
    
    md += "## 🔬 1. Ablation Study\n"
    md += "Adding complexity only makes sense if it improves metrics. We track Accuracy, F1 Score, and Expected PnL as we move from a simple baseline to the full institutional pipeline. The chart separates economics from classification scores because accuracy near 50% is not sufficient if expected PnL remains negative after friction.\n\n"
    md += "### Expected PnL After Friction\n"
    md += "![Ablation Expected PnL](ablation_pnl.png)\n\n"
    md += "### Classification Metrics\n"
    md += "![Ablation Classification Metrics](ablation_classification.png)\n\n"
    
    md += "| Configuration | Accuracy | F1 Score | Brier | Trades | Win Rate | Exp. PnL |\n"
    md += "|---|---|---|---|---|---|---|\n"
    
    for conf, m in results.items():
        md += f"| {conf.replace('_', ' ')} | {m['Accuracy']:.4f} | {m['F1']:.4f} | {m['Brier']:.4f} | {m['Trades']} | {m['Win Rate']:.2%} | {m['Expected PnL/Trade']:.4%} |\n"
        
    md += "\n## 📊 2. Performance Visualization\n"
    md += "### Cumulative Equity Curve (Out-of-Sample)\n"
    md += "The following chart shows the simulated growth of a $10,000 account using the final optimized configuration (Config 6).\n\n"
    md += "![Equity Curve](equity_curve.png)\n\n"
    
    md += "### Drawdown Profile\n"
    md += "Understanding risk is more important than understanding profit. This chart highlights the peak-to-trough declines during the validation period.\n\n"
    md += "![Drawdown Profile](drawdown_profile.png)\n\n"
    
    md += "\n## 🎯 3. Model Diagnostics\n"
    md += "### Confusion Matrix\n"
    md += "A look at the raw classification performance for the final pipeline. We prioritize avoiding 'False Wins' (Type I errors) to preserve capital.\n\n"
    md += "![Confusion Matrix](confusion_matrix.png)\n\n"
    
    md += "### Top 5 Maximum Drawdowns\n"
    md += "| Start Date | End Date | Max Drawdown (%) |\n"
    md += "|---|---|---|\n"
    for _, row in dd_table.iterrows():
        md += f"| {row['Start Date']} | {row['End Date']} | {row['Max Drawdown (%)']:.2f}% |\n"
        
    md += "\n## ⚠️ 4. Methodology & Limitations\n"
    md += "- **Data Integrity:** Walk-forward validation was performed over a 1-year historical window (~8,760 hours). All features were computed causally to prevent look-ahead bias.\n"
    md += "- **Execution Realism:** A static friction of **9 basis points (bps)** was applied (1bp slippage + 2bp spread + 6bp fees). Real-world slippage can vary significantly with liquidity.\n"
    md += "- **Calibration:** Probability outputs are calibrated using Platt Scaling (Sigmoid) to ensure that confidence levels correspond to actual win frequencies.\n"
    md += "- **Risk Warning:** Past performance does not guarantee future results. High drawdown periods in the out-of-sample data indicate significant volatility risks.\n"
    
    return md

def main():
    print("🚀 Starting Research Validation Pipeline...")
    
    # 1. Fetch 365 Days of Data for BTC, ETH, SOL
    universe = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    days = 365
    interval = "1h"
    
    loader = BinanceDataLoader()
    all_data = []
    
    for sym in universe:
        print(f"Fetching data for {sym}...")
        df_ohlcv = loader.fetch_ohlcv(sym, interval, days)
        df_funding = loader.fetch_funding(sym, days)
        
        df = df_ohlcv.copy()
        if not df_funding.empty:
            df["funding_rate"] = df_funding["funding_rate"].reindex(df.index, method="ffill").fillna(0.0)
        else:
            df["funding_rate"] = 0.0
            
        df = FeatureEngineer.engineer_features(df, interval)
        
        df["fixed_24h_return"] = df["close"].shift(-24) / df["close"] - 1.0
        df["fixed_24h_label"] = (df["fixed_24h_return"] > 0).astype(int)
        
        df_tbm = apply_triple_barrier(df.copy(), tp_pct=0.02, sl_pct=-0.02, time_limit=48)
        df["tbm_label"] = df_tbm["tbm_label"]
        df["tbm_return"] = np.where(df["tbm_label"] == 1, 0.02, -0.02)
        
        df["symbol"] = sym
        df = df.dropna()
        all_data.append(df)
        
    global_df = pd.concat(all_data).sort_index()
    print(f"Dataset prepared: {len(global_df)} rows.")
    
    # 2. Run Ablation Study
    results, config6_preds = run_ablation_study(global_df)
    
    # 3. Process Equity Curve
    config6_preds = config6_preds.sort_values("timestamp")
    
    starting_equity = 10000.0
    equity = [starting_equity]
    
    taken_trades = config6_preds[config6_preds["proba"] >= 0.5].copy()
    
    for _, row in taken_trades.iterrows():
        pnl = equity[-1] * row["net_return"]
        equity.append(equity[-1] + pnl)
        
    taken_trades["equity"] = equity[1:]
    
    if taken_trades.empty:
        taken_trades = pd.DataFrame({"timestamp": global_df.index[:10], "equity": [10000.0]*10})
        
    taken_trades.to_csv("equity_curve.csv", index=False)
    
    # 4. Drawdown Table
    dd_table = get_drawdown_table(taken_trades)
    dd_table.to_csv("drawdown_table.csv", index=False)
    
    # 5. Visualizations
    create_visualizations(results, taken_trades, dd_table)
    
    # 6. Markdown Report
    md_content = generate_markdown(results, dd_table)
    with open("RESEARCH_VALIDATION.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print("✨ Validation Complete. Report saved to RESEARCH_VALIDATION.md")

if __name__ == "__main__":
    main()
