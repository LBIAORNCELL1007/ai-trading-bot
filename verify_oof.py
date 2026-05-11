"""Quick verification of OOF win rates from the v2 model."""
import pandas as pd

oof = pd.read_csv('tbm_xgboost_model_v2_oof.csv')
print(f"Total OOF rows: {len(oof):,}")
print(f"Columns: {list(oof.columns)}")
print(f"Label distribution: {oof['y'].value_counts().to_dict()}")
print(f"Base rate (P(y=1)): {oof['y'].mean():.4f}")
print()

print("=" * 60)
print("  GLOBAL OOF WIN RATES (all symbols combined)")
print("=" * 60)
for t in [0.45, 0.50, 0.52, 0.55, 0.56, 0.58, 0.60, 0.65, 0.70]:
    sel = oof[oof['oof_proba'] >= t]
    n = len(sel)
    if n == 0:
        print(f"  thr={t:.2f}: No trades")
        continue
    wr = sel['y'].mean()
    print(f"  thr={t:.2f}: trades={n:,}  WR={wr*100:.1f}%")

print()
print("=" * 60)
print("  CONFIDENCE DISTRIBUTION")
print("=" * 60)
print(f"  Min proba:    {oof['oof_proba'].min():.4f}")
print(f"  Max proba:    {oof['oof_proba'].max():.4f}")
print(f"  Mean proba:   {oof['oof_proba'].mean():.4f}")
print(f"  Median proba: {oof['oof_proba'].median():.4f}")

# Check raw vs calibrated
if 'oof_proba_raw' in oof.columns:
    print()
    print(f"  Raw min:      {oof['oof_proba_raw'].min():.4f}")
    print(f"  Raw max:      {oof['oof_proba_raw'].max():.4f}")
    print(f"  Raw mean:     {oof['oof_proba_raw'].mean():.4f}")
