"""
Smoke test for the institutional data pipeline.
Runs ingestion on 2 symbols for a short period.
"""
import subprocess
import os
import pandas as pd
import json
from pathlib import Path

def run_smoke_test():
    print("🚀 Starting Data Pipeline Smoke Test...")
    
    # 1. Run build_global_dataset.py on 2 major symbols for 30 days
    cmd = [
        "python", "build_global_dataset.py",
        "--universe", "manual",
        "--symbols", "BTCUSDT,ETHUSDT",
        "--days", "30",
        "--output", "smoke_test.parquet"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Pipeline failed!")
        print(result.stdout)
        print(result.stderr)
        return False
        
    print(result.stdout)
    
    # 2. Validate Artifacts
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    report_file = Path("ingestion_report.json")
    final_file = processed_dir / "smoke_test.parquet"
    
    expected_files = [
        raw_dir / "BTCUSDT_1h.parquet",
        raw_dir / "ETHUSDT_1h.parquet",
        report_file,
        final_file
    ]
    
    for f in expected_files:
        if not f.exists():
            print(f"❌ Missing expected artifact: {f}")
            return False
        print(f"✅ Found artifact: {f}")
        
    # 3. Inspect Data Quality
    with open(report_file, "r") as f:
        report = json.load(f)
        for entry in report:
            print(f"📊 QA Report for {entry['symbol']}: {entry['rows']} rows, issues: {entry['issues']}")

    # 4. Check Final Dataset structure
    df = pd.read_parquet(final_file)
    print(f"✅ Final dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    if "tbm_label" not in df.columns:
        print("❌ Missing tbm_label in final dataset")
        return False
        
    if "close_fd_04" not in df.columns:
        print("❌ Missing close_fd_04 in final dataset")
        return False
        
    print("\n✨ Smoke Test PASSED!")
    return True

if __name__ == "__main__":
    success = run_smoke_test()
    if not success:
        exit(1)
