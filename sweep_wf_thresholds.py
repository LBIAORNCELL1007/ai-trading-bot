"""Quick WF threshold sweep on the sigmoid-calibrated model."""

import subprocess
import re

DATA = "global_alpha_dataset_1h_2pct.csv"
TP = "0.02"
SL = "0.02"


def parse(out):
    stats = {}
    for line in out.splitlines():
        if "Trades executed:" in line:
            stats["trades"] = int(line.split(":")[-1].strip().replace(",", ""))
        elif "Win rate:" in line:
            stats["wr"] = float(line.split(":")[-1].strip())
        elif "Mean NET return per trade:" in line:
            stats["net_pct"] = float(re.sub(r"[^0-9.\-]", "", line.split(":")[-1]))
        elif "Annualised Sharpe (approx):" in line:
            stats["sharpe"] = float(re.sub(r"[^0-9.\-]", "", line.split(":")[-1]))
        elif "Total return:" in line:
            stats["ret_pct"] = float(re.sub(r"[^0-9.\-]", "", line.split(":")[-1]))
        elif "Max drawdown:" in line:
            stats["dd_pct"] = float(re.sub(r"[^0-9.\-]", "", line.split(":")[-1]))
    return stats


print(
    f"{'thr':>6} {'trades':>7} {'WR':>7} {'net/trd':>10} {'Sharpe':>8} {'total':>9} {'maxDD':>8}"
)
print("-" * 60)
for thr in [
    0.460,
    0.470,
    0.475,
    0.480,
    0.485,
    0.488,
    0.490,
    0.492,
    0.495,
    0.498,
    0.500,
    0.505,
    0.510,
]:
    out = subprocess.run(
        [
            "C:/Python314/python.exe",
            "walk_forward_simulator.py",
            "--data",
            DATA,
            "--threshold",
            str(thr),
            "--tp-pct",
            TP,
            "--sl-pct",
            SL,
        ],
        capture_output=True,
        text=True,
    ).stdout
    s = parse(out)
    if "trades" not in s:
        print(f"{thr:>6.3f}  parse-fail")
        continue
    print(
        f"{thr:>6.3f} {s.get('trades', 0):>7} {s.get('wr', 0):>7.4f} "
        f"{s.get('net_pct', 0):>+9.4f}% {s.get('sharpe', 0):>+8.3f} "
        f"{s.get('ret_pct', 0):>+8.2f}% {s.get('dd_pct', 0):>+8.2f}%"
    )
