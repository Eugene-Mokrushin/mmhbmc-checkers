import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from connectome.controls import CONTROLS
from train.evaluate import RUNS_DIR

METRICS = ("random", "greedy", "minimax2", "untrained", "safety", "material")


def load_runs(pattern: str, runs_dir: Path = RUNS_DIR) -> list[dict]:
    runs = []
    for config in sorted(runs_dir.glob(f"{pattern}/config.json")):
        cfg = json.loads(config.read_text())
        rows = [r for r in csv.DictReader((config.parent / "log.csv").open()) if r["random"]]
        if len(rows) >= 2:
            runs.append({"control": cfg.get("control", "real"), "seed": cfg["seed"], "first": rows[0], "last": rows[-1]})
    return runs


def compare(runs: list[dict], metric: str) -> list[dict]:
    runs = [r for r in runs if r["first"].get(metric) and r["last"].get(metric)]
    gain = {c: np.array([float(r["last"][metric]) - float(r["first"][metric]) for r in runs if r["control"] == c]) for c in CONTROLS}
    final = {c: np.array([float(r["last"][metric]) for r in runs if r["control"] == c]) for c in CONTROLS}
    out = []
    for c in CONTROLS:
        if not len(gain[c]):
            continue
        p = np.nan
        if c != "real" and len(gain[c]) > 1 and len(gain["real"]) > 1:
            p = stats.ttest_ind(gain["real"], gain[c], equal_var=False).pvalue
        out.append({"control": c, "n": len(gain[c]), "final": final[c], "gain": gain[c], "p": p})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="*", help="which run directories to compare, e.g. 'c2048-*'")
    args = parser.parse_args()
    runs = load_runs(args.pattern)
    print(f"{len(runs)} runs matching {args.pattern!r}; win rates and skill AUCs after training, gain over the untrained fly")
    for metric in METRICS:
        rows = compare(runs, metric)
        if not rows:
            continue
        print(f"\n{metric}")
        print(f"{'':14}{'n':>3}{'final':>16}{'gain':>16}{'p vs real':>11}")
        for row in rows:
            f, g = row["final"], row["gain"]
            p = "" if np.isnan(row["p"]) else f"{row['p']:.3f}"
            print(f"{row['control']:14}{row['n']:>3}{f.mean():>9.1%} ± {f.std(ddof=1) if len(f) > 1 else 0:.1%}"
                  f"{g.mean():>+9.1%} ± {g.std(ddof=1) if len(g) > 1 else 0:.1%}{p:>11}")


if __name__ == "__main__":
    main()
