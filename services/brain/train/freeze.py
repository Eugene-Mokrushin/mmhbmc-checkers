import argparse
import csv
import dataclasses
import json
import subprocess

import numpy as np

from connectome.load import load
from game.arena import play, tally
from game.fly import Fly
from paths import ARTIFACTS_DIR, REPO
from train.controls import setup
from train.curve import restore
from train.evaluate import RUNS_DIR, baselines
from train.pack import load_gradient_fly
from train.plasticity import Rule

FLIES_DIR = ARTIFACTS_DIR / "flies"


def shortlist(runs: list[str], keep: int = 3) -> list[tuple[str, int]]:
    # the checkpoints that did best against random and greedy in the in-run evaluations
    rows = []
    for run in runs:
        for row in csv.DictReader((RUNS_DIR / run / "log.csv").open()):
            if int(row["games"]) and row["random"] and row["greedy"]:
                rows.append((float(row["random"]) + float(row["greedy"]), run, int(row["games"])))
    return [(run, games) for _, run, games in sorted(rows, reverse=True)[:keep]]


def strength(fly: Fly, games: int) -> dict[str, float]:
    opponents = baselines(seed=11)
    return {name: tally(play(fly, opponents[name], games, f"freeze vs {name}"), fly)["win"] for name in ("random", "greedy")}


def load_fly(name: str, c=None, device: str | None = None) -> Fly:
    meta = json.loads((FLIES_DIR / f"{name}.json").read_text())
    if meta.get("kind") == "gradient":
        return load_gradient_fly(name, device)
    fly, plastic = setup(meta["control"], meta["seed"], Rule(), 0.0, c, meta.get("brain", "mb"), device)
    with np.load(FLIES_DIR / f"{name}.npz") as z:
        plastic.load(z["weights"])
    return fly


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--name", default="plain")
    parser.add_argument("--games", type=int, default=300)
    args = parser.parse_args()

    c = load()
    tested = []
    for run, games in shortlist(args.runs):
        fly, _ = restore(run, games, c)
        result = strength(fly, args.games)
        tested.append((result["random"] + result["greedy"], run, games, result))
        print(f"{run} @ {games}: " + ", ".join(f"{k} {v:.1%}" for k, v in result.items()))
    _, run, games, result = max(tested)

    config = json.loads((RUNS_DIR / run / "config.json").read_text())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    fly, _ = restore(run, games, c)
    FLIES_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(RUNS_DIR / run / f"fly-{games:07d}.npz") as z:
        np.savez_compressed(FLIES_DIR / f"{args.name}.npz", weights=z["weights"])
    meta = {
        "name": args.name,
        "run": run,
        "games": games,
        "brain": config.get("brain", "mb"),
        "control": config.get("control", "real"),
        "seed": config["seed"],
        "wins": result,
        "sim": dataclasses.asdict(fly.sim.p),
        "commit": commit,
    }
    (FLIES_DIR / f"{args.name}.json").write_text(json.dumps(meta, indent=2))
    print(f"froze {args.name}: {run} @ {games} games, " + ", ".join(f"{k} {v:.1%}" for k, v in result.items()))


if __name__ == "__main__":
    main()
