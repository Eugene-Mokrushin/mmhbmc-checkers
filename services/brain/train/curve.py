import argparse
import json

import numpy as np

from connectome.load import load
from game.arena import play, tally
from game.fly import Fly
from progress import Progress
from train.controls import setup
from train.evaluate import RUNS_DIR, baselines
from train.plasticity import Plasticity, Rule

OPPONENTS = ("random", "greedy", "untrained")


def restore(run: str, games: int, c=None) -> tuple[Fly, Plasticity]:
    config = json.loads((RUNS_DIR / run / "config.json").read_text())
    fly, plastic = setup(config.get("control", "real"), config["seed"], Rule(), 0.0, c, config.get("brain", "mb"))
    with np.load(RUNS_DIR / run / f"fly-{games:07d}.npz") as z:
        plastic.load(z["weights"])
    return fly, plastic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--games", type=int, default=300)
    args = parser.parse_args()

    c = load()
    checkpoints = sorted(int(p.stem.split("-")[1]) for p in (RUNS_DIR / args.run).glob("fly-*.npz"))
    fly, _ = restore(args.run, checkpoints[0], c)
    points = [0, *checkpoints]
    rows = []
    with Progress(len(points), f"curve {args.run}") as bar:
        for games in points:
            if games:
                fly, _ = restore(args.run, games, c)
            else:
                fly = fly.untrained()
            opponents = {**baselines(seed=7), "untrained": fly.untrained()}
            rows.append((games, {name: tally(play(fly, opponents[name], args.games, f"curve {games} vs {name}"), fly)["win"] for name in OPPONENTS}))
            bar.update(1, games=games)

    margin = 1.96 * np.sqrt(0.25 / args.games)
    print(f"win rate over {args.games} games per point (95% margin about ±{margin:.0%})")
    print(f"{'games':>7}" + "".join(f"{name:>11}" for name in OPPONENTS))
    for games, result in rows:
        print(f"{games:>7}" + "".join(f"{result[name]:>11.1%}" for name in OPPONENTS))


if __name__ == "__main__":
    main()
