import csv
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from game.arena import play, tally
from game.fly import Fly
from game.players import GreedyPlayer, MinimaxPlayer, RandomPlayer
from paths import ARTIFACTS_DIR
from train.plasticity import Plasticity

RUNS_DIR = ARTIFACTS_DIR / "runs"
COLUMNS = ["minutes", "games", "random", "greedy", "minimax2", "untrained", "train_win", "kc_active", "mbon_hz", "depressed"]


def baselines(seed: int) -> dict:
    return {"random": RandomPlayer(seed), "greedy": GreedyPlayer(seed), "minimax2": MinimaxPlayer(2, seed)}


def evaluate(fly: Fly, games: int, seed: int) -> dict[str, float]:
    if not games:
        return {}
    explore, fly.explore = fly.explore, 0.0
    try:
        opponents = {**baselines(seed), "untrained": fly.untrained()}
        return {name: tally(play(fly, opponent, games, f"eval vs {name}"), fly)["win"] for name, opponent in opponents.items()}
    finally:
        fly.explore = explore


class Run:
    def __init__(self, name: str, config: dict):
        self.dir = RUNS_DIR / name
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(config, indent=2))
        self.log = self.dir / "log.csv"
        self.start = time.monotonic()

    def record(self, row: dict) -> None:
        row = {"minutes": round((time.monotonic() - self.start) / 60, 1), **row}
        new = not self.log.exists()
        with self.log.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
            if new:
                writer.writeheader()
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

    def checkpoint(self, plastic: Plasticity, games: int, seed: int) -> Path:
        path = self.dir / f"fly-{games:07d}.npz"
        rule = json.dumps(dataclasses.asdict(plastic.rule))
        np.savez_compressed(path, weights=plastic.state(), games=games, seed=seed, rule=np.array(rule))
        return path
