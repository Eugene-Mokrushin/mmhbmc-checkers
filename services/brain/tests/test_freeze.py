import json

import numpy as np

from train import evaluate, freeze
from train.controls import setup
from train.plasticity import Rule


def write_log(root, run, rows):
    (root / run).mkdir()
    lines = [",".join(evaluate.COLUMNS)]
    for games, random_win, greedy_win in rows:
        values = {"games": games, "random": random_win, "greedy": greedy_win}
        lines.append(",".join(str(values.get(col, "")) for col in evaluate.COLUMNS))
    (root / run / "log.csv").write_text("\n".join(lines) + "\n")


def test_shortlist_takes_the_best_evaluated_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "RUNS_DIR", tmp_path)
    write_log(tmp_path, "a", [(0, 0.9, 0.9), (1024, 0.6, 0.5), (2048, 0.7, 0.6)])
    write_log(tmp_path, "b", [(1024, 0.8, 0.7)])
    assert freeze.shortlist(["a", "b"], keep=2) == [("b", 1024), ("a", 2048)]


def test_frozen_fly_loads_back(brain, tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "FLIES_DIR", tmp_path)
    _, plastic = setup("real", 3, Rule(), explore=0.0, c=brain)
    plastic.update(np.array([[1, 0]]), np.array([1.0]), np.array([0.0]))
    np.savez(tmp_path / "plain.npz", weights=plastic.state())
    (tmp_path / "plain.json").write_text(json.dumps({"control": "real", "seed": 3}))
    fly = freeze.load_fly("plain", c=brain)
    loaded = fly.sim.w[plastic.kc, plastic.mbon].numpy()
    assert np.array_equal(loaded, plastic.state())
    assert fly.explore == 0.0
