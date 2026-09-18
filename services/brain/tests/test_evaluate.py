import csv

import numpy as np

from game.fly import Fly
from train import evaluate
from train.plasticity import Plasticity


def test_log_keeps_the_same_columns_on_every_row(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RUNS_DIR", tmp_path)
    run = evaluate.Run("test", {"games": 10})
    run.record({"games": 0, "random": 0.4})
    run.record({"games": 64, "random": 0.5, "train_win": 0.25, "depressed": 0.1})
    rows = list(csv.DictReader((tmp_path / "test" / "log.csv").open()))
    assert list(rows[0]) == evaluate.COLUMNS
    assert rows[0]["random"] == "0.4" and rows[0]["train_win"] == ""
    assert rows[1]["train_win"] == "0.25" and rows[1]["depressed"] == "0.1"


def test_checkpoint_holds_the_plastic_weights(tmp_path, monkeypatch, brain):
    monkeypatch.setattr(evaluate, "RUNS_DIR", tmp_path)
    plastic = Plasticity(Fly(brain))
    path = evaluate.Run("test", {}).checkpoint(plastic, 128, seed=3)
    with np.load(path) as z:
        assert np.array_equal(z["weights"], plastic.state())
        assert int(z["games"]) == 128 and int(z["seed"]) == 3


def test_no_evaluation_when_zero_games(brain):
    assert evaluate.evaluate(Fly(brain), 0, seed=0) == {}
