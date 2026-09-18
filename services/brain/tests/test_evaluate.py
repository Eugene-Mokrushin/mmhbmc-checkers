import csv

import numpy as np

import progress
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


def test_untrained_copy_keeps_the_starting_wiring(brain):
    fly = Fly(brain)
    plastic = Plasticity(fly)
    start = plastic.state()
    plastic.update(np.array([[1, 0]]), np.array([1.0]))
    copy = fly.untrained()
    assert np.array_equal(Plasticity(copy).state(), start)
    assert not np.array_equal(plastic.state(), start)
    assert copy.rng is not fly.rng


def test_evaluation_includes_the_untrained_self(brain, tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "PROGRESS_FILE", tmp_path / "progress.txt")
    fly = Fly(brain)
    fly.explore = 0.3
    result = evaluate.evaluate(fly, 2, seed=0)
    assert set(result) == {"random", "greedy", "minimax2", "untrained"}
    assert fly.explore == 0.3


def test_restore_rebuilds_a_saved_fly(brain, tmp_path, monkeypatch):
    from train import curve

    monkeypatch.setattr(evaluate, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(curve, "RUNS_DIR", tmp_path)
    fly, plastic = curve.setup("real", 4, curve.Rule(), explore=0.0, c=brain)
    plastic.update(np.array([[1, 0]]), np.array([1.0]))
    run = evaluate.Run("saved", {"control": "real", "seed": 4})
    run.checkpoint(plastic, 256, seed=4)
    restored, restored_plastic = curve.restore("saved", 256, c=brain)
    assert np.array_equal(restored_plastic.state(), plastic.state())
    assert restored.untrained() is not restored
