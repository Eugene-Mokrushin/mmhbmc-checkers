import numpy as np
import pytest
import torch

import progress
from game.fly import Fly
from game.players import RandomPlayer
from train.controls import setup
from train.plasticity import Plasticity, Rule
from train.block import block


@pytest.fixture
def fly(brain):
    return Fly(brain)


def test_only_kc_to_mbon_synapses_are_plastic(fly):
    plastic = Plasticity(fly)
    assert plastic.weights.shape == (2, 1)
    assert torch.allclose(plastic.weights[:, 0] / fly.sim.p.w_syn, torch.tensor([5.0, 3.0]))


def test_better_than_expected_weakens_active_kcs_onto_reward_mbons(fly):
    plastic = Plasticity(fly, Rule(rate=0.5, recovery=0))
    start = plastic.weights.clone()
    plastic.update(np.array([[1, 0]]), np.array([1.0]), np.array([0.0]))
    assert torch.allclose(plastic.weights[:, 0], start[:, 0] * torch.tensor([0.5, 1.0]))


def test_worse_than_expected_strengthens_active_kcs_onto_reward_mbons(fly):
    plastic = Plasticity(fly, Rule(rate=0.5, recovery=0))
    start = plastic.weights.clone()
    plastic.update(np.array([[1, 0]]), np.array([-1.0]), np.array([0.0]))
    assert torch.allclose(plastic.weights[:, 0], start[:, 0] * torch.tensor([1.5, 1.0]))


def test_an_expected_outcome_changes_nothing(fly):
    plastic = Plasticity(fly, Rule(rate=0.5, recovery=0))
    start = plastic.weights.clone()
    plastic.update(np.array([[1, 0], [1, 0]]), np.array([0.0, 0.0]), np.array([5.0, 5.0]))
    assert torch.equal(plastic.weights, start)


def test_weights_stay_between_zero_and_ceiling(fly):
    plastic = Plasticity(fly, Rule(rate=0.9, recovery=0.01))
    for _ in range(50):
        plastic.update(np.array([[1, 0]]), np.array([1.0]), np.array([0.0]))
    assert (plastic.weights >= 0).all()
    assert plastic.depressed() == 0.5
    for _ in range(50):
        plastic.update(np.array([[1, 0]]), np.array([-1.0]), np.array([0.0]))
    assert (plastic.weights <= 2 * plastic.start).all()
    assert torch.isclose(plastic.weights[0, 0], 2 * plastic.start[0, 0], rtol=0.02)


def test_state_round_trip(fly):
    plastic = Plasticity(fly)
    saved = plastic.state()
    plastic.update(np.array([[1, 0]]), np.array([1.0]), np.array([0.0]))
    plastic.load(saved)
    assert np.array_equal(plastic.state(), saved)


def test_training_block_plays_to_the_end(fly, tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "PROGRESS_FILE", tmp_path / "progress.txt")
    stats = block(fly, Plasticity(fly), RandomPlayer(1), 2, shaped=True)
    assert set(stats) == {"train_win", "kc_active", "mbon_hz"}
    assert 0 <= stats["train_win"] <= 1


@pytest.mark.parametrize("control", ["real", "degree", "random", "compartments"])
def test_every_control_builds_a_working_learner(brain, control):
    fly, plastic = setup(control, seed=0, rule=Rule(), explore=0.1, c=brain)
    assert fly.explore == 0.1
    assert plastic.weights.shape == (2, 1)
    plastic.update(np.array([[1, 0]]), np.array([1.0]), np.array([0.0]))
