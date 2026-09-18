import numpy as np
import pytest
import torch

import progress
from connectome.load import NPZ_PATH, load
from flycore.board import INITIAL, apply_move, flip
from flycore.encode import N_LINES
from flycore.moves import legal_moves
from game.arena import play
from game.fly import WINDOW, Fly
from game.players import RandomPlayer

pytestmark = [
    pytest.mark.data,
    pytest.mark.slow,
    pytest.mark.skipif(not NPZ_PATH.exists(), reason="run python -m connectome.build first"),
]


@pytest.fixture(scope="module")
def fly():
    return Fly(load())


def test_valence_split(fly):
    assert (fly.valence == 1).sum() == 23
    assert (fly.valence == -1).sum() == 25


def test_mbons_rest_near_ten_hz(fly):
    silence = torch.zeros((WINDOW, 1, N_LINES), dtype=torch.bool)
    rates = fly.sim.counts(silence, fly.rest).numpy()[0, fly.mb.members("MBON")] / (WINDOW * fly.sim.p.dt)
    assert 5 <= np.median(rates) <= 20


def test_same_board_same_score(fly):
    after = [flip(apply_move(INITIAL, m)) for m in legal_moves(INITIAL)]
    assert fly.scores(after) == fly.scores(after)


def test_raster_covers_every_neuron_for_the_window(fly):
    raster = fly.raster(INITIAL)
    assert raster.shape == (WINDOW, fly.mb.graph.n)
    assert raster[:, fly.mb.members("KC")].any(axis=0).mean() > 0.05


def test_plays_legal_games(fly, tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "PROGRESS_FILE", tmp_path / "progress.txt")
    games = play(fly, RandomPlayer(1), 4)
    assert all(g.over for g in games)
