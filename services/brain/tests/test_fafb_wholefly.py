import numpy as np
import pytest

import progress
from connectome.coordinates import one_per_neuron, read_markers
from connectome.load import NPZ_PATH, load
from flycore.board import INITIAL, apply_move, flip
from flycore.moves import legal_moves
from game.arena import play
from game.players import RandomPlayer
from game.wholefly import WholeFly
from train.plasticity import Plasticity

pytestmark = [
    pytest.mark.data,
    pytest.mark.slow,
    pytest.mark.skipif(not NPZ_PATH.exists(), reason="run python -m connectome.build first"),
]


@pytest.fixture(scope="module")
def full():
    return load()


@pytest.fixture(scope="module")
def fly(full):
    return WholeFly(load(min_syn=5), full, one_per_neuron(read_markers()))


@pytest.fixture(scope="module")
def opening():
    return [flip(apply_move(INITIAL, m)) for m in legal_moves(INITIAL)]


def test_both_mushroom_bodies_keep_every_kc_to_mbon_synapse(fly, full):
    kc, mbon = fly.kcs, fly.mbons
    assert len(kc) > 5000 and len(mbon) == 96
    assert (fly.wiring[kc][:, mbon] != full.weights()[kc][:, mbon]).nnz == 0


def test_same_board_same_score(fly, opening):
    assert fly.scores(opening) == fly.scores(opening)


def test_counts_hold_kenyon_cells_then_mbons(fly, opening):
    counts = fly.counts(opening[:2])
    assert counts.shape == (2, len(fly.kcs) + len(fly.mbons))
    assert counts[:, fly.kc_cols].any() and counts[:, fly.mbon_cols].any()


def test_dopamine_changes_only_synapses_of_active_kenyon_cells(fly, opening):
    plastic = Plasticity(fly.untrained())
    counts = plastic.fly.counts(opening)
    active = counts[:, plastic.fly.kc_cols] > 0
    before = plastic.state()
    plastic.update(active, np.ones(len(opening)), plastic.fly.judge(counts))
    changed = (plastic.state() != before).any(axis=1)
    assert changed.any() and not (changed & ~active.any(axis=0)).any()


def test_plays_a_legal_game(fly, tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "PROGRESS_FILE", tmp_path / "progress.txt")
    assert all(g.over for g in play(fly, RandomPlayer(1), 2))
