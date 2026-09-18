import numpy as np
import pytest
import scipy.sparse as sp
from tiny_brain import random_mushroom_body

from connectome.controls import rewire, scramble_compartments
from connectome.extract import POPULATIONS


@pytest.fixture
def mb():
    return random_mushroom_body()


def blocks(mb, weights):
    coo = sp.coo_array(weights)
    for a in POPULATIONS:
        for b in POPULATIONS:
            sel = (mb.population[coo.row] == a) & (mb.population[coo.col] == b)
            yield (a, b), coo.row[sel], coo.col[sel], np.abs(coo.data[sel])


def no_duplicates_or_self_loops(weights):
    coo = sp.coo_array(weights)
    assert (coo.row != coo.col).all()
    assert len(set(zip(coo.row.tolist(), coo.col.tolist()))) == coo.nnz


def test_degree_shuffle_keeps_every_degree_and_weight(mb):
    real = mb.graph.weights()
    shuffled = rewire(mb, "degree", np.random.default_rng(1))
    no_duplicates_or_self_loops(shuffled)
    for (_, r1, c1, w1), (_, r2, c2, w2) in zip(blocks(mb, real), blocks(mb, shuffled)):
        assert np.array_equal(np.bincount(r1, minlength=mb.graph.n), np.bincount(r2, minlength=mb.graph.n))
        assert np.array_equal(np.bincount(c1, minlength=mb.graph.n), np.bincount(c2, minlength=mb.graph.n))
        assert np.allclose(np.bincount(r1, w1, mb.graph.n), np.bincount(r2, w2, mb.graph.n))
    kc, mbon = mb.members("KC"), mb.members("MBON")
    assert (real[kc][:, mbon] != shuffled[kc][:, mbon]).nnz > 0


def test_random_graph_keeps_edge_and_synapse_totals_per_block(mb):
    real = mb.graph.weights()
    scattered = rewire(mb, "random", np.random.default_rng(1))
    no_duplicates_or_self_loops(scattered)
    for (_, r1, _, w1), (_, r2, _, w2) in zip(blocks(mb, real), blocks(mb, scattered)):
        assert len(r1) == len(r2)
        assert np.isclose(w1.sum(), w2.sum())


def test_signs_follow_the_new_source(mb):
    scattered = sp.coo_array(rewire(mb, "random", np.random.default_rng(2)))
    assert (np.sign(scattered.data) == mb.graph.sign[scattered.row]).all()


def test_compartment_scramble_is_a_permutation():
    share = np.array([0.0, 0.2, 0.9, 1.0])
    scrambled = scramble_compartments(share, np.random.default_rng(0))
    assert sorted(scrambled) == sorted(share)
    assert not np.array_equal(scrambled, share)
