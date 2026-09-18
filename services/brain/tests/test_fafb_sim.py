import numpy as np
import pytest

from connectome.extract import extract
from connectome.load import NPZ_PATH, load
from sim.fast import FastLIF
from sim.inputs import projection
from sim.sparsity import measure, without_apl

pytestmark = [
    pytest.mark.data,
    pytest.mark.slow,
    pytest.mark.skipif(not NPZ_PATH.exists(), reason="run python -m connectome.build first"),
]


@pytest.fixture(scope="module")
def setup():
    c = load()
    mb = extract(c)
    return c, mb, projection(c, mb, np.random.default_rng(0))


def test_projection_keeps_each_kcs_real_claws(setup):
    c, mb, proj = setup
    kcs = mb.members("KC")
    claws = c.counts[np.flatnonzero(c.cell_class == "ALPN")][:, mb.neurons[kcs]]
    assert np.array_equal(proj[:, kcs].sum(axis=0), claws.sum(axis=0))
    assert np.array_equal(np.diff(proj[:, kcs].tocsc().indptr), np.diff(claws.tocsc().indptr))
    assert proj[:, mb.population != "KC"].nnz == 0


@pytest.mark.parametrize("lines", [16, 24])
def test_kenyon_cells_are_sparse_and_distinct(setup, lines):
    _, mb, proj = setup
    r = measure(FastLIF(mb.graph.weights(), proj), mb, lines, np.random.default_rng(1))
    assert 0.05 <= r["active"] <= 0.10
    assert r["overlap"] < 0.15
    assert r["repeat"] > 3 * r["overlap"]


def test_apl_is_what_keeps_the_code_sparse(setup):
    _, mb, proj = setup
    r = measure(FastLIF(without_apl(mb), proj), mb, 16, np.random.default_rng(1))
    assert r["active"] > 0.5
