import numpy as np
import pytest
import scipy.sparse as sp

from connectome.load import NPZ_PATH, load
from flycore.encode import square_lines
from game.fly import Fly
from sim.inputs import DIAGONALS
from sim.sparsity import BINS, game_positions, measure, without_apl

pytestmark = [
    pytest.mark.data,
    pytest.mark.slow,
    pytest.mark.skipif(not NPZ_PATH.exists(), reason="run python -m connectome.build first"),
]


@pytest.fixture(scope="module")
def c():
    return load()


@pytest.fixture(scope="module")
def fly(c):
    return Fly(c)


def test_each_kc_reads_one_diagonal_with_its_real_claws(c, fly):
    mb, proj = fly.mb, sp.csc_array(fly.projection)
    kcs = mb.members("KC")
    claws = c.counts[np.flatnonzero(c.cell_class == "ALPN")][:, mb.neurons[kcs]].tocsc()
    fields = [{line for square in d for line in square_lines(square)} for d in DIAGONALS]
    for j, kc in enumerate(kcs):
        lines = set(proj.indices[proj.indptr[kc] : proj.indptr[kc + 1]].tolist())
        real = np.sort(claws.data[claws.indptr[j] : claws.indptr[j + 1]])[::-1][:15]
        assert np.array_equal(np.sort(proj.data[proj.indptr[kc] : proj.indptr[kc + 1]])[::-1], real)
        assert not lines or any(lines <= field for field in fields)
    assert proj[:, mb.population != "KC"].nnz == 0


@pytest.mark.parametrize("pieces", list(BINS))
def test_kenyon_cells_are_sparse_at_every_stage_of_a_game(fly, pieces):
    assert 0.05 <= measure(fly, game_positions(BINS[pieces], 64))["active"] <= 0.10


def test_different_positions_get_different_codes(fly):
    assert measure(fly, game_positions(BINS["9-16"], 64))["overlap"] < 0.4


def test_apl_is_what_keeps_the_code_sparse(fly):
    assert measure(fly.with_weights(without_apl(fly.mb)), game_positions(BINS["9-16"], 32))["active"] > 0.5
