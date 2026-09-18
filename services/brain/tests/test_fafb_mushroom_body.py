import numpy as np
import pytest

from connectome.coordinates import one_per_neuron, read_markers
from connectome.extract import extract
from connectome.load import NPZ_PATH, load

pytestmark = [
    pytest.mark.data,
    pytest.mark.slow,
    pytest.mark.skipif(not NPZ_PATH.exists(), reason="run python -m connectome.build first"),
]


@pytest.fixture(scope="module")
def fafb():
    return load()


@pytest.fixture(scope="module")
def mb(fafb):
    return extract(fafb)


def test_population_sizes(mb):
    sizes = {p: (mb.population == p).sum() for p in ("KC", "MBON", "DAN", "APL")}
    assert sizes == {"KC": 2597, "MBON": 48, "DAN": 309, "APL": 1}
    assert len(set(mb.graph.cell_type[mb.members("MBON")])) == 35
    assert mb.graph.counts.nnz == 490_752


def test_kc_to_mbon_is_sparse_and_reaches_every_kc(mb):
    kc_mbon = mb.block("KC", "MBON")
    assert kc_mbon.nnz / np.prod(kc_mbon.shape) < 0.25
    assert (np.diff(kc_mbon.indptr) > 0).all()


def test_each_mbon_type_has_its_own_dopamine_input(mb):
    contact = mb.dan_mbon_contact()
    assert (contact.sum(axis=0) > 0).all()
    types = mb.graph.cell_type[mb.members("MBON")]
    per_type = np.stack([contact[:, types == t].sum(axis=1) for t in sorted(set(types))], axis=1)
    per_type /= np.linalg.norm(per_type, axis=0)
    similarity = per_type.T @ per_type
    assert np.median(similarity[~np.eye(len(similarity), dtype=bool)]) < 0.1


def test_kenyon_cells_sit_on_their_own_side(fafb, mb):
    points = one_per_neuron(read_markers())
    kc = fafb.cell_class == "Kenyon_Cell"
    left = points.loc[fafb.root_id[kc & (fafb.side == "left")], "x"]
    right = points.loc[mb.graph.root_id[mb.members("KC")], "x"]
    midline = (left.mean() + right.mean()) / 2
    assert (right > midline).all() and (left < midline).all()
    assert (points.loc[mb.graph.root_id[mb.members("APL")], "x"] > midline).all()

