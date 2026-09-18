import numpy as np
import pytest
from conftest import assert_same

from connectome import raw
from connectome.build import build
from connectome.load import load, save

pytestmark = [
    pytest.mark.data,
    pytest.mark.slow,
    pytest.mark.skipif(
        not all((raw.RAW_DIR / name).exists() for name in raw.SHA256),
        reason="FlyWire CSVs not in data/raw/",
    ),
]


@pytest.fixture(scope="module")
def fafb():
    return build()


def test_neuron_count(fafb):
    assert fafb.n == 139_255


def test_unthresholded_totals(fafb):
    assert fafb.counts.nnz == 19_773_733
    assert int(fafb.counts.data.sum()) == 76_944_499


def test_min_syn_5_totals(fafb):
    t = fafb.thresholded(5)
    assert t.counts.nnz == 3_732_460
    assert int(t.counts.data.sum()) == 50_666_648


def test_min_syn_5_matches_the_filtered_download(fafb):
    filtered = build(edges_file=raw.EDGES_MIN5)
    assert (fafb.thresholded(5).counts != filtered.counts).nnz == 0


def test_every_neuron_with_outputs_has_a_sign(fafb):
    has_output = np.diff(fafb.counts.indptr) > 0
    assert (fafb.sign[has_output] != 0).all()
    assert (~has_output).sum() == 252


def test_mushroom_body_counts_per_side(fafb):
    for cell_class, per_side in [("Kenyon_Cell", (2580, 2597)), ("MBON", (48, 48)), ("DAN", (166, 165))]:
        found = [((fafb.cell_class == cell_class) & (fafb.side == side)).sum() for side in ("left", "right")]
        assert tuple(found) == per_side, cell_class


def test_mushroom_body_signs(fafb):
    kc = np.char.startswith(fafb.cell_type, "KC")
    apl = fafb.cell_type == "APL"
    assert kc.sum() == 5_177 and (fafb.nt_type[kc] == "ACH").all()
    assert apl.sum() == 2 and (fafb.sign[apl] == -1).all()


def test_round_trip(fafb, tmp_path):
    save(fafb, tmp_path / "connectome.npz")
    assert_same(load(tmp_path / "connectome.npz"), fafb)
