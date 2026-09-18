import pytest
from conftest import assert_same

from connectome.load import load, save


def test_round_trip(tiny, tmp_path):
    save(tiny, tmp_path / "c.npz")
    assert_same(load(tmp_path / "c.npz"), tiny)


def test_threshold_on_load(tiny, tmp_path):
    save(tiny, tmp_path / "c.npz")
    assert load(tmp_path / "c.npz", min_syn=5).counts.nnz == 2


def test_only_unthresholded_graphs_are_saved(tiny, tmp_path):
    with pytest.raises(ValueError, match="unthresholded"):
        save(tiny.thresholded(5), tmp_path / "c.npz")


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="connectome.build"):
        load(tmp_path / "nope.npz")
