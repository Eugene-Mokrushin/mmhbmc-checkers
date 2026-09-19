import numpy as np
import pytest

from connectome.extract import extract


def test_members(brain):
    mb = extract(brain)
    assert mb.neurons.tolist() == [0, 1, 4, 5, 7]
    assert mb.population.tolist() == ["KC", "KC", "MBON", "DAN", "APL"]
    assert mb.graph.root_id.tolist() == [100, 101, 104, 105, 107]


def types(mb, population):
    return mb.graph.cell_type[mb.members(population)].tolist()


def test_mbon_follows_its_kc_input_not_its_label(brain):
    assert types(extract(brain), "MBON") == ["MBON07"]
    assert types(extract(brain, "left"), "MBON") == ["MBON01"]


def test_bilateral_dan_is_in_both_mushroom_bodies(brain):
    assert types(extract(brain), "DAN") == ["PAM01"]
    assert types(extract(brain, "left"), "DAN") == ["PAM01", "PAM02"]


def test_blocks(brain):
    mb = extract(brain)
    assert mb.block("KC", "MBON").toarray().tolist() == [[5], [3]]
    assert mb.block("DAN", "KC").toarray().tolist() == [[4, 0]]
    assert mb.dan_mbon_contact().tolist() == [[1.0]]


def test_subgraph_needs_sorted_indices(brain):
    with pytest.raises(ValueError, match="ascending"):
        brain.subgraph(np.array([4, 0]))


def test_both_mushroom_bodies_together(brain):
    mb = extract(brain, "both")
    assert types(mb, "MBON") == ["MBON01", "MBON07"]
    assert types(mb, "DAN") == ["PAM01", "PAM02"]
    assert mb.members("KC").tolist() == [0, 1, 2]
