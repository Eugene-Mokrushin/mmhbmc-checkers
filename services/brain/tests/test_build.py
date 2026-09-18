import numpy as np
import pytest
from conftest import EDGES

from connectome.build import build


def test_neurons_are_indexed_by_ascending_root_id(tiny):
    assert tiny.root_id.tolist() == [10, 20, 30, 40]
    assert tiny.index_of([40, 10]).tolist() == [3, 0]


def test_unknown_root_id(tiny):
    with pytest.raises(ValueError, match="not in the neuron table"):
        tiny.index_of([99])


def test_pairs_are_summed_across_neuropils(tiny):
    kc, apl, _ = tiny.index_of([10, 20, 30])
    assert tiny.counts[kc, apl] == 6
    assert tiny.counts.nnz == 4
    assert tiny.counts.data.sum() == 18


def test_threshold_applies_to_the_pair_total(tiny):
    t = tiny.thresholded(5)
    kc, _, mbon = tiny.index_of([10, 20, 30])
    assert t.counts.nnz == 2
    assert t.counts[mbon, kc] == 7
    assert t.meta["min_syn"] == 5
    assert tiny.counts.nnz == 4


def test_transmitters_and_signs(tiny):
    assert tiny.nt_type.tolist() == ["ACH", "GABA", "GLUT", ""]
    assert tiny.sign.tolist() == [1, -1, -1, 0]
    assert tiny.nt_confident.tolist() == [True, True, False, False]


def test_weights_take_the_presynaptic_sign(tiny):
    w = tiny.weights()
    kc, apl, mbon = tiny.index_of([10, 20, 30])
    assert w[kc, apl] == 6
    assert w[apl, kc] == -4
    assert w[mbon, kc] == -7
    assert w.dtype == np.float32


def test_annotations(tiny):
    assert tiny.cell_type.tolist() == ["KCg-m", "APL", "MBON01", ""]
    assert tiny.side.tolist() == ["left", "right", "left", "right"]


def test_neuron_with_two_transmitters(make_raw):
    with pytest.raises(ValueError, match="more than one nt_type"):
        build(make_raw(EDGES + [(10, 40, "SMP_L", 2, "GABA")]), verify=False)


def test_edge_transmitter_contradicting_neurons_csv(make_raw):
    with pytest.raises(ValueError, match="contradicts neurons.csv"):
        build(make_raw([(10, 20, "SMP_L", 5, "GLUT")]), verify=False)


def test_edge_to_unknown_neuron(make_raw):
    with pytest.raises(ValueError, match="not in the neuron table"):
        build(make_raw(EDGES + [(10, 99, "SMP_L", 5, "ACH")]), verify=False)


def test_changed_raw_files(make_raw):
    with pytest.raises(ValueError, match="differ from the pinned download"):
        build(make_raw())
