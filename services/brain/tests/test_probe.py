import numpy as np

from sim.probe import events, from_bytes, to_bytes


def test_events_are_neuron_step_pairs_in_time_order():
    raster = np.zeros((4, 3), dtype=bool)
    raster[0, 2] = raster[1, 0] = raster[1, 1] = raster[3, 2] = True
    assert events(raster).tolist() == [[2, 0], [0, 1], [1, 1], [2, 3]]


def test_bytes_round_trip():
    ev = np.array([[2955, 0], [7, 999]], dtype=np.int32)
    data = to_bytes(ev)
    assert len(data) == 16
    assert np.array_equal(from_bytes(data), ev)
