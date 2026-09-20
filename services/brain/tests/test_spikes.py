import numpy as np

from api.spikes import gapped, varints


def spread(blob: bytes) -> np.ndarray:
    # the website's decoder, written plainly here so the test checks the wire itself
    out, value, shift, running = [], 0, 0, 0
    for byte in blob:
        value |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            continue
        running += value
        out.append(running)
        value, shift = 0, 0
    return np.asarray(out, dtype=np.int64)


def test_a_frame_of_firing_survives_the_wire():
    rng = np.random.default_rng(3)
    ids = np.sort(rng.choice(139255, 14000, replace=False))
    assert np.array_equal(spread(gapped(ids)), ids)


def test_the_gaps_cost_about_a_byte_each():
    ids = np.arange(0, 139255, 10)
    assert len(gapped(ids)) < 1.1 * len(ids)


def test_numbers_of_every_size_come_back():
    for value in (0, 1, 127, 128, 16383, 16384, 2097151, 2097152, 268435455):
        assert spread(varints(np.array([value])))[0] == value


def test_an_empty_frame_is_empty():
    assert gapped(np.array([], dtype=np.int64)) == b""
