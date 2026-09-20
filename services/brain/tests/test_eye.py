import numpy as np

from flycore.board import INITIAL
from sim.eye import EMPTY, LIGHT, MAX_RATE, SHADES, Rhythm, image, projection, sample
from sim.retina import rank


def test_board_image_shades_each_square():
    img = image(INITIAL)
    assert img[0, 0] == LIGHT and img[3, 0] == EMPTY
    assert img[7, 0] == SHADES[0]
    assert img[0, 1] == SHADES[2]


def test_each_eye_sees_its_half_of_the_board():
    where = np.array([[0.0, 0.0], [0.0, 1.0], [0.99, 0.0], [0.99, 1.0]])
    row, col = sample(where, np.array(["right", "right", "left", "left"]))
    assert row.tolist() == [0, 0, 7, 7]
    assert col.tolist() == [3, 7, 4, 0]


def test_rhythm_fires_at_each_lines_rate():
    rhythm = Rhythm(np.array([[MAX_RATE, 10.0, 0.0]]), steps=10_000, dt=1e-4, phase=np.full(3, 0.5))
    assert rhythm.shape == (10_000, 1, 3)
    fired = sum(rhythm[t][0].int() for t in range(10_000))
    assert fired.tolist() == [150, 10, 0]


def test_each_photoreceptor_gets_its_own_input_line():
    p = projection(np.array([4, 1]), n=6)
    assert p.shape == (2, 6)
    rows, cols = p.toarray().nonzero()
    assert rows.tolist() == [0, 1] and cols.tolist() == [4, 1]


def test_rank_spreads_values_over_zero_to_one():
    assert rank(np.array([5.0, -1.0, 3.0])).tolist() == [1.0, 0.0, 0.5]
