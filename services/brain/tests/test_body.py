import numpy as np

from connectome.body import facing, turn
from connectome.carve import clustered


def test_thinning_merges_what_sits_together():
    points = np.array([[0.0, 0, 0], [0.01, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 2, 3], [1, 2, 3], [0, 1, 4]])  # the last one collapses
    kept, made = clustered(points, faces, cells=4)
    assert len(kept) == 4
    assert len(made) == 2  # the two that were the same triangle stay, the flat one goes


def test_a_quarter_turn_is_a_quarter_turn():
    half = np.sqrt(0.5)
    turned = turn(f"{half} 0 0 {half}") @ np.array([1.0, 0, 0])
    assert np.allclose(turned, [0, 1, 0], atol=1e-9)
    assert np.allclose(turn(None), np.eye(3))


def test_the_model_is_turned_to_face_the_connectome():
    # the model looks along +x with +z up; the connectome has y down the head and z back
    ahead, left, up = np.array([[1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]])
    assert np.allclose(facing(ahead[None])[0], [0, 0, -1])  # forward is towards the front
    assert np.allclose(facing(up[None])[0], [0, -1, 0])  # up is towards the top of the head
    assert np.allclose(facing(left[None])[0], [1, 0, 0])  # across the head
