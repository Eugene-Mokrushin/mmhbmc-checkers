import numpy as np
import pandas as pd

from api.atlas import SPAN, Drawings, atlas, frame
from api.symmetry import Envelope


def markers(roots) -> pd.DataFrame:
    return pd.DataFrame({"x": [1000.0 * i for i in range(len(roots))], "y": [0.0] * len(roots), "z": [0.0] * len(roots)}, index=pd.Index(roots, name="root_id"))


def store(path, roots, counts, xyz):
    np.savez(path, root_id=np.asarray(roots, dtype=np.int64), counts=np.asarray(counts, dtype=np.int64), xyz=np.asarray(xyz, dtype=np.float32))
    return Drawings(path)


def read(blob: bytes):
    neurons, points, quiet = np.frombuffer(blob, dtype="<u4", count=3)
    spans = np.frombuffer(blob, dtype="<u2", offset=12, count=neurons)
    xyz = np.frombuffer(blob, dtype="<i2", offset=12 + neurons * 2, count=(points + quiet) * 3).reshape(-1, 3)
    return spans, xyz


def test_a_neuron_is_drawn_along_its_skeleton(tmp_path):
    roots = [10, 20]
    drawn = store(tmp_path / "s.npz", roots, [3, 0], [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
    points = markers(roots)
    middle, spread = frame(points)
    spans, xyz = read(atlas(np.array(roots), points, middle, spread, drawn))
    assert list(spans) == [3, 1]  # the one without a skeleton keeps its marker
    assert len(xyz) == 4


def test_a_crowded_brain_shares_the_points_out(tmp_path):
    roots = list(range(40))
    drawn = store(tmp_path / "s.npz", roots, [8] * 40, np.zeros((320, 3)))
    points = markers(roots)
    middle, spread = frame(points)
    spans, xyz = read(atlas(np.array(roots), points, middle, spread, drawn, budget=80))
    assert list(spans) == [2] * 40
    assert len(xyz) == 80


def test_without_any_skeletons_every_neuron_is_its_marker():
    roots = [7, 8, 9]
    points = markers(roots)
    middle, spread = frame(points)
    spans, xyz = read(atlas(np.array(roots), points, middle, spread, None))
    assert list(spans) == [1, 1, 1]
    assert abs(xyz[:, 0]).max() <= 1.02 * SPAN


def test_a_side_the_knife_spared_is_cut_back_to_the_other():
    # a brain with a fringe on the left that has no counterpart on the right
    rng = np.random.default_rng(0)
    core = rng.uniform(-1, 1, (60000, 3)) * [100.0, 60.0, 60.0]
    fringe = rng.uniform(-1, 1, (3000, 3)) * [10.0, 60.0, 60.0] - [150.0, 0.0, 0.0]
    inside = Envelope(np.concatenate([core, fringe]))
    assert inside.holds(core).mean() > 0.9
    assert inside.holds(fringe).mean() < 0.01


def test_what_both_sides_have_is_kept():
    rng = np.random.default_rng(1)
    ball = rng.normal(0, 1, (60000, 3))
    ball = ball / np.linalg.norm(ball, axis=1, keepdims=True) * rng.uniform(0, 1, (60000, 1)) ** (1 / 3) * 100
    inside = Envelope(ball)
    assert inside.holds(ball).mean() > 0.93


def test_the_half_that_is_not_simulated_is_drawn_behind_it(tmp_path):
    roots, others = [10, 20], [30, 40]
    drawn = store(tmp_path / "s.npz", roots + others, [2, 2, 2, 2], np.zeros((8, 3)))
    points = markers(roots + others)
    middle, spread = frame(points)
    spans, xyz = read(atlas(np.array(roots), points, middle, spread, drawn, None, np.array(others)))
    assert list(spans) == [2, 2]  # only the fly's own neurons can fire
    assert len(xyz) == 8  # but the other half is there to look at
