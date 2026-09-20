import numpy as np
import pandas as pd

from api.atlas import SPAN, Drawings, atlas, frame


def markers(roots) -> pd.DataFrame:
    return pd.DataFrame({"x": [1000.0 * i for i in range(len(roots))], "y": [0.0] * len(roots), "z": [0.0] * len(roots)}, index=pd.Index(roots, name="root_id"))


def store(path, roots, counts, xyz):
    np.savez(path, root_id=np.asarray(roots, dtype=np.int64), counts=np.asarray(counts, dtype=np.int64), xyz=np.asarray(xyz, dtype=np.float32))
    return Drawings(path)


def read(blob: bytes):
    neurons, points = np.frombuffer(blob, dtype="<u4", count=2)
    spans = np.frombuffer(blob, dtype="<u2", offset=8, count=neurons)
    xyz = np.frombuffer(blob, dtype="<i2", offset=8 + neurons * 2, count=points * 3).reshape(-1, 3)
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
