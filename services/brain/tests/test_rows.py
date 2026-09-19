import numpy as np
import pytest
import torch

from sim.fast import FastLIF
from sim.kernels import Rows


def test_locate_read_write_touch_only_existing_connections():
    rows = Rows(np.array([[0, 2, 0], [3, 0, 4], [0, 0, 5]], dtype=float), torch.float32)
    where = rows.locate(np.array([0, 1]), np.array([1, 2]))
    assert where.tolist() == [[0, -1], [-1, 2]]
    assert rows.read(where).tolist() == [[2.0, 0.0], [0.0, 4.0]]
    rows.write(where, torch.tensor([[7.0, 9.0], [9.0, 8.0]]))
    assert rows.read(where).tolist() == [[7.0, 0.0], [0.0, 8.0]]
    assert rows.data.tolist() == [7.0, 3.0, 8.0, 5.0]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a second device")
def test_another_device_gives_the_same_spikes():
    rng = np.random.default_rng(0)
    weights = rng.integers(1, 40, (60, 60)) * (rng.random((60, 60)) < 0.1) * np.where(rng.random((60, 1)) < 0.7, 1, -1)
    projection = rng.integers(0, 60, (16, 60)) * (rng.random((16, 60)) < 0.3)
    inputs = torch.from_numpy(rng.random((300, 2, 16)) < 0.05)
    cpu = FastLIF(weights, projection).raster(inputs)
    mps = FastLIF(weights, projection, device="mps").raster(inputs)
    assert torch.equal(cpu, mps)
