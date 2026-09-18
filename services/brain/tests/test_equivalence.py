import numpy as np
import pytest
import scipy.sparse as sp
import torch

from connectome.extract import extract
from connectome.load import NPZ_PATH, load
from sim import reference
from sim.fast import FastLIF
from sim.inputs import poisson, projection, random_lines
from sim.params import LIF

P = LIF()


def same_raster(weights, proj, inputs):
    fast = FastLIF(weights, proj, dtype=torch.float64).raster(torch.from_numpy(inputs[:, None]))[:, 0].numpy()
    ref = reference.raster(weights, proj, inputs)
    assert fast.sum() > 0
    assert np.array_equal(fast, ref), f"{(fast != ref).sum()} of {ref.sum()} spikes differ"


def test_random_network():
    rng = np.random.default_rng(0)
    n = 60
    counts = rng.integers(1, 40, (n, n)) * (rng.random((n, n)) < 0.1)
    sign = np.where(rng.random((n, 1)) < 0.7, 1.0, -1.0)
    weights = sp.csr_array(sign * counts)
    proj = sp.csr_array(rng.integers(0, 60, (128, n)) * (rng.random((128, n)) < 0.05))
    inputs = poisson(random_lines(1, 24, rng), 500, P.input_rate, P.dt, rng)[:, 0]
    same_raster(weights, proj, inputs)


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.skipif(not NPZ_PATH.exists(), reason="run python -m connectome.build first")
def test_right_mushroom_body():
    rng = np.random.default_rng(0)
    c = load()
    mb = extract(c)
    inputs = poisson(random_lines(1, 24, rng), 1000, P.input_rate, P.dt, rng)[:, 0]
    same_raster(mb.graph.weights(), projection(c, mb, rng), inputs)
