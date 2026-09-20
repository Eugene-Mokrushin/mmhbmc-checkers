import numpy as np
import pytest
import scipy.sparse as sp
import torch

from sim.fast import FastLIF
from sim.grad import GradLIF
from sim.params import LIF
from sim.surrogate import Pattern, Propagate


def network(seed=0, n=80, lines=12):
    rng = np.random.default_rng(seed)
    mask = (rng.random((n, n)) < 0.15) & ~np.eye(n, dtype=bool)
    w = mask * rng.integers(1, 40, (n, n)) * rng.choice([-1, 1], (n, n), p=[0.3, 0.7])
    projection = sp.csr_array((np.full(lines, 250.0), (np.arange(lines), rng.choice(n, lines, replace=False))), shape=(lines, n))
    inputs = torch.from_numpy(rng.random((400, 3, lines)) < 0.03)
    return sp.csr_array(w), projection, inputs


@pytest.mark.parametrize("dt", [1e-4, 5e-4])
def test_same_spikes_as_the_fast_simulator(dt):
    w, projection, inputs = network()
    p = LIF(dt=dt, input_gain=1.0)
    readout = np.arange(80)
    with torch.no_grad():
        counts = GradLIF(w, projection, readout, p)(inputs).numpy()
    expected = FastLIF(w, projection, p).counts(inputs).numpy()
    assert counts.sum() > 100
    assert np.array_equal(counts, expected)


def test_propagation_gradients_are_exact():
    w, _, _ = network(1, n=15)
    pattern = Pattern(w.indptr, w.indices, 15, "cpu")
    x = torch.randn(2, 15, dtype=torch.float64, requires_grad=True)
    spikes = (torch.rand(2, 15, dtype=torch.float64) * (torch.rand(2, 15) < 0.5)).requires_grad_()
    values = torch.randn(w.nnz, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b, c: Propagate.apply(a, b, c, pattern), (x, spikes, values))


def test_learning_raises_a_readout_neurons_firing():
    w, projection, inputs = network(2)
    brain = GradLIF(w, projection, np.array([40]), LIF(dt=5e-4, input_gain=1.0))
    optimizer = torch.optim.Adam(brain.parameters(), lr=0.05)
    first = brain(inputs[:200]).sum().item()
    for _ in range(15):
        optimizer.zero_grad()
        (-brain(inputs[:200]).sum()).backward()
        optimizer.step()
    assert brain(inputs[:200]).sum().item() > first + 3
