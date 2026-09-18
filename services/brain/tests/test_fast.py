import numpy as np
import scipy.sparse as sp
import torch

from sim.fast import FastLIF
from sim.params import LIF

P = LIF()


def network(weights, projection):
    return FastLIF(sp.csr_array(np.array(weights, dtype=np.float64)), sp.csr_array(np.array(projection, dtype=np.float64)))


def regular(steps, every, lines=1):
    inputs = np.zeros((steps, 1, lines), dtype=bool)
    inputs[::every, 0, 0] = True
    return torch.from_numpy(inputs)


def test_silent_without_input():
    sim = network([[0, 50], [50, 0]], [[0, 0]])
    assert sim.counts(regular(500, 10) & False).sum() == 0


def test_driven_neuron_fires_and_respects_refractory_period():
    sim = network([[0]], [[400]])
    spikes = sim.raster(regular(2000, 5))[:, 0, 0].numpy()
    times = np.flatnonzero(spikes)
    assert len(times) > 10
    assert np.diff(times).min() >= P.ref_steps


def test_excitation_propagates():
    sim = network([[0, 300], [0, 0]], [[400, 0]])
    raster = sim.raster(regular(1000, 5))[:, 0].numpy()
    assert raster[:, 1].sum() > 0
    assert np.flatnonzero(raster[:, 1])[0] > np.flatnonzero(raster[:, 0])[0]


def test_inhibition_suppresses():
    free = network([[0, 0], [0, 0]], [[400, 12]]).counts(regular(2000, 20))[0, 1]
    inhibited = network([[0, -300], [0, 0]], [[400, 12]]).counts(regular(2000, 20))[0, 1]
    assert free >= 10
    assert inhibited == 0


def test_batch_entries_are_independent():
    sim = network([[0, 300], [-300, 0]], [[400, 0], [0, 400]])
    rng = np.random.default_rng(0)
    inputs = torch.from_numpy(rng.random((1000, 3, 2)) < 0.1)
    together = sim.counts(inputs)
    for b in range(3):
        assert torch.equal(sim.counts(inputs[:, b : b + 1])[0], together[b])
