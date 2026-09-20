import numpy as np
import scipy.sparse as sp
import torch

from sim.fast import FastLIF
from sim.params import LIF
from sim.watch import Watcher


def network(seed=0, n=90, lines=14):
    rng = np.random.default_rng(seed)
    mask = (rng.random((n, n)) < 0.15) & ~np.eye(n, dtype=bool)
    weights = mask * rng.integers(1, 40, (n, n)) * rng.choice([-1, 1], (n, n), p=[0.3, 0.7])
    projection = sp.csr_array((np.full(lines, 250.0), (np.arange(lines), rng.choice(n, lines, replace=False))), shape=(lines, n))
    inputs = torch.from_numpy(rng.random((300, 4, lines)) < 0.03)
    return sp.csr_array(weights), projection, inputs


def test_the_serving_path_counts_the_same_spikes():
    weights, projection, inputs = network()
    sim = FastLIF(weights, projection, LIF(input_gain=1.0))
    counts, frames = Watcher(sim).run(inputs)
    expected = sim.counts(inputs)
    assert counts.sum() > 200
    assert torch.equal(counts, expected)


def test_frames_say_when_a_neuron_fired():
    weights, projection, inputs = network(1)
    sim = FastLIF(weights, projection, LIF(input_gain=1.0))
    counts, frames = Watcher(sim).run(inputs)
    raster = sim.raster(inputs)
    per_frame = round(0.005 / sim.p.dt)
    for board in range(counts.shape[0]):
        for neuron in torch.nonzero(counts[board] > 0)[:6, 0].tolist():
            when = torch.nonzero(raster[:, board, neuron])[:, 0] // per_frame
            kept = [bit for bit in range(32) if frames[board, neuron].item() >> bit & 1]
            assert kept == sorted(set(when.tolist()))


def test_every_fly_the_website_serves_can_be_watched():
    # the watcher asks the fly for its input spikes; a fly that cannot answer cannot play
    from game.fly import Fly
    from game.gradfly import GradFly
    from game.wholefly import WholeFly

    for kind in (Fly, WholeFly, GradFly):
        assert callable(getattr(kind, "inputs", None)), kind.__name__


def test_the_website_is_shown_a_frame_every_two_milliseconds():
    weights, projection, inputs = network()
    sim = FastLIF(weights, projection, LIF(input_gain=1.0))
    watcher = Watcher(sim)
    steps = inputs.shape[0]
    for every, expected in ((2.0, steps // round(0.002 / sim.p.dt)), (5.0, steps // round(0.005 / sim.p.dt))):
        seen = [kind for kind, _ in watcher.stream(inputs, every_ms=every) if kind == "frame"]
        assert len(seen) == expected
