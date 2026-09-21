import numpy as np
import scipy.sparse as sp
import torch

from flycore.board import INITIAL, apply_move
from flycore.moves import legal_moves
from game.remember import Remembering, run
from sim.fast import FastLIF
from sim.params import LIF


class Eyes:
    # a stand-in retina: every board becomes the same steady drizzle of input
    def __init__(self, lines: int, steps: int):
        self.lines, self.steps = lines, steps

    def inputs(self, boards):
        rng = np.random.default_rng(len(boards))
        return torch.from_numpy(rng.random((self.steps, len(boards), self.lines)) < 0.05)


class Fly:
    # the smallest thing Remembering can drive: a brain, an eye and a way to read it
    def __init__(self, n: int = 60, lines: int = 8, steps: int = 40, seed: int = 0):
        rng = np.random.default_rng(seed)
        weights = sp.csr_array(rng.normal(0, 0.0008, (n, n)) * (rng.random((n, n)) < 0.2))
        projection = sp.csr_array((np.full(lines, 250.0), (np.arange(lines), rng.choice(n, lines, replace=False))), shape=(lines, n))
        self.sim = FastLIF(weights, projection, LIF(input_gain=1.0))
        self.eyes = Eyes(lines, steps)
        self.head = rng.normal(0, 1, n)

    def read(self, counts: np.ndarray) -> np.ndarray:
        return counts @ self.head


def test_the_state_it_ends_in_is_the_state_it_starts_from_next_time():
    fly = Fly()
    mind = Remembering(fly)
    mind.choose_many([INITIAL], [7])
    first = mind.kept[7]
    assert first[0].shape == (1, fly.sim.n)
    after = apply_move(INITIAL, legal_moves(INITIAL)[0])
    mind.choose_many([after], [7])
    assert not torch.equal(first[0], mind.kept[7][0])  # it moved on rather than resetting


def test_a_forgetting_brain_starts_the_same_way_every_time():
    fly = Fly()
    mind = Remembering(fly)
    first = mind.choose_many([INITIAL], [1])
    mind.forget()
    again = mind.choose_many([INITIAL], [1])
    assert [m.origin for m in first] == [m.origin for m in again]


def test_each_game_keeps_its_own_state():
    fly = Fly()
    mind = Remembering(fly)
    mind.choose_many([INITIAL, INITIAL], [1, 2])
    assert set(mind.kept) == {1, 2}
    mind.forget(1)
    assert set(mind.kept) == {2}


def test_the_run_hands_back_where_it_finished():
    fly = Fly()
    inputs = fly.eyes.inputs([INITIAL])
    counts, ending = run(fly.sim, inputs, None)
    assert counts.shape == (1, fly.sim.n)
    assert ending[0].shape == (1, fly.sim.n)
    # starting from that state again is not the same as starting from rest
    again, _ = run(fly.sim, inputs, ending)
    fresh, _ = run(fly.sim, inputs, None)
    assert not torch.equal(again, fresh)
