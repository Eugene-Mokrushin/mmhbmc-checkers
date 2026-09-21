import numpy as np
import pandas as pd
import scipy.sparse as sp

from connectome.graph import Connectome
from game.players import Player
from sim.eye import Rhythm, projection, rates
from sim.fast import FastLIF
from sim.params import LIF
from sim.retina import field

DT = 5e-4
WINDOW = 200  # 100 ms in 0.5 ms steps
CHUNK = 512
KINDS = {"R1-6": 0, "R7": 1, "R8": 2}
COLOUR = False  # R7 and R8 become two colour channels, one for each side
SPLIT = False  # and R1-6 drop to bare occupancy, so only colour says whose piece it is
SWEEP = 0.0  # how far the board drifts across the eye during the window, in squares
SWEEPS = 4


class Eyes:
    # the board shown to the photoreceptors, shared by the trained fly and its trainer
    def __init__(self, c: Connectome, points: pd.DataFrame, seed: int = 0):
        self.receptors, self.where = field(c, points)
        self.sides = c.side[self.receptors]
        self.kinds = np.array([KINDS.get(t, 0) for t in c.cell_type[self.receptors]])
        self.projection = projection(self.receptors, c.n)
        self.phase = np.random.default_rng(seed).random(len(self.receptors))

    def inputs(self, positions) -> Rhythm:
        shifts = np.linspace(-SWEEP, SWEEP, SWEEPS) if SWEEP else (0.0,)
        return Rhythm(rates(positions, self.where, self.sides, self.kinds if COLOUR else None, shifts, SPLIT), WINDOW, DT, self.phase)


def descending(c: Connectome) -> np.ndarray:
    return np.flatnonzero(c.super_class == "descending")


class GradFly(Player):
    # Version 2, gradient variant: the whole brain with connection strengths learned by
    # gradient descent (signs and wiring as in FlyWire), seeing the board through its eyes;
    # a board's score is a learned weighted sum of descending-neuron spikes.
    window = WINDOW

    def __init__(self, c: Connectome, points: pd.DataFrame, weights: sp.csr_array, head: np.ndarray, seed=0, device="cpu"):
        super().__init__(seed)
        self.eyes, self.dn, self.head = Eyes(c, points), descending(c), np.asarray(head)
        self.sim = FastLIF(weights, self.eyes.projection, LIF(dt=DT, input_gain=1.0), device=device)

    def inputs(self, positions) -> Rhythm:
        return self.eyes.inputs(positions)

    def counts(self, positions) -> np.ndarray:
        unique = list(dict.fromkeys(positions))
        out = [self.sim.counts(self.eyes.inputs(unique[i : i + CHUNK])).numpy()[:, self.dn] for i in range(0, len(unique), CHUNK)]
        where = {pos: i for i, pos in enumerate(unique)}
        return np.concatenate(out)[[where[pos] for pos in positions]]

    def read(self, counts: np.ndarray) -> np.ndarray:
        return counts[:, self.dn] @ self.head[:-1] + self.head[-1]

    def judge(self, counts: np.ndarray) -> np.ndarray:
        return counts @ self.head[:-1] + self.head[-1]

    def scores(self, after):
        return self.judge(self.counts(after)).tolist()
