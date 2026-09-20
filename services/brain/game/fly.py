import copy

import numpy as np
import scipy.sparse as sp
import torch

from connectome.extract import extract
from connectome.graph import Connectome
from flycore.board import Move
from flycore.encode import N_LINES
from game.decode import valence
from game.players import Player, candidates
from sim.fast import FastLIF
from sim.inputs import background, board_lines, projection, regular
from sim.params import LIF

WINDOW = 500
SETTLE = 3000
CHUNK = 1024


class Fly(Player):
    window = WINDOW

    def __init__(self, c: Connectome, p: LIF = LIF(), seed: int = 0, side: str = "right"):
        super().__init__(seed)
        self.mb = extract(c, side)
        self.projection = projection(c, self.mb, self.rng)
        self.phase = self.rng.integers(0, p.input_period, N_LINES)
        self.valence = valence(self.mb)
        self.wire(self.mb.graph.weights(), p)

    def wire(self, weights: sp.csr_array, p: LIF) -> None:
        self.wiring = weights
        self.sim = FastLIF(weights, self.projection, p, bias=background(self.mb, p))
        self.rest = self.sim.settle(SETTLE)

    def with_weights(self, weights: sp.csr_array) -> "Fly":
        other = copy.copy(self)
        other.rng = self.rng.spawn(1)[0]
        other.wire(weights, self.sim.p)
        return other

    @property
    def kcs(self) -> np.ndarray:
        # KC and MBON indices in the simulator (where plasticity acts) and as columns of counts()
        return self.mb.members("KC")

    @property
    def mbons(self) -> np.ndarray:
        return self.mb.members("MBON")

    kc_cols, mbon_cols = kcs, mbons

    def untrained(self) -> "Fly":
        # plasticity edits the simulator's weights, never the wiring the fly started with
        return self.with_weights(self.wiring)

    def inputs(self, positions) -> torch.Tensor:
        return torch.from_numpy(regular(board_lines(positions), WINDOW, self.sim.p.input_period, self.phase))

    def counts(self, positions) -> np.ndarray:
        # the fly is deterministic, so each distinct board is simulated once
        unique = list(dict.fromkeys(positions))
        out = []
        for i in range(0, len(unique), CHUNK):
            out.append(self.sim.counts(self.inputs(unique[i : i + CHUNK]), self.rest).numpy())
        where = {pos: i for i, pos in enumerate(unique)}
        return np.concatenate(out)[[where[pos] for pos in positions]]

    def raster(self, position) -> np.ndarray:
        return self.sim.raster(self.inputs([position]), self.rest)[:, 0].numpy()

    def read(self, counts: np.ndarray) -> np.ndarray:
        # the same judgement, from the spike counts of every neuron rather than the few
        # the player keeps
        return counts[:, self.mbons] @ self.valence

    def judge(self, counts: np.ndarray) -> np.ndarray:
        # approach MBON spikes minus avoid MBON spikes
        return counts[:, self.mbon_cols] @ self.valence

    def scores(self, after):
        return self.judge(self.counts(after)).tolist()

    def choose_recorded(self, positions) -> tuple[list[Move], np.ndarray]:
        options, after = candidates(positions)
        counts = self.counts(after)
        picks = self.pick(options, self.judge(counts))
        flat = [m for moves in options for m in moves]
        return [flat[i] for i in picks], counts[picks]
