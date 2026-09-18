import copy

import numpy as np
import scipy.sparse as sp
import torch

from connectome.extract import extract
from connectome.graph import Connectome
from flycore.board import Move
from flycore.encode import N_LINES
from game.decode import score, valence
from game.players import Player, candidates
from sim.fast import FastLIF
from sim.inputs import background, board_lines, normalization, projection, regular
from sim.params import LIF

WINDOW = 500
SETTLE = 3000
CHUNK = 1024


class Fly(Player):
    def __init__(self, c: Connectome, p: LIF = LIF(), seed: int = 0):
        super().__init__(seed)
        self.mb = extract(c)
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

    def untrained(self) -> "Fly":
        # plasticity edits the simulator's weights, never the wiring the fly started with
        return self.with_weights(self.wiring)

    def inputs(self, positions) -> tuple[torch.Tensor, np.ndarray]:
        lines = board_lines(positions)
        return torch.from_numpy(regular(lines, WINDOW, self.sim.p.input_period, self.phase)), normalization(lines)

    def counts(self, positions) -> np.ndarray:
        # the fly is deterministic, so each distinct board is simulated once
        unique = list(dict.fromkeys(positions))
        out = []
        for i in range(0, len(unique), CHUNK):
            spikes, scale = self.inputs(unique[i : i + CHUNK])
            out.append(self.sim.counts(spikes, self.rest, scale).numpy())
        where = {pos: i for i, pos in enumerate(unique)}
        return np.concatenate(out)[[where[pos] for pos in positions]]

    def raster(self, position) -> np.ndarray:
        spikes, scale = self.inputs([position])
        return self.sim.raster(spikes, self.rest, scale)[:, 0].numpy()

    def scores(self, after):
        return score(self.counts(after), self.mb, self.valence).tolist()

    def choose_recorded(self, positions) -> tuple[list[Move], np.ndarray]:
        options, after = candidates(positions)
        counts = self.counts(after)
        picks = self.pick(options, score(counts, self.mb, self.valence))
        flat = [m for moves in options for m in moves]
        return [flat[i] for i in picks], counts[picks]
