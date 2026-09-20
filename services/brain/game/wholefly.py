import numpy as np
import pandas as pd
import scipy.sparse as sp

from connectome.extract import MushroomBody, extract
from connectome.graph import Connectome
from game.decode import valence
from game.fly import Fly
from game.players import Player
from sim.eye import Rhythm, projection, rates
from sim.fast import FastLIF
from sim.params import LIF
from sim.retina import field

WINDOW = 1000  # 100 ms: the board has to travel from the eyes through the optic lobes first
CHUNK = 512


def wiring(thin: Connectome, full: Connectome, mb: MushroomBody) -> sp.csr_array:
    # the brain at 5+ synapses per connection, but every KC->MBON synapse, since that
    # is the layer that learns
    kc, mbon = mb.neurons[mb.members("KC")], mb.neurons[mb.members("MBON")]
    missing = sp.coo_array(full.weights()[kc][:, mbon] - thin.weights()[kc][:, mbon])
    extra = sp.csr_array((missing.data, (kc[missing.row], mbon[missing.col])), shape=(thin.n, thin.n))
    out = (thin.weights() + extra).tocsr()
    out.eliminate_zeros()
    return out


class WholeFly(Fly):
    # The whole brain looks at the board through its eyes; learning is the same as in the
    # mushroom-body fly, dopamine at the KC->MBON synapses, here of both mushroom bodies.
    window = WINDOW

    def __init__(self, thin: Connectome, full: Connectome, points: pd.DataFrame, seed: int = 0, device: str = "cpu"):
        Player.__init__(self, seed)
        self.mb, self.device = extract(full, "both"), device
        self.receptors, self.where = field(thin, points)
        self.sides = thin.side[self.receptors]
        self.projection = projection(self.receptors, thin.n)
        self.phase = self.rng.random(len(self.receptors))
        self.valence = valence(self.mb)
        self.readout = np.concatenate([self.kcs, self.mbons])
        self.wire(wiring(thin, full, self.mb), LIF(input_gain=1.0))

    @property
    def kcs(self) -> np.ndarray:
        return self.mb.neurons[self.mb.members("KC")]

    @property
    def mbons(self) -> np.ndarray:
        return self.mb.neurons[self.mb.members("MBON")]

    @property
    def kc_cols(self) -> np.ndarray:
        return np.arange(len(self.mb.members("KC")))

    @property
    def mbon_cols(self) -> np.ndarray:
        return np.arange(len(self.mb.members("KC")), len(self.readout))

    def wire(self, weights: sp.csr_array, p: LIF) -> None:
        self.wiring = weights
        self.sim = FastLIF(weights, self.projection, p, device=self.device)
        self.rest = None

    def inputs(self, positions) -> Rhythm:
        return Rhythm(rates(positions, self.where, self.sides), WINDOW, self.sim.p.dt, self.phase)

    def counts(self, positions) -> np.ndarray:
        unique = list(dict.fromkeys(positions))
        out = [self.sim.counts(self.inputs(unique[i : i + CHUNK])).numpy()[:, self.readout] for i in range(0, len(unique), CHUNK)]
        where = {pos: i for i, pos in enumerate(unique)}
        return np.concatenate(out)[[where[pos] for pos in positions]]
