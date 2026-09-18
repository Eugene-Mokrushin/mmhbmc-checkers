import numpy as np
import scipy.sparse as sp

from connectome.extract import MushroomBody
from connectome.graph import Connectome
from flycore.board import Position
from flycore.encode import encode
from sim.params import LIF

N_LINES = 128
FULL_BOARD = 24


def projection(c: Connectome, mb: MushroomBody, rng: np.random.Generator, n_lines: int = N_LINES) -> sp.csr_array:
    # Each KC keeps its real claws, one per projection neuron with that neuron's
    # synapse count, but the claws are rewired to random board lines.
    kcs = mb.members("KC")
    pns = np.flatnonzero(c.cell_class == "ALPN")
    claws = c.counts[pns][:, mb.neurons[kcs]].tocsc()
    lines, targets, counts = [], [], []
    for j, kc in enumerate(kcs):
        synapses = claws.data[claws.indptr[j] : claws.indptr[j + 1]]
        lines.append(rng.choice(n_lines, size=len(synapses), replace=False))
        targets.append(np.full(len(synapses), kc))
        counts.append(synapses)
    coo = (np.concatenate(counts), (np.concatenate(lines), np.concatenate(targets)))
    return sp.csr_array(coo, shape=(n_lines, mb.graph.n))


def random_lines(batch: int, active: int, rng: np.random.Generator, n_lines: int = N_LINES) -> np.ndarray:
    out = np.zeros((batch, n_lines), dtype=bool)
    for row in out:
        row[rng.choice(n_lines, size=active, replace=False)] = True
    return out


def board_lines(positions: list[Position], n_lines: int = N_LINES) -> np.ndarray:
    out = np.zeros((len(positions), n_lines), dtype=bool)
    for row, pos in zip(out, positions):
        row[encode(pos)] = True
    return out


def normalization(lines: np.ndarray) -> np.ndarray:
    # The antennal lobe we bypass normalizes its output across odor strengths
    # (Olsen et al. 2010). Scaling by sqrt(24 / pieces) keeps 5-10% of KCs active
    # from opening to endgame.
    return np.sqrt(FULL_BOARD / np.maximum(lines.sum(axis=1), 1))


def regular(lines: np.ndarray, steps: int, period: int, phase: np.ndarray) -> np.ndarray:
    # A board isn't noisy: each active line fires at a fixed rate with its own fixed
    # offset, so the same board always produces the same spikes.
    fire = (np.arange(steps)[:, None] + phase[None, :]) % period == 0
    return fire[:, None, :] & lines[None]


def poisson(lines: np.ndarray, steps: int, rate: float, dt: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random((steps, *lines.shape)) < rate * dt) & lines


def background(mb: MushroomBody, p: LIF) -> np.ndarray:
    # stands in for the third of MBON input that comes from outside the mushroom body
    return np.where(mb.population == "MBON", p.mbon_drive, 0.0)
