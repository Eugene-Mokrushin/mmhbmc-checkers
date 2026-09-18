import numpy as np
import scipy.sparse as sp

from connectome.extract import MushroomBody
from connectome.graph import Connectome
from flycore.board import Position
from flycore.encode import N_LINES, encode, square_lines
from flycore.squares import JUMP
from sim.params import LIF

# every run of three squares along a diagonal: a piece, its neighbour, the square behind
DIAGONALS = [(s, over, land) for s in range(32) for over, land in JUMP[s] if land >= 0]


def projection(c: Connectome, mb: MushroomBody, rng: np.random.Generator) -> sp.csr_array:
    # Each KC keeps its real claws, one per projection neuron with that neuron's
    # synapse count, but they read three squares on one diagonal. A KC can then
    # fire for a local shape like "my man, their man, empty square behind".
    kcs = mb.members("KC")
    pns = np.flatnonzero(c.cell_class == "ALPN")
    claws = c.counts[pns][:, mb.neurons[kcs]].tocsc()
    lines, targets, counts = [], [], []
    for j, kc in enumerate(kcs):
        field = [line for square in DIAGONALS[rng.integers(len(DIAGONALS))] for line in square_lines(square)]
        synapses = np.sort(claws.data[claws.indptr[j] : claws.indptr[j + 1]])[::-1][: len(field)]
        lines.append(rng.choice(field, size=len(synapses), replace=False))
        targets.append(np.full(len(synapses), kc))
        counts.append(synapses)
    coo = (np.concatenate(counts), (np.concatenate(lines), np.concatenate(targets)))
    return sp.csr_array(coo, shape=(N_LINES, mb.graph.n))


def board_lines(positions: list[Position]) -> np.ndarray:
    out = np.zeros((len(positions), N_LINES), dtype=bool)
    for row, pos in zip(out, positions):
        row[encode(pos)] = True
    return out


def random_lines(batch: int, active: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((batch, N_LINES), dtype=bool)
    for row in out:
        row[rng.choice(N_LINES, size=active, replace=False)] = True
    return out


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
