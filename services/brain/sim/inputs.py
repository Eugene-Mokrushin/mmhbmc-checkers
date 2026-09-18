import numpy as np
import scipy.sparse as sp

from connectome.extract import MushroomBody
from connectome.graph import Connectome

N_LINES = 128


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


def poisson(lines: np.ndarray, steps: int, rate: float, dt: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random((steps, *lines.shape)) < rate * dt) & lines
