import numpy as np
import scipy.sparse as sp

from connectome.extract import MushroomBody
from connectome.graph import Connectome

NEURONS = [  # cell_class, cell_type, side
    ("Kenyon_Cell", "KCg-m", "right"),
    ("Kenyon_Cell", "KCab", "right"),
    ("Kenyon_Cell", "KCg-m", "left"),
    ("MBON", "MBON01", "right"),
    ("MBON", "MBON07", "left"),
    ("DAN", "PAM01", "right"),
    ("DAN", "PAM02", "right"),
    ("MBIN", "APL", "right"),
    ("MBIN", "DPM", "right"),
]
EDGES = [  # pre, post, synapses
    (0, 4, 5), (1, 4, 3),
    (2, 3, 6), (0, 3, 1),
    (5, 0, 4), (5, 2, 4),
    (6, 2, 19), (6, 1, 1),
    (0, 7, 2), (7, 1, 2),
    (5, 4, 3),
    (8, 0, 5),
]


def tiny_brain() -> Connectome:
    n = len(NEURONS)
    pre, post, syn = zip(*EDGES)
    column = [np.array(values) for values in zip(*NEURONS)]
    return Connectome(
        root_id=np.arange(n, dtype=np.int64) + 100,
        counts=sp.coo_array((syn, (pre, post)), shape=(n, n)).tocsr(),
        sign=np.ones(n, dtype=np.int8),
        nt_confident=np.ones(n, dtype=bool),
        super_class=np.full(n, "central"),
        cell_class=column[0],
        sub_class=np.full(n, ""),
        side=column[2],
        cell_type=column[1],
        nt_type=np.full(n, "ACH"),
        meta={"min_syn": 1},
    )


def random_mushroom_body(seed: int = 0) -> MushroomBody:
    rng = np.random.default_rng(seed)
    sizes = {"KC": 60, "MBON": 8, "DAN": 6, "APL": 1}
    population = np.concatenate([np.full(k, p) for p, k in sizes.items()])
    n = len(population)
    density = {("KC", "MBON"): 0.3, ("KC", "KC"): 0.1, ("KC", "APL"): 1.0, ("APL", "KC"): 1.0, ("DAN", "MBON"): 0.5, ("MBON", "MBON"): 0.3}
    counts = np.zeros((n, n), dtype=np.int32)
    for (a, b), d in density.items():
        block = np.ix_(population == a, population == b)
        counts[block] = rng.integers(1, 20, counts[block].shape) * (rng.random(counts[block].shape) < d)
    np.fill_diagonal(counts, 0)
    sign = np.where(population == "APL", -1, 1).astype(np.int8)
    sign[np.flatnonzero(population == "MBON")[:3]] = -1
    graph = Connectome(
        root_id=np.arange(n, dtype=np.int64),
        counts=sp.csr_array(counts),
        sign=sign,
        nt_confident=np.ones(n, dtype=bool),
        super_class=np.full(n, "central"),
        cell_class=population,
        sub_class=np.full(n, ""),
        side=np.full(n, "right"),
        cell_type=population,
        nt_type=np.full(n, "ACH"),
        meta={"min_syn": 1},
    )
    return MushroomBody(side="right", neurons=np.arange(n), population=population, graph=graph)
