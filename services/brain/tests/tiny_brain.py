import numpy as np
import scipy.sparse as sp

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
