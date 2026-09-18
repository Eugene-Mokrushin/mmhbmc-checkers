import numpy as np
import scipy.sparse as sp

from connectome.extract import POPULATIONS, MushroomBody

CONTROLS = ("real", "degree", "random", "compartments")


def rewire(mb: MushroomBody, kind: str, rng: np.random.Generator) -> sp.csr_array:
    # Shuffles stay inside each population pair (KC->MBON, APL->KC, ...). Mixing
    # them would hand KCs random inhibition and fail the sparsity gate trivially.
    shuffle = {"degree": swap_targets, "random": scatter}[kind]
    counts = sp.coo_array(mb.graph.counts)
    rows, cols, data = [], [], []
    for a in POPULATIONS:
        for b in POPULATIONS:
            sel = (mb.population[counts.row] == a) & (mb.population[counts.col] == b)
            if sel.any():
                r, c, d = shuffle(counts.row[sel], counts.col[sel], counts.data[sel], mb.members(a), mb.members(b), rng)
                rows.append(r)
                cols.append(c)
                data.append(d)
    new = sp.csr_array((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=counts.shape)
    return (sp.diags_array(mb.graph.sign.astype(np.float32)) @ new).tocsr()


def swap_targets(rows, cols, data, sources, targets, rng, sweeps=10):
    # Maslov-Sneppen: swap the targets of two edges unless that makes a duplicate
    # or a self-loop. Every neuron keeps its in- and out-degree, every edge its weight.
    r, c = rows.tolist(), cols.tolist()
    n = max(r + c) + 1
    edges = {a * n + b for a, b in zip(r, c)}
    for i, j in rng.integers(0, len(r), size=(sweeps * len(r), 2)).tolist():
        a, b, x, y = r[i], c[i], r[j], c[j]
        if a == y or x == b or a * n + y in edges or x * n + b in edges:
            continue
        edges -= {a * n + b, x * n + y}
        edges |= {a * n + y, x * n + b}
        c[i], c[j] = y, b
    return rows, np.array(c), data


def scatter(rows, cols, data, sources, targets, rng):
    # Erdos-Renyi: the same number of edges and synapses, placed uniformly at random
    n = int(max(sources.max(), targets.max())) + 1
    chosen: set[int] = set()
    while len(chosen) < len(rows):
        r = rng.choice(sources, size=len(rows))
        c = rng.choice(targets, size=len(rows))
        for code in (r[r != c] * n + c[r != c]).tolist():
            if len(chosen) == len(rows):
                break
            chosen.add(code)
    codes = np.array(sorted(chosen))
    return codes // n, codes % n, rng.permutation(data)


def scramble_compartments(reward_share: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # each MBON keeps its KC input and its valence but gets another MBON's DANs
    return rng.permutation(reward_share)
