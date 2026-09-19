from __future__ import annotations

import dataclasses

import numpy as np
import scipy.sparse as sp

from connectome.graph import Connectome

POPULATIONS = ("KC", "MBON", "DAN", "APL")

# Most DANs split their KC synapses between both mushroom bodies; MBONs and APL don't.
MIN_SHARE = {"MBON": 0.5, "DAN": 0.1, "APL": 0.5}


@dataclasses.dataclass(frozen=True, eq=False)
class MushroomBody:
    side: str
    neurons: np.ndarray  # indices into the full connectome
    population: np.ndarray
    graph: Connectome

    def members(self, population: str) -> np.ndarray:
        return np.flatnonzero(self.population == population)

    def block(self, pre: str, post: str) -> sp.csr_array:
        return self.graph.counts[self.members(pre)][:, self.members(post)]

    def dan_mbon_contact(self) -> np.ndarray:
        contact = (self.block("DAN", "MBON") + self.block("MBON", "DAN").T).toarray().astype(np.float64)
        return contact / np.maximum(contact.sum(axis=0), 1)


def candidates(c: Connectome) -> dict[str, np.ndarray]:
    return {
        "MBON": c.cell_class == "MBON",
        "DAN": np.strings.startswith(c.cell_type, "PAM") | np.strings.startswith(c.cell_type, "PPL1"),
        "APL": c.cell_type == "APL",
    }


def kc_contact(counts: sp.csr_array, idx: np.ndarray, kcs: np.ndarray) -> np.ndarray:
    out = counts[idx][:, kcs].sum(axis=1)
    inp = counts[kcs][:, idx].sum(axis=0)
    return np.asarray(out).ravel() + np.asarray(inp).ravel()


def extract(c: Connectome, side: str = "right") -> MushroomBody:
    # side "both" takes the two mushroom bodies together
    kc = c.cell_class == "Kenyon_Cell"
    own = np.flatnonzero(kc & ((c.side == side) | (side == "both")))
    other = np.flatnonzero(kc & (c.side != side) & (side != "both"))

    parts = {"KC": own}
    for population, mask in candidates(c).items():
        idx = np.flatnonzero(mask)
        mine, theirs = kc_contact(c.counts, idx, own), kc_contact(c.counts, idx, other)
        share = mine / np.maximum(mine + theirs, 1)
        parts[population] = idx[share >= MIN_SHARE[population]]

    neurons = np.concatenate([parts[p] for p in POPULATIONS])
    population = np.concatenate([np.full(len(parts[p]), p) for p in POPULATIONS])
    order = np.argsort(neurons)
    neurons, population = neurons[order], population[order]
    return MushroomBody(side=side, neurons=neurons, population=population, graph=c.subgraph(neurons))
