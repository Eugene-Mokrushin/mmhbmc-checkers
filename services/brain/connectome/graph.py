from __future__ import annotations

import dataclasses

import numpy as np
import scipy.sparse as sp

ANNOTATIONS = ("super_class", "cell_class", "sub_class", "side", "cell_type", "nt_type")


@dataclasses.dataclass(frozen=True, eq=False)
class Connectome:
    root_id: np.ndarray
    counts: sp.csr_array  # [pre, post], synapses summed over neuropils
    sign: np.ndarray
    nt_confident: np.ndarray
    super_class: np.ndarray
    cell_class: np.ndarray
    sub_class: np.ndarray
    side: np.ndarray
    cell_type: np.ndarray
    nt_type: np.ndarray
    meta: dict

    @property
    def n(self) -> int:
        return len(self.root_id)

    def index_of(self, root_ids) -> np.ndarray:
        return index(self.root_id, np.asarray(root_ids, dtype=np.int64))

    def weights(self) -> sp.csr_array:
        c = self.counts
        sign = np.repeat(self.sign, np.diff(c.indptr))
        data = c.data.astype(np.float32) * sign
        return sp.csr_array((data, c.indices, c.indptr), shape=c.shape)

    def subgraph(self, idx: np.ndarray) -> Connectome:
        if (np.diff(idx) <= 0).any():
            raise ValueError("subgraph indices must be ascending so root IDs stay sorted")
        fields = {f.name: getattr(self, f.name)[idx] for f in dataclasses.fields(self) if f.name not in ("counts", "meta")}
        return Connectome(counts=self.counts[idx][:, idx], meta=dict(self.meta), **fields)

    def thresholded(self, min_syn: int) -> Connectome:
        if min_syn <= self.meta["min_syn"]:
            return self
        counts = self.counts.copy()
        counts.data[counts.data < min_syn] = 0
        counts.eliminate_zeros()
        return dataclasses.replace(self, counts=counts, meta={**self.meta, "min_syn": min_syn})


def index(sorted_ids: np.ndarray, ids: np.ndarray) -> np.ndarray:
    idx = np.minimum(np.searchsorted(sorted_ids, ids), len(sorted_ids) - 1)
    unknown = sorted_ids[idx] != ids
    if unknown.any():
        raise ValueError(f"{unknown.sum()} root IDs not in the neuron table: {ids[unknown][:3].tolist()}")
    return idx
