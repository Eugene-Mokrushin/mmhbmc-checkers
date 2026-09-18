from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from connectome import raw
from connectome.graph import Connectome, index
from connectome.load import NPZ_PATH, save

# Shiu et al. 2024
SIGN = {"ACH": 1, "DA": 1, "SER": 1, "OCT": 1, "GABA": -1, "GLUT": -1}


def build(raw_dir: Path = raw.RAW_DIR, edges_file: str = raw.EDGES, verify: bool = True) -> Connectome:
    raw_dir = Path(raw_dir)
    names = [raw.CLASSIFICATION, raw.CELL_TYPES, raw.TRANSMITTERS, edges_file]
    hashes = raw.check_hashes(raw_dir, names) if verify else {}

    neurons = raw.read_neurons(raw_dir)
    edges = raw.read_edges(raw_dir, edges_file)
    root_id = neurons["root_id"].to_numpy()
    if (np.diff(root_id) == 0).any():
        raise ValueError("duplicate root IDs in the neuron table")
    n = len(root_id)

    pre = index(root_id, edges["pre_root_id"].to_numpy())
    post = index(root_id, edges["post_root_id"].to_numpy())
    counts = sp.coo_array((edges["syn_count"].to_numpy(), (pre, post)), shape=(n, n)).tocsr()
    counts.sum_duplicates()
    counts.sort_indices()

    predicted = neurons["predicted_nt"].to_numpy(dtype=str)
    nt = transmitters(pre, edges["nt_type"], predicted, root_id)
    return Connectome(
        root_id=root_id,
        counts=counts,
        sign=np.array([SIGN.get(t, 0) for t in nt], dtype=np.int8),
        nt_confident=predicted != "",
        super_class=neurons["super_class"].to_numpy(dtype=str),
        cell_class=neurons["cell_class"].to_numpy(dtype=str),
        sub_class=neurons["sub_class"].to_numpy(dtype=str),
        side=neurons["side"].to_numpy(dtype=str),
        cell_type=neurons["cell_type"].to_numpy(dtype=str),
        nt_type=nt,
        meta={**raw.SOURCE, "edges": edges_file, "sha256": hashes, "min_syn": 1},
    )


def transmitters(pre: np.ndarray, edge_nt: pd.Series, predicted: np.ndarray, root_id: np.ndarray) -> np.ndarray:
    codes = edge_nt.cat.codes.to_numpy()
    if (codes < 0).any():
        raise ValueError("connections with a blank nt_type")
    per_neuron = np.full(len(predicted), -1, dtype=np.int16)
    per_neuron[pre] = codes
    mixed = per_neuron[pre] != codes
    if mixed.any():
        bad = root_id[np.unique(pre[mixed])]
        raise ValueError(f"{len(bad)} neurons have more than one nt_type: {bad[:3].tolist()}")

    names = np.append(np.asarray(edge_nt.cat.categories, dtype=str), "")
    from_edges = names[per_neuron]
    conflict = (from_edges != "") & (predicted != "") & (from_edges != predicted)
    if conflict.any():
        raise ValueError(f"connections nt_type contradicts neurons.csv for {root_id[conflict][:3].tolist()}")

    nt = np.where(from_edges != "", from_edges, predicted)
    unknown = set(nt) - SIGN.keys() - {""}
    if unknown:
        raise ValueError(f"no sign convention for {sorted(unknown)}")
    return nt


def main() -> None:
    c = build()
    save(c)
    for min_syn in (1, 5):
        t = c.thresholded(min_syn)
        print(f"min_syn={min_syn}: {c.n:,} neurons, {t.counts.nnz:,} pairs, {int(t.counts.data.sum()):,} synapses")
    print(f"wrote {NPZ_PATH}")


if __name__ == "__main__":
    main()
