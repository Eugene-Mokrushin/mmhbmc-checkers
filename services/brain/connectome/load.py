import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from connectome.graph import ANNOTATIONS, Connectome
from paths import DATA_DIR

NPZ_PATH = DATA_DIR / "connectome.npz"
FORMAT = 1


def save(c: Connectome, path: Path = NPZ_PATH) -> None:
    if c.meta["min_syn"] != 1:
        raise ValueError("save the unthresholded connectome and threshold at load time")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        root_id=c.root_id,
        counts_data=c.counts.data,
        counts_indices=c.counts.indices,
        counts_indptr=c.counts.indptr,
        sign=c.sign,
        nt_confident=c.nt_confident,
        meta=np.array(json.dumps({**c.meta, "format": FORMAT})),
        **{name: getattr(c, name) for name in ANNOTATIONS},
    )


def load(path: Path = NPZ_PATH, min_syn: int = 1) -> Connectome:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing, run `python -m connectome.build`")
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if meta.pop("format", None) != FORMAT:
            raise ValueError(f"{path} is from an older build, run `python -m connectome.build`")
        n = len(z["root_id"])
        counts = sp.csr_array((z["counts_data"], z["counts_indices"], z["counts_indptr"]), shape=(n, n))
        c = Connectome(
            root_id=z["root_id"],
            counts=counts,
            sign=z["sign"],
            nt_confident=z["nt_confident"],
            meta=meta,
            **{name: z[name] for name in ANNOTATIONS},
        )
    return c.thresholded(min_syn)
