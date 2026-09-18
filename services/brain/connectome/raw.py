import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from paths import RAW_DIR

CLASSIFICATION = "classification.csv.gz"
CELL_TYPES = "consolidated_cell_types.csv.gz"
TRANSMITTERS = "neurons.csv.gz"
EDGES = "connections_princeton_no_threshold.csv.gz"
EDGES_MIN5 = "connections_princeton.csv.gz"

SOURCE = {"dataset": "FAFB", "snapshot": "v783", "downloaded": "2026-09-18"}
SHA256 = {
    "classification.csv.gz": "e946b552f4056dfc977707be0674609832c3f64332a22d69dc0d9615e7aae663",
    "connections_princeton.csv.gz": "445f996bf6c4b1803b9ba186189138a3061ff8623aa94c0abcf38af30a5bd48b",
    "connections_princeton_no_threshold.csv.gz": "62c2e562a7470bfd32cbb98e36af3daa607ba0822b37223b34f2aa45250a920e",
    "consolidated_cell_types.csv.gz": "8aba246d71dc40361677493629972ce3883048c3d02010adc42bda22962a1a2d",
    "coordinates.csv.gz": "14337121f451f98c2576cee72c24409ada5aaf7948b7c7ca8de9040296840e05",
    "neurons.csv.gz": "6a6b3759e635f0f35a677d169052362131ec61d95f55919298b55c43fce4e719",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_hashes(raw_dir: Path, names: list[str]) -> dict[str, str]:
    hashes = {name: sha256(raw_dir / name) for name in names}
    changed = [name for name, h in hashes.items() if h != SHA256.get(name)]
    if changed:
        raise ValueError(f"raw files differ from the pinned download: {changed}")
    return hashes


def _strings(path: Path, columns: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["root_id", *columns],
        dtype={"root_id": np.int64, **{c: str for c in columns}},
        keep_default_na=False,
    )
    return df.rename(columns=columns)


def read_neurons(raw_dir: Path) -> pd.DataFrame:
    neurons = _strings(
        raw_dir / CLASSIFICATION,
        {"super_class": "super_class", "class": "cell_class", "sub_class": "sub_class", "side": "side"},
    )
    for name, columns in [
        (CELL_TYPES, {"primary_type": "cell_type"}),
        (TRANSMITTERS, {"nt_type": "predicted_nt"}),
    ]:
        extra = _strings(raw_dir / name, columns)
        if not extra["root_id"].isin(neurons["root_id"]).all():
            raise ValueError(f"{name} has root IDs missing from {CLASSIFICATION}")
        neurons = neurons.merge(extra, on="root_id", how="left", validate="one_to_one")
    return neurons.fillna("").sort_values("root_id", ignore_index=True)


def read_edges(raw_dir: Path, name: str = EDGES) -> pd.DataFrame:
    return pd.read_csv(
        raw_dir / name,
        usecols=["pre_root_id", "post_root_id", "syn_count", "nt_type"],
        dtype={
            "pre_root_id": np.int64,
            "post_root_id": np.int64,
            "syn_count": np.int32,
            "nt_type": "category",
        },
    )
