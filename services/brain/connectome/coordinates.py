from pathlib import Path

import numpy as np
import pandas as pd

from paths import RAW_DIR

COORDINATES = "coordinates.csv.gz"


def read_markers(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    markers = pd.read_csv(raw_dir / COORDINATES, dtype={"root_id": np.int64, "supervoxel_id": np.int64})
    xyz = markers.pop("position").str.strip("[]").str.split(expand=True).astype(np.int64)
    markers[["x", "y", "z"]] = xyz.to_numpy()
    return markers


def one_per_neuron(markers: pd.DataFrame) -> pd.DataFrame:
    # FlyWire markers are proofreading points, several per neuron. Keep the one
    # nearest the neuron's median marker, so the point always lies on the neuron.
    markers = markers.sort_values(["root_id", "supervoxel_id"], ignore_index=True)
    xyz = markers[["x", "y", "z"]]
    median = xyz.groupby(markers["root_id"]).transform("median")
    dist = ((xyz - median) ** 2).sum(axis=1)
    best = dist.groupby(markers["root_id"]).idxmin()
    return markers.loc[best, ["root_id", "x", "y", "z"]].set_index("root_id")
