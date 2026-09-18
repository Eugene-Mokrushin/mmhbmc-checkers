import json
import os
from pathlib import Path

import pandas as pd

from connectome import raw
from connectome.coordinates import one_per_neuron, read_markers
from connectome.extract import MushroomBody, extract
from connectome.load import load

ARTIFACTS_DIR = Path(os.environ.get("FLY_ARTIFACTS_DIR", raw.REPO / "artifacts"))
NEURONS_JSON = ARTIFACTS_DIR / "neurons.json"


def neurons_json(mb: MushroomBody, points: pd.DataFrame) -> dict:
    g = mb.graph
    missing = set(g.root_id) - set(points.index)
    if missing:
        raise ValueError(f"{len(missing)} neurons have no coordinates: {sorted(missing)[:3]}")
    xyz = points.loc[g.root_id, ["x", "y", "z"]].to_numpy().tolist()
    return {
        **raw.SOURCE,
        "hemisphere": mb.side,
        "units": "nm",
        "neurons": [
            {
                "root_id": str(g.root_id[i]),
                "population": str(mb.population[i]),
                "cell_type": str(g.cell_type[i]),
                "side": str(g.side[i]),
                "position": xyz[i],
            }
            for i in range(g.n)
        ],
    }


def main() -> None:
    mb = extract(load())
    doc = neurons_json(mb, one_per_neuron(read_markers()))
    NEURONS_JSON.parent.mkdir(parents=True, exist_ok=True)
    NEURONS_JSON.write_text(json.dumps(doc, separators=(",", ":")))
    sizes = {p: int((mb.population == p).sum()) for p in ("KC", "MBON", "DAN", "APL")}
    print(f"{mb.side} mushroom body: {sizes}, {mb.graph.counts.nnz:,} connections")
    print(f"wrote {NEURONS_JSON} ({NEURONS_JSON.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
