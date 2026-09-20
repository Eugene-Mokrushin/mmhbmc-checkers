import numpy as np
import pandas as pd

from api.symmetry import Envelope
from paths import DATA_DIR

SPAN = 32000  # a whole brain half-width, comfortably inside a 16-bit number
BUDGET = 700_000  # points in one picture of a brain, shared out among its neurons
SKELETONS = DATA_DIR / "skeletons.npz"


def frame(points: pd.DataFrame) -> tuple[np.ndarray, float]:
    # one scale for everything, so a fly that holds part of the brain still draws its
    # neurons where they belong inside it
    xyz = points.to_numpy(dtype=np.float64)
    middle = np.nanmedian(xyz, axis=0)
    return middle, float(np.nanmax(np.abs(xyz - middle)))


def scaled(xyz: np.ndarray, middle: np.ndarray, spread: float) -> np.ndarray:
    # 16-bit numbers wrap around past 32767, which would throw a point to the far side
    # of the brain, so anything beyond the frame is held at its edge
    return np.clip(np.nan_to_num((xyz - middle) / spread, nan=0.0), -1.02, 1.02) * SPAN


class Drawings:
    # FlyWire's own skeletons, thinned: the places each cell runs through, not one dot
    def __init__(self, path=SKELETONS):
        with np.load(path) as held:
            counts = held["counts"].astype(np.int64)
            self.root_id, self.xyz = held["root_id"], held["xyz"]
            self.starts = np.cumsum(counts) - counts
            self.counts = counts
        self.order = np.argsort(self.root_id)
        self.sorted = self.root_id[self.order]
        self.keep = np.ones(len(self.xyz), dtype=bool)

    def trim(self, inside: "Envelope | None") -> None:
        # asked once for the whole store, not once per neuron
        self.keep = np.ones(len(self.xyz), dtype=bool) if inside is None else inside.holds(self.xyz)

    def of(self, root: int) -> np.ndarray:
        found = np.searchsorted(self.sorted, root)
        if found >= len(self.sorted) or self.sorted[found] != root:
            return self.xyz[:0]
        at = self.order[found]
        first, last = self.starts[at], self.starts[at] + self.counts[at]
        return self.xyz[first:last][self.keep[first:last]]


def spread_out(root_id: np.ndarray, points: pd.DataFrame, drawn: "Drawings | None", inside: "Envelope | None" = None, budget: int = BUDGET):
    # every neuron gets the same share of the picture; one with no skeleton keeps its marker
    share = max(1, budget // max(1, len(root_id)))
    marks = np.nan_to_num(points.reindex(root_id).to_numpy(dtype=np.float64))
    fits = inside.holds(marks) if inside is not None else np.ones(len(marks), dtype=bool)
    counts = np.zeros(len(root_id), dtype=np.uint16)
    out = []
    for i, root in enumerate(root_id):
        line = drawn.of(int(root)) if drawn is not None else marks[:0]
        if not len(line):
            line = marks[i : i + 1] if fits[i] else marks[:0]
        if not len(line):
            continue
        take = np.linspace(0, len(line) - 1, min(len(line), share)).astype(int)
        out.append(line[take])
        counts[i] = len(take)
    return (np.concatenate(out) if out else marks[:0]), counts


def atlas(root_id: np.ndarray, points: pd.DataFrame, middle: np.ndarray, spread: float, drawn: "Drawings | None" = None, inside: "Envelope | None" = None, budget: int = BUDGET) -> bytes:
    # where each of a fly's neurons runs, in the order its simulator numbers them, so a
    # spike can be drawn along the cell that made it
    xyz, counts = spread_out(root_id, points, drawn, inside, budget)
    head = np.uint32([len(root_id), len(xyz)]).tobytes()
    return head + counts.astype("<u2").tobytes() + scaled(xyz, middle, spread).astype("<i2").tobytes()
