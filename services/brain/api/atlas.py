import numpy as np
import pandas as pd

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


class Envelope:
    # The brain is symmetrical; FAFB is not — more of the left eye was preserved than the
    # right. Each column across the head is cut back to whichever side reaches less far,
    # so the picture is the brain rather than what happened to survive the knife.
    def __init__(self, xyz: np.ndarray, cells: int = 48, bins: int = 96, fringe: float = 0.02):
        self.cells, self.middle = cells, float(np.median(xyz[:, 0]))
        self.low = xyz[:, 1:].min(axis=0)
        self.step = (xyz[:, 1:].max(axis=0) - self.low) / cells
        off = np.abs(xyz[:, 0] - self.middle)
        self.widest = float(off.max())
        band = np.clip((off / self.widest * bins).astype(np.int64), 0, bins - 1)
        at = self.cell(xyz)
        column, side = at[:, 0] * cells + at[:, 1], xyz[:, 0] < self.middle
        reach = np.full(cells * cells, np.inf)
        for half in (side, ~side):
            counts = np.zeros((cells * cells, bins))
            np.add.at(counts, (column[half], band[half]), 1)
            outward = np.cumsum(counts[:, ::-1], axis=1)[:, ::-1]  # points at least this far out
            enough = np.maximum(1.0, fringe * outward[:, :1])  # a stray point is not a side
            far = (outward >= enough).sum(axis=1)  # the outermost band that still holds cells
            reach = np.minimum(reach, far * self.widest / bins)
        self.reach = smoothed(reach.reshape(cells, cells))

    def cell(self, xyz: np.ndarray) -> np.ndarray:
        return np.clip(((xyz[:, 1:] - self.low) / self.step).astype(np.int64), 0, self.cells - 1)

    def holds(self, xyz: np.ndarray) -> np.ndarray:
        # read the edge between columns rather than at them, or the brain gets a staircase
        where = np.clip((xyz[:, 1:] - self.low) / self.step - 0.5, 0, self.cells - 1.001)
        corner = where.astype(np.int64)
        part = where - corner
        a, b = corner[:, 0], corner[:, 1]
        near = self.reach[a, b] * (1 - part[:, 1]) + self.reach[a, b + 1] * part[:, 1]
        far = self.reach[a + 1, b] * (1 - part[:, 1]) + self.reach[a + 1, b + 1] * part[:, 1]
        return np.abs(xyz[:, 0] - self.middle) <= near * (1 - part[:, 0]) + far * part[:, 0]


def smoothed(grid: np.ndarray) -> np.ndarray:
    # round the column edges outwards, so the cut is a curve and not a staircase, and so
    # no cell is dropped only because its neighbour's column ended sooner
    padded = np.pad(grid, 1, mode="edge")
    return np.maximum.reduce([grid, padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]])


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

    def of(self, root: int) -> np.ndarray:
        found = np.searchsorted(self.sorted, root)
        if found >= len(self.sorted) or self.sorted[found] != root:
            return self.xyz[:0]
        at = self.order[found]
        return self.xyz[self.starts[at] : self.starts[at] + self.counts[at]]


def spread_out(root_id: np.ndarray, points: pd.DataFrame, drawn: "Drawings | None", inside: "Envelope | None" = None, budget: int = BUDGET):
    # every neuron gets the same share of the picture; one with no skeleton keeps its marker
    share = max(1, budget // max(1, len(root_id)))
    marks = np.nan_to_num(points.reindex(root_id).to_numpy(dtype=np.float64))
    counts = np.zeros(len(root_id), dtype=np.uint16)
    out = []
    for i, root in enumerate(root_id):
        line = drawn.of(int(root)) if drawn is not None else marks[:0]
        line = line if len(line) else marks[i : i + 1]
        if inside is not None:
            line = line[inside.holds(line)]
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
