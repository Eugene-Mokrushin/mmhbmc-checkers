import numpy as np


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
            counts = np.bincount(column[half] * bins + band[half], minlength=cells * cells * bins).reshape(cells * cells, bins)
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
