import numpy as np


def clustered(points: np.ndarray, faces: np.ndarray, cells: int) -> tuple[np.ndarray, np.ndarray]:
    # points that land in the same small box become one; it is an outline, not a specimen
    low, high = points.min(axis=0), points.max(axis=0)
    step = (high - low).max() / cells
    at = np.round((points - low) / step).astype(np.int64)
    key = (at[:, 0] * 100003 + at[:, 1]) * 100003 + at[:, 2]
    order = np.argsort(key, kind="stable")
    first = np.ones(len(key), dtype=bool)
    first[1:] = key[order][1:] != key[order][:-1]
    where = np.zeros(len(key), dtype=np.int64)
    where[order] = np.cumsum(first) - 1
    kept = np.zeros((int(first.sum()), 3))
    np.add.at(kept, where, points)
    kept /= np.bincount(where, minlength=len(kept))[:, None]
    made = where[faces]
    flat = (made[:, 0] == made[:, 1]) | (made[:, 1] == made[:, 2]) | (made[:, 0] == made[:, 2])
    return kept, made[~flat]
