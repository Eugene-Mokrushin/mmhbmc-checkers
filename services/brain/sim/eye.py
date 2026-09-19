import numpy as np
import pandas as pd
import scipy.sparse as sp

from connectome.graph import Connectome
from flycore.board import Position
from flycore.squares import bits, square_to_rc

PHOTORECEPTORS = ("R1-6", "R7", "R8")
LAMINA = ("L1", "L2", "L3", "L4", "L5")
KICK = 250  # one input spike fires the photoreceptor
MAX_RATE = 150.0
# brightness of each square: light squares, dark empty squares, then own and opponent pieces
LIGHT, EMPTY, SHADES = 0.5, 0.3, (0.8, 1.0, 0.05, 0.15)


def rank(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values)) / max(len(values) - 1, 1)


def sheet(xyz: np.ndarray) -> np.ndarray:
    # a curved sheet of terminals flattened onto its two main directions, the first
    # oriented dorsal to ventral (+y) and the second anterior to posterior (+z)
    centred = xyz - xyz.mean(axis=0)
    axes = np.linalg.svd(centred, full_matrices=False)[2][:2]
    axes *= np.sign([axes[0, 1], axes[1, 2]])[:, None]
    flat = centred @ axes.T
    return np.stack([rank(flat[:, 0]), rank(flat[:, 1])], axis=1)


def fill(pos: np.ndarray, members: np.ndarray, xyz: np.ndarray, k: int = 5) -> None:
    # a photoreceptor with no mapped partner takes the mean position of its k nearest mapped neighbours
    known = ~np.isnan(pos[members, 0])
    for i in np.flatnonzero(~known):
        d = np.linalg.norm(xyz[known] - xyz[i], axis=1)
        pos[members[i]] = pos[members[known][np.argsort(d)[:k]]].mean(axis=0)


def inherit(counts: sp.csr_array, targets: np.ndarray, sources: np.ndarray, pos: np.ndarray, outgoing=False) -> np.ndarray:
    # each target takes the synapse-weighted mean position of its partners among the sources
    known = sources[~np.isnan(pos[sources, 0])]
    w = counts[targets][:, known] if outgoing else counts[known][:, targets].T
    total = np.asarray(w.sum(axis=1)).ravel()
    with np.errstate(invalid="ignore"):
        return (w @ pos[known]) / total[:, None]


def field(c: Connectome, points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # Where each photoreceptor looks, as (elevation, azimuth) in [0, 1] within its eye,
    # 0 being dorsal and frontal. R1-6 come from the lamina's geometry (the retina isn't
    # in the data); R7 and R8 inherit positions through L-cells and medulla neurons,
    # which also undoes the optic chiasm's flip.
    pos = np.full((c.n, 2), np.nan)
    r16 = np.flatnonzero(c.cell_type == "R1-6")
    for side in ("left", "right"):
        eye = r16[c.side[r16] == side]
        pos[eye] = sheet(points.loc[c.root_id[eye], ["x", "y", "z"]].to_numpy())
    lamina = np.flatnonzero(np.isin(c.cell_type, LAMINA))
    pos[lamina] = inherit(c.counts, lamina, r16, pos)
    medulla = np.flatnonzero(np.asarray(c.counts[lamina].sum(axis=0)).ravel() > 0)
    medulla = medulla[~np.isin(c.cell_type[medulla], PHOTORECEPTORS + LAMINA)]
    pos[medulla] = inherit(c.counts, medulla, lamina, pos)
    r78 = np.flatnonzero(np.isin(c.cell_type, ("R7", "R8")))
    pos[r78] = inherit(c.counts, r78, np.concatenate([lamina, medulla]), pos, outgoing=True)
    for kind in ("R7", "R8"):
        for side in ("left", "right"):
            members = np.flatnonzero((c.cell_type == kind) & (c.side == side))
            fill(pos, members, points.loc[c.root_id[members], ["x", "y", "z"]].to_numpy())
    receptors = np.flatnonzero(np.isin(c.cell_type, PHOTORECEPTORS) & np.isin(c.side, ("left", "right")))
    return receptors, pos[receptors]


def image(board: Position) -> np.ndarray:
    out = np.full((8, 8), LIGHT)
    out[np.add.outer(np.arange(8), np.arange(8)) % 2 == 1] = EMPTY
    for shade, pieces in zip(SHADES, board):
        for s in bits(pieces):
            out[square_to_rc(s)] = shade
    return out


def sample(where: np.ndarray, sides: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # the board fills the frontal field: each eye sees its half plus half a square of overlap
    row = np.clip(where[:, 0] * 8, 0, 7.999).astype(int)
    reach = where[:, 1] * 4.5 - 0.5
    col = np.clip(np.where(sides == "right", 4 + reach, 4 - reach), 0, 7.999).astype(int)
    return row, col


def rates(boards: list[Position], where: np.ndarray, sides: np.ndarray) -> np.ndarray:
    row, col = sample(np.nan_to_num(where, nan=0.5), sides)
    return MAX_RATE * np.stack([image(b)[row, col] for b in boards])


def rhythm(rate: np.ndarray, steps: int, dt: float, phase: np.ndarray) -> np.ndarray:
    # steady firing at each line's own rate and offset: [steps, boards, lines]
    count = np.floor((np.arange(steps + 1)[:, None, None] * dt + phase) * rate)
    return np.diff(count, axis=0) > 0
