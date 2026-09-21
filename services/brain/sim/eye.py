import numpy as np
import scipy.sparse as sp
import torch

from flycore.board import Position
from flycore.squares import bits, square_to_rc

KICK = 250  # one input spike fires the photoreceptor
MAX_RATE = 150.0
# brightness of each square: light squares, dark empty squares, then own and opponent pieces
LIGHT, EMPTY, SHADES = 0.5, 0.3, (0.8, 1.0, 0.05, 0.15)
# the same board split the way a fly's eye is. R7 and R8 are the two colour channels, one
# for each side's pieces. R1-6 either keep the board as it was, which only adds colour to
# what the fly already had, or drop to bare occupancy, which makes colour carry whose
# piece it is on its own.
COLOURS = ((1.0, 1.0, 0.05, 0.05), (0.05, 0.05, 1.0, 1.0))
BARE = (0.9, 1.0, 0.85, 0.95)


def image(board: Position, channel: int = -1, split: bool = False) -> np.ndarray:
    shades = SHADES if channel < 0 else BARE if channel == 0 and split else SHADES if channel == 0 else COLOURS[channel - 1]
    out = np.full((8, 8), LIGHT)
    out[np.add.outer(np.arange(8), np.arange(8)) % 2 == 1] = EMPTY
    for shade, pieces in zip(shades, board):
        for s in bits(pieces):
            out[square_to_rc(s)] = shade
    return out


def sample(where: np.ndarray, sides: np.ndarray, shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    # the board fills the frontal field: each eye sees its half plus half a square of overlap
    row = np.clip(where[:, 0] * 8, 0, 7.999).astype(int)
    reach = where[:, 1] * 4.5 - 0.5
    col = np.clip(np.where(sides == "right", 4 + reach, 4 - reach) + shift, 0, 7.999).astype(int)
    return row, col


def rates(boards, where, sides, kinds=None, shifts=(0.0,), split=False) -> np.ndarray:
    # (sweeps, boards, photoreceptors). With one shift the board stands still; with several
    # it drifts across the eye during the window, which is what a motion detector needs.
    where = np.nan_to_num(where, nan=0.5)
    out = []
    for shift in shifts:
        row, col = sample(where, sides, shift)
        if kinds is None:
            out.append(np.stack([image(b)[row, col] for b in boards]))
            continue
        seen = np.empty((len(boards), len(row)))
        for c in range(3):
            at = kinds == c
            if at.any():
                seen[:, at] = np.stack([image(b, c, split)[row[at], col[at]] for b in boards])
        out.append(seen)
    return MAX_RATE * np.stack(out)


class Rhythm:
    # steady firing at each line's own rate and offset, worked out a step at a time so the
    # input for thousands of photoreceptors never has to be stored. The rate carries a
    # leading sweep, so a board that drifts changes rate partway through the window.
    def __init__(self, rate: np.ndarray, steps: int, dt: float, phase: np.ndarray):
        rate = torch.as_tensor(np.asarray(rate), dtype=torch.float32)
        self.rate = rate if rate.ndim == 3 else rate[None]
        self.phase = torch.as_tensor(phase, dtype=torch.float32)
        self.dt, self.steps = dt, steps
        self.shape = (steps, *self.rate.shape[1:])

    def to(self, device) -> "Rhythm":
        self.rate, self.phase = self.rate.to(device), self.phase.to(device)
        return self

    def columns(self) -> "Rhythm":
        # the same trains with lines down the rows and boards across, which is the shape
        # the serving path multiplies by
        other = Rhythm(self.rate.transpose(1, 2).contiguous(), self.steps, self.dt, self.phase.reshape(-1, 1))
        other.shape = (self.steps, self.shape[2], self.shape[1])
        return other

    def __getitem__(self, t: int) -> torch.Tensor:
        rate = self.rate[min(t * len(self.rate) // self.steps, len(self.rate) - 1)]
        return torch.floor(((t + 1) * self.dt + self.phase) * rate) > torch.floor((t * self.dt + self.phase) * rate)


def projection(receptors: np.ndarray, n: int) -> sp.csr_array:
    lines = np.arange(len(receptors))
    return sp.csr_array((np.full(len(receptors), KICK), (lines, receptors)), shape=(len(receptors), n))
