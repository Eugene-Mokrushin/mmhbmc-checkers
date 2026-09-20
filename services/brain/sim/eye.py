import numpy as np
import scipy.sparse as sp
import torch

from flycore.board import Position
from flycore.squares import bits, square_to_rc

KICK = 250  # one input spike fires the photoreceptor
MAX_RATE = 150.0
# brightness of each square: light squares, dark empty squares, then own and opponent pieces
LIGHT, EMPTY, SHADES = 0.5, 0.3, (0.8, 1.0, 0.05, 0.15)


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


class Rhythm:
    # steady firing at each line's own rate and offset, worked out a step at a time
    # so the input for thousands of photoreceptors never has to be stored
    def __init__(self, rate: np.ndarray, steps: int, dt: float, phase: np.ndarray):
        self.rate = torch.as_tensor(rate, dtype=torch.float32)
        self.phase = torch.as_tensor(phase, dtype=torch.float32)
        self.dt, self.shape = dt, (steps, *rate.shape)

    def to(self, device) -> "Rhythm":
        self.rate, self.phase = self.rate.to(device), self.phase.to(device)
        return self

    def columns(self) -> "Rhythm":
        # the same trains with lines down the rows and boards across, which is the shape
        # the serving path multiplies by
        other = Rhythm(self.rate.T.contiguous(), self.shape[0], self.dt, self.phase.reshape(-1, 1))
        other.shape = (self.shape[0], self.shape[2], self.shape[1])
        return other

    def __getitem__(self, t: int) -> torch.Tensor:
        return torch.floor(((t + 1) * self.dt + self.phase) * self.rate) > torch.floor((t * self.dt + self.phase) * self.rate)


def projection(receptors: np.ndarray, n: int) -> sp.csr_array:
    lines = np.arange(len(receptors))
    return sp.csr_array((np.full(len(receptors), KICK), (lines, receptors)), shape=(len(receptors), n))
