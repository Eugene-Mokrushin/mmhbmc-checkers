import numpy as np
import torch

from flycore.board import Position
from train.teacher import teacher_file

MOVES = 8
TEMPERATURE = 1.0  # teacher values are in pieces; a piece apart is a strong preference
SCALE = 3.0  # a value of 3 pieces counts as nearly won


class Lessons:
    # the teacher's positions, each with the boards its legal moves leave behind and the
    # teacher's value of each board for the side that moved
    def __init__(self, depth: int, holdout: float = 0.02, seed: int = 0):
        with np.load(teacher_file(depth)) as z:
            self.boards, groups, self.values = z["boards"], z["groups"], z["values"]
        order = np.argsort(groups, kind="stable")
        self.boards, groups, self.values = self.boards[order], groups[order], self.values[order]
        self.starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
        self.ends = np.r_[self.starts[1:], len(groups)]
        held = np.random.default_rng(seed).random(len(self.starts)) < holdout
        self.train, self.test = np.flatnonzero(~held), np.flatnonzero(held)

    def batch(self, which: np.ndarray, rng: np.random.Generator) -> tuple[list[Position], np.ndarray, np.ndarray]:
        # up to MOVES boards per position, always including the teacher's favourite
        boards, groups, values = [], [], []
        for k, g in enumerate(which):
            idx = np.arange(self.starts[g], self.ends[g])
            if len(idx) > MOVES:
                best = idx[np.argmax(self.values[idx])]
                idx = np.r_[best, rng.choice(idx[idx != best], MOVES - 1, replace=False)]
            boards += [Position(*map(int, b)) for b in self.boards[idx]]
            groups += [k] * len(idx)
            values += self.values[idx].tolist()
        return boards, np.array(groups), np.array(values, dtype=np.float32)


def log_softmax(x: torch.Tensor, g: torch.Tensor, n: int) -> torch.Tensor:
    # log-softmax within each position's group of boards
    top = torch.full((n,), -torch.inf, device=x.device).scatter_reduce(0, g, x.detach(), "amax")
    total = torch.zeros(n, device=x.device).index_add(0, g, torch.exp(x - top[g]))
    return x - top[g] - torch.log(total[g])


def loss(scores: torch.Tensor, groups: np.ndarray, values: np.ndarray) -> tuple[torch.Tensor, dict]:
    # rank the moves of each position as the teacher does, and put a value on each board
    # that means the same across positions, so the fly can also judge lines it imagines;
    # a score of 1 stands for SCALE pieces in both
    g = torch.as_tensor(groups, device=scores.device)
    v = torch.as_tensor(values, device=scores.device)
    n = int(g.max()) + 1
    target = torch.exp(log_softmax(v / TEMPERATURE, g, n))
    rank = -(target * log_softmax(scores * SCALE / TEMPERATURE, g, n)).sum() / n
    value = torch.mean((torch.tanh(scores) - torch.tanh(v / SCALE)) ** 2)
    best = lambda x: x == torch.full((n,), -torch.inf, device=x.device).scatter_reduce(0, g, x, "amax")[g]
    agree = torch.zeros(n, device=scores.device).index_add(0, g, (best(scores.detach()) & best(v)).float()) > 0
    return rank + value, {"rank": rank.item(), "value": value.item(), "agree": agree.float().mean().item()}
