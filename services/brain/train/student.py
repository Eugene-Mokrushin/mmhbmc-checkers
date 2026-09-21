from pathlib import Path

import numpy as np
import pandas as pd
import torch

from connectome.graph import Connectome
from game.gradfly import DT, Eyes, GradFly, descending
from sim.grad import GradLIF
from sim.params import LIF
from train.lessons import SCALE, Lessons, loss


class Student:
    # the whole brain being trained, with the readout that turns descending-neuron
    # spikes into a score
    def __init__(self, c: Connectome, points: pd.DataFrame, device: str, readout: np.ndarray | None = None):
        self.c, self.points, self.device = c, points, device
        self.eyes = Eyes(c, points)
        where = descending(c) if readout is None else readout
        self.brain = GradLIF(c.weights(), self.eyes.projection, where, LIF(dt=DT, input_gain=1.0), device)
        self.head = torch.nn.Linear(len(self.brain.readout), 1).to(device)
        self.mean = torch.zeros(len(self.brain.readout), device=device)
        self.spread = torch.ones(len(self.brain.readout), device=device)
        self.carry = None  # an rng here means it never judges a board from a brain at rest

    def parameters(self) -> list[torch.Tensor]:
        return [self.brain.gain, *self.head.parameters()]

    def scores(self, boards, state=None) -> torch.Tensor:
        if state is None and self.carry is not None:
            state = self.busy(boards)
        counts = self.brain.run(self.eyes.inputs(boards), state)[0]
        return self.head((counts - self.mean) / self.spread).squeeze(1)

    @torch.no_grad()
    def busy(self, boards):
        # the state another board leaves the brain in, so it judges with a head still full
        other = [boards[i] for i in self.carry.permutation(len(boards))]
        return self.brain.run(self.eyes.inputs(other))[1]

    @torch.no_grad()
    def calibrate(self, lessons: Lessons, rng: np.random.Generator, positions: int = 1000) -> float:
        # scale each descending neuron's spike count, and start the readout as the best
        # linear fit of the untrained brain's spikes to the teacher's values
        boards, _, values = lessons.batch(rng.choice(lessons.train, positions, replace=False), rng)
        counts = torch.cat([self.brain(self.eyes.inputs(boards[i : i + 256])) for i in range(0, len(boards), 256)])
        self.mean, self.spread = counts.mean(0), counts.std(0) + 1.0
        x = (counts - self.mean) / self.spread
        y = torch.tanh(torch.as_tensor(values, device=x.device) / SCALE)
        w = torch.linalg.solve(x.T @ x + 0.1 * len(x) * torch.eye(x.shape[1], device=x.device), x.T @ (y - y.mean()))
        self.head.weight.copy_(w[None])
        self.head.bias.fill_(y.mean().item())
        return float(torch.corrcoef(torch.stack([x @ w, y]))[0, 1])

    def fly(self, seed: int = 0) -> GradFly:
        w = self.head.weight.detach().ravel() / self.spread
        head = np.r_[w.cpu().numpy(), self.head.bias.item() - (self.mean * w).sum().item()]
        return GradFly(self.c, self.points, self.brain.weights(), head, seed, self.device)

    @torch.no_grad()
    def exam(self, lessons: Lessons, n: int, rng: np.random.Generator) -> dict[str, float]:
        results, sample = [], rng.choice(lessons.test, min(n, len(lessons.test)), replace=False)
        for which in np.array_split(sample, -(-len(sample) // 8)):
            boards, groups, values = lessons.batch(which, rng)
            total, parts = loss(self.scores(boards), groups, values)
            results.append({"loss": total.item(), **parts})
        return {f"test_{k}": float(np.mean([r[k] for r in results])) for k in results[0]}

    def save(self, path: Path, **extra) -> None:
        state = {"gain": self.brain.gain.detach().cpu(), "head": self.head.state_dict(), "mean": self.mean.cpu(), "spread": self.spread.cpu()}
        torch.save({**state, **extra}, path)

    def load(self, path: Path) -> dict:
        state = torch.load(path, map_location=self.device)
        with torch.no_grad():
            self.brain.gain.copy_(state.pop("gain"))
        self.head.load_state_dict(state.pop("head"))
        self.mean, self.spread = state.pop("mean").to(self.device), state.pop("spread").to(self.device)
        return state
