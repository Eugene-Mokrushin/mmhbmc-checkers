from dataclasses import dataclass

import numpy as np
import torch

from game.decode import reward_share
from game.fly import Fly


@dataclass(frozen=True)
class Rule:
    rate: float = 1.0
    decay: float = 0.6
    recovery: float = 0.0002
    ceiling: float = 2.0


class Plasticity:
    # Only KC->MBON synapses change. Dopamine depresses synapses from recently
    # active KCs (Hige et al. 2015) and strengthens those from silent ones (Cohn
    # et al. 2015). Reward reaches MBONs in PAM compartments, punishment those in
    # PPL1 compartments, each through its real DAN contacts.
    def __init__(self, fly: Fly, rule: Rule = Rule(), share: np.ndarray | None = None):
        self.fly, self.rule = fly, rule
        self.kc = torch.from_numpy(fly.mb.members("KC"))[:, None]
        self.mbon = torch.from_numpy(fly.mb.members("MBON"))
        self.start = self.weights.clone()
        self.reward_share = torch.from_numpy(reward_share(fly.mb) if share is None else share).float()

    @property
    def weights(self) -> torch.Tensor:
        return self.fly.sim.w[self.kc, self.mbon]

    def dopamine(self, reward: np.ndarray) -> torch.Tensor:
        r = torch.as_tensor(reward, dtype=torch.float32)[:, None]
        return r.clamp(min=0) * self.reward_share + (-r).clamp(min=0) * (1 - self.reward_share)

    def update(self, eligibility: np.ndarray, reward: np.ndarray) -> None:
        e = torch.as_tensor(eligibility, dtype=torch.float32)
        e = e - e.mean(dim=1, keepdim=True)
        drive = e.T @ self.dopamine(reward) / len(e)
        w = (self.weights * (1 - self.rule.rate * drive)).clamp(min=0)
        w = torch.minimum(w, self.rule.ceiling * self.start)
        w += self.rule.recovery * (self.start - w)
        self.fly.sim.w[self.kc, self.mbon] = w
        self.fly.sim.rewired()

    def depressed(self) -> float:
        existing = self.start > 0
        return float(((self.weights < 0.05 * self.start) & existing).sum() / existing.sum())

    def state(self) -> np.ndarray:
        return self.weights.numpy().copy()

    def load(self, weights: np.ndarray) -> None:
        self.fly.sim.w[self.kc, self.mbon] = torch.from_numpy(weights)
        self.fly.sim.rewired()
