from dataclasses import dataclass

import numpy as np
import torch

from game.decode import reward_share
from game.fly import Fly


@dataclass(frozen=True)
class Rule:
    rate: float = 1.0
    decay: float = 0.6
    recovery: float = 0.003
    ceiling: float = 2.0


class Plasticity:
    # Only KC->MBON synapses change, and only those from recently active KCs.
    # Dopamine carries surprise: the outcome minus what the fly's own score
    # predicted. Better than expected weakens those synapses in reward (PAM)
    # compartments, which hold the avoid MBONs, and strengthens them in punishment
    # (PPL1) compartments; worse than expected does the opposite (Hige et al. 2015,
    # Handler et al. 2019). Each MBON's share comes from its real DAN contacts.
    # Weights stay within [0, ceiling x start] and slowly drift back to the start.
    def __init__(self, fly: Fly, rule: Rule = Rule(), share: np.ndarray | None = None):
        self.fly, self.rule = fly, rule
        self.kc = torch.from_numpy(fly.mb.members("KC"))[:, None]
        self.mbon = torch.from_numpy(fly.mb.members("MBON"))
        self.start = self.weights.clone()
        self.reward_share = torch.from_numpy(reward_share(fly.mb) if share is None else share).float()
        self.mean, self.spread = None, None

    @property
    def weights(self) -> torch.Tensor:
        return self.fly.sim.w[self.kc, self.mbon]

    def dopamine(self, surprise: np.ndarray) -> torch.Tensor:
        # per MBON: +surprise in a pure reward compartment, -surprise in a pure punishment one
        s = torch.as_tensor(surprise, dtype=torch.float32)[:, None]
        return s * (2 * self.reward_share - 1)

    def expect(self, scores: np.ndarray) -> np.ndarray:
        # the fly's score as an expected outcome between -1 and 1
        scores = np.asarray(scores, dtype=np.float64)
        if self.mean is None:
            self.mean, self.spread = scores.mean(), scores.std()
        self.mean = 0.99 * self.mean + 0.01 * scores.mean()
        self.spread = 0.99 * self.spread + 0.01 * scores.std()
        return np.tanh((scores - self.mean) / (self.spread + 1e-9))

    def update(self, eligibility: np.ndarray, outcome: np.ndarray, scores: np.ndarray) -> None:
        surprise = np.clip(outcome, -1, 1) - self.expect(scores)
        e = torch.as_tensor(eligibility, dtype=torch.float32)
        drive = e.T @ self.dopamine(surprise) / len(e)
        w = (self.weights - self.rule.rate * drive * self.start).clamp(min=0)
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
