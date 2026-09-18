import numpy as np

from connectome.controls import rewire, scramble_compartments
from connectome.load import load
from game.decode import reward_share
from game.fly import Fly
from train.plasticity import Plasticity, Rule


def setup(control: str, seed: int, rule: Rule, explore: float, c=None) -> tuple[Fly, Plasticity]:
    fly = Fly(load() if c is None else c, seed=seed)
    rng = np.random.default_rng([seed, 5])
    share = reward_share(fly.mb)
    if control in ("degree", "random"):
        fly = fly.with_weights(rewire(fly.mb, control, rng))
    elif control == "compartments":
        share = scramble_compartments(share, rng)
    fly.explore = explore
    return fly, Plasticity(fly, rule, share)
