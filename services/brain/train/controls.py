import numpy as np

from connectome.controls import rewire, scramble_compartments
from connectome.coordinates import one_per_neuron, read_markers
from connectome.load import load
from game.decode import reward_share
from game.fly import Fly
from game.wholefly import WholeFly
from sim.kernels import best_device
from train.plasticity import Plasticity, Rule

BRAINS = ("mb", "whole")


def build(brain: str, seed: int, c=None, device: str | None = None) -> Fly:
    full = load() if c is None else c
    if brain == "mb":
        return Fly(full, seed=seed)
    return WholeFly(load(min_syn=5), full, one_per_neuron(read_markers()), seed, device or best_device())


def setup(control: str, seed: int, rule: Rule, explore: float, c=None, brain: str = "mb", device: str | None = None) -> tuple[Fly, Plasticity]:
    if brain == "whole" and control in ("degree", "random"):
        raise ValueError("rewired controls exist only for the mushroom-body fly")
    fly = build(brain, seed, c, device)
    rng = np.random.default_rng([seed, 5])
    share = reward_share(fly.mb)
    if control in ("degree", "random"):
        fly = fly.with_weights(rewire(fly.mb, control, rng))
    elif control == "compartments":
        share = scramble_compartments(share, rng)
    fly.explore = explore
    return fly, Plasticity(fly, rule, share)
