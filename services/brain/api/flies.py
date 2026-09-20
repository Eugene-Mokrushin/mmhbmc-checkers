import itertools
from dataclasses import dataclass

import numpy as np
import torch

from connectome.load import load
from flycore.board import Move, Position
from game.imagine import Centered, Imagination
from game.players import Player, candidates
from train.freeze import FLIES_DIR, load_fly
from train.skills import exam

BREADTH = 2
TYPICAL = 400


@dataclass
class Stabled:
    # one fly, ready to play, with its score already put on a common scale so it can
    # also judge the lines it imagines
    name: str
    player: Player
    centred: Centered
    device: str
    root_id: np.ndarray  # the neurons its simulator holds, in its own order

    def thinking(self, depth: int) -> Player:
        return self.player if depth <= 1 else Imagination(self.centred, depth=depth, breadth=BREADTH, seed=depth)


def devices() -> list[str]:
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]


def kept() -> list[str]:
    return sorted(path.stem for path in FLIES_DIR.glob("*.json"))


class Stable:
    # the flies the website can play against, spread over the graphics cards
    def __init__(self, names: list[str] | None = None, cards: list[str] | None = None):
        c, cards = load(), cards or devices()
        self.c, self.flies = c, {}
        for name, device in zip(names or kept(), itertools.cycle(cards)):
            player = load_fly(name, c, device)
            mb = getattr(player, "mb", None)
            root_id = mb.graph.root_id if mb is not None and player.sim.n == mb.graph.n else c.root_id
            self.flies[name] = Stabled(name, player, Centered(player, exam(TYPICAL, seed=3)[0]), device, root_id)

    def choose(self, name: str, depth: int, position: Position) -> tuple[Move | None, float, Position | None]:
        fly = self.flies[name]
        options, after = candidates([position])
        if not after:
            return None, -1.0, None
        scores = fly.thinking(depth).scores(after)
        pick = int(max(range(len(after)), key=lambda i: scores[i]))
        return options[0][pick], float(scores[pick]), after[pick]

    def state(self) -> dict:
        cards = [{"name": torch.cuda.get_device_name(i), "memory_used_mb": round(torch.cuda.memory_allocated(i) / 2**20)} for i in range(torch.cuda.device_count())]
        return {"gpus": cards, "flies": [{"name": f.name, "device": f.device} for f in self.flies.values()]}
