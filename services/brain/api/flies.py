import itertools
from dataclasses import dataclass, field

import numpy as np
import torch

from connectome.load import load
from flycore.board import Move, Position
from game.imagine import Centered, Imagination
from game.players import Player, candidates
from sim.watch import Watcher
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
    watcher: Watcher | None = field(default=None)

    def thinking(self, depth: int) -> Player:
        return self.player if depth <= 1 else Imagination(self.centred, depth=depth, breadth=BREADTH, seed=depth)

    def watching(self, boards):
        # the simulation as it happens, a frame at a time
        if self.watcher is None:
            self.watcher = Watcher(self.player.sim)
        return self.watcher.stream(self.player.inputs(boards), getattr(self.player, "rest", None))

    def watch(self, boards) -> tuple[np.ndarray, torch.Tensor]:
        for kind, payload in self.watching(boards):
            if kind == "end":
                return payload[0].numpy(), payload[1]
        raise RuntimeError("the simulation ended without a result")


def spoken(move: Move) -> dict:
    return {"origin": move.origin, "destination": move.destination, "captured": move.captured, "path": list(move.path), "promotes": move.promotes}


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

    def choose(self, name: str, depth: int, position: Position) -> tuple[Move | None, float, "torch.Tensor | None"]:
        # every legal move is judged in one pass, which also records what fired for each
        fly = self.flies[name]
        options, after = candidates([position])
        if not after:
            return None, -1.0, None
        counts, frames = fly.watch(after)
        move, score, pick = self.pick(fly, options[0], after, counts, depth)
        return move, score, frames[pick]

    def pick(self, fly: Stabled, moves: list[Move], after: list[Position], counts, depth: int) -> tuple[Move, float, int]:
        scores = fly.player.read(counts if isinstance(counts, np.ndarray) else counts.numpy()) if depth <= 1 else np.asarray(fly.thinking(depth).scores(after))
        chosen = int(np.argmax(scores))
        return moves[chosen], float(scores[chosen]), chosen

    def state(self) -> dict:
        cards = [{"name": torch.cuda.get_device_name(i), "memory_used_mb": round(torch.cuda.memory_allocated(i) / 2**20)} for i in range(torch.cuda.device_count())]
        return {"gpus": cards, "flies": [{"name": f.name, "device": f.device} for f in self.flies.values()]}
