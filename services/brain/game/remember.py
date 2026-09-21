import numpy as np
import torch

from flycore.board import Move, Position
from game.players import Player, candidates

State = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def run(sim, inputs, state: State | None) -> tuple[torch.Tensor, State]:
    # the same run the simulator always does, but handing back the state it ends in
    total = torch.zeros((inputs.shape[1], sim.n), dtype=torch.int32, device=sim.device)
    flat = total.view(-1)
    spikes = sim.spikes(inputs, state)
    while True:
        try:
            _, (row, source) = next(spikes)
        except StopIteration as done:
            return total.cpu(), done.value
        flat.index_add_(0, row * sim.n + source, torch.ones_like(row, dtype=torch.int32))


class Remembering(Player):
    # Version 4. The brain is not put back to rest between moves: it carries on from
    # where the last move left it, so what it answers depends on the game so far and not
    # only on the board in front of it — which is how the animal is, and is not how
    # versions one to three were.
    def __init__(self, fly, seed: int = 0, keep: float = 1.0):
        # keep says how much of the brain survives the gap between moves: one carries it
        # whole, nought is the fly we had before, and in between the gap costs it something
        super().__init__(seed)
        self.fly, self.keep = fly, keep
        self.kept: dict[int, State] = {}

    def forget(self, key: int | None = None) -> None:
        self.kept.pop(key, None) if key is not None else self.kept.clear()

    def scores(self, after: list[Position]) -> list[float]:
        return self.fly.scores(after)

    def choose_many(self, positions: list[Position], keys: list[int] | None = None) -> list[Move]:
        keys = list(range(len(positions))) if keys is None else keys
        options, after = candidates(positions)
        if not after:
            return []
        inputs = self.fly.eyes.inputs(after).to(self.fly.sim.device)
        counts, ending = run(self.fly.sim, inputs, self._batch(options, keys))
        scores = self.fly.read(counts.numpy())
        flat = [m for moves in options for m in moves]
        picks = self.pick(options, scores)
        self._keep(options, keys, picks, ending)
        return [flat[i] for i in picks]

    def _batch(self, options, keys) -> State | None:
        # every board a game is weighing starts from that game's own state
        if not any(key in self.kept for key in keys):
            return None
        held = [self.kept.get(key) for key, moves in zip(keys, options) for _ in moves]
        rest = self.fly.sim.start(1)
        return tuple(torch.cat([(one or rest)[part] for one in held]) for part in range(3))

    def _keep(self, options, keys, picks, ending: State) -> None:
        # only the move actually played leaves its mark; the others were imagined
        rest = self.fly.sim.start(1)
        for key, pick in zip(keys, picks):
            held = tuple(part[pick : pick + 1].clone() for part in ending)
            self.kept[key] = held if self.keep >= 1 else self._faded(held, rest)

    def _faded(self, held: State, rest: State) -> State:
        # the brain settles back towards rest while it waits for you to move
        voltage = rest[0] + (held[0] - rest[0]) * self.keep
        return voltage, held[1] * self.keep, torch.where(torch.rand_like(held[1]) < self.keep, held[2], rest[2])
