from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flycore.board import Position, apply_move, flip
from flycore.moves import legal_moves
from game.players import Player

WIN = 1000.0


@dataclass
class Node:
    board: Position  # just left by whoever moved, seen from their side
    left: int  # replies still to imagine
    extensions: int  # forced captures that may still be followed past the depth
    score: float = 0.0
    value: float | None = None
    kids: list[Node] = field(default_factory=list)


class Centered(Player):
    # The fly's raw score sits around a large baseline that varies little from
    # board to board. Lines of different length end on different sides' judgement,
    # so compare them only after putting the score on a common, centred scale.
    def __init__(self, judge: Player, typical: list[Position]):
        super().__init__()
        raw = np.asarray(judge.scores(typical), dtype=float)
        self.judge, self.mean, self.spread = judge, raw.mean(), raw.std() + 1e-9

    def scores(self, after):
        return list((np.asarray(self.judge.scores(after), dtype=float) - self.mean) / self.spread)


class Imagination(Player):
    # The fly thinks ahead by itself. For each of its moves it imagines the
    # opponent's most plausible replies, judging them from the opponent's side
    # with its own brain, then its own best answer, and so on. Code only lists
    # legal moves and applies them. A line isn't cut off while a capture is
    # pending, since a forced move isn't a decision.
    def __init__(self, judge: Player, depth: int = 3, breadth: int = 2, extensions: int = 4, seed: int = 0):
        super().__init__(seed)
        self.judge, self.depth, self.breadth, self.extensions = judge, depth, breadth, extensions

    def scores(self, after: list[Position]) -> list[float]:
        roots = [Node(board, self.depth - 1, self.extensions) for board in after]
        parents, level = [], roots
        while level:
            for node, score in zip(level, self.judge.scores([n.board for n in level])):
                node.score = score
            for parent in parents:
                parent.kids = sorted(parent.kids, key=lambda n: n.score, reverse=True)[: self.breadth]
            kept = [kid for parent in parents for kid in parent.kids] if parents else roots
            parents = [node for node in kept if grow(node)]
            level = [kid for parent in parents for kid in parent.kids]
        return [value(root) for root in roots]


def grow(node: Node) -> bool:
    to_move = flip(node.board)
    moves = legal_moves(to_move)
    if not moves:
        node.value = WIN
        return False
    if node.left > 0:
        left, extensions = node.left - 1, node.extensions
    elif moves[0].is_capture and node.extensions > 0:
        left, extensions = 0, node.extensions - 1
    else:
        node.value = node.score
        return False
    node.kids = [Node(flip(apply_move(to_move, m)), left, extensions) for m in moves]
    return True


def value(node: Node) -> float:
    if node.value is None:
        node.value = -max(value(kid) for kid in node.kids)
    return node.value
