import numpy as np

from flycore.board import Move, Position, apply_move, flip
from flycore.moves import legal_moves


def candidates(positions: list[Position]) -> tuple[list[list[Move]], list[Position]]:
    # the board each legal move leaves behind, seen from the mover's side
    options = [legal_moves(pos) for pos in positions]
    return options, [flip(apply_move(pos, m)) for pos, moves in zip(positions, options) for m in moves]


class Player:
    def __init__(self, seed: int = 0, explore: float = 0.0):
        self.rng = np.random.default_rng(seed)
        self.explore = explore

    def pick(self, options: list[list[Move]], scores) -> list[int]:
        picks, start = [], 0
        for moves in options:
            s = np.asarray(scores[start : start + len(moves)])
            best = np.flatnonzero(s == s.max()) if self.rng.random() >= self.explore else np.arange(len(moves))
            picks.append(start + int(self.rng.choice(best)))
            start += len(moves)
        return picks

    def choose_many(self, positions: list[Position]) -> list[Move]:
        options, after = candidates(positions)
        flat = [m for moves in options for m in moves]
        return [flat[i] for i in self.pick(options, self.scores(after))]

    def scores(self, after: list[Position]) -> list[float]:
        raise NotImplementedError


class RandomPlayer(Player):
    def scores(self, after):
        return [0.0] * len(after)


def material(pos: Position) -> float:
    own = pos.own_men.bit_count() + 1.5 * pos.own_kings.bit_count()
    return own - pos.opp_men.bit_count() - 1.5 * pos.opp_kings.bit_count()


class GreedyPlayer(Player):
    def scores(self, after):
        return [material(pos) for pos in after]


class MinimaxPlayer(Player):
    def __init__(self, depth: int = 2, seed: int = 0):
        super().__init__(seed)
        self.depth = depth

    def scores(self, after):
        return [-negamax(flip(pos), self.depth - 1, -np.inf, np.inf) for pos in after]


def negamax(pos: Position, depth: int, alpha: float, beta: float) -> float:
    moves = legal_moves(pos)
    if not moves:
        return -1000.0
    if depth <= 0:
        return material(pos)
    best = -np.inf
    for m in moves:
        best = max(best, -negamax(apply_move(pos, m), depth - 1, -beta, -alpha))
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return best
