import numpy as np

from flycore.board import Move, Position, apply_move, flip
from flycore.moves import legal_moves


class Player:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def choose_many(self, positions: list[Position]) -> list[Move]:
        options = [legal_moves(pos) for pos in positions]
        results = self.scores([flip(apply_move(pos, m)) for pos, moves in zip(positions, options) for m in moves])
        picks, i = [], 0
        for moves in options:
            s = np.asarray(results[i : i + len(moves)])
            i += len(moves)
            picks.append(moves[self.rng.choice(np.flatnonzero(s == s.max()))])
        return picks

    def scores(self, after: list[Position]) -> list[float]:
        # each position is the board a move leaves behind, seen from the mover's side
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
