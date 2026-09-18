import functools
import random

import numpy as np
from scipy.stats import rankdata

from flycore.board import INITIAL, Position, apply_move, flip
from flycore.moves import legal_moves
from game.fly import Fly
from game.players import material


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    ranks = rankdata(scores)
    ahead, behind = (labels > 0).sum(), (labels < 0).sum()
    return float((ranks[labels > 0].sum() - ahead * (ahead + 1) / 2) / (ahead * behind))


@functools.cache
def exam(n: int = 1500, seed: int = 7) -> tuple[list[Position], np.ndarray, np.ndarray]:
    # boards a move leaves behind in random games, labelled by whether the opponent
    # can now capture (safety) and by who is ahead in pieces (material)
    rnd, boards, safe = random.Random(seed), [], []
    while len(boards) < n:
        pos = INITIAL
        for _ in range(rnd.randint(2, 60)):
            moves = legal_moves(pos)
            if not moves:
                break
            pos = apply_move(pos, rnd.choice(moves))
        moves = legal_moves(pos)
        if not moves:
            continue
        after = apply_move(pos, rnd.choice(moves))
        boards.append(flip(after))
        safe.append(-1 if any(m.is_capture for m in legal_moves(after)) else 1)
    return boards, np.array(safe), np.sign([material(b) for b in boards])


def skills(fly: Fly) -> dict[str, float]:
    boards, safe, ahead = exam()
    scores = np.array(fly.scores(boards))
    return {"safety": auc(scores, safe), "material": auc(scores[ahead != 0], ahead[ahead != 0])}
