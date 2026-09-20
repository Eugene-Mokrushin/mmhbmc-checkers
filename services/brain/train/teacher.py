import argparse
import functools
import multiprocessing as mp
import random

import numpy as np

from flycore.board import INITIAL, Position, apply_move
from flycore.moves import legal_moves
from game.players import MinimaxPlayer, candidates, negamax
from paths import DATA_DIR
import progress
from progress import Progress

TEACHER_DEPTH = 7
MAX_PLIES = 150


def teacher_file(depth: int = TEACHER_DEPTH):
    return DATA_DIR / f"teacher-d{depth}.npz"


def positions(seed: int, keep: float = 0.3) -> list[Position]:
    # positions from a game between two players of random strength and carelessness,
    # so the set covers what a fly meets against beginners and stronger players alike
    rnd = random.Random(seed)
    depths, careless = [rnd.randint(0, 4) for _ in range(2)], [rnd.choice([0.05, 0.2, 0.5]) for _ in range(2)]
    pos, out = INITIAL, []
    for ply in range(MAX_PLIES):
        moves = legal_moves(pos)
        if not moves:
            break
        if len(moves) > 1 and rnd.random() < keep:
            out.append(pos)
        side = ply % 2
        if depths[side] == 0 or rnd.random() < careless[side]:
            move = rnd.choice(moves)
        else:
            values = [-negamax(apply_move(pos, m), depths[side] - 1, -np.inf, np.inf) for m in moves]
            move = rnd.choice([m for m, v in zip(moves, values) if v == max(values)])
        pos = apply_move(pos, move)
    return out


def label(seed: int, depth: int = TEACHER_DEPTH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # every board a position's legal moves leave behind (seen by the mover) with the
    # teacher's value of it for the mover
    boards, groups, values, teacher = [], [], [], MinimaxPlayer(depth)
    for k, pos in enumerate(positions(seed)):
        _, after = candidates([pos])
        boards += after
        groups += [k] * len(after)
        values += teacher.scores(after)
    return np.array(boards, dtype=np.uint32).reshape(-1, 4), np.array(groups), np.array(values, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=6000)
    parser.add_argument("--depth", type=int, default=TEACHER_DEPTH)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    progress.use(f"teacher-d{args.depth}")
    parts, offset = [], 0
    with mp.Pool(args.workers) as pool, Progress(args.games, f"teacher d{args.depth}") as bar:
        for boards, groups, values in pool.imap_unordered(functools.partial(label, depth=args.depth), range(args.games), chunksize=4):
            parts.append((boards, groups + offset, values))
            offset += groups.max() + 1 if len(groups) else 0
            bar.update(1, positions=offset)
    boards, groups, values = (np.concatenate(x) for x in zip(*parts))
    np.savez_compressed(teacher_file(args.depth), boards=boards, groups=groups, values=values)
    print(f"{offset} positions, {len(boards)} boards -> {teacher_file(args.depth)}")


if __name__ == "__main__":
    main()
