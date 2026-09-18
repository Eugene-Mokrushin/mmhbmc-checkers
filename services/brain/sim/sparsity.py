import random

import numpy as np
import scipy.sparse as sp

from connectome.controls import rewire
from connectome.extract import MushroomBody
from connectome.load import load
from flycore.board import INITIAL, Position, apply_move
from flycore.moves import legal_moves
from game.fly import Fly
from progress import Progress

BINS = {"2-8": (2, 8), "9-16": (9, 16), "17-24": (17, 24)}
PER_BIN = 64


def jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a & b).sum(-1) / np.maximum((a | b).sum(-1), 1)


def without_apl(mb: MushroomBody) -> sp.csr_array:
    keep = (mb.population != "APL").astype(np.float32)
    return (sp.diags_array(keep) @ mb.graph.weights()).tocsr()


def game_positions(pieces: tuple[int, int], n: int, seed: int = 0) -> list[Position]:
    rnd, out = random.Random(seed), []
    while len(out) < n:
        pos = INITIAL
        for _ in range(rnd.randint(0, 80)):
            moves = legal_moves(pos)
            if not moves:
                break
            pos = apply_move(pos, rnd.choice(moves))
        if pieces[0] <= pos.occupied.bit_count() <= pieces[1]:
            out.append(pos)
    return out


def measure(fly: Fly, positions: list[Position]) -> dict:
    kc = fly.counts(positions)[:, fly.mb.members("KC")] > 0
    return {"active": kc.mean(), "overlap": jaccard(kc, np.roll(kc, 1, axis=0)).mean()}


def main() -> None:
    fly = Fly(load())
    rng = np.random.default_rng(0)
    variants = {
        "real": fly,
        "no APL": fly.with_weights(without_apl(fly.mb)),
        "degree": fly.with_weights(rewire(fly.mb, "degree", rng)),
        "random": fly.with_weights(rewire(fly.mb, "random", rng)),
    }
    rows = []
    with Progress(len(variants) * len(BINS), "sparsity gate") as bar:
        for name, f in variants.items():
            for pieces, bounds in BINS.items():
                rows.append((name, pieces, measure(f, game_positions(bounds, PER_BIN))))
                bar.update(active_kc=float(rows[-1][2]["active"]))
    print(f"{'':8}{'pieces':>7}{'KCs active':>12}{'overlap':>10}")
    for name, pieces, r in rows:
        print(f"{name:8}{pieces:>7}{r['active']:>12.1%}{r['overlap']:>10.2f}")


if __name__ == "__main__":
    main()
