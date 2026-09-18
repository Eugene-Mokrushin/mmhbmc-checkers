import argparse

import numpy as np

from connectome.load import load
from flycore.board import Position
from game.fly import Fly
from game.players import material
from progress import Progress
from sim.sparsity import game_positions
from train.plasticity import Plasticity, Rule
from train.skills import auc

BATCH = 64


def labelled(n: int, seed: int) -> tuple[list[Position], np.ndarray]:
    positions = [p for p in game_positions((2, 24), 3 * n, seed) if material(p) != 0][:n]
    return positions, np.sign([material(p) for p in positions])


def evaluate(fly: Fly, positions: list[Position], labels: np.ndarray) -> float:
    return auc(np.array(fly.scores(positions)), labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=2048)
    parser.add_argument("--test", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rate", type=float, default=Rule.rate)
    parser.add_argument("--recovery", type=float, default=Rule.recovery)
    args = parser.parse_args()

    fly = Fly(load())
    plastic = Plasticity(fly, Rule(rate=args.rate, decay=0.0, recovery=args.recovery))
    train, train_labels = labelled(args.train, seed=1)
    test, test_labels = labelled(args.test, seed=2)
    kc, mbon = fly.mb.members("KC"), fly.mb.members("MBON")

    print(f"before: AUC {evaluate(fly, test, test_labels):.3f} (0.5 = chance, 1.0 = perfect)")
    with Progress(args.epochs * len(train), "sanity: material") as bar:
        for epoch in range(args.epochs):
            for i in range(0, len(train), BATCH):
                counts = fly.counts(train[i : i + BATCH])
                plastic.update(counts[:, kc] > 0, train_labels[i : i + BATCH], counts[:, mbon] @ fly.valence)
                bar.update(len(counts), epoch=epoch + 1, depressed=plastic.depressed())
            print(f"epoch {epoch + 1}: AUC {evaluate(fly, test, test_labels):.3f}, depressed {plastic.depressed():.1%}")


if __name__ == "__main__":
    main()
