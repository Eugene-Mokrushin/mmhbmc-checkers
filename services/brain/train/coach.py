import argparse

import numpy as np

import progress
from progress import Progress
from train.controls import setup
from train.evaluate import Run, evaluate
from train.lessons import Lessons
from train.plasticity import Rule
from train.skills import skills

SPREAD = 1.0  # a move a piece better than the average one is clearly approved of


def choose(scores: np.ndarray, groups: np.ndarray, explore: float, rng) -> np.ndarray:
    # the board the fly goes for in each position, sometimes trying another one
    picks = []
    for k in range(groups.max() + 1):
        where = np.flatnonzero(groups == k)
        picks.append(rng.choice(where) if rng.random() < explore else where[np.argmax(scores[where])])
    return np.array(picks)


def approval(values: np.ndarray, groups: np.ndarray, picks: np.ndarray) -> np.ndarray:
    # how the teacher sees the move the fly went for, against the average move available
    # in that position: +1 clearly better, -1 clearly worse
    means = np.array([values[groups == k].mean() for k in range(groups.max() + 1)])
    return np.tanh((values[picks] - means) / SPREAD)


def main() -> None:
    # The fly learns the way a fly does, at its KC->MBON synapses, but the dopamine now
    # reports whether a search engine would have approved of the move it chose, instead
    # of what that move won or lost.
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="w-coach")
    parser.add_argument("--positions", type=int, default=60000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=Rule.decay)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--explore", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    progress.use(args.run)
    rng, lessons = np.random.default_rng(args.seed), Lessons(args.depth)
    fly, plastic = setup("real", args.seed, Rule(rate=args.rate, decay=args.decay), 0.0, brain="whole")
    run = Run(args.run, vars(args))

    def measure(seen: int, **extra) -> None:
        run.record({"games": seen, **evaluate(fly, args.eval_games, args.seed + 2, ["random", "greedy", "minimax2"]), **skills(fly), **extra})

    measure(0)
    with Progress(args.positions, "coached") as bar:
        for seen in range(args.batch, args.positions + 1, args.batch):
            boards, groups, values = lessons.batch(rng.choice(lessons.train, args.batch, replace=False), rng)
            counts = fly.counts(boards)
            scores = fly.judge(counts)
            picks = choose(scores, groups, args.explore, rng)
            liked = approval(values, groups, picks)
            plastic.update(counts[picks][:, fly.kc_cols] > 0, liked, scores[picks])
            bar.update(args.batch, approval=round(float(liked.mean()), 3))
            if seen % args.eval_every < args.batch:
                run.checkpoint(plastic, seen, args.seed)
                measure(seen, depressed=plastic.depressed())


if __name__ == "__main__":
    main()
