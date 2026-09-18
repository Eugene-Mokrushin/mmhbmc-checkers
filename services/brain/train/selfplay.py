import argparse

import numpy as np

import progress
from connectome.controls import CONTROLS
from game.arena import OPPONENTS
from progress import Progress
from train.block import block
from train.controls import setup
from train.evaluate import Run, checkpoint_file, evaluate
from train.plasticity import Rule
from train.skills import skills


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="shaped-random")
    parser.add_argument("--games", type=int, default=20000)
    parser.add_argument("--opponent", choices=OPPONENTS, default="random")
    parser.add_argument("--control", choices=CONTROLS, default="real")
    parser.add_argument("--reward", choices=("shaped", "terminal"), default="shaped")
    parser.add_argument("--init", help="start from a saved fly: RUN (latest checkpoint) or RUN@GAMES")
    parser.add_argument("--parallel", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=2048)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--eval-against", default="random,greedy,minimax2,untrained")
    parser.add_argument("--explore", type=float, default=0.05)
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--decay", type=float, default=Rule.decay)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    progress.use(args.run)
    fly, plastic = setup(args.control, args.seed, Rule(rate=args.rate, decay=args.decay), args.explore)
    if args.init:
        with np.load(checkpoint_file(args.init)) as z:
            plastic.load(z["weights"])
    opponent = OPPONENTS[args.opponent](seed=args.seed + 1)
    run = Run(args.run, vars(args))
    against = args.eval_against.split(",")

    def measure(games: int, **extra) -> None:
        run.record({"games": games, **evaluate(fly, args.eval_games, args.seed + 2, against), **skills(fly), **extra})

    measure(0)
    done, next_eval = 0, args.eval_every
    with Progress(args.games, "train") as bar:
        while done < args.games:
            n = min(args.parallel, args.games - done)
            stats = block(fly, plastic, opponent, n, args.reward == "shaped")
            done += n
            bar.update(n, **stats)
            if done >= next_eval or done == args.games:
                next_eval += args.eval_every
                run.checkpoint(plastic, done, args.seed)
                measure(done, **stats, depressed=plastic.depressed())


if __name__ == "__main__":
    main()
