import argparse

import numpy as np

from game.arena import play, tally
from game.players import MinimaxPlayer
from train.freeze import load_fly
import progress


def main() -> None:
    # The two-move engine breaks ties at random, so which games it plays depends on its
    # seed. This asks how much of a published win rate is the fly and how much is that.
    parser = argparse.ArgumentParser()
    parser.add_argument("--fly", default="whole-gradient")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 5, 11, 17, 23])
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    progress.use(f"seeds-{args.fly}")
    fly = load_fly(args.fly, device=args.device)
    rows = []
    for seed in args.seeds:
        r = tally(play(fly, MinimaxPlayer(args.depth, seed=seed), args.games, f"seed {seed}"), fly)
        rows.append((seed, r["win"], r["draw"], r["loss"]))
        print(f"seed {seed:3d}: won {r['win']:.0%} drew {r['draw']:.0%} lost {r['loss']:.0%}", flush=True)

    score = [w + d / 2 for _, w, d, _ in rows]
    print()
    print(f"score over {len(rows)} opponents: {np.mean(score):.2f} ± {np.std(score):.2f}  (worst {min(score):.2f}, best {max(score):.2f})")


if __name__ == "__main__":
    main()
