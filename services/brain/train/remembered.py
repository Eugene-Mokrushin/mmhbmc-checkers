import argparse

from game.arena import play, tally
from game.remember import Remembering
from train.evaluate import baselines
from train.freeze import load_fly
import progress


def main() -> None:
    # Version 4 against version 3: the same brain and the same weights, the only
    # difference being whether it is put back to rest between moves.
    parser = argparse.ArgumentParser()
    parser.add_argument("--fly", default="whole-gradient")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--against", nargs="+", default=["random", "greedy", "minimax2"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--keep", type=float, nargs="+", default=[1.0], help="how much of the brain survives between moves")
    args = parser.parse_args()

    progress.use(f"remembered-{args.fly}")
    fly = load_fly(args.fly, device=args.device)
    minds = {"forgetting": fly} | {f"keeping {k:g}": Remembering(fly, keep=k) for k in args.keep}

    rows = {}
    for name, mind in minds.items():
        for other in args.against:
            against = baselines(seed=11)[other]
            result = tally(play(mind, against, args.games, f"{name} vs {other}"), mind)
            rows[(name, other)] = result
            if hasattr(mind, "forget"):
                mind.forget()
            print(f"{name:12s} vs {other:9s}  won {result['win']:.0%}  drew {result['draw']:.0%}  lost {result['loss']:.0%}", flush=True)

    print()
    print(f"{'':12s}" + "".join(f"{other:>18s}" for other in args.against))
    for name in minds:
        line = f"{name:12s}"
        for other in args.against:
            r = rows[(name, other)]
            line += f"{r['win']:>8.0%} W {r['draw']:.0%} D"
        print(line)


if __name__ == "__main__":
    main()
