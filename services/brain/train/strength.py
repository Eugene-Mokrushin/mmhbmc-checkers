import argparse

import progress
from connectome.load import load
from game.arena import play, tally
from game.imagine import Centered, Imagination
from game.players import MinimaxPlayer
from paths import ARTIFACTS_DIR
from train.freeze import load_fly
from train.skills import exam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fly", default="plain")
    parser.add_argument("--depths", default="1,2,3", help="how many moves the fly imagines")
    parser.add_argument("--against", default="2,3,4", help="minimax depths to play")
    parser.add_argument("--breadth", type=int, default=2, help="opponent replies imagined per move")
    parser.add_argument("--games", type=int, default=100)
    args = parser.parse_args()

    name = f"strength-{args.fly}-d{args.depths.replace(',', '')}"
    progress.use(name)
    fly = Centered(load_fly(args.fly, load()), exam(600, seed=3)[0])
    depths = [int(d) for d in args.depths.split(",")]
    against = [int(d) for d in args.against.split(",")]
    out = ARTIFACTS_DIR / "runs" / f"{name}.txt"
    lines = [f"{args.fly} fly, {args.games} games per cell, imagining {args.breadth} replies per move", ""]
    lines.append(f"{'':12}" + "".join(f"{'minimax ' + str(m):>18}" for m in against))
    for depth in depths:
        player = Imagination(fly, depth=depth, breadth=args.breadth, seed=depth)
        cells = []
        for m in against:
            r = tally(play(player, MinimaxPlayer(m, seed=m), args.games, f"depth {depth} vs minimax {m}"), player)
            cells.append(f"{r['win']:.0%} W {r['draw']:.0%} D")
            out.write_text("\n".join(lines + [f"depth {depth:<6}" + "".join(f"{c:>18}" for c in cells)]) + "\n")
        lines.append(f"depth {depth:<6}" + "".join(f"{c:>18}" for c in cells))
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
