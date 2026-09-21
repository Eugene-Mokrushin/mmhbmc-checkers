import argparse

import numpy as np

from flycore.board import INITIAL, apply_move
from flycore.moves import legal_moves
from game.players import RandomPlayer
from game.remember import Remembering
from train.freeze import load_fly


def main() -> None:
    # A brain that is never put back to rest could quietly wind down over a long game.
    # This plays one out and watches how much it fires, move by move.
    parser = argparse.ArgumentParser()
    parser.add_argument("--fly", default="whole-gradient")
    parser.add_argument("--moves", type=int, default=40)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    fly = load_fly(args.fly, device=args.device)
    mind, other = Remembering(fly), RandomPlayer(5)
    pos, rows = INITIAL, []
    for turn in range(args.moves):
        if not legal_moves(pos):
            break
        move = mind.choose_many([pos], [1])[0]
        voltage, current, _ = mind.kept[1]
        rows.append((turn, float(voltage.mean()) * 1000, float(current.mean()), float((voltage > -0.05).float().mean())))
        pos = apply_move(pos, move)
        if not legal_moves(pos):
            break
        pos = apply_move(pos, other.choose_many([pos])[0])

    print(f"{'move':>5} {'mean mV':>9} {'mean input':>11} {'near firing':>12}")
    for turn, mv, current, near in rows:
        print(f"{turn:5d} {mv:9.2f} {current:11.4f} {near:11.2%}")
    early, late = rows[: len(rows) // 3], rows[-len(rows) // 3 :]
    print()
    print(f"first third {np.mean([r[1] for r in early]):.2f} mV, last third {np.mean([r[1] for r in late]):.2f} mV")


if __name__ == "__main__":
    main()
