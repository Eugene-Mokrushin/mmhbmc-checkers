from flycore.board import Position
from flycore.squares import FULL, bits

N_LINES = 160
EMPTY = 128


def square_lines(square: int) -> list[int]:
    return [4 * square + kind for kind in range(4)] + [EMPTY + square]


def encode(pos: Position) -> list[int]:
    # one line per square: 4 * square + kind for a piece (own man, own king,
    # opponent man, opponent king), 128 + square when it's empty
    pieces = [4 * s + kind for kind, mask in enumerate(pos) for s in bits(mask)]
    return sorted(pieces + [EMPTY + s for s in bits(FULL & ~pos.occupied)])
