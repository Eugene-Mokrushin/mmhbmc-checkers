from flycore.board import Position
from flycore.squares import bits

N_LINES = 128


def encode(pos: Position) -> list[int]:
    # line 4 * square + kind, kind: own man, own king, opponent man, opponent king
    return sorted(4 * s + kind for kind, pieces in enumerate(pos) for s in bits(pieces))
