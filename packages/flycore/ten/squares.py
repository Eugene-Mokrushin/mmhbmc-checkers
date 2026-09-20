from __future__ import annotations

from typing import Iterator

UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT = range(4)
ALL_DIRS = (UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT)
FORWARD = (UP_LEFT, UP_RIGHT)
_DELTAS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

SIDE = 10
SQUARES = 50
FULL = (1 << SQUARES) - 1
PROMOTION_ROW = 0b11111  # the five squares of the far row, from the side to move

#    .  0  .  1  .  2  .  3  .  4
#    5  .  6  .  7  .  8  .  9  .
#    . 10  . 11  . 12  . 13  . 14
#   15  . 16  . 17  . 18  . 19  .
#    . 20  . 21  . 22  . 23  . 24
#   25  . 26  . 27  . 28  . 29  .
#    . 30  . 31  . 32  . 33  . 34
#   35  . 36  . 37  . 38  . 39  .
#    . 40  . 41  . 42  . 43  . 44
#   45  . 46  . 47  . 48  . 49  .


def square_to_rc(s: int) -> tuple[int, int]:
    row = s // 5
    return row, 2 * (s % 5) + (1 if row % 2 == 0 else 0)


def rc_to_square(row: int, col: int) -> int:
    if not (0 <= row < SIDE and 0 <= col < SIDE) or (row + col) % 2 == 0:
        return -1
    return row * 5 + col // 2


def _rays() -> list[list[tuple[int, ...]]]:
    # every square a king could reach going one way, in the order it would pass them
    out = []
    for s in range(SQUARES):
        row, col = square_to_rc(s)
        here = []
        for dr, dc in _DELTAS:
            line, step = [], 1
            while True:
                t = rc_to_square(row + dr * step, col + dc * step)
                if t < 0:
                    break
                line.append(t)
                step += 1
            here.append(tuple(line))
        out.append(here)
    return out


RAYS = _rays()
STEP = [[ray[0] if ray else -1 for ray in square] for square in RAYS]


def bits(mask: int) -> Iterator[int]:
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def mask(*squares: int) -> int:
    out = 0
    for s in squares:
        out |= 1 << s
    return out


def reverse(m: int) -> int:
    # the board as the other player sees it
    return mask(*(SQUARES - 1 - s for s in bits(m)))
