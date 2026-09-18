from __future__ import annotations

from typing import Iterator

UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT = range(4)
FORWARD = (UP_LEFT, UP_RIGHT)
ALL_DIRS = (UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT)
_DELTAS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

PROMOTION_ROW = 0x0000000F
FULL = 0xFFFFFFFF

#    .  0  .  1  .  2  .  3
#    4  .  5  .  6  .  7  .
#    .  8  .  9  . 10  . 11
#   12  . 13  . 14  . 15  .
#    . 16  . 17  . 18  . 19
#   20  . 21  . 22  . 23  .
#    . 24  . 25  . 26  . 27
#   28  . 29  . 30  . 31  .


def square_to_rc(s: int) -> tuple[int, int]:
    row = s // 4
    return row, 2 * (s % 4) + (1 if row % 2 == 0 else 0)


def rc_to_square(row: int, col: int) -> int:
    if not (0 <= row < 8 and 0 <= col < 8) or (row + col) % 2 == 0:
        return -1
    return row * 4 + col // 2


def _tables() -> tuple[list[list[int]], list[list[tuple[int, int]]]]:
    step = [[-1] * 4 for _ in range(32)]
    jump = [[(-1, -1)] * 4 for _ in range(32)]
    for s in range(32):
        row, col = square_to_rc(s)
        for d, (dr, dc) in enumerate(_DELTAS):
            over = rc_to_square(row + dr, col + dc)
            land = rc_to_square(row + 2 * dr, col + 2 * dc)
            step[s][d] = over
            if over >= 0 and land >= 0:
                jump[s][d] = (over, land)
    return step, jump


STEP, JUMP = _tables()


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


def reverse32(m: int) -> int:
    return mask(*(31 - s for s in bits(m)))
