from __future__ import annotations

from typing import NamedTuple

from flycore.ten.squares import SIDE, bits, rc_to_square, reverse


class Position(NamedTuple):
    own_men: int
    own_kings: int
    opp_men: int
    opp_kings: int

    @property
    def own(self) -> int:
        return self.own_men | self.own_kings

    @property
    def opp(self) -> int:
        return self.opp_men | self.opp_kings

    @property
    def occupied(self) -> int:
        return self.own | self.opp


class Move(NamedTuple):
    origin: int
    destination: int
    captured: int
    path: tuple[int, ...]
    promotes: bool

    @property
    def is_capture(self) -> bool:
        return self.captured != 0


# twenty a side, four rows each, the side to move at the bottom
INITIAL = Position(own_men=((1 << 20) - 1) << 30, own_kings=0, opp_men=(1 << 20) - 1, opp_kings=0)


def apply_move(pos: Position, move: Move) -> Position:
    origin, dest = 1 << move.origin, 1 << move.destination
    own_men, own_kings = pos.own_men, pos.own_kings
    if own_kings & origin:
        own_kings = (own_kings & ~origin) | dest
    else:
        own_men &= ~origin
        if move.promotes:
            own_kings |= dest
        else:
            own_men |= dest

    # positions are always seen from the side to move
    return flip(Position(own_men, own_kings, pos.opp_men & ~move.captured, pos.opp_kings & ~move.captured))


def flip(pos: Position) -> Position:
    return Position(
        own_men=reverse(pos.opp_men),
        own_kings=reverse(pos.opp_kings),
        opp_men=reverse(pos.own_men),
        opp_kings=reverse(pos.own_kings),
    )


def render(pos: Position) -> str:
    glyph = {}
    for pieces, char in zip(pos, "oOxX"):
        for s in bits(pieces):
            glyph[s] = char
    rows = []
    for row in range(SIDE):
        squares = (rc_to_square(row, col) for col in range(SIDE))
        rows.append(" ".join(" " if s < 0 else glyph.get(s, ".") for s in squares))
    return "\n".join(rows)
