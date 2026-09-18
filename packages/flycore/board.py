from __future__ import annotations

from typing import NamedTuple

from flycore.squares import bits, rc_to_square, reverse32


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


INITIAL = Position(own_men=0xFFF00000, own_kings=0, opp_men=0x00000FFF, opp_kings=0)


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
    return Position(
        own_men=reverse32(pos.opp_men & ~move.captured),
        own_kings=reverse32(pos.opp_kings & ~move.captured),
        opp_men=reverse32(own_men),
        opp_kings=reverse32(own_kings),
    )


def render(pos: Position) -> str:
    glyph = {}
    for pieces, char in zip(pos, "oOxX"):
        for s in bits(pieces):
            glyph[s] = char
    rows = []
    for row in range(8):
        squares = (rc_to_square(row, col) for col in range(8))
        rows.append(" ".join(" " if s < 0 else glyph.get(s, ".") for s in squares))
    return "\n".join(rows)
