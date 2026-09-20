from __future__ import annotations

from flycore.ten.board import Move, Position
from flycore.ten.squares import ALL_DIRS, RAYS, bits


def captures(pos: Position) -> list[Move]:
    # every capture there is; the rules then keep only the longest
    out: list[Move] = []
    for s in bits(pos.own):
        _jump(pos, s, s, bool(pos.own_kings >> s & 1), 0, (s,), out)
    return out


def _jump(pos: Position, origin: int, at: int, king: bool, taken: int, path: tuple[int, ...], out: list[Move]) -> None:
    # a jumped piece stays on the board until the move ends: it cannot be jumped twice,
    # and it still blocks the way
    stays = pos.occupied & ~(1 << origin)
    landings = _landings(pos, at, king, taken, stays)
    for over, land in landings:
        _jump(pos, origin, land, king, taken | (1 << over), path + (land,), out)
    if taken and not landings:
        out.append(Move(origin, at, taken, path, False))


def _landings(pos: Position, at: int, king: bool, taken: int, stays: int) -> list[tuple[int, int]]:
    # (the piece jumped, where we land) for every jump available from here
    out = []
    for d in ALL_DIRS:
        ray = RAYS[at][d]
        for i, square in enumerate(ray):
            if taken >> square & 1 or pos.own >> square & 1:
                break  # our own piece, or one we have already jumped
            if pos.opp >> square & 1:
                for land in ray[i + 1 :]:
                    if stays >> land & 1:
                        break
                    out.append((square, land))
                    if not king:
                        break  # a man lands on the square immediately beyond
                break  # only one piece is jumped at a time
            if not king:
                break  # a man jumps its neighbour, it cannot reach a piece further off
    return out
