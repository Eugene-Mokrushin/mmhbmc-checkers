from __future__ import annotations

from flycore.board import Move, Position, apply_move
from flycore.squares import ALL_DIRS, FORWARD, FULL, JUMP, PROMOTION_ROW, STEP, bits


def legal_moves(pos: Position) -> list[Move]:
    return _captures(pos) or _quiet_moves(pos)


def is_terminal(pos: Position) -> bool:
    return pos.own == 0 or not legal_moves(pos)


def perft(pos: Position, depth: int) -> int:
    if depth == 0:
        return 1
    moves = legal_moves(pos)
    if depth == 1:
        return len(moves)
    return sum(perft(apply_move(pos, m), depth - 1) for m in moves)


def _quiet_moves(pos: Position) -> list[Move]:
    empty = FULL & ~pos.occupied
    moves = []
    for s in bits(pos.own):
        king = bool(pos.own_kings >> s & 1)
        for d in ALL_DIRS if king else FORWARD:
            t = STEP[s][d]
            if t >= 0 and empty >> t & 1:
                promotes = not king and bool(PROMOTION_ROW >> t & 1)
                moves.append(Move(s, t, 0, (s, t), promotes))
    return moves


def _captures(pos: Position) -> list[Move]:
    out: list[Move] = []
    for s in bits(pos.own):
        _jump(pos, s, s, bool(pos.own_kings >> s & 1), 0, (s,), out)
    return out


def _jump(
    pos: Position,
    origin: int,
    at: int,
    king: bool,
    captured: int,
    path: tuple[int, ...],
    out: list[Move],
) -> None:
    # jumped pieces stay on the board until the move is applied, but can't be jumped twice
    blockers = (pos.occupied & ~(1 << origin) & ~captured) | (1 << at)
    targets = pos.opp & ~captured
    extended = False

    for d in ALL_DIRS if king else FORWARD:
        over, land = JUMP[at][d]
        if over < 0 or not targets >> over & 1 or blockers >> land & 1:
            continue
        extended = True
        taken = captured | (1 << over)
        if not king and PROMOTION_ROW >> land & 1:
            out.append(Move(origin, land, taken, path + (land,), True))
        else:
            _jump(pos, origin, land, king, taken, path + (land,), out)

    if captured and not extended:
        out.append(Move(origin, at, captured, path, False))
