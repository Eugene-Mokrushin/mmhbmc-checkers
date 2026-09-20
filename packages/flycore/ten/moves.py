from __future__ import annotations

from flycore.ten.board import Move, Position, apply_move
from flycore.ten.captures import captures
from flycore.ten.squares import FORWARD, FULL, PROMOTION_ROW, RAYS, STEP, bits


def legal_moves(pos: Position) -> list[Move]:
    # taking is compulsory, and of the ways to take you must choose one that takes the most
    taken = captures(pos)
    if not taken:
        return _quiet_moves(pos)
    most = max(bin(move.captured).count("1") for move in taken)
    return [_crowned(move) for move in taken if bin(move.captured).count("1") == most]


def is_terminal(pos: Position) -> bool:
    return pos.own == 0 or not legal_moves(pos)


def perft(pos: Position, depth: int) -> int:
    if depth == 0:
        return 1
    moves = legal_moves(pos)
    if depth == 1:
        return len(moves)
    return sum(perft(apply_move(pos, m), depth - 1) for m in moves)


def _crowned(move: Move) -> Move:
    # a man crowns only if it finishes on the far row; passing through it mid-jump is not
    # enough, and it carries on as a man
    if PROMOTION_ROW >> move.destination & 1:
        return move._replace(promotes=True)
    return move


def _quiet_moves(pos: Position) -> list[Move]:
    empty = FULL & ~pos.occupied
    moves = []
    for s in bits(pos.own):
        if pos.own_kings >> s & 1:
            for ray in RAYS[s]:
                for t in ray:
                    if not empty >> t & 1:
                        break
                    moves.append(Move(s, t, 0, (s, t), False))
        else:
            for d in FORWARD:
                t = STEP[s][d]
                if t >= 0 and empty >> t & 1:
                    moves.append(Move(s, t, 0, (s, t), bool(PROMOTION_ROW >> t & 1)))
    return moves
