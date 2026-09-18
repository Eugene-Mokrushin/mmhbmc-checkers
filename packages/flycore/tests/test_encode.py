from flycore.board import INITIAL, Position, apply_move
from flycore.encode import N_LINES, encode
from flycore.moves import legal_moves
from flycore.squares import mask


def test_one_line_per_piece():
    pos = Position(own_men=mask(20), own_kings=mask(3), opp_men=mask(9), opp_kings=mask(31))
    assert encode(pos) == [4 * 3 + 1, 4 * 9 + 2, 4 * 20 + 0, 4 * 31 + 3]


def test_opening():
    lines = encode(INITIAL)
    assert len(lines) == 24
    assert all(0 <= line < N_LINES for line in lines)
    assert [line % 4 for line in lines] == [2] * 12 + [0] * 12


def test_lines_are_always_from_the_side_to_move():
    after = apply_move(INITIAL, legal_moves(INITIAL)[0])
    assert sum(1 for line in encode(after) if line % 4 == 0) == 12
    assert encode(after) != encode(INITIAL)
