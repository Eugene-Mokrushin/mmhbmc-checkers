from flycore.board import INITIAL, Position, apply_move
from flycore.encode import EMPTY, N_LINES, encode, square_lines
from flycore.moves import legal_moves
from flycore.squares import mask


def test_one_line_per_square():
    pos = Position(own_men=mask(20), own_kings=mask(3), opp_men=mask(9), opp_kings=mask(31))
    lines = encode(pos)
    assert len(lines) == 32
    assert {4 * 3 + 1, 4 * 9 + 2, 4 * 20 + 0, 4 * 31 + 3} <= set(lines)
    assert sum(line >= EMPTY for line in lines) == 28


def test_opening():
    lines = encode(INITIAL)
    assert len(lines) == 32
    assert all(0 <= line < N_LINES for line in lines)
    assert [line % 4 for line in lines if line < EMPTY] == [2] * 12 + [0] * 12
    assert [line - EMPTY for line in lines if line >= EMPTY] == list(range(12, 20))


def test_each_square_lights_exactly_one_of_its_lines():
    pos = apply_move(INITIAL, legal_moves(INITIAL)[0])
    lines = set(encode(pos))
    for square in range(32):
        assert len(lines & set(square_lines(square))) == 1


def test_lines_are_always_from_the_side_to_move():
    after = apply_move(INITIAL, legal_moves(INITIAL)[0])
    assert sum(1 for line in encode(after) if line < EMPTY and line % 4 == 0) == 12
    assert encode(after) != encode(INITIAL)
