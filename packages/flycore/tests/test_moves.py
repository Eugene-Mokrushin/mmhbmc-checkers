from flycore.board import INITIAL, Position, apply_move
from flycore.moves import is_terminal, legal_moves
from flycore.squares import mask


def moves_of(pos):
    return [(m.path, m.captured) for m in legal_moves(pos)]


def test_capture_is_mandatory():
    pos = Position(mask(22, 28), 0, mask(17), 0)
    assert moves_of(pos) == [((22, 13), mask(17))]


def test_multi_jump_is_one_move():
    pos = Position(mask(29), 0, mask(25, 17), 0)
    assert moves_of(pos) == [((29, 22, 13), mask(25, 17))]


def test_branching_chain_gives_every_complete_path():
    pos = Position(mask(29), 0, mask(25, 17, 18), 0)
    assert {m.path for m in legal_moves(pos)} == {(29, 22, 13), (29, 22, 15)}


def test_promotion_ends_the_chain():
    pos = Position(mask(10), 0, mask(6, 5), 0)
    (move,) = legal_moves(pos)
    assert (move.path, move.captured, move.promotes) == ((10, 1), mask(6), True)

    after = apply_move(pos, move)
    assert after.opp_kings == mask(30)
    assert after.own_men == mask(26)


def test_king_keeps_jumping_through_back_row():
    pos = Position(0, mask(10), mask(6, 5), 0)
    assert (10, 1, 8) in {m.path for m in legal_moves(pos)}


def test_men_move_forward_only():
    pos = Position(mask(18), 0, 0, 0)
    assert {m.destination for m in legal_moves(pos)} == {14, 15}


def test_kings_move_both_ways():
    pos = Position(0, mask(18), 0, 0)
    assert {m.destination for m in legal_moves(pos)} == {14, 15, 22, 23}


def test_men_cannot_capture_backward():
    pos = Position(mask(13), 0, mask(17), 0)
    assert not any(m.is_capture for m in legal_moves(pos))


def test_kings_capture_backward():
    pos = Position(0, mask(13), mask(17), 0)
    assert moves_of(pos) == [((13, 22), mask(17))]


def test_terminal_without_pieces():
    assert is_terminal(Position(0, 0, mask(5), 0))


def test_terminal_when_blocked():
    pos = Position(mask(4), 0, mask(0), 0)
    assert legal_moves(pos) == []
    assert is_terminal(pos)


def test_opening_is_not_terminal():
    assert not is_terminal(INITIAL)
