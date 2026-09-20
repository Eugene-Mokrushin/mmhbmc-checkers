import pytest

from flycore.ten.board import INITIAL, Position, apply_move, flip
from flycore.ten.moves import legal_moves, perft
from flycore.ten.squares import PROMOTION_ROW, mask

# the counts every draughts engine is checked against, from the opening position
PERFT = {1: 9, 2: 81, 3: 658, 4: 4265, 5: 27117, 6: 167140}


@pytest.mark.parametrize("depth,want", PERFT.items())
def test_the_opening_branches_the_way_the_books_say(depth, want):
    assert perft(INITIAL, depth) == want


def test_the_board_starts_with_twenty_a_side():
    assert bin(INITIAL.own_men).count("1") == 20
    assert bin(INITIAL.opp_men).count("1") == 20
    assert flip(flip(INITIAL)) == INITIAL


def test_a_king_slides_as_far_as_it_likes():
    # a lone king in a corner-ish square reaches everything on its diagonals
    pos = Position(own_men=0, own_kings=mask(27), opp_men=0, opp_kings=0)
    reach = {move.destination for move in legal_moves(pos)}
    assert reach == {21, 16, 10, 5, 22, 18, 13, 9, 4, 31, 36, 40, 45, 32, 38, 43, 49}


def test_a_man_takes_backwards_as_well_as_forwards():
    # our man at 27 with theirs behind it at 32, the square beyond empty
    pos = Position(own_men=mask(27), own_kings=0, opp_men=mask(32), opp_kings=0)
    moves = legal_moves(pos)
    assert [(m.origin, m.destination) for m in moves] == [(27, 38)]
    assert moves[0].captured == mask(32)


def test_you_must_take_as_many_as_you_can():
    # one way takes a single piece, the other takes two
    pos = Position(own_men=mask(27), own_kings=0, opp_men=mask(21, 22, 13), opp_kings=0)
    moves = legal_moves(pos)
    assert all(bin(m.captured).count("1") == 2 for m in moves)
    assert {m.destination for m in moves} == {9}


def test_a_man_crowns_only_where_it_stops():
    # it lands on the far row mid-chain, with another piece to take, so it carries on as a
    # man and is not crowned: 12 takes 7, stands on 1 which is the far row, takes 6, stops on 10
    pos = Position(own_men=mask(12), own_kings=0, opp_men=mask(7, 6), opp_kings=0)
    moves = legal_moves(pos)
    assert len(moves) == 1
    assert list(moves[0].path) == [12, 1, 10]
    assert PROMOTION_ROW >> moves[0].path[1] & 1  # it really did stand on the far row
    assert moves[0].promotes is False
    after = apply_move(pos, moves[0])
    assert after.opp_kings == 0 and bin(after.opp_men).count("1") == 1


def test_a_piece_already_jumped_cannot_be_jumped_again():
    pos = Position(own_men=0, own_kings=mask(49), opp_men=mask(43, 32), opp_kings=0)
    for move in legal_moves(pos):
        assert bin(move.captured).count("1") <= 2


def test_what_is_taken_leaves_the_board():
    pos = Position(own_men=mask(27), own_kings=0, opp_men=mask(22), opp_kings=0)
    move = legal_moves(pos)[0]
    after = apply_move(pos, move)
    assert after.opp_men | after.opp_kings == mask(49 - move.destination)
    assert after.own == 0
