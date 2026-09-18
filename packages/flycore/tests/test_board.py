import random

from flycore.board import INITIAL, apply_move, flip, render
from flycore.moves import legal_moves
from flycore.squares import mask, reverse32


def count(m):
    return bin(m).count("1")


def test_reverse32_is_its_own_inverse():
    rng = random.Random(0)
    for _ in range(1000):
        m = rng.getrandbits(32)
        assert reverse32(reverse32(m)) == m
    assert reverse32(mask(0)) == mask(31)
    assert reverse32(INITIAL.own_men) == INITIAL.opp_men


def test_flip_twice_is_identity():
    rng = random.Random(1)
    for _ in range(100):
        pos = INITIAL
        for _ in range(rng.randint(0, 40)):
            moves = legal_moves(pos)
            if not moves:
                break
            pos = apply_move(pos, rng.choice(moves))
        assert flip(flip(pos)) == pos
    assert flip(INITIAL) == INITIAL


def test_apply_move_hands_the_board_to_the_opponent():
    for m in legal_moves(INITIAL):
        after = apply_move(INITIAL, m)
        moved = (INITIAL.own_men & ~mask(m.origin)) | mask(m.destination)
        assert after.own_men == INITIAL.own_men
        assert after.opp_men == reverse32(moved)


def test_two_quiet_moves_return_to_the_first_frame():
    first = legal_moves(INITIAL)[0]
    p1 = apply_move(INITIAL, first)
    reply = legal_moves(p1)[0]
    p2 = apply_move(p1, reply)
    assert p2.own_men == (INITIAL.own_men & ~mask(first.origin)) | mask(first.destination)
    assert p2.opp_men == reverse32((p1.own_men & ~mask(reply.origin)) | mask(reply.destination))


def test_render_opening():
    lines = render(INITIAL).splitlines()
    assert lines[0] == "  x   x   x   x"
    assert lines[3] == ".   .   .   .  "
    assert lines[7] == "o   o   o   o  "


def test_random_games_keep_the_board_consistent():
    rng = random.Random(1234)
    for _ in range(200):
        pos = INITIAL
        for _ in range(300):
            assert pos.own_men & pos.own_kings == 0
            assert pos.opp_men & pos.opp_kings == 0
            assert pos.own & pos.opp == 0
            assert pos.occupied >> 32 == 0
            assert pos.own_men & 0x0000000F == 0
            assert pos.opp_men & 0xF0000000 == 0
            assert count(pos.own) <= 12 and count(pos.opp) <= 12

            moves = legal_moves(pos)
            if not moves:
                break
            move = rng.choice(moves)
            before = count(pos.opp)
            pos = apply_move(pos, move)
            assert count(pos.own) == before - count(move.captured)
