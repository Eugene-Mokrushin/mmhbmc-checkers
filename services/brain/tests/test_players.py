from flycore.board import INITIAL, Position
from flycore.moves import legal_moves
from flycore.squares import mask
from game.players import GreedyPlayer, MinimaxPlayer, RandomPlayer, material


def test_material_counts_kings_as_one_and_a_half():
    assert material(INITIAL) == 0
    assert material(Position(mask(1, 2), mask(3), mask(4), 0)) == 2.5


def test_random_player_moves_legally():
    player = RandomPlayer(seed=0)
    assert player.choose_many([INITIAL] * 20)[0] in legal_moves(INITIAL)


def test_greedy_takes_the_king_over_the_man():
    pos = Position(mask(22, 23), 0, mask(17), mask(18))
    assert {m.captured for m in legal_moves(pos)} == {mask(17), mask(18)}
    for seed in range(10):
        (move,) = GreedyPlayer(seed).choose_many([pos])
        assert move.captured == mask(18)


def test_minimax_does_not_hang_a_piece():
    # 22-17 walks into 13x17; 22-18 is safe. Greedy can't tell them apart.
    pos = Position(mask(22), 0, mask(13), 0)
    for seed in range(10):
        (move,) = MinimaxPlayer(depth=2, seed=seed).choose_many([pos])
        assert move.path == (22, 18)
    greedy = {GreedyPlayer(seed).choose_many([pos])[0].path for seed in range(20)}
    assert greedy == {(22, 17), (22, 18)}
