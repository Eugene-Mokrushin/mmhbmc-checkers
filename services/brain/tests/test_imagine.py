from flycore.board import Position, apply_move, flip
from flycore.moves import legal_moves
from flycore.squares import mask
from game.imagine import WIN, Imagination
from game.players import GreedyPlayer, Player, material

HANGING = Position(mask(22), 0, mask(13), 0)  # 22-17 walks into 13x17, 22-18 is safe


class Recorder(Player):
    def __init__(self):
        super().__init__()
        self.seen = []

    def scores(self, after):
        self.seen += after
        return [material(b) for b in after]


def after_boards(pos):
    return [flip(apply_move(pos, m)) for m in legal_moves(pos)]


def test_depth_one_without_extensions_is_the_judge_alone():
    boards = after_boards(HANGING)
    assert Imagination(GreedyPlayer(), depth=1, extensions=0).scores(boards) == [material(b) for b in boards]


def test_imagining_the_reply_avoids_hanging_a_piece():
    for seed in range(8):
        (move,) = Imagination(GreedyPlayer(), depth=2, breadth=10, extensions=0, seed=seed).choose_many([HANGING])
        assert move.path == (22, 18)


def test_forced_captures_are_followed_past_the_depth():
    for seed in range(8):
        (move,) = Imagination(GreedyPlayer(), depth=1, extensions=4, seed=seed).choose_many([HANGING])
        assert move.path == (22, 18)


def test_the_judge_also_sees_the_board_from_the_opponents_side():
    judge = Recorder()
    Imagination(judge, depth=2, breadth=10, extensions=0).choose_many([HANGING])
    replies = [flip(apply_move(flip(b), m)) for b in after_boards(HANGING) for m in legal_moves(flip(b))]
    assert set(replies) <= set(judge.seen)
    assert any(b.own_men & mask(13) == 0 and b.own for b in replies)


def test_a_move_that_leaves_the_opponent_stuck_wins():
    pos = Position(mask(22), 0, mask(17), 0)
    assert Imagination(GreedyPlayer(), depth=3).scores(after_boards(pos)) == [WIN]


def test_breadth_limits_the_replies_imagined():
    judge = Recorder()
    Imagination(judge, depth=3, breadth=1, extensions=0).choose_many([Position(mask(21, 22, 23), 0, mask(9, 10), 0)])
    wide = Recorder()
    Imagination(wide, depth=3, breadth=10, extensions=0).choose_many([Position(mask(21, 22, 23), 0, mask(9, 10), 0)])
    assert len(judge.seen) < len(wide.seen)
