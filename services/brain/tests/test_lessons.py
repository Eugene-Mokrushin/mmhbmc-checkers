import numpy as np
import torch

from flycore.board import INITIAL, apply_move, flip
from flycore.moves import legal_moves
from game.arena import Game
from game.players import RandomPlayer
from train.lessons import loss
from train.outcomes import DISCOUNT, lived

GROUPS = np.array([0, 0, 0, 1, 1])
VALUES = np.array([1.0, -2.0, 0.0, 1000.0, -1000.0], dtype=np.float32)


def test_agreeing_with_the_teacher_costs_less():
    good, _ = loss(torch.tensor([0.3, -0.6, 0.0, 1.0, -1.0]), GROUPS, VALUES)
    bad, parts = loss(torch.tensor([-0.5, 0.5, 0.0, -1.0, 1.0]), GROUPS, VALUES)
    assert good < bad and parts["agree"] == 0.0


def test_gradient_moves_scores_toward_the_teacher():
    scores = torch.zeros(5, requires_grad=True)
    loss(scores, GROUPS, VALUES)[0].backward()
    assert scores.grad[0] < 0 < scores.grad[1] and scores.grad[3] < 0 < scores.grad[4]


def test_own_games_pair_each_chosen_board_with_the_result():
    fly, other = RandomPlayer(1), RandomPlayer(2)
    game, pos = Game(other, fly), INITIAL
    for _ in range(5):
        move = legal_moves(pos)[0]
        game.moves.append(move)
        pos = apply_move(pos, move)
    game.winner = other
    boards, targets = lived(game, fly)
    first = apply_move(INITIAL, game.moves[0])
    assert boards[0] == flip(apply_move(first, game.moves[1]))
    assert len(boards) == 2 and targets == [-(DISCOUNT**4), -(DISCOUNT**2)]
