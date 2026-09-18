import numpy as np

from flycore.board import flip
from flycore.moves import legal_moves
from game.fly import Fly
from game.players import material
from train.skills import auc, exam, skills


def test_auc():
    labels = np.array([1, 1, -1, -1])
    assert auc(np.array([4, 3, 2, 1]), labels) == 1.0
    assert auc(np.array([1, 2, 3, 4]), labels) == 0.0
    assert auc(np.array([1, 1, 1, 1]), labels) == 0.5


def test_exam_labels_match_the_rules():
    boards, safe, ahead = exam(60, seed=1)
    assert len(boards) == len(safe) == len(ahead) == 60
    for board, s, a in zip(boards, safe, ahead):
        opponent_can_capture = any(m.is_capture for m in legal_moves(flip(board)))
        assert (s == -1) == opponent_can_capture
        assert a == np.sign(material(board))


def test_skills_are_auc_values(brain):
    result = skills(Fly(brain))
    assert set(result) == {"safety", "material"}
    assert all(0 <= v <= 1 for v in result.values())
