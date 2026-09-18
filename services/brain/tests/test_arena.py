import pytest

import progress
from flycore.moves import legal_moves
from game.arena import MAX_PLIES, play, tally
from game.players import MinimaxPlayer, RandomPlayer


@pytest.fixture(autouse=True)
def quiet_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "PROGRESS_FILE", tmp_path / "progress.txt")


def test_games_end_properly():
    a, b = RandomPlayer(1), RandomPlayer(2)
    games = play(a, b, 30)
    assert all(g.over for g in games)
    assert sum(g.first is a for g in games) == 15
    for g in games:
        if g.winner is None:
            assert len(g.moves) == MAX_PLIES
        else:
            assert not legal_moves(g.pos)
            assert g.winner is not g.to_move


def test_tally_adds_up():
    a = RandomPlayer(1)
    result = tally(play(a, RandomPlayer(2), 20), a)
    assert sum(result.values()) == pytest.approx(1)


def test_minimax_beats_random():
    a = MinimaxPlayer(depth=2, seed=1)
    assert tally(play(a, RandomPlayer(2), 20), a)["win"] > 0.8


def test_self_play_needs_two_objects():
    a = RandomPlayer(1)
    with pytest.raises(ValueError):
        play(a, a, 2)
