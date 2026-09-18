import io

import progress
from progress import Progress, duration


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_duration():
    assert duration(None) == "?"
    assert duration(42) == "42s"
    assert duration(185) == "3m05s"
    assert duration(3 * 3600 + 7 * 60) == "3h07m"


def test_file_shows_count_bar_eta_and_stats(tmp_path):
    clock, path = Clock(), tmp_path / "progress.txt"
    with Progress(100, "selfplay", path=path, every=5, clock=clock, stream=io.StringIO()) as p:
        clock.now = 60
        p.update(25, win_random=0.614, games=25)
        line = path.read_text()
    assert line.startswith("selfplay [#####               ] 25/100 25.0% | 1m00s elapsed, 3m00s left")
    assert "win_random=0.614 games=25" in line


def test_file_is_rewritten_at_most_every_interval(tmp_path):
    clock, path = Clock(), tmp_path / "progress.txt"
    with Progress(10, "run", path=path, every=5, clock=clock, stream=io.StringIO()) as p:
        clock.now = 1
        p.update()
        assert " 0/10 " in path.read_text()
        clock.now = 6
        p.update()
        assert " 2/10 " in path.read_text()


def test_final_line_goes_to_logs_when_not_a_terminal(tmp_path):
    stream = io.StringIO()
    with Progress(3, "run", path=tmp_path / "p.txt", clock=Clock(), stream=stream) as p:
        p.update(3)
    assert stream.getvalue().startswith("run [####################] 3/3 100.0%")
    assert stream.getvalue().count("\n") == 1


def test_named_runs_write_their_own_file(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "PROGRESS_DIR", tmp_path)
    monkeypatch.setattr(progress, "PROGRESS_FILE", tmp_path / "latest.txt")
    monkeypatch.setattr(progress, "RUN", "")
    progress.use("b-random")
    with Progress(4, "eval vs greedy", clock=Clock(), stream=io.StringIO()):
        pass
    assert (tmp_path / "b-random.txt").read_text().startswith("b-random: eval vs greedy [")
