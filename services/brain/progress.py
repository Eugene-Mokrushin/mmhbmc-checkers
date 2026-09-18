import os
import sys
import time
from datetime import datetime
from pathlib import Path

from paths import ARTIFACTS_DIR

PROGRESS_DIR = ARTIFACTS_DIR / "progress"
PROGRESS_FILE = PROGRESS_DIR / "latest.txt"
RUN = ""


def use(name: str) -> None:
    # from here on this process writes to its own file, so parallel runs don't overwrite each other
    global PROGRESS_FILE, RUN
    PROGRESS_FILE, RUN = PROGRESS_DIR / f"{name}.txt", name


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    h, m, s = seconds // 3600, seconds // 60 % 60, seconds % 60
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s" if m else f"{s}s"


class Progress:
    def __init__(self, total, label, path=None, every=5.0, clock=time.monotonic, stream=sys.stderr):
        self.total, self.label, self.every = total, label, every
        self.path = Path(path) if path else PROGRESS_FILE
        self.clock, self.stream = clock, stream
        self.done, self.stats = 0, {}
        self.start = self.last = clock()

    def __enter__(self):
        self.write()
        return self

    def __exit__(self, *exc):
        self.write(final=True)

    def update(self, n=1, **stats):
        self.done += n
        self.stats.update(stats)
        if self.clock() - self.last >= self.every:
            self.write()

    def line(self) -> str:
        elapsed = self.clock() - self.start
        frac = min(self.done / self.total, 1) if self.total else 1
        left = elapsed / self.done * (self.total - self.done) if self.done else None
        bar = "#" * int(frac * 20)
        stats = " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.stats.items())
        parts = [
            f"{RUN + ': ' if RUN else ''}{self.label} [{bar:<20}] {self.done:,}/{self.total:,} {frac:.1%}",
            f"{duration(elapsed)} elapsed, {duration(left)} left",
            stats,
            datetime.now().strftime("%H:%M:%S"),
        ]
        return " | ".join(p for p in parts if p)

    def write(self, final=False):
        self.last = self.clock()
        text = self.line()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(text + "\n")
        os.replace(tmp, self.path)
        if self.stream.isatty():
            print("\r" + text, end="\n" if final else "", file=self.stream, flush=True)
        elif final:
            print(text, file=self.stream, flush=True)
