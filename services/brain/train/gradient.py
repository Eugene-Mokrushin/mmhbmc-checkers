import argparse
import csv
import json
import time

import numpy as np
import torch

import progress
from connectome.coordinates import one_per_neuron, read_markers
from connectome.load import load
from game import gradfly
from game.arena import play, tally
from progress import Progress
from sim.kernels import best_device
from train.evaluate import RUNS_DIR, baselines
from train.lessons import Lessons, loss
from train.student import Student

COLUMNS = ["minutes", "step", "loss", "rank", "value", "agree", "own", "train_win", "test_loss", "test_rank", "test_value", "test_agree", "random", "greedy", "minimax2"]


class Log:
    def __init__(self, run_dir, config: dict):
        self.dir, self.file, self.began = run_dir, run_dir / "log.csv", time.monotonic()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    def record(self, step: int, **row) -> None:
        new = not self.file.exists()
        with self.file.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
            if new:
                writer.writeheader()
            writer.writerow({"minutes": round((time.monotonic() - self.began) / 60, 1), "step": step, **{k: round(v, 4) for k, v in row.items()}})


def games(student: Student, n: int, seed: int) -> dict[str, float]:
    fly, opponents = student.fly(seed), baselines(seed)
    return {name: tally(play(fly, opponent, n, f"eval vs {name}"), fly)["win"] for name, opponent in opponents.items()} if n else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="g-teacher")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--positions", type=int, default=6, help="positions per step, up to 8 boards each")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--init", help="checkpoint to continue from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, help="brain steps per board; 200 is a tenth of a second")
    parser.add_argument("--carry", action="store_true", help="judge every board from a brain another board left busy")
    parser.add_argument("--colour", action="store_true", help="R1-6 see that a piece is there, R7 and R8 whose it is")
    parser.add_argument("--sweep", type=float, default=0.0, help="how far the board drifts across the eye, in squares")
    parser.add_argument("--recalibrate", action="store_true", help="refit the readout after loading, for a changed window")
    args = parser.parse_args()

    if args.window:
        gradfly.WINDOW = args.window
    gradfly.COLOUR, gradfly.SWEEP = args.colour, args.sweep
    progress.use(args.run)
    log, rng, lessons = Log(RUNS_DIR / args.run, vars(args)), np.random.default_rng(args.seed), Lessons(args.depth)
    student = Student(load(min_syn=5), one_per_neuron(read_markers()), best_device())
    start = student.load(args.init).get("step", 0) if args.init else 0
    if args.recalibrate:
        print(f"readout refitted for a {gradfly.WINDOW}-step window: correlation {student.calibrate(lessons, rng):.2f}", flush=True)
    if args.carry:
        student.carry = np.random.default_rng(args.seed + 7)
    if not args.init:
        print(f"readout fitted to the untrained brain: correlation {student.calibrate(lessons, rng):.2f}", flush=True)
        student.save(log.dir / "g-0000000.pt", step=0)
        log.record(0, **student.exam(lessons, 200, rng), **games(student, args.eval_games, args.seed + 2))
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    recent = []

    with Progress(args.steps - start, "teacher lessons") as bar:
        for step in range(start + 1, args.steps + 1):
            boards, groups, values = lessons.batch(rng.choice(lessons.train, args.positions, replace=False), rng)
            optimizer.zero_grad()
            total, parts = loss(student.scores(boards), groups, values)
            total.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                student.brain.gain.clamp_(-4, 4)
            recent.append({"loss": total.item(), **parts})
            bar.update(1, **{k: round(v, 3) for k, v in recent[-1].items()})
            if step % args.eval_every == 0 or step == args.steps:
                student.save(log.dir / f"g-{step:07d}.pt", step=step)
                mean = {k: float(np.mean([r[k] for r in recent])) for k in recent[0]}
                log.record(step, **mean, **student.exam(lessons, 200, rng), **games(student, args.eval_games, args.seed + 2))
                recent = []


if __name__ == "__main__":
    main()
