import argparse

import numpy as np
import torch

import progress
from connectome.coordinates import one_per_neuron, read_markers
from connectome.load import load
from flycore.board import INITIAL, apply_move, flip
from game.arena import Game, play
from game.players import GreedyPlayer, MinimaxPlayer
from progress import Progress
from sim.kernels import best_device
from train.evaluate import RUNS_DIR
from train.gradient import Log, games
from train.lessons import Lessons, loss
from train.student import Student

DISCOUNT = 0.98


def lived(game: Game, fly) -> tuple[list, list[float]]:
    # the boards the fly chose in a game, each paired with how the game went for it,
    # counted less the further the end was
    z = 0.0 if game.winner is None else 1.0 if game.winner is fly else -1.0
    pos, boards, targets = INITIAL, [], []
    for ply, move in enumerate(game.moves):
        if (ply % 2 == 0) == (game.first is fly):
            boards.append(flip(apply_move(pos, move)))
            targets.append(z * DISCOUNT ** (len(game.moves) - ply))
        pos = apply_move(pos, move)
    return boards, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="g-games")
    parser.add_argument("--init", required=True, help="checkpoint of the teacher phase")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--games", type=int, default=64, help="games per round, half against each opponent")
    parser.add_argument("--steps", type=int, default=40, help="training steps per round")
    parser.add_argument("--positions", type=int, default=8, help="teacher positions per step, so earlier lessons stay")
    parser.add_argument("--boards", type=int, default=48, help="boards from the fly's own games per step")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--explore", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    progress.use(args.run)
    log, rng, lessons = Log(RUNS_DIR / args.run, vars(args)), np.random.default_rng(args.seed), Lessons(args.depth)
    student = Student(load(min_syn=5), one_per_neuron(read_markers()), best_device())
    student.load(args.init)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    opponents = [MinimaxPlayer(2, args.seed), MinimaxPlayer(3, args.seed), GreedyPlayer(args.seed)]
    with Progress(args.rounds, "own games") as bar:
        for r in range(1, args.rounds + 1):
            fly = student.fly(args.seed + r)
            fly.explore = args.explore
            boards, targets, wins = [], [], []
            for opponent in rng.choice(opponents, 2, replace=False):
                for game in play(fly, opponent, args.games // 2, f"round {r} vs {type(opponent).__name__}"):
                    b, t = lived(game, fly)
                    boards, targets, wins = boards + b, targets + t, wins + [game.winner is fly]
            recent = []
            for _ in range(args.steps):
                pick = rng.choice(len(boards), min(args.boards, len(boards)), replace=False)
                taught, groups, values = lessons.batch(rng.choice(lessons.train, args.positions, replace=False), rng)
                scores = student.scores([boards[i] for i in pick] + taught)
                own = torch.mean((torch.tanh(scores[: len(pick)]) - torch.as_tensor(np.array(targets)[pick], dtype=torch.float32, device=scores.device)) ** 2)
                lesson, parts = loss(scores[len(pick) :], groups, values)
                optimizer.zero_grad()
                (own + lesson).backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                with torch.no_grad():
                    student.brain.gain.clamp_(-4, 4)
                recent.append({"own": own.item(), **parts})
            mean = {k: float(np.mean([x[k] for x in recent])) for k in recent[0]}
            bar.update(1, train_win=round(float(np.mean(wins)), 2), **{k: round(v, 3) for k, v in mean.items()})
            if r % args.eval_every == 0 or r == args.rounds:
                student.save(log.dir / f"g-{r:05d}.pt", round=r)
                log.record(step=r, train_win=float(np.mean(wins)), **mean, **student.exam(lessons, 200, rng), **games(student, args.eval_games, args.seed + 2))


if __name__ == "__main__":
    main()
