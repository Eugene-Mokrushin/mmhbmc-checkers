import argparse

import numpy as np

from connectome.controls import CONTROLS
from flycore.board import apply_move
from game.arena import OPPONENTS, Game, finish, step
from game.fly import WINDOW, Fly
from game.players import Player, material
from progress import Progress
from train.controls import setup
from train.evaluate import Run, evaluate
from train.plasticity import Plasticity, Rule

WIN = 3.0


def block(fly: Fly, plastic: Plasticity, opponent: Player, n: int, shaped: bool) -> dict:
    kc, mbon = fly.mb.members("KC"), fly.mb.members("MBON")
    games = [Game(fly, opponent) if i % 2 == 0 else Game(opponent, fly) for i in range(n)]
    traces = np.zeros((n, len(kc)), dtype=np.float32)
    before = np.zeros(n)
    kc_active, mbon_spikes = [], []
    while not all(g.over for g in games):
        finish(games)
        mine = [i for i, g in enumerate(games) if not g.over and g.to_move is fly]
        if mine:
            moves, counts = fly.choose_recorded([games[i].pos for i in mine])
            active = counts[:, kc] > 0
            kc_active.append(active.mean())
            mbon_spikes.append(counts[:, mbon].mean())
            for j, i in enumerate(mine):
                traces[i] = plastic.rule.decay * traces[i] + (1 - plastic.rule.decay) * active[j]
                before[i] = material(games[i].pos)
                games[i].moves.append(moves[j])
                games[i].pos = apply_move(games[i].pos, moves[j])
        finish(games)
        step(games, opponent)
        finish(games)
        if mine:
            reward = np.array([outcome(games[i], fly, before[i], shaped) for i in mine])
            plastic.update(traces[mine], reward)
    wins = sum(g.winner is fly for g in games) / n
    return {"train_win": wins, "kc_active": float(np.mean(kc_active)), "mbon_hz": float(np.mean(mbon_spikes)) / (WINDOW * fly.sim.p.dt)}


def outcome(game: Game, fly: Fly, before: float, shaped: bool) -> float:
    if game.over:
        return 0.0 if game.winner is None else WIN if game.winner is fly else -WIN
    return material(game.pos) - before if shaped else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="shaped-random")
    parser.add_argument("--games", type=int, default=20000)
    parser.add_argument("--opponent", choices=OPPONENTS, default="random")
    parser.add_argument("--control", choices=CONTROLS, default="real")
    parser.add_argument("--reward", choices=("shaped", "terminal"), default="shaped")
    parser.add_argument("--parallel", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=2048)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--explore", type=float, default=0.05)
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--decay", type=float, default=Rule.decay)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fly, plastic = setup(args.control, args.seed, Rule(rate=args.rate, decay=args.decay), args.explore)
    opponent = OPPONENTS[args.opponent](seed=args.seed + 1)
    run = Run(args.run, vars(args))
    run.record({"games": 0, **evaluate(fly, args.eval_games, args.seed + 2)})

    done, next_eval = 0, args.eval_every
    with Progress(args.games, f"train {args.run}") as bar:
        while done < args.games:
            n = min(args.parallel, args.games - done)
            stats = block(fly, plastic, opponent, n, args.reward == "shaped")
            done += n
            bar.update(n, **stats)
            if done >= next_eval or done == args.games:
                next_eval += args.eval_every
                run.checkpoint(plastic, done, args.seed)
                run.record({"games": done, **evaluate(fly, args.eval_games, args.seed + 2), **stats, "depressed": plastic.depressed()})


if __name__ == "__main__":
    main()
