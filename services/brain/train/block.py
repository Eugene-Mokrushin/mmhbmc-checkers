import numpy as np

from flycore.board import apply_move
from game.arena import Game, finish, step
from game.fly import WINDOW, Fly
from game.players import Player, material
from train.plasticity import Plasticity

WIN = 3.0


def block(fly: Fly, plastic: Plasticity, opponent: Player, n: int, shaped: bool) -> dict:
    kc, mbon = fly.mb.members("KC"), fly.mb.members("MBON")
    games = [Game(fly, opponent) if i % 2 == 0 else Game(opponent, fly) for i in range(n)]
    traces = np.zeros((n, len(kc)), dtype=np.float32)
    before, expected = np.zeros(n), np.zeros(n)
    kc_active, mbon_spikes = [], []
    while not all(g.over for g in games):
        finish(games)
        mine = [i for i, g in enumerate(games) if not g.over and g.to_move is fly]
        if mine:
            moves, counts = fly.choose_recorded([games[i].pos for i in mine])
            active = counts[:, kc] > 0
            kc_active.append(active.mean())
            mbon_spikes.append(counts[:, mbon].mean())
            scores = counts[:, mbon] @ fly.valence
            for j, i in enumerate(mine):
                traces[i] = plastic.rule.decay * traces[i] + (1 - plastic.rule.decay) * active[j]
                before[i], expected[i] = material(games[i].pos), scores[j]
                games[i].moves.append(moves[j])
                games[i].pos = apply_move(games[i].pos, moves[j])
        finish(games)
        step(games, opponent)
        finish(games)
        if mine:
            reward = np.array([outcome(games[i], fly, before[i], shaped) for i in mine])
            plastic.update(traces[mine], reward, expected[mine])
    wins = sum(g.winner is fly for g in games) / n
    return {"train_win": wins, "kc_active": float(np.mean(kc_active)), "mbon_hz": float(np.mean(mbon_spikes)) / (WINDOW * fly.sim.p.dt)}


def outcome(game: Game, fly: Fly, before: float, shaped: bool) -> float:
    if game.over:
        return 0.0 if game.winner is None else WIN if game.winner is fly else -WIN
    return material(game.pos) - before if shaped else 0.0
