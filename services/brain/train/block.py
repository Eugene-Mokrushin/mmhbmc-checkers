import numpy as np

from flycore.board import apply_move
from game.arena import Game, finish, step
from game.fly import WINDOW, Fly
from game.players import Player, material
from train.plasticity import Plasticity

def block(fly: Fly, plastic: Plasticity, opponent: Player, n: int, reward: str) -> dict:
    kc, mbon = fly.mb.members("KC"), fly.mb.members("MBON")
    games = [Game(fly, opponent) if i % 2 == 0 else Game(opponent, fly) for i in range(n)]
    rule = plastic.rule
    traces = np.zeros((n, len(kc)), dtype=np.float32)
    slow = np.zeros((n, len(kc)), dtype=np.float32)
    before, expected, dosed = np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)
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
                traces[i] = rule.decay * traces[i] + (1 - rule.decay) * active[j]
                slow[i] = rule.game_decay * slow[i] + (1 - rule.game_decay) * active[j]
                before[i], expected[i] = material(games[i].pos), scores[j]
                games[i].moves.append(moves[j])
                games[i].pos = apply_move(games[i].pos, moves[j])
        finish(games)
        step(games, opponent)
        finish(games)
        if mine:
            outcomes = np.array([outcome(games[i], before[i], reward) for i in mine])
            plastic.update(traces[mine], outcomes, expected[mine])
        ended = [i for i, g in enumerate(games) if g.over and not dosed[i]]
        if ended and rule.game_dose:
            results = [0 if games[i].winner is None else 1 if games[i].winner is fly else -1 for i in ended]
            plastic.game_over(slow[ended], np.array(results))
        dosed[ended] = True
    wins = sum(g.winner is fly for g in games) / n
    return {"train_win": wins, "kc_active": float(np.mean(kc_active)), "mbon_hz": float(np.mean(mbon_spikes)) / (WINDOW * fly.sim.p.dt)}


def outcome(game: Game, before: float, reward: str) -> float:
    # the small dose after each exchange. shaped: material won or lost;
    # balance: who is ahead, a piece worth a quarter; terminal: nothing
    if reward == "shaped":
        return material(game.pos) - before if not game.over else 0.0
    return material(game.pos) / 4 if reward == "balance" and not game.over else 0.0
