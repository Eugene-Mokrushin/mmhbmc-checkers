import argparse
import time
from dataclasses import dataclass, field

from flycore.board import INITIAL, Move, Position, apply_move
from flycore.moves import legal_moves
from game.players import GreedyPlayer, MinimaxPlayer, Player, RandomPlayer
from progress import Progress

MAX_PLIES = 200


@dataclass
class Game:
    first: Player
    second: Player
    pos: Position = INITIAL
    moves: list[Move] = field(default_factory=list)
    winner: Player | None = None
    over: bool = False

    @property
    def to_move(self) -> Player:
        return self.first if len(self.moves) % 2 == 0 else self.second


def finish(games: list[Game]) -> None:
    for g in games:
        if g.over:
            continue
        if not legal_moves(g.pos):
            g.over, g.winner = True, g.second if g.to_move is g.first else g.first
        elif len(g.moves) >= MAX_PLIES:
            g.over = True


def step(games: list[Game], player: Player) -> None:
    mine = [g for g in games if not g.over and g.to_move is player]
    if not mine:
        return
    for g, move in zip(mine, player.choose_many([g.pos for g in mine])):
        if move not in legal_moves(g.pos):
            raise RuntimeError(f"illegal move {move} after {len(g.moves)} plies")
        g.moves.append(move)
        g.pos = apply_move(g.pos, move)


def tally(games: list[Game], player: Player) -> dict[str, float]:
    over = [g for g in games if g.over]
    if not over:
        return {}
    wins = sum(g.winner is player for g in over)
    losses = sum(g.winner is not None and g.winner is not player for g in over)
    return {"win": wins / len(over), "draw": (len(over) - wins - losses) / len(over), "loss": losses / len(over)}


def play(a: Player, b: Player, n: int, label: str = "games") -> list[Game]:
    if a is b:
        raise ValueError("self-play needs two player objects")
    games = [Game(a, b) if i % 2 == 0 else Game(b, a) for i in range(n)]
    done, ply = 0, 0
    with Progress(n, label) as bar:
        while done < n:
            for player in (a, b):
                finish(games)
                step(games, player)
            finish(games)
            ply += 2
            now = sum(g.over for g in games)
            bar.update(now - done, ply=ply, **tally(games, a))
            done = now
    return games


OPPONENTS = {"random": RandomPlayer, "greedy": GreedyPlayer, "minimax": MinimaxPlayer}


def main() -> None:
    from connectome.load import load
    from game.fly import Fly

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--opponent", choices=OPPONENTS, default="random")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fly = Fly(load(), seed=args.seed)
    start = time.monotonic()
    games = play(fly, OPPONENTS[args.opponent](seed=args.seed + 1), args.games, f"fly vs {args.opponent}")
    result = tally(games, fly)
    plies = sum(len(g.moves) for g in games) / len(games)
    print(f"fly vs {args.opponent}: {len(games)} games, " + ", ".join(f"{k} {v:.1%}" for k, v in result.items()))
    print(f"{plies:.0f} plies per game, {(time.monotonic() - start) / len(games):.1f} s per game")


if __name__ == "__main__":
    main()
