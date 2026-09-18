"""对手之间的对拍工具:两边都用同一套"生成器回合"协议驱动引擎。

用来回答"移植过来的官方 AI 到底强不强"这类问题 —— 与训练无关,纯对拍。
    python tests/duel.py            # 默认全体循环赛
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monet.engine.game import Game  # noqa: E402
from monet.env import obs as O  # noqa: E402
from monet.env.obs import ObsBuilder  # noqa: E402
from monet.training.selfplay import STATIC_OPPONENTS  # noqa: E402


def play_side(game: Game, color: str, opp) -> None:
    """让 opp 走完它在 color 侧的一整个行动阶段。"""
    ob = ObsBuilder()
    ob.act_start(game.view(color), color)
    gen = opp.turn(lambda: game.view(color), color, ob)
    last = None
    try:
        for _ in range(24):
            if game.phase_over():
                break
            try:
                a = gen.send(last)
            except StopIteration:
                break
            res, _ = game.apply(color, a)
            ob.on_observation(res.observation, res.consumed, a)
            last = res
            if a == O.END:
                break
    finally:
        gen.close()
    game.end_phase(color)


def duel(name_a: str, name_b: str, games: int = 40, seed: int = 0) -> dict:
    wins = draws = 0
    diff = 0
    for g in range(games):
        color = "R" if g % 2 == 0 else "B"
        a = STATIC_OPPONENTS[name_a](seed + g)
        b = STATIC_OPPONENTS[name_b](seed + g)
        game = Game()
        for _ in range(64):
            if game.done:
                break
            play_side(game, game.phase, a if game.phase == color else b)
        if game.winner == color:
            wins += 1
        elif game.winner is None:
            draws += 1
        r, bl = game.score
        mine, theirs = (r, bl) if color == "R" else (bl, r)
        diff += mine - theirs
    return {"winrate": wins / games, "drawrate": draws / games, "diff": diff / games}


def main() -> int:
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else [
        "baseline",
        "hunter",
        "camper",
        "official_baseline",
        "official_hunter",
    ]
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    print(f"循环赛 {games} 局/对,每格 = 胜率/平局率/净胜分(行方视角)\n")
    print(" " * 19 + "".join(f"{n:>21}" for n in names))
    for a in names:
        cells = []
        for b in names:
            if a == b:
                cells.append(" " * 21)
                continue
            r = duel(a, b, games)
            cells.append(f"{r['winrate']:.2f}/{r['drawrate']:.2f}/{r['diff']:+.1f}".rjust(21))
        print(f"{a:<19}" + "".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
