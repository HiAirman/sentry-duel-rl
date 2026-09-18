"""`RL_best040` 移植对手的行为诊断:它到底打得怎么样,观测量对不对。

    python tests/diag_rlbest.py                 # 默认:6 局/对手 + 对照
    python tests/diag_rlbest.py --games 3       # 更省 CPU
    python tests/diag_rlbest.py --no-control --no-fair

## 为什么不用 `tests/duel.py::play_side`

`play_side` 每个行动阶段都新建一个 `ObsBuilder`,而观测构建器**必须一局一个**
(`reset` 只在开局、`first_act_` 只判一次、信念靠跨阶段的 `dilate` 推进)。
用 `play_side` 驱动整回合对手,对手会退化成僵尸:信念永远是开局那一格、
死亡/占点标量恒 0 —— 量出来的强度不是它真实的强度。所以这里自带驱动
`play_game`,builder 一局一个。

## 驱动语义(env 一致,不是"更公平"的版本)

`monet/env/sentry_env.py::_run_opponent_phase` 的循环是

    for _ in range(3 * max_attempts):
        if game.phase_over(): break      # ← 额度用完就在这里退出
        a = gen.send(last)
        res = game.apply(color, a)
        ob.on_observation(...)

也就是说**每个阶段最后一个动作的结果不会再 send 回生成器**(退出时 `gen.close()`)。
整回合对手内部那一步 `on_observation` 因此被丢掉:下一个阶段开头的
`act_start` 会从视图重新取大部分状态,但**信念/情报会陈旧一个动作**。
部署侧的 C++ `rl_act()` 不是这样,它自己驱动引擎、每个动作的观测都吃到。
所以本诊断默认按 env 的真实驱动跑(量出来的就是训练里会遇到的强度),
`--fair` 再跑一遍"最后一个结果也发回去"的版本,两者的差值就是这条差异的代价。

## 对照(必须做的)

同一个网络,喂**固定随机置换后的 412 维**(掩码不变,动作依然合法):
行为退化到近乎乱走、得分率掉到 0 —— 说明分数来自观测语义正确,而不是
"随便动动也能蒙到分"。这条不成立的话,上面所有分数都没有意义。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monet.engine.game import END, Game  # noqa: E402
from monet.env import obs_rlbest as OB  # noqa: E402
from monet.env.obs import ObsBuilder  # noqa: E402
from monet.env.opponents import Opponent  # noqa: E402
from monet.env.rlbest import RLBestOpponent, load_rlbest_net  # noqa: E402
from monet.training.selfplay import STATIC_OPPONENTS  # noqa: E402

# 每个阶段最多允许的尝试次数(被拒的动作不消耗额度,要留重试余量)
MAX_ATTEMPTS = 24


@dataclass
class Result:
    ai_color: str
    ai_score: int = 0
    opp_score: int = 0
    turns: int = 0
    actions: int = 0  # 我方成功消耗额度的动作数
    illegal: int = 0  # 被引擎拒绝的动作数(拒绝是合法的,计数只为看频率)
    hist: List[int] = field(default_factory=lambda: [0] * 8)
    rejects: List[int] = field(default_factory=lambda: [0] * 8)
    obs_checked: int = 0
    obs_violations: List[str] = field(default_factory=list)

    @property
    def win(self) -> bool:
        return self.ai_score > self.opp_score

    @property
    def draw(self) -> bool:
        return self.ai_score == self.opp_score


class ShuffledObsNet:
    """对照:同一个网络,但输入维被固定随机置换(掩码不受影响)。"""

    def __init__(self, net, seed: int = 0):
        self.net = net
        self.perm = np.random.default_rng(seed).permutation(OB.OBS_DIM)

    def logits(self, obs):
        return self.net.logits(np.asarray(obs, dtype=np.float32)[self.perm])


def _check_obs(ob: RLBestOpponent, res: Result) -> None:
    """决策点的观测不变量。任何一条破了都要说出来,而不是只让分数变低。"""
    res.obs_checked += 1
    obs = ob.encode()
    if obs.shape != (OB.OBS_DIM,):
        res.obs_violations.append(f"shape={obs.shape}")
        return
    if not np.isfinite(obs).all():
        res.obs_violations.append("含 nan/inf")
    plane = lambda i: obs[i * 49 : (i + 1) * 49]  # noqa: E731
    if plane(2).sum() != 1.0:
        res.obs_violations.append(f"平面2 我方位置 {plane(2).sum()}")
    if not set(np.unique(plane(3))) <= {0.0, 1.0}:
        res.obs_violations.append("平面3 信念非 0/1")
    if plane(4).sum() > 1.0 or plane(5).sum() > 1.0:
        res.obs_violations.append("平面4/5 多于一个 1")
    if plane(6)[0] != 1.0 or plane(7)[6 * 7 + 6] != 1.0:
        res.obs_violations.append("出生点常量平面不对")
    s = obs[8 * 49 :]
    if s.shape != (20,):
        res.obs_violations.append(f"标量段 {s.shape}")
    elif s.min() < 0.0 or s.max() > 1.6:  # turn/24 在加时里会略过 1
        res.obs_violations.append(f"标量越界 {s.min()}/{s.max()}")


def play_game(
    ai: Opponent,
    other: Opponent,
    ai_color: str = "R",
    fair: bool = False,
    check_obs: bool = False,
) -> Result:
    """一局。双方各自常驻一个 ObsBuilder,生成器逐个交出动作、环境执行。

    `fair=True` 时在阶段结束时把最后一个动作的结果也 `send` 回生成器
    (部署侧 C++ 的真实流程);默认与 `sentry_env._run_opponent_phase` 一致。
    """
    res = Result(ai_color=ai_color)
    game = Game()
    obs = {"R": ObsBuilder(), "B": ObsBuilder()}
    for who in (ai, other):
        reset = getattr(who, "reset", None)
        if callable(reset):
            reset()

    while not game.done:
        c = game.phase
        ob = obs[c]
        ob.act_start(game.view(c), c)
        who = ai if c == ai_color else other
        gen = who.turn(lambda: game.view(c), c, ob)
        last = None
        try:
            for _ in range(MAX_ATTEMPTS):
                if game.phase_over():
                    if fair and last is not None:
                        try:
                            gen.send(last)
                        except StopIteration:
                            pass
                    break
                try:
                    a = gen.send(last)
                except StopIteration:
                    break
                if c == ai_color and check_obs:
                    _check_obs(ai.ob, res)
                r, _ev = game.apply(c, a)
                ob.on_observation(r.observation, r.consumed, a)
                if c == ai_color:
                    res.hist[a] += 1
                    if r.success and r.consumed:
                        res.actions += 1
                    else:
                        res.illegal += 1
                        res.rejects[a] += 1
                last = r
                if a == END:
                    break
        finally:
            gen.close()
        game.end_phase(c)

    res.turns = game.turn
    r_score, b_score = game.score  # (红, 蓝)
    res.ai_score = r_score if ai_color == "R" else b_score
    res.opp_score = b_score if ai_color == "R" else r_score
    return res


def _summary(name: str, results: List[Result]) -> str:
    wins = sum(r.win for r in results)
    draws = sum(r.draw for r in results)
    losses = len(results) - wins - draws
    pts = sum(r.ai_score for r in results) / len(results)
    opp = sum(r.opp_score for r in results) / len(results)
    hist = np.zeros(8, dtype=int)
    for r in results:
        hist += np.asarray(r.hist, dtype=int)
    total = max(1, hist.sum())
    names = ["MOVE", "TN", "TE", "TS", "TW", "FIRE", "SCAN", "END"]
    hist_s = " ".join(f"{n}:{h}({100 * h // total}%)" for n, h in zip(names, hist))
    viol = sum(len(r.obs_violations) for r in results)
    return (
        f"  {name:<18} 胜率 {wins / len(results):.3f} ({wins}胜{draws}平{losses}负)"
        f"  均分 {pts:.2f} : {opp:.2f}\n"
        f"    动作 {hist_s}   被拒 {sum(r.illegal for r in results)}"
        f"   观测校验 {sum(r.obs_checked for r in results)} 次,异常 {viol}"
    )


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description="RL_best040 移植对手诊断")
    ap.add_argument("--games", type=int, default=6, help="每个对手的对局数(上下半场各半)")
    ap.add_argument(
        "--opponents",
        default="baseline,camper,m3_v1,official_baseline",
        help="对手名单(逗号分隔,取自 selfplay.STATIC_OPPONENTS)",
    )
    ap.add_argument("--no-control", action="store_true", help="跳过「置换观测」对照")
    ap.add_argument("--no-fair", action="store_true", help="跳过「最后一个观测也回传」的对照")
    args = ap.parse_args()

    net = load_rlbest_net()
    print("权重:", " ".join(f"{p}{tuple(getattr(net, p).shape)}" for p in
                            ("W0", "b0", "W1", "b1", "W2", "b2")))
    print(f"对局 {args.games} 局/对手,红蓝各半\n")

    names = [n.strip() for n in args.opponents.split(",") if n.strip()]
    for name in names:
        if name not in STATIC_OPPONENTS:
            print(f"  ? 未知对手 {name},可选:{sorted(STATIC_OPPONENTS)}")
            return 2

    summary = {}
    for name in names:
        results = []
        for g in range(args.games):
            color = "R" if g % 2 == 0 else "B"
            other = STATIC_OPPONENTS[name](seed=100 + g)
            results.append(
                play_game(RLBestOpponent(seed=0), other, ai_color=color, check_obs=True)
            )
        summary[name] = results
        print(_summary(f"env 驱动 vs {name}", results))

    if not args.no_fair:
        print("\n最后一个动作的观测也回传(部署侧 C++ 的流程):")
        for name in names[:2]:
            results = [
                play_game(
                    RLBestOpponent(seed=0),
                    STATIC_OPPONENTS[name](seed=100 + g),
                    ai_color="R" if g % 2 == 0 else "B",
                    fair=True,
                    check_obs=True,
                )
                for g in range(args.games)
            ]
            print(_summary(f"fair 驱动 vs {name}", results))

    if not args.no_control:
        print("\n对照:同一权重 + 固定随机置换的 412 维观测(应显著退化):")
        for name in names[:2]:
            results = [
                play_game(
                    RLBestOpponent(net=ShuffledObsNet(net, seed=1), seed=0),
                    STATIC_OPPONENTS[name](seed=100 + g),
                    ai_color="R" if g % 2 == 0 else "B",
                )
                for g in range(args.games)
            ]
            print(_summary(f"置换观测 vs {name}", results))

    allres = [r for rs in summary.values() for r in rs]
    print(
        f"\n合计 {len(allres)} 局:胜率 "
        f"{sum(r.win for r in allres) / len(allres):.3f},均分 "
        f"{sum(r.ai_score for r in allres) / len(allres):.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
