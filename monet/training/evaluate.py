"""评测:让策略与指定对手对打若干局,红蓝各半。"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..env.opponents import Opponent
from ..env.sentry_env import SentryEnv
from ..models.mlp import MLP
from ..env import obs as O


class NetPolicy:
    """把裸网络包成可选温度的采样策略(与部署侧的温度采样一致)。

    `rules` 非空时变成 rl_VG_v1.0 的混合策略:每个决策点先问规则,命中就用收窄后的
    掩码采样,没命中则关掉 SCAN 再交给网络。默认 None = 纯网络。

    带规则和不带规则是**两个策略**,得分率不可直接比较。联盟里的陪练(rl_v5 /
    m3_v1 / m3_v2 / rl_VG_v0.2)与快照对手都必须传 None —— 给它们加规则,量到的
    就不是这些对手本来的强度了。
    """

    def __init__(
        self,
        net: MLP,
        temperature: float = 1.0,
        deterministic: bool = False,
        seed: int = 0,
        rules=None,
    ):
        self.net = net
        self.temperature = temperature
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        self.rules = rules
        # 带记忆的网络必须在**一局之内**把隐状态传下去。丢掉它等于把 RNN 当
        # 无记忆的 MLP 用:不报错、不崩,只是评测出来的分数比训练时低一截
        # ("离线测的 ≠ 发出去的",本仓库反复踩的坑)。
        self.recurrent = getattr(net, "is_recurrent", False)
        self.h = None

    def reset(self) -> None:
        """一局开始。规则 2 的失明计数必须跟着归零,否则会跨局累加。"""
        if self.rules is not None:
            self.rules.reset()
        if self.recurrent:
            self.h = self.net.new_hidden()

    def _logits(self, obs: np.ndarray) -> np.ndarray:
        if not self.recurrent:
            return self.net.logits(obs)
        if self.h is None:      # 没走 reset 就直接 act(老调用方):补一个零初值
            self.h = self.net.new_hidden()
        lg, _v, self.h = self.net.step(obs, self.h)
        return lg

    def act(self, obs: np.ndarray, mask: np.ndarray, ob=None) -> int:
        if self.rules is not None and ob is not None:
            mask, _ = self.rules.act_mask(ob, mask)
        lg = self._logits(obs)
        lg = np.where(mask > 0, lg, -1e9)
        if self.deterministic:
            return int(np.argmax(lg))
        if self.temperature <= 1e-3:
            return int(np.argmax(lg))
        z = (lg - lg.max()) / self.temperature
        p = np.exp(z)
        p /= p.sum()
        return int(self.rng.choice(len(p), p=p))


def play_match(
    policy,
    opponent: Opponent,
    games: int = 100,
    seed: int = 0,
    max_steps: int = 400,
) -> Dict[str, float]:
    """红蓝各打一半,返回胜/平/负与净胜分。

    `max_steps` 打满还没分出胜负的局按**平局**计(`env.result()` 在无 winner 时
    返回 0.5),所以这个上限会直接改写 `winrate`。它必须与训练侧一致,不然评测
    得分率和训练日志里的 `ep_win` 不是同一个口径。
    """
    env = SentryEnv(opponent, agent_color=None, seed=seed, max_steps=max_steps)
    win = draw = loss = 0
    score_for = score_against = 0
    returns: List[float] = []
    lengths: List[int] = []
    for g in range(games):
        color = "R" if g % 2 == 0 else "B"
        obs, mask, info = env.reset(seed=seed + g, agent_color=color)
        if hasattr(policy, "reset"):
            policy.reset()  # 规则的失明计数必须逐局清零,否则跨局累加(见 NetPolicy.reset)
        total = 0.0
        n = 0
        while True:
            a = policy.act(obs, mask, env.ag_ob)
            obs, mask, r, term, trunc, info = env.step(a)
            total += r
            n += 1
            if term or trunc:
                break
        res = env.result()
        win += res == 1.0
        draw += res == 0.5
        loss += res == 0.0
        score_for += info["my_score"]
        score_against += info["opp_score"]
        returns.append(total)
        lengths.append(n)
    n = max(1, games)
    return {
        "games": games,
        "win": win,
        "draw": draw,
        "loss": loss,
        "winrate": (win + 0.5 * draw) / n,  # 与排行榜一致的综合得分率
        "score_for": score_for / n,
        "score_against": score_against / n,
        "net_score": (score_for - score_against) / n,
        "return": float(np.mean(returns)) if returns else 0.0,
        "length": float(np.mean(lengths)) if lengths else 0.0,
    }


def evaluate(
    net: MLP,
    opponents: Dict[str, Opponent],
    games: int = 40,
    seed: int = 0,
    deterministic: bool = False,
    temperature: float = 1.0,
    rules=None,
) -> Dict[str, Dict[str, float]]:
    """`rules` 非空 = 按 rl_VG_v1.0 的混合策略评测。

    带规则与不带规则的数字不可直接比较。要 A/B 就保持名单一致、只切 `rules`。
    """
    # 全程共用**这一条** policy 随机流,按 opponents 的字典顺序依次消耗 —— 所以
    # "某个对手的胜率"取决于它前面跑了多少局:换名单、换顺序、换局数,读数就不可比
    # (实测同一份权重对 rl_v5 落在 0.650~0.775)。报数必须连名单 + 局数 + 种子一起报;
    # 要最干净的读数就用 tests/diag_one_opp.py 的单对手名单,局数可以放心堆。
    #
    # 推论:同种子**不等于**逐局配对。棋盘对两边是同一份(play_match 按 seed+g 重播
    # 环境),但一局消耗几个随机数取决于局长,而局长取决于策略本身 —— 两个不同的
    # 网络跑到第 k 个对手时,流的位置已经错开了。想压方差只能堆局数。
    policy = NetPolicy(
        net, temperature=temperature, deterministic=deterministic, seed=seed, rules=rules
    )
    out = {}
    for name, opp in opponents.items():
        opp.reset()
        out[name] = play_match(policy, opp, games=games, seed=seed)
    return out


def summary_line(name: str, res: Dict[str, float]) -> str:
    return (
        f"{name:<10} 得分率 {res['winrate']:.3f}  "
        f"({res['win']:.0f}胜/{res['draw']:.0f}平/{res['loss']:.0f}负)  "
        f"净胜分 {res['net_score']:+.2f}  回报 {res['return']:+.2f}"
    )
