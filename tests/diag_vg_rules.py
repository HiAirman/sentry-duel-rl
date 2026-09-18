"""先量后训:在起训权重上量两条规则的命中率,以及规则带来多少棋力。

**为什么要有这个脚本。** 规则写错了不会崩,只会让 AI 悄悄变弱或变懒 —— 在训练里
它表现为"跑得好好的但学不到东西",在部署里表现为"比测出来的弱"。所以在开一轮几
小时的训练之前,先用同一份权重、同一份名单,把 `rules=on` 和 `rules=off` 各测一遍:

- `rules=off` 是**基线**(纯网络),也是与既有 `metrics.csv` 数字可比的口径;
- `rules=on` 是 `rl_VG_v1.0` 的真实形态;
- 两者的差 = 规则单独贡献的棋力;`强制%` 说明规则吃掉了多少决策。

用法:
    python tests/diag_vg_rules.py [--games 100] [--pack rl_VG_v0.2] [--opponents rl_v5,...]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from monet.env.rules_vg import VGRules
from monet.env import obs as O
from monet.pack import load_pack_net, resolve_pack_dir
from monet.training.evaluate import NetPolicy, play_match
from monet.training.selfplay import STATIC_OPPONENTS


class _CountingPolicy:
    """数决策点。**必须实测**:阶段数取决于局有多长(20 回合终局 vs 400 步截断),
    拿"每局 3 个动作 × 猜一个阶段数"去推分母,得到的百分比是编出来的。"""

    def __init__(self, policy):
        self.p = policy
        self.n = 0

    def reset(self):
        self.p.reset()

    def act(self, obs, mask, ob=None):
        self.n += 1
        return self.p.act(obs, mask, ob)


def measure(net, opp_name, games, seed, mode: str):
    """跑一组对局,返回(结果字典, 规则计数 + 决策数)。

    `mode` ∈ {"off", "kill", "scan", "both"}:单条规则的贡献只能这样分开量 ——
    两条一起开时,一个合成的 Δ 说不清是谁干的。
    """
    rules = None if mode == "off" else VGRules(
        enable_kill=mode in ("kill", "both"), enable_scan=mode in ("scan", "both")
    )
    policy = _CountingPolicy(
        NetPolicy(net, temperature=1.0, deterministic=False, seed=seed, rules=rules)
    )
    opp = STATIC_OPPONENTS[opp_name](seed)
    t0 = time.time()
    res = play_match(policy, opp, games=games, seed=seed)
    res["secs"] = time.time() - t0
    counts = {"decisions": policy.n}
    if rules is not None:
        counts.update(
            forced=rules.forced_count, kill=rules.kill_count, scan=rules.scan_count
        )
    return res, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="rl_VG_v0.2", help="被测权重来自哪个参赛包")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--opponents", default="rl_v5,m3_v1,m3_v2,official_hunter")
    args = ap.parse_args()

    print(f"[diag] 权重包 {args.pack} → {resolve_pack_dir(name=args.pack)}")
    net = load_pack_net(name=args.pack, hidden=512)
    print(f"[diag] obs={O.OBS_DIM} 对手={args.opponents} 每格 {args.games} 局\n")

    names = [s for s in args.opponents.split(",") if s]
    hdr = (
        f"{'对手':<16}{'纯网络':>9}{'只开击杀':>10}{'只开扫描':>10}{'两条都开':>10}"
        f"{'强制%':>9}{'击杀/局':>9}{'扫描/局':>9}"
    )
    print(hdr)
    print("-" * 96)
    for n in names:
        out = {}
        for mode in ("off", "kill", "scan", "both"):
            out[mode] = measure(net, n, args.games, args.seed, mode)
        c = out["both"][1]
        per, dec = args.games, max(1, out["both"][1]["decisions"])
        print(
            f"{n:<16}{out['off'][0]['winrate']:>9.3f}{out['kill'][0]['winrate']:>10.3f}"
            f"{out['scan'][0]['winrate']:>10.3f}{out['both'][0]['winrate']:>10.3f}"
            f"{c['forced'] / dec * 100:>8.1f}%"
            f"{c['kill'] / per:>9.2f}{c['scan'] / per:>9.2f}"
        )
    print(
        f"\n『纯网络』才是能与历史 metrics.csv 对比的口径;『两条都开』是 v1.0 的形态。\n"
        f"单列两栏用来定位:哪条规则把胜率拉下去了,就调它(而不是一起调)。\n"
        f"强制% 的分母是实测决策点({out['both'][1]['decisions']} 个),不是估的阶段数×3。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
