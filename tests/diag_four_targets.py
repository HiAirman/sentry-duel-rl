"""四个目标的胜率测量:rl_v5 / rl_VG_v0.2 / rl_best040 / rl_VG_v1.0。

**为什么名单必须钉死成这四个、且写在这个文件里。** `evaluate()` 里 `NetPolicy` 只有
一条采样流,按 `opponents` 的 **dict 顺序**逐个对手取随机数(`evaluate.py`),而一局
要抽多少个随机数取决于**这一局打了多长** —— 也就是取决于策略本身。所以:

  * 换一个对手、换一个顺序、加减一个名字,后面所有对手的读数都会跟着变;
  * 两个不同的网络**即使种子相同也不是配对比较**,差值的方差不能按配对算。

这不是缺陷,是评测口径(`README` §五)。代价是任何一次报数都必须带上"名单 + 局数 +
种子"三件套,否则数字之间不可比 —— 本文件存在的意义就是把这三件套固定下来。

对手种子取 `cfg.seed + 12345`,与训练器 `_eval_opponents()` 一致,这样这里的读数
和训练日志里的 `eval_*` 列是同一个口径。

用法:
    python tests/diag_four_targets.py [games] [seed] [--rules off] [ckpt...]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monet.env.rules_vg import VGRules  # noqa: E402
from monet.pack import load_pack_net  # noqa: E402
from monet.store import load_checkpoint  # noqa: E402
from monet.training.evaluate import evaluate  # noqa: E402
from monet.training.selfplay import STATIC_OPPONENTS  # noqa: E402

# 顺序即口径:改动这个列表等于改动全部读数,任何历史数字都不再可比。
TARGETS = ["rl_v5", "rl_VG_v0.2", "rl_best040", "rl_VG_v1.0"]

# 与训练器 `SelfPlay._eval_opponents()` 的 `cfg.seed + 12345` 对齐(两次跑都是 seed 0)。
OPP_SEED = 0 + 12345

DEFAULT_CKPTS = [
    "runs/v1.1/final.npz",
    "runs/v2.2/final.npz",
    "runs/v2.0/final.npz",
]


def _winrate_se(r, games: int) -> float:
    """得分率的标准误。单局得分 ∈ {0, 0.5, 1},所以用二阶矩算,不用二项近似。

    `winrate = (win + 0.5*draw)/n` 的分布不是二项(平局是半个胜场),拿
    `sqrt(wr*(1-wr)/n)` 会**低估**标准误,正好在"要不要判它达标"的边界上骗人。
    """
    wr = r["winrate"]
    e_x2 = (r["win"] * 1.0 + r["draw"] * 0.25) / games
    var = max(e_x2 - wr * wr, 0.0)
    return math.sqrt(var / games)


def main() -> int:
    argv = sys.argv[1:]
    use_rules = True
    if "--rules" in argv:
        i = argv.index("--rules")
        use_rules = argv[i + 1].lower() != "off"
        del argv[i : i + 2]
    games = int(argv[0]) if len(argv) > 0 else 400
    seed = int(argv[1]) if len(argv) > 1 else 90000
    ckpts = argv[2:] or DEFAULT_CKPTS

    nets = [("rl_VG_v1.0(已交付包)", load_pack_net("rl_VG_v1.0"))]
    for c in ckpts:
        net, meta, _a = load_checkpoint(c)
        cfg = json.loads(meta["cfg"]) if meta and "cfg" in meta else {}
        nets.append((f"{c} [step {meta.get('step', '?')} {cfg.get('init_pack', '?')}]"
                     if meta else c, net))

    print(f"名单 {TARGETS}   games={games}   seed={seed}   "
          f"对手种子={OPP_SEED}   规则={'on' if use_rules else 'off'}")
    print("-" * 78)
    print(f"{'权重':<46}" + "".join(f"{t[:11]:>12}" for t in TARGETS) + f"{'均值':>9}{'最低':>8}")
    print("-" * 78)

    for name, net in nets:
        pool = {n: STATIC_OPPONENTS[n](OPP_SEED) for n in TARGETS}
        res = evaluate(net, pool, games=games, seed=seed,
                       rules=VGRules() if use_rules else None)
        wr = [res[t]["winrate"] for t in TARGETS]
        cells = "".join(f"{res[t]['winrate']:>8.3f}±{_winrate_se(res[t], games):.3f}"
                        for t in TARGETS)
        print(f"{name:<46}{cells}{sum(wr) / len(wr):>9.3f}{min(wr):>8.3f}")
        print(f"{'':<46}" + "".join(
            f"{'%d/%d/%d' % (res[t]['win'], res[t]['draw'], res[t]['loss']):>12}"
            for t in TARGETS))
    print("-" * 78)
    print("胜/平/负 行在该权重下方。达标线 0.7 —— 判读时看 ±,不要只看点估计。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
