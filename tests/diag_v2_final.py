"""v2.x 终局对比:同一名单、同一种子下,把 v1.0 基线和新权重放在一条采样流上量。

**为什么必须这么量**:`evaluate()` 全程共用一条 `NetPolicy` RNG 流,按对手在
字典里的顺序依次消耗。也就是说"某个对手的读数"取决于它前面跑了多少局 ——
换名单、换顺序、换局数,读数就不可比。所以这里的名单顺序是写死的,报数必须
连名单一起报(见 memory: evaluate-shares-one-sampling-stream)。

用法:
    python tests/diag_v2_final.py [games] [seed]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from monet.env.rules_vg import VGRules  # noqa: E402
from monet.pack import load_pack_net  # noqa: E402
from monet.store import load_checkpoint  # noqa: E402
from monet.training.evaluate import evaluate  # noqa: E402

# 顺序即采样流顺序,**不要重排**(重排 = 换了一个测量口径)
ROSTER = [
    "official_baseline", "official_hunter", "official_stalker", "official_patrol",
    "rl_v5", "m3_v1", "m3_v2", "rl_VG_v0.2", "rl_best040", "rl_VG_v1.0",
]

# 用户的目标对手(前三个 + 冻结的 v1.0)
TARGETS = ["rl_v5", "rl_VG_v0.2", "rl_best040", "rl_VG_v1.0"]


def main() -> int:
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    from monet.training.selfplay import STATIC_OPPONENTS

    def run(net):
        # 对手对象**每次重建**:它们自带 RNG 和隐状态,复用同一个对象等于让第二次
        # 测量从"已经打过 400 局"的位置接着跑 —— A/B 就不再是同一条采样流了。
        # 种子只由 seed 决定(与训练器同一条约定:cfg.seed + 12345),不随局数变。
        pool = {n: STATIC_OPPONENTS[n](seed + 12345) for n in ROSTER}
        res = evaluate(net, pool, games=games, seed=seed, rules=VGRules())
        return {n: res[n]["winrate"] for n in ROSTER}

    base = load_pack_net("rl_VG_v1.0")
    final, _meta, _ = load_checkpoint("runs/v2.0/final.npz")

    print(f"名单({len(ROSTER)}): {', '.join(ROSTER)}   games={games} seed={seed}")
    print(f"{'对手':<20}{'v1.0 基线':>12}{'v2.0':>12}{'差':>10}")
    print("-" * 54)
    b, f = run(base), run(final)
    for n in ROSTER:
        mark = " ★" if n in TARGETS else ""
        print(f"{n:<20}{b[n]:>12.3f}{f[n]:>12.3f}{f[n] - b[n]:>+10.3f}{mark}")
    mb, mf = float(np.mean(list(b.values()))), float(np.mean(list(f.values())))
    print("-" * 54)
    print(f"{'综合':<20}{mb:>12.4f}{mf:>12.4f}{mf - mb:>+10.4f}")
    print()
    ok = all(f[t] >= 0.70 for t in TARGETS)
    for t in TARGETS:
        print(f"  {t:<16}{f[t]:.3f}  {'✓' if f[t] >= 0.70 else '✗'} (需 ≥0.70)")
    print(f"\n目标{'全部达成' if ok else '未达成'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
