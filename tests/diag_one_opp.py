"""单对手高局数测量:给"某个对手到底掉没掉"这件事一个够紧的置信区间。

名单只有一个对手时,`evaluate` 那条共享采样流只服务它一个人 —— 读数最干净,
且局数可以放心往上堆(没有别的对手在稀释预算)。

用法:
    python tests/diag_one_opp.py <对手名> [games] [seed] [ckpt...]
    # ckpt 省略时默认量 rl_VG_v1.0 包和 runs/v2.0/final.npz 两个
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monet.env.rules_vg import VGRules  # noqa: E402
from monet.pack import load_pack_net  # noqa: E402
from monet.store import load_checkpoint  # noqa: E402
from monet.training.evaluate import evaluate  # noqa: E402


def main() -> int:
    opp = sys.argv[1] if len(sys.argv) > 1 else "rl_v5"
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    ckpts = sys.argv[4:] or ["runs/v2.0/final.npz"]

    from monet.training.selfplay import STATIC_OPPONENTS

    nets = [("rl_VG_v1.0(基线)", load_pack_net("rl_VG_v1.0"))]
    for c in ckpts:
        net, _m, _a = load_checkpoint(c)
        nets.append((c, net))

    print(f"对手 {opp}   games={games} seed={seed}")
    print(f"{'权重':<28}{'得分率':>10}{'±1SE':>9}{'胜/平/负':>16}")
    print("-" * 64)
    for name, net in nets:
        pool = {opp: STATIC_OPPONENTS[opp](seed + 12345)}
        r = evaluate(net, pool, games=games, seed=seed, rules=VGRules())[opp]
        wr = r["winrate"]
        se = math.sqrt(max(wr * (1 - wr), 1e-9) / games)
        print(f"{name:<28}{wr:>10.3f}{se:>9.3f}"
              f"{'%d/%d/%d' % (r['win'], r['draw'], r['loss']):>16}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
