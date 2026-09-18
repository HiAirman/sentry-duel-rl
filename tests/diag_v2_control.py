"""v2.x 终局的**对照组**:把 v1.0 和 v2.0 放在与早停判据**完全相同**的口径下量。

早停那一次读数是 `evaluate(net, 四个目标对手, games=stop_confirm_games, seed=cfg.seed+90000)`。
但"v2.0 在四对手上全过线"这件事,只有配上"v1.0 在同一口径下是什么样"才有意义 ——
否则无法区分"记忆确实带来了提升"和"这条口径本身对谁都宽松"。

用法:
    python tests/diag_v2_control.py [games] [seed] [ckpt]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monet.env.rules_vg import VGRules  # noqa: E402
from monet.pack import load_pack_net  # noqa: E402
from monet.store import load_checkpoint  # noqa: E402
from monet.training.evaluate import evaluate  # noqa: E402

# 与早停判据同一份名单、同一顺序
TARGETS = ["rl_v5", "rl_VG_v0.2", "rl_best040", "rl_VG_v1.0"]


def main() -> int:
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 90000
    ckpt = sys.argv[3] if len(sys.argv) > 3 else "runs/v2.0/final.npz"

    from monet.training.selfplay import STATIC_OPPONENTS

    def run(net):
        pool = {n: STATIC_OPPONENTS[n](seed + 12345) for n in TARGETS}
        res = evaluate(net, pool, games=games, seed=seed, rules=VGRules())
        return {n: res[n]["winrate"] for n in TARGETS}

    base = run(load_pack_net("rl_VG_v1.0"))
    final, _m, _a = load_checkpoint(ckpt)
    cand = run(final)

    print(f"名单 {TARGETS}   games={games} seed={seed}   候选={ckpt}")
    print(f"{'对手':<18}{'v1.0':>10}{'候选':>10}{'差':>10}")
    print("-" * 48)
    for n in TARGETS:
        print(f"{n:<18}{base[n]:>10.3f}{cand[n]:>10.3f}{cand[n] - base[n]:>+10.3f}")
    print("-" * 48)
    print(f"{'平均':<18}{sum(base.values())/4:>10.4f}"
          f"{sum(cand.values())/4:>10.4f}"
          f"{(sum(cand.values())-sum(base.values()))/4:>+10.4f}")
    print()
    for label, r in (("v1.0", base), ("候选", cand)):
        miss = [n for n in TARGETS if r[n] < 0.70]
        print(f"  {label:<6}{'全部 ≥0.70' if not miss else '未达:' + '、'.join(miss)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
