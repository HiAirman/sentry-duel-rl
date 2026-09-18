"""记忆到底贡献了多少?—— 把 P/bP 强制归零(等价于"有记忆层但不用它")再测一遍。

`feat = a3 + P·h + bP`,所以 `P=bP=0` 时前向逐位退化成**纯编码器**:网络结构还在、
参数还在,只是记忆那一支的输出被关掉。这是个干净的消融 —— 两组权重只差这一项。

如果消融组的分数与完整组几乎一样,说明这一轮的成绩来自"继续在含 rl_VG_v1.0 的
联盟里训练",而不是来自记忆本身。这个结论要写进报告,不能只报好消息。

用法:
    python tests/diag_mem_ablation.py [games] [seed] [ckpt|pack名]

第三项缺省是包 `rl_VG_v2.0`;给一个 `runs/…/final.npz` 之类的检查点路径也行
(还没打包的候选就只能这样量)。
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

TARGETS = ["rl_v5", "rl_VG_v0.2", "rl_best040", "rl_VG_v1.0"]

# 检查点里藏着 GRU 隐层宽度,第三项若是路径就不该猜 —— 读出来直接用。
SOURCE = "rl_VG_v2.0"


def load_twice():
    """返回两份**互不共享内存**的同权重网络(完整组 / 消融组)。

    必须真的读两遍而不是 `copy.deepcopy`:要消融的是 `p` 里的两个数组,
    共享了内存就变成"把两组一起归零",量出来两组一模一样 —— 那正是这个脚本
    最不该得出的结论。
    """
    # `is_file()` 而不是 `exists()`:包名 `rl_VG_v2.0` 在仓库里是个**目录**,
    # 用 exists 判断会把它当检查点送进 np.load,报一句看不懂的 PermissionError。
    if Path(SOURCE).is_file():
        a, _m, _arch = load_checkpoint(SOURCE)
        b, _m2, _arch2 = load_checkpoint(SOURCE)
        return a, b
    return load_pack_net(name=SOURCE), load_pack_net(name=SOURCE)


def rel_memory_magnitude(net, steps: int = 60, seed: int = 0) -> float:
    """实测 |P·h + bP| 相对 |a3| 的比值 —— 记忆那一支在 feat 里占多大分量。"""
    rng = np.random.default_rng(seed)
    h = net.new_hidden()
    ratios = []
    for _ in range(steps):
        obs = rng.normal(0, 1, net.obs_dim).astype(np.float32)
        X = obs.reshape(1, -1)
        a3 = net.enc.encode(X, cache=False)
        _H, h = net.gru.forward(a3.reshape(1, 1, -1), h, cache=False)
        mem = h @ net.p["P"].T + net.p["bP"]     # h 是 (1,G),P 是 (H,G)
        ratios.append(float(np.abs(mem).mean() / (np.abs(a3).mean() + 1e-9)))
    return float(np.mean(ratios))


def main() -> int:
    global SOURCE
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 90000
    if len(sys.argv) > 3:
        SOURCE = sys.argv[3]

    from monet.training.selfplay import STATIC_OPPONENTS

    full, ablated = load_twice()
    print(f"权重来源 {SOURCE}")
    print(f"记忆支 |P·h+bP| / |a3| 实测比值 = {rel_memory_magnitude(full):.4f}")

    ablated.p["P"][:] = 0.0
    ablated.p["bP"][:] = 0.0

    def run(net):
        # 对手对象是**有状态**的(内部 RNG 会随局数前进),复用同一池会让第二次
        # 测量从已经推进过的流位置开始 —— 两组比的就不再是同一个口径了。
        pool = {n: STATIC_OPPONENTS[n](seed + 12345) for n in TARGETS}
        r = evaluate(net, pool, games=games, seed=seed, rules=VGRules())
        return {n: r[n]["winrate"] for n in TARGETS}

    a, b = run(full), run(ablated)
    print(f"名单 {TARGETS}  games={games} seed={seed}")
    print(f"{'对手':<18}{'带记忆':>10}{'P=bP=0':>10}{'差':>10}")
    print("-" * 48)
    for n in TARGETS:
        print(f"{n:<18}{a[n]:>10.3f}{b[n]:>10.3f}{a[n] - b[n]:>+10.3f}")
    print("-" * 48)
    print(f"{'平均':<18}{sum(a.values())/4:>10.4f}{sum(b.values())/4:>10.4f}"
          f"{(sum(a.values())-sum(b.values()))/4:>+10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
