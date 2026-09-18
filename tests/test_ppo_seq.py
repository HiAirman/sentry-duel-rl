"""PPOSeqAgent:记忆层能不能真的被训到。

这一版 PPO 与 `ppo.py` 的差异只有"批怎么切",而**切错不会报错**:
* 段跨了局 -> 上一局末尾的隐状态漏进下一局开局;
* `h0` 存成了"离开这一步"的状态 -> 段的起点比它该在的位置晚一步,记忆被平移;
* 补零的步没被 `valid` 剔出归一化 -> 损失随每批补齐多少而缩放,学得慢且不规律;
* 梯度缓冲没共享 -> Adam 看不到 GRU 的梯度,权重永远不动(已修,见
  tests/test_rnn_policy.py)。

以上每一条的症状都是"加了 RNN 但没变强"。所以这里的判据是**数值**的:
`update` 算出来的梯度必须等于它声称在优化的那个标量目标的数值梯度。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monet.models.ppo import compute_gae, log_softmax, masked_logits  # noqa: E402
from monet.models.ppo_seq import PPOSeqAgent, SeqRollout, episode_segments  # noqa: E402
from monet.models.rnn_policy import RNNPolicy  # noqa: E402

_SMALL = dict(obs_dim=5, hidden=6, act_dim=3, gru_hidden=3)


def _net(seed=4):
    return RNNPolicy(seed=seed, **_SMALL)


def _to_f64(net):
    """转 float64 并保持合并视图共享(不能写 `net.p[k] = ...`,那会把共享打断,
    有限差分就会扰动到没人读的副本上,整条检查空转通过)。"""
    for k in list(net.enc.p):
        net.enc.p[k] = net.enc.p[k].astype(np.float64)
    for k in list(net.gru.p):
        net.gru.p[k] = net.gru.p[k].astype(np.float64)
    net._P = net._P.astype(np.float64)
    net._bP = net._bP.astype(np.float64)
    net._relink()
    net.zero_grad()
    return net


# --------------------------------------------------------------------- 分段

def test_segments_partition_the_rollout_and_never_cross_a_done():
    """段必须**恰好铺满** rollout:不重、不漏、不跨局。

    漏了 -> 那些步一个 epoch 里根本没被训练(而且是静默漏);
    重了 -> 同一批数据被算两遍,等价于给那段加权;
    跨局 -> 把上一局末尾的隐状态带进新的一局,而新局开局必须归零。
    """
    dones = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)  # 局:4,3,4
    segs = episode_segments(dones, 2)
    assert segs == [(0, 2), (2, 4), (4, 6), (6, 7), (7, 9), (9, 11)], segs

    # 铺满且不重叠
    covered = []
    for a, b in segs:
        assert 0 <= a < b <= len(dones)
        covered.extend(range(a, b))
    assert sorted(covered) == list(range(len(dones))), "段没有恰好覆盖 rollout"
    assert len(covered) == len(set(covered)), "段之间有重叠"

    # 每个段都落在**同一局**内部:段区间里不能含有 done(最后一步除外)
    for a, b in segs:
        assert not np.any(dones[a:b - 1] > 0), f"段 {(a, b)} 跨了局"


def test_a_long_episode_is_split_into_many_segments():
    """一局比 seg_len 长得多时要切满,且最后一段不足额。"""
    dones = np.zeros(10, dtype=np.float32)
    dones[-1] = 1.0
    segs = episode_segments(dones, 4)
    assert segs == [(0, 4), (4, 8), (8, 10)], segs


def test_stored_h0_reproduces_the_sampled_logp():
    """**这条是本文件的主判据。**

    采样时存的 `h` 必须是"**进入**这一步时"的隐状态。把它存成"离开这一步"的
    (即误用 `h_new`)是个极容易犯、又完全不报错的错:段的起点整体后移一步,
    记忆被平移、段与段之间的信息接错,训练照常跑,只是永远学不到东西。

    检验方式:拿存下来的 `h0` 重新跑一遍整段前向,必须逐字复现当时采样出的
    `logp`。h0 错位一步就对不上。
    """
    net = _net(seed=1)
    rng = np.random.default_rng(4)
    # P 必须非零,否则记忆通路整条恒为 0,这条测试会退化成"两条路输出都是 a3 的头",
    # 存什么 h0 都能对上。
    net.p["P"][:] = rng.standard_normal(net.p["P"].shape).astype(np.float32) * 0.3
    agent = PPOSeqAgent(net, seg_len=2, segs_per_mb=2, seed=0)

    dones = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]      # 局:4,3,4 -> 段长 2 会切进局内部
    roll = SeqRollout()
    h = net.new_hidden()
    for t in range(len(dones)):
        obs = rng.standard_normal(_SMALL["obs_dim"]).astype(np.float32)
        mask = np.ones(_SMALL["act_dim"], np.float32)
        mask[t % _SMALL["act_dim"]] = 0.0
        a, logp, v, h_new = agent.act(obs, mask, h)
        roll.add(obs, mask, a, logp, v, 0.0, dones[t], h)
        h = net.new_hidden() if dones[t] else h_new

    obs, mask, act, old_logp, _v, _r, dones_a, hs = roll.arrays()
    segs = episode_segments(dones_a, 2)
    assert (2, 4) in segs, "这一局的第 2 段必须落在局**内部**,否则测不到错位的 h0"

    for a, b in segs:
        lg, _ = net.forward_seq(obs[a:b][None], hs[a][None], cache=False)
        lp = log_softmax(masked_logits(lg[0], mask[a:b]))
        got = lp[np.arange(b - a), act[a:b]]
        assert np.allclose(got, old_logp[a:b], atol=1e-5), (
            f"段 {(a, b)} 用存下来的 h0 复现不出当时的 logp:"
            f"最大偏差 {np.abs(got - old_logp[a:b]).max():.3e}。"
            f"`h` 大概是存成了离开这一步的状态(应存进入时的)。"
        )

        # 反向对照:把 h0 挪后一步**必须**对不上。这一条是防"测试空转"的 ——
        # 若 P 太接近零、或 h 根本没进前向,上面那种"复现成功"是白给的。
        if a + 1 < len(hs):
            lg2, _ = net.forward_seq(obs[a:b][None], hs[a + 1][None], cache=False)
            lp2 = log_softmax(masked_logits(lg2[0], mask[a:b]))
            got2 = lp2[np.arange(b - a), act[a:b]]
            assert not np.allclose(got2, old_logp[a:b], atol=1e-5), (
                f"段 {(a, b)} 用一个**错位**的 h0 也能复现 logp —— 这条测试是空转的,"
                f"隐状态根本没影响输出(检查 P 是否为零)"
            )


# --------------------------------------------------------------------- 梯度

class _CaptureGrad(PPOSeqAgent):
    """只抓梯度、不更新参数 —— 有限差分要比较的是**同一个**参数点上的梯度。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.captured = None

    def _clip_and_step(self):
        self.captured = {k: v.copy() for k, v in self.net.g.items()}
        return 0.0


def _ref_objective(net, obs, mask, act, olp, adv, ret, h0, clip, vf, ent):
    """`update` 声称在最小化的标量,用**独立**代码重写一遍。

    全段铺满、无补齐(Vld 全 1),所以 Nv == L,批量均值就是 `.mean()`。
    """
    X = obs[None].astype(np.float64)
    M = mask[None].astype(np.float64)
    Aa = act[None]
    OLP = olp[None].astype(np.float64)
    ADV = adv[None].astype(np.float64)
    R = ret[None].astype(np.float64)

    logits, vpred = net.forward_seq(X, h0, cache=False)
    lp_all = log_softmax(masked_logits(logits, M))
    lp_a = np.take_along_axis(lp_all, Aa[..., None], axis=-1)[..., 0]
    p = np.exp(lp_all)

    ratio = np.exp(lp_a - OLP)
    surr1 = ratio * ADV
    surr2 = np.clip(ratio, 1.0 - clip, 1.0 + clip) * ADV
    policy_loss = -np.minimum(surr1, surr2).mean()
    ent_row = -(p * lp_all).sum(axis=-1)
    value_loss = 0.5 * ((vpred - R) ** 2).mean()
    return policy_loss + vf * value_loss - ent * float(ent_row.mean())


def test_update_gradient_matches_finite_differences():
    """`update` 的解析梯度 == 它声称在优化的那个目标的数值梯度。

    一次覆盖:GAE/优势标准化的接法、ratio/surr/clip、**行熵**(不是批量均值熵)、
    价值项、以及 `w = Vld/Nv` 的归一化。符号写反在训练里只表现为"变弱",看不出来。
    """
    net = _to_f64(_net(seed=7))
    rng = np.random.default_rng(31)
    for k in net.p:                       # 别让 P 还是零:那样 GRU 整条无梯度
        net.p[k] += rng.standard_normal(net.p[k].shape) * 0.05

    agent = _CaptureGrad(net, seg_len=4, segs_per_mb=1, epochs=1, seed=0)
    clip, vf, ent = agent.clip, agent.vf_coef, agent.ent_coef
    last_value = 0.37

    L = 4
    roll = SeqRollout()
    h = net.new_hidden()
    dones = [0, 0, 0, 1]
    for t in range(L):
        obs = rng.standard_normal(_SMALL["obs_dim"]).astype(np.float32)
        mask = np.ones(_SMALL["act_dim"], np.float32)
        mask[(t + 1) % _SMALL["act_dim"]] = 0.0
        a, logp, v, h_new = agent.act(obs, mask, h)
        roll.add(obs, mask, a, logp, v, float(rng.standard_normal()), dones[t], h)
        h = net.new_hidden() if dones[t] else h_new

    agent.update(roll, last_value)
    assert agent.captured is not None, "update 没有走到 _clip_and_step"

    obs, mask, act, olp, values, rewards, dones_a, hs = roll.arrays()
    adv, ret = compute_gae(rewards, dones_a, values, last_value, agent.gamma, agent.lam)
    adv_norm = ((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float64)

    def objective():
        return _ref_objective(net, obs, mask, act, olp, adv_norm, ret,
                              hs[0][None], clip, vf, ent)

    # 中心差分的绝对噪声下限是 eps_mach*|f|/(2h) ≈ 2.2e-16/2e-5 ≈ 1e-11,而截断误差
    # 是 O(h^2)。所以判据用 atol+rtol 的组合形式,不能只按每个张量自己的最大值做
    # 相对比较:像 W_hr 这种梯度本来就小的张量(max≈5e-5),1e-10 的差分噪声就能
    # 换算成 2e-6 的"相对误差",那不是错,是浮点下限。
    eps = 1e-5
    atol, rtol = 1e-8, 1e-6
    worst = (0.0, None)
    for key in net.p:
        flat = net.p[key].reshape(-1)
        num = np.zeros(flat.size)
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + eps
            lp = objective()
            flat[i] = old - eps
            lm = objective()
            flat[i] = old
            num[i] = (lp - lm) / (2 * eps)
        num = num.reshape(net.p[key].shape)
        nmax = np.abs(num).max()
        # 绊线:数值梯度恒为零 => 这条比较是空转的,不能算通过。
        assert nmax > 1e-9, f"{key} 的数值梯度恒为零,这条检查没有测到东西"
        diff = np.abs(agent.captured[key] - num).max()
        rel = diff / nmax
        if rel > worst[0]:
            worst = (rel, key)
        assert diff < atol + rtol * nmax, (
            f"{key} 的解析梯度与数值梯度不符:绝对差 {diff:.3e}(相对 {rel:.3e},"
            f"容差 {atol + rtol * nmax:.3e};解析 max={np.abs(agent.captured[key]).max():.3e},"
            f"数值 max={nmax:.3e})"
        )
    print(f"    (最大相对误差 {worst[0]:.2e} @ {worst[1]})")


# ------------------------------------------------------------------ 端到端

def test_update_reaches_the_gru_after_the_first_step():
    """跑两次 update,第二次必须动到 GRU 的参数。

    这是"记忆层到底有没有被训"的端到端判据,也是 `net.g` 与 `gru.g` 不共享那个
    真错误的回归绊线 —— 那时 `P`/`W3` 照常更新(它们写在 `net.g` 里),但
    `W_ir/W_hn/...` 的梯度恒为零,两轮、二十轮之后都纹丝不动,且不报任何错。

    为什么是**第二次**:P 零初始化,所以第一步 `dH = dfeat @ P = 0`,GRU 本来就
    拿不到梯度(这正是"初始化等于 v1.0"的代价,是设计而非缺陷)。第一步更新完 P
    离开零,第二步 GRU 才开始学。
    """
    net = _net(seed=11)
    agent = PPOSeqAgent(net, seg_len=4, segs_per_mb=2, seed=1, lr=1e-3)
    rng = np.random.default_rng(2)

    def one_rollout():
        roll = SeqRollout()
        h = net.new_hidden()
        for t in range(12):
            obs = rng.standard_normal(_SMALL["obs_dim"]).astype(np.float32)
            mask = np.ones(_SMALL["act_dim"], np.float32)
            a, logp, v, h_new = agent.act(obs, mask, h)
            done = 1 if t % 6 == 5 else 0
            roll.add(obs, mask, a, logp, v, float(rng.standard_normal()), done, h)
            h = net.new_hidden() if done else h_new
        return roll

    assert np.abs(net.p["P"]).max() == 0, "P 必须是零初始化"
    agent.update(one_rollout(), 0.0)
    assert np.abs(net.p["P"]).max() > 0, "第一次 update 之后 P 必须离开零"

    before_gru = {k: net.gru.p[k].copy() for k in net.gru.p}
    agent.update(one_rollout(), 0.0)
    moved = [k for k in net.gru.p if not np.array_equal(net.gru.p[k], before_gru[k])]
    assert moved, (
        "第二次 update 之后 GRU 的参数一个都没动 —— 记忆层根本没被训练。"
        "先查 net.g 与 gru.g 是不是同一批数组(tests/test_rnn_policy.py 有一条专测)。"
    )
    print(f"    (GRU 中被更新到的张量:{len(moved)}/{len(net.gru.p)})")


def test_update_runs_on_a_rollout_with_ragged_episodes():
    """段长不整除局长时的冒烟:补齐、valid 掩码、不足一段的尾段都得能跑。

    局长为 5/1/9,seg_len=4 —— 尾段分别是 1、1、1 步,走的是补齐那条路径。
    """
    net = _net(seed=5)
    agent = PPOSeqAgent(net, seg_len=4, segs_per_mb=3, seed=3)
    rng = np.random.default_rng(8)

    dones = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    roll = SeqRollout()
    h = net.new_hidden()
    for t in range(len(dones)):
        obs = rng.standard_normal(_SMALL["obs_dim"]).astype(np.float32)
        mask = np.ones(_SMALL["act_dim"], np.float32)
        a, logp, v, h_new = agent.act(obs, mask, h)
        roll.add(obs, mask, a, logp, v, float(rng.standard_normal()), dones[t], h)
        h = net.new_hidden() if dones[t] else h_new

    segs = episode_segments(np.asarray(dones, np.float32), 4)
    stats = agent.update(roll, 0.0)
    assert stats["samples"] == float(len(dones))
    assert stats["segments"] == float(len(segs)), (
        f"段数不对:{stats['segments']} != {len(segs)}"
    )
    assert all(np.isfinite(v) for k, v in stats.items()), f"统计量出现 nan/inf:{stats}"
    assert stats["grad_norm"] > 0, "整轮 update 没有产生梯度"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
