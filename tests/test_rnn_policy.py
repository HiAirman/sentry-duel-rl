"""RNNPolicy(编码器 + GRU)的三条命根子。

1. **初始化逐位等于起点 MLP。** 这是"最差也不会更差"的全部依据 —— P/bP 零初始化
   保证 feat ≡ a3。它一旦不成立(比如有人把 P 改成 randn),模型从作废的策略头
   起步,训练曲线照常往上爬,只是永远追不上 v1.0,而且没有任何报错。
2. **逐步推理 == 整段推理。** 采样走 `step()`、训练走 `forward_seq()`,两条路径
   各写一遍前向必然漂移,而漂移的表现是"离线测出来的分数 ≠ 发出去的分数"。
3. **反向对得上有限差分。** 记忆通路和残差直连通路**都要**回传到编码器,漏掉
   残差那条不会报错,只是编码器学得慢。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monet.models.mlp import MLP  # noqa: E402
from monet.models.rnn_policy import RNNPolicy  # noqa: E402

_SMALL = dict(obs_dim=6, hidden=8, act_dim=3, gru_hidden=4)


def _small(seed=0):
    return RNNPolicy(seed=seed, **_SMALL)


def test_init_is_bit_identical_to_the_mlp_it_started_from():
    """`from_mlp` 之后,逐位等于那份 MLP 的输出。

    用 `array_equal` 而不是 allclose:P=0 时 feat 就是 a3 加两个精确的 0.0,
    float32 下加法恒等,所以这里**必须**逐位相同。放宽成 allclose 就等于放过了
    "P 其实不是零"这种错。
    """
    rng = np.random.default_rng(0)
    mlp = MLP(obs_dim=_SMALL["obs_dim"], hidden=_SMALL["hidden"],
              act_dim=_SMALL["act_dim"], seed=3)
    net = RNNPolicy.from_mlp(mlp, gru_hidden=_SMALL["gru_hidden"])

    B, L = 5, 1
    X = rng.standard_normal((B, _SMALL["obs_dim"])).astype(np.float32)
    want, want_v = mlp.forward(X, cache=False)

    # --- 真正该钉的不变量:feat ≡ a3,逐位 ---
    # P/bP 为零时 `a3 + (H@P.T) + bP` 里的两个加数都是精确的 0.0,float32 加法恒等,
    # 所以这一条**必须**逐位成立。它一旦不成立,就是 P/bP 真的不是零。
    X3 = X.reshape(B, L, -1)
    net.forward_seq(X3, net.new_hidden(B), cache=True)
    feat = net._feat.copy()
    a3 = net.enc.encode(X.reshape(B, -1), cache=False).reshape(B, L, -1)
    assert np.array_equal(feat, a3), (
        "P/bP 为零时 feat 必须逐位等于编码器输出。"
        f"最大偏差 {np.abs(feat - a3).max():.3e} —— P/bP 大概不是零初始化的。"
    )

    # --- 头的输出:只能是"浮点末位级"的一致,不能要求逐位 ---
    # feat 是 3 维 (B,L,H),`feat @ W3.T` 走 numpy 的**批量 matmul**;起点 MLP 的
    # a3 是 2 维,走普通 matmul。BLAS 为两者选的 kernel/求和顺序不同,末位差 1e-7。
    # 那是浮点求和顺序,不是结构性差异 —— 结构性差异会是 0.1 量级。
    got, got_v = net.forward_seq(X3, net.new_hidden(B), cache=False)
    d = np.abs(got[:, 0] - want).max()
    assert d < 1e-5, (
        f"整段路径的 logits 与起点 MLP 差得太多:最大偏差 {d:.3e}(只允许末位级)。"
    )
    assert np.abs(got_v[:, 0] - want_v).max() < 1e-5

    # 逐步路径:形状又不同,同样只要求数值一致。
    for i in range(len(X)):
        lg, v, _ = net.step(X[i], net.new_hidden(1))
        assert np.allclose(lg, want[i], atol=1e-5), (
            f"第 {i} 步的 logits 与起点 MLP 不一致:{lg} != {want[i]}"
        )
        assert abs(v - float(want_v[i])) < 1e-5, f"第 {i} 步的 value 与起点 MLP 不一致"


def test_uninitialized_p_is_zero():
    """直接钉 P/bP 的初值。上一条只在"装载的 MLP 恰好对齐"时才有效,
    这条不管装了什么都会查。"""
    net = _small()
    assert np.array_equal(net.p["P"], np.zeros_like(net.p["P"])), "P 必须是零初始化"
    assert np.array_equal(net.p["bP"], np.zeros_like(net.p["bP"])), "bP 必须是零初始化"


def test_step_and_forward_seq_agree():
    """逐步走完 == 一次性走完(含隐状态传递)。

    先给 P 一个非零值,否则整条记忆通路恒为 0,这条测试就退化成"两条路都返回
    a3 的头输出",什么也没查。
    """
    net = _small(seed=1)
    rng = np.random.default_rng(5)
    net.p["P"][:] = rng.standard_normal(net.p["P"].shape).astype(np.float32) * 0.1
    net.p["bP"][:] = rng.standard_normal(net.p["bP"].shape).astype(np.float32) * 0.1

    B, L = 2, 5
    X = rng.standard_normal((B, L, _SMALL["obs_dim"])).astype(np.float32)
    h0 = rng.standard_normal((B, _SMALL["gru_hidden"])).astype(np.float32) * 0.3

    logits, values = net.forward_seq(X, h0, cache=False)

    for b in range(B):
        h = h0[b:b + 1].copy()
        for t in range(L):
            lg, v, h = net.step(X[b, t], h)
            assert np.allclose(lg, logits[b, t], atol=1e-5), f"b={b} t={t} logits 不一致"
            assert abs(v - float(values[b, t])) < 1e-5, f"b={b} t={t} value 不一致"


def test_hidden_state_actually_changes_the_output():
    """记忆通路真的在起作用 —— 不同 h0 必须给出不同输出。

    这条挡的是"实现上把 h 传丢了"(比如 forward_seq 里忘了把 h0 递给 GRU),
    那种错会让整条记忆通路变成恒等,而 1/2 两条测试都还是绿的。
    """
    net = _small(seed=2)
    rng = np.random.default_rng(9)
    net.p["P"][:] = rng.standard_normal(net.p["P"].shape).astype(np.float32) * 0.1

    X = rng.standard_normal((1, 4, _SMALL["obs_dim"])).astype(np.float32)
    h_zero = np.zeros((1, _SMALL["gru_hidden"]), np.float32)
    h_rand = rng.standard_normal((1, _SMALL["gru_hidden"])).astype(np.float32)

    a, _ = net.forward_seq(X, h_zero, cache=False)
    b, _ = net.forward_seq(X, h_rand, cache=False)
    assert not np.allclose(a, b), "换一个初始隐状态,输出完全没变 —— h0 没被用上"


def _to_float64_in_place(net):
    """把参数换成 float64,**并且保持合并视图的共享**。

    必须走 `enc.p` / `gru.p` 这两个子字典再 `_relink()`,不能写 `net.p[k] = ...`:
    那样只重新绑定合并视图,子对象还是原来的 float32 数组,而前向读的是子对象
    —— 于是有限差分扰动的是"没人读的副本",数值梯度恒为零。这条测试曾经就是
    这么写的,结果编码器和整个 GRU 的检查全部**空转通过**(0 对 0),把一个真
    错误盖了过去:当时 `net.g` 与 `enc.g`/`gru.g` 不共享,Adam 根本看不到那些
    梯度。下面 `assert np.abs(num).max() > 1e-7` 就是防止再次空转的绊线。
    """
    for k in list(net.enc.p):
        net.enc.p[k] = net.enc.p[k].astype(np.float64)
    for k in list(net.gru.p):
        net.gru.p[k] = net.gru.p[k].astype(np.float64)
    net._P = net._P.astype(np.float64)
    net._bP = net._bP.astype(np.float64)
    net._relink()
    net.zero_grad()
    return net


def _seq_loss(net, X, h0, dlogits, dvalue):
    logits, values = net.forward_seq(X, h0, cache=True)
    return float((logits * dlogits).sum() + (values * dvalue).sum())


def test_backward_seq_matches_finite_differences():
    """整条链的数值梯度检查,重点看**编码器**和**GRU**那几项。

    编码器的梯度有两条来源:残差直连(dfeat -> a3)和记忆通路(dfeat -> h -> GRU
    -> a3)。只实现前者会让 W0/W1/W2 的梯度偏小,数值检查立刻抓到。
    """
    net = _small(seed=4)
    rng = np.random.default_rng(13)
    for k in net.p:
        net.p[k][:] = (net.p[k] + rng.standard_normal(net.p[k].shape) * 0.05).astype(np.float32)
    _to_float64_in_place(net)           # P 也被扰动过,记忆通路整条都有梯度

    B, L = 2, 4
    X = rng.standard_normal((B, L, _SMALL["obs_dim"]))
    h0 = rng.standard_normal((B, _SMALL["gru_hidden"])) * 0.4
    dlogits = rng.standard_normal((B, L, _SMALL["act_dim"]))
    dvalue = rng.standard_normal((B, L))

    net.zero_grad()
    _seq_loss(net, X, h0, dlogits, dvalue)
    net.backward_seq(dlogits, dvalue)
    ana = {k: net.g[k].copy() for k in net.p}

    eps = 1e-5
    worst = (0.0, None)
    for key in net.p:
        arr = net.p[key]
        flat = arr.reshape(-1)
        num = np.zeros(flat.size)
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + eps
            lp = _seq_loss(net, X, h0, dlogits, dvalue)
            flat[i] = old - eps
            lm = _seq_loss(net, X, h0, dlogits, dvalue)
            flat[i] = old
            num[i] = (lp - lm) / (2 * eps)
        num = num.reshape(arr.shape)
        # 绊线:数值梯度恒为零说明这次比较是空的 —— 要么扰动没落到前向真正读的
        # 数组上,要么该参数的梯度通路根本没接上。两种都必须当场报错,不能算过。
        nmax = np.abs(num).max()
        assert nmax > 1e-7, (
            f"{key} 的数值梯度恒为零(最大 {nmax:.3e})—— 这条检查是空转的,不算通过。"
            f"常见原因:参数被重新绑定过,前向读的已不是被扰动的那个数组。"
        )
        err = np.abs(ana[key] - num).max() / nmax
        if err > worst[0]:
            worst = (err, key)
        assert err < 1e-5, (
            f"{key} 的解析梯度与数值梯度不符:相对误差 {err:.3e}"
            f"(解析 max={np.abs(ana[key]).max():.3e},数值 max={nmax:.3e})"
        )
    print(f"    (最大相对误差 {worst[0]:.2e} @ {worst[1]})")


def test_gradients_are_shared_with_the_submodules():
    """`net.g` 必须与 `enc.g` / `gru.g` 共享同一批数组 —— 这条比参数共享更容易漏。

    `backward_seq` 把 GRU 的梯度写进 `self.gru.g`、编码器的写进 `self.enc.g`,
    而 `_clip_and_step` 读的是 `net.g`。三者不共享时:
      * Adam 看到的 W0..W2/LN/GRU 梯度**恒为零** —— 那些权重永远不动;
      * `zero_grad()` 清不掉它们 —— 跨 minibatch 无限累加。
    症状是"加了记忆层但完全没变强",不报错、不崩、曲线照常往上爬。
    """
    net = _small(seed=6)
    rng = np.random.default_rng(21)
    net.p["P"][:] = rng.standard_normal(net.p["P"].shape).astype(np.float32)
    X = rng.standard_normal((2, 3, _SMALL["obs_dim"])).astype(np.float32)
    dl = rng.standard_normal((2, 3, _SMALL["act_dim"])).astype(np.float32)
    dv = rng.standard_normal((2, 3)).astype(np.float32)

    net.zero_grad()
    net.forward_seq(X, None, cache=True)
    net.backward_seq(dl, dv)

    for k in net.p:
        # P/bP 是 RNNPolicy 自己的,不在任何子模块里;其余按归属查对应的子字典。
        # (注意用 is None 逐级判断,`or` 会在 ndarray 上触发"真值不唯一"。)
        if k == "P":
            want = net._gP
        elif k == "bP":
            want = net._gbP
        else:
            want = net.enc.g.get(k)
            if want is None:
                want = net.gru.g.get(k)
        assert net.g[k] is want, f"{k} 的梯度缓冲没有共享"
    for k in ("W0", "ln1_g", "W1", "W_ir", "W_hn", "b_in", "P", "bP"):
        assert np.abs(net.g[k]).max() > 0, (
            f"{k} 经一次 backward_seq 后梯度仍为零 —— 要么通路没接上,要么缓冲没共享"
        )

    net.zero_grad()
    for k in ("W0", "ln1_g", "W_ir", "W_hn"):
        assert np.abs(net.g[k]).max() == 0, f"zero_grad() 没清掉 {k}"
    # 子模块自己的视图也得是干净的,否则下一轮会带着上一轮的残留继续加
    assert np.abs(net.enc.g["W0"]).max() == 0 and np.abs(net.gru.g["W_ir"]).max() == 0


def test_load_state_dict_tolerates_a_checkpoint_without_the_gru():
    """从 rl_VG_v1.0 的 checkpoint 起训:那里没有 GRU 和 P/bP。

    缺的键必须**保留初始化值**(包括 P/bP 的零),已有的键必须真的被装载 ——
    写反了会静默地让"从 v1.0 起训"变成"从随机起训"。
    """
    mlp = MLP(obs_dim=_SMALL["obs_dim"], hidden=_SMALL["hidden"],
              act_dim=_SMALL["act_dim"], seed=8)
    net = RNNPolicy(seed=99, **_SMALL)
    before_gru = {k: v.copy() for k, v in net.gru.p.items()}

    net.load_state_dict(mlp.state_dict())

    for k in ("W0", "b0", "W2", "ln3_g", "W3", "bv"):
        assert np.array_equal(net.p[k], mlp.p[k]), f"{k} 没有被装载"
        # 装载必须**同时**改到子对象,否则前向读旧数组、Adam 更新新数组
        assert np.array_equal(net.enc.p[k], mlp.p[k]), f"{k} 装进了合并视图但没进 enc.p"
    for k in before_gru:
        assert np.array_equal(net.gru.p[k], before_gru[k]), f"checkpoint 里没有 {k},不该被改动"
    assert np.array_equal(net.p["P"], np.zeros_like(net.p["P"])), "P 应保持零初始化"
    # 共享必须是"同一个对象",不是"值相等"
    net.p["W0"][0, 0] += 1.0
    assert net.enc.p["W0"][0, 0] == net.p["W0"][0, 0], "p 与 enc.p 没有共享同一个数组"


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
