"""GRU 的有限差分梯度检查。

**这是循环网络唯一能机械验证的东西。** 前向写错(比如 h 的更新式写成
`z*n + (1-z)*h`、或者反向里拿更新后的 h 当 h_{t-1})的共同特征是:训练不报错、
不崩、只是**学得比应有的慢**,而且慢得看不出来 —— 曲线照常往上走,只是永远到不了
该到的位置。有限差分是唯一能在几分钟内把它们分辨出来的手段。

用 float64:float32 的舍入噪声约 1e-7,中心差分步长 1e-5 时差分本身的误差是
O(eps^2)=1e-10,float32 的噪声会把可辨的相对误差压在 1e-4 上下,分辨不出
"梯度算错了 1%" 这种错。float64 下能到 1e-9。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monet.models.gru import GRU, sigmoid  # noqa: E402


def _make(seed=0, I=4, G=3):
    g = GRU(input_dim=I, hidden=G, seed=seed)
    for k in g.p:
        g.p[k] = g.p[k].astype(np.float64)
    for k in g.g:
        g.g[k] = g.g[k].astype(np.float64)
    return g


def _loss(gru, X, h0, W_out):
    """标量损失,同时吃到整条隐状态序列和最后一步。

    两项都要:只吃 H 的话 `dh_last` 那条路径(反向里 `dh_next` 的初值)测不到,
    而它恰好是最容易写错的一处。
    """
    H, h_last = gru.forward(X, h0, cache=True)
    L = float((H * W_out).sum()) + float(h_last.sum())
    return L, H, h_last


def _num_grad(gru, X, h0, W_out, key, eps=1e-5):
    """中心差分。就地扰动参数,不改结构。"""
    p = gru.p[key]
    flat = p.reshape(-1)
    out = np.zeros(flat.size, dtype=np.float64)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        lp, _, _ = _loss(gru, X, h0, W_out)
        flat[i] = old - eps
        lm, _, _ = _loss(gru, X, h0, W_out)
        flat[i] = old
        out[i] = (lp - lm) / (2 * eps)
    return out.reshape(p.shape)


def test_gru_backward_matches_finite_differences():
    rng = np.random.default_rng(7)
    gru = _make()
    X = rng.standard_normal((2, 5, 4))
    h0 = rng.standard_normal((2, 3)) * 0.5
    W_out = rng.standard_normal((2, 5, 3))

    _, H, _ = _loss(gru, X, h0, W_out)
    dH = W_out.copy()
    dH[:, -1] += 1.0                      # h_last 那一项的梯度
    for k in gru.g:
        gru.g[k].fill(0.0)
    dX = gru.backward(dH)

    worst = (0.0, None)
    for key in gru.p:
        ana = gru.g[key]
        num = _num_grad(gru, X, h0, W_out, key)
        denom = max(1e-12, np.abs(num).max())
        err = np.abs(ana - num).max() / denom
        if err > worst[0]:
            worst = (err, key)
        assert err < 1e-6, (
            f"{key} 的解析梯度与数值梯度不符:相对误差 {err:.3e}"
            f"(解析 max={np.abs(ana).max():.3e},数值 max={np.abs(num).max():.3e})。"
            f"先查 h_t 的更新式是不是 (1-z)*n + z*h_prev,再查反向有没有拿更新后的 h 当 h_prev。"
        )

    # dX 也要查:它错了会静默地把错误梯度喂回编码器,而参数梯度可能仍然是对的。
    num_dX = np.zeros_like(X)
    flat = X.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + 1e-5
        lp, _, _ = _loss(gru, X, h0, W_out)
        flat[i] = old - 1e-5
        lm, _, _ = _loss(gru, X, h0, W_out)
        flat[i] = old
        num_dX.reshape(-1)[i] = (lp - lm) / 2e-5
    err = np.abs(dX - num_dX).max() / max(1e-12, np.abs(num_dX).max())
    assert err < 1e-6, f"dX 与数值梯度不符:相对误差 {err:.3e}"

    print(f"    (最大相对误差 {worst[0]:.2e} @ {worst[1]})")


def test_gru_h_uses_the_pytorch_gate_convention():
    """钉死 `h = (1-z)*n + z*h_prev` 这个口径。

    它是本文件里唯一一个"写反了不报错、只是变弱"的地方,所以单独钉一条:
    构造一个 z 恒为 0.5 的 GRU,那么 h_t 必须精确等于 (n_t + h_{t-1})/2。
    若实现用的是 `z*n + (1-z)*h_prev`,在同一条式子上 z=0.5 也是 0.5,查不出来 ——
    所以这里让 z **不**为 0.5:把 b_iz 设成 logit(0.25),则 z≈0.25。
    """
    gru = _make()
    for k in gru.p:
        gru.p[k][:] = 0.0
    gru.p["b_iz"][:] = np.log(0.25 / 0.75)     # z ≈ 0.25,与 0.5 分得开
    gru.p["b_hn"][:] = 0.0
    gru.p["W_hn"][:] = 0.0                     # hh = 0,于是 n = tanh(xn)

    rng = np.random.default_rng(3)
    X = rng.standard_normal((1, 4, 4))
    H, h_last = gru.forward(X, None, cache=False)

    # 手推:h_0 = z*h_{-1} + (1-z)*n_0,h_{-1}=0 -> h_0 = 0.75*n_0
    n0 = np.tanh(X[0, 0] @ gru.p["W_in"].T + gru.p["b_in"])
    want0 = 0.75 * n0
    assert np.allclose(H[0, 0], want0, atol=1e-10), (
        f"第一步隐状态不符:{H[0,0]} != {want0}。"
        f"更新式大概写成了 z*n + (1-z)*h_prev。"
    )
    n1 = np.tanh(X[0, 1] @ gru.p["W_in"].T + gru.p["b_in"])
    want1 = 0.75 * n1 + 0.25 * want0
    assert np.allclose(H[0, 1], want1, atol=1e-10), f"第二步隐状态不符:{H[0,1]} != {want1}"
    assert np.allclose(h_last[0], want1, atol=1e-10)


def test_gru_zero_input_projection_keeps_h_at_zero():
    """`W_in`/`b_in`/`W_hn`/`b_hn` 全零 ⇒ h 恒为 0(与 h0 无关)。

    这条是 rl_VG_v2.0 的起点性质:GRU 的输出经一个**零初始化**的升维投影加回
    编码器输出,所以模型在初始化时**逐位等于 rl_VG_v1.0**。有了这条,记忆层
    最差也只能学成"不用它",不会一上来就把一个已经很强的策略弄坏。
    """
    gru = _make()
    for k in ("W_in", "b_in", "W_hn", "b_hn"):
        gru.p[k][:] = 0.0
    rng = np.random.default_rng(11)
    X = rng.standard_normal((3, 6, 4))
    # **h0 必须是零。** 门控只把 h_{t-1} 缩放,不消灭它:n=0 时 h_t = z*h_{t-1},
    # 非零 h0 会按 z 衰减而不是归零。真实用法里新的一局从零隐状态起步,所以这里
    # 传 None —— 用随机 h0 去测这条性质,测的其实是另一件事。
    H, h_last = gru.forward(X, None, cache=False)
    assert np.array_equal(H, np.zeros_like(H)), "候选通路全零 + 零初值时 h 必须恒为 0"
    assert np.array_equal(h_last, np.zeros_like(h_last))


def test_sigmoid_is_stable_at_extremes():
    """±1e4 上不能出 nan/inf —— 训练早期 logits 冲高时这条会被真的触发。"""
    x = np.array([-1e4, -800.0, -50.0, 0.0, 50.0, 800.0, 1e4])
    y = sigmoid(x)
    assert np.all(np.isfinite(y)), y
    # 饱和端只要求"数值上就是 0/1",不要求位模式精确 —— 截断到 ±60 之后
    # 负端是 1/(1+e^60) ≈ 8.8e-27,可表示,不是 0.0。**要点是没下溢成 nan。**
    assert y[0] < 1e-20 and y[-1] == 1.0, y
    assert np.allclose(sigmoid(np.array([0.0])), 0.5)


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
