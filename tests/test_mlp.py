"""MLP 反向传播的数值梯度校验 —— 手写 LN/残差/GELU 是最容易写错的地方。

用小网络(obs=7, hidden=5, act=3)做中心差分,误差阈值 1e-4(相对误差)。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monet.models.mlp import MLP  # noqa: E402

OBS, HID, ACT, B = 7, 5, 3, 4


def _loss(net, X, D1, D2):
    logits, value = net.forward(X, cache=False)
    return float((logits * D1).sum() + (value * D2).sum())


def _as_float64(net):
    """梯度校验在 float64 下做:float32 的中心差分截断误差会淹没小梯度。"""
    net.p = {k: v.astype(np.float64) for k, v in net.p.items()}
    net.g = {k: np.zeros_like(v) for k, v in net.p.items()}
    return net


def test_numerical_gradient():
    rng = np.random.default_rng(0)
    net = _as_float64(MLP(obs_dim=OBS, hidden=HID, act_dim=ACT, seed=1))
    X = rng.normal(size=(B, OBS))
    D1 = rng.normal(size=(B, ACT))
    D2 = rng.normal(size=(B,))

    net.zero_grad()
    logits, value = net.forward(X, cache=True)
    net.backward(D1.copy(), D2.copy())

    eps = 1e-6
    worst = 0.0
    worst_name = ""
    for name, p in net.p.items():
        flat = p.reshape(-1)
        idxs = rng.choice(flat.size, size=min(6, flat.size), replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp = _loss(net, X, D1, D2)
            flat[i] = orig - eps
            lm = _loss(net, X, D1, D2)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = float(net.g[name].reshape(-1)[i])
            denom = max(1e-6, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > worst:
                worst, worst_name = rel, f"{name}[{i}]"
    assert worst < 1e-6, f"梯度校验失败:{worst_name} 相对误差 {worst:.2e}"


def test_forward_matches_reference_formula():
    """逐层对齐 rl_ai_v5.cpp 的 forward():LN 顺序、残差位置、GELU 形状。"""
    from monet.models.mlp import gelu

    rng = np.random.default_rng(3)
    net = MLP(obs_dim=OBS, hidden=HID, act_dim=ACT, seed=2)
    X = rng.normal(size=(2, OBS)).astype(np.float32)

    def ln(x, g, b):
        m = x.mean(axis=1, keepdims=True)
        v = x.var(axis=1, keepdims=True)
        return (x - m) / np.sqrt(v + 1e-5) * g + b

    p = net.p
    a1 = gelu(ln(X @ p["W0"].T + p["b0"], p["ln1_g"], p["ln1_b"]))
    z1 = a1 @ p["W1"].T + p["b1"] + a1
    a2 = gelu(ln(z1, p["ln2_g"], p["ln2_b"]))
    z2 = a2 @ p["W2"].T + p["b2"] + a2
    a3 = gelu(ln(z2, p["ln3_g"], p["ln3_b"]))
    ref = a3 @ p["W3"].T + p["b3"]
    got, _ = net.forward(X, cache=False)
    assert np.allclose(ref, got, atol=1e-5)


def test_gelu_derivative():
    from monet.models.mlp import gelu, gelu_grad

    x = np.linspace(-4, 4, 41)  # float64:float32 下中心差分精度不够
    eps = 1e-6
    num = (gelu(x + eps) - gelu(x - eps)) / (2 * eps)
    assert np.allclose(num, gelu_grad(x), atol=1e-8)


def test_masked_policy_never_samples_invalid():
    from monet.models.ppo import PPOAgent

    net = MLP(obs_dim=OBS, hidden=HID, act_dim=ACT, seed=4)
    agent = PPOAgent(net, seed=0)
    obs = np.zeros(OBS, dtype=np.float32)
    mask = np.array([0, 1, 0], dtype=np.float32)
    for _ in range(50):
        a, _, _ = agent.act(obs, mask)
        assert a == 1


def test_ppo_update_reduces_kl_on_first_epoch():
    """一次 update 后,策略对同一批数据的 KL 应当是小量(信任域生效)。"""
    from monet.models.ppo import PPOAgent, Rollout

    rng = np.random.default_rng(7)
    net = MLP(obs_dim=OBS, hidden=HID, act_dim=ACT, seed=5)
    agent = PPOAgent(net, lr=1e-3, seed=0)
    roll = Rollout()
    mask = np.ones(ACT, dtype=np.float32)
    for _ in range(32):
        obs = rng.normal(size=OBS).astype(np.float32)
        a, lp, v = agent.act(obs, mask)
        roll.add(obs, mask, a, lp, v, float(rng.normal()), 0.0)
    stats = agent.update(roll)
    assert stats["approx_kl"] < 0.5
    assert np.isfinite(stats["policy_loss"]) and np.isfinite(stats["value_loss"])


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
