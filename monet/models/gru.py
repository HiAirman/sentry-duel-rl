"""GRU(门控循环单元)—— 纯 NumPy,前向 + 沿时间的反向。

公式与 `torch.nn.GRU` 逐字对齐,便于日后拿 PyTorch 对拍:

    r_t = σ(x_t W_ir^T + b_ir + h_{t-1} W_hr^T + b_hr)
    z_t = σ(x_t W_iz^T + b_iz + h_{t-1} W_hz^T + b_hz)
    n_t = tanh(x_t W_in^T + b_in + r_t ⊙ (h_{t-1} W_hn^T + b_hn))
    h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}

注意 `h_t` 的写法是 `(1-z)*n + z*h_{t-1}`(PyTorch/原论文的口径),不是
`z*n + (1-z)*h`。两者只差 z 的语义,写反**不会报错**,只会让门学成反的 ——
症状是收敛慢而不是崩,极易误判成"循环网络不适合这个任务"。改这里要看两处:
前向这一行,和反向里 `dz = dh * (h_prev - n)` 那一行。

依赖:仅 numpy。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# σ 的朴素写法在 |x| 很大时 np.exp(-x) 会溢出、刷一屏运行时警告;先把输入夹到
# ±60。夹紧只是把饱和区截得早一点(两端最大偏差 ~1e-26),换来干净的数值路径。
_SIG_CLIP = 60.0

# 导出符号表:(C++ 符号名, 参数名)。与 mlp.TENSORS 同构,由 export_cpp 与 pack 共用
# ——两边各写一份迟早分叉,而分叉的表现是"包里的 GRU 权重装错了门",网络照常前向、
# 只是变弱,几乎查不出来。顺序按门分组(r/z/n),与 __init__ 的声明顺序一致。
TENSORS = [
    ("kGruWIr", "W_ir"), ("kGruBIr", "b_ir"),
    ("kGruWHr", "W_hr"), ("kGruBHr", "b_hr"),
    ("kGruWIz", "W_iz"), ("kGruBIz", "b_iz"),
    ("kGruWHz", "W_hz"), ("kGruBHz", "b_hz"),
    ("kGruWIn", "W_in"), ("kGruBIn", "b_in"),
    ("kGruWHn", "W_hn"), ("kGruBHn", "b_hn"),
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -_SIG_CLIP, _SIG_CLIP)))


class GRU:
    """一层 GRU。参数命名与 PyTorch 的 `weight_ih_l0` / `weight_hh_l0` 分块对应。

    参数形状一律 (out, in),前向用 `x @ W.T` 消费 —— 与 MLP 同约定。
    """

    def __init__(self, input_dim: int, hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.input_dim, self.hidden = input_dim, hidden
        I, G = input_dim, hidden

        def lin(fan_in, fan_out):
            bound = 1.0 / np.sqrt(fan_in)
            return rng.uniform(-bound, bound, size=(fan_out, fan_in)).astype(np.float32)

        # 12 个张量各自独立命名(r/z/n 各一组 W_ih/W_hh + 偏置),对应 PyTorch 一个
        # nn.GRU 内部按 `chunk(3)` 切出来的三块 —— 名字对得上,才能直接吃
        # `state_dict()` 做对拍。**门序必须是 r / z / n**:既是 PyTorch 的顺序,也是
        # gru.py::TENSORS 和部署侧 `rl_VG.cpp` 的读权重顺序,三处对上才不出错位。
        self.p: Dict[str, np.ndarray] = {
            "W_ir": lin(I, G), "b_ir": np.zeros(G, np.float32),
            "W_hr": lin(G, G), "b_hr": np.zeros(G, np.float32),
            "W_iz": lin(I, G), "b_iz": np.zeros(G, np.float32),
            "W_hz": lin(G, G), "b_hz": np.zeros(G, np.float32),
            "W_in": lin(I, G), "b_in": np.zeros(G, np.float32),
            "W_hn": lin(G, G), "b_hn": np.zeros(G, np.float32),
        }
        self.g: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._cache = None

    # ------------------------------------------------------------------ 前向

    def forward(self, X: np.ndarray, h0: Optional[np.ndarray] = None, cache: bool = True):
        """X: (B, L, input_dim) -> (H: (B, L, hidden), h_last: (B, hidden))。

        `h0=None` 表示"每段从零隐状态开始"。**调用方必须保证这个零是"这一局的
        开头"**,否则段与段之间就断了记忆 —— 截断 BPTT 的正确性全押在这一点上。
        段中间起头的片段要传上一段存下来的 `h0`(见 ppo_seq.py)。
        """
        p = self.p
        B, L, I = X.shape
        G = self.hidden
        dt = X.dtype   # 跟着输入走,不写死 float32:float64 下才做得动 1e-9 的梯度检查
        if B == 0 or L == 0:
            return np.zeros((B, L, G), dt), np.zeros((B, G), dt)

        # 输入侧的三个投影与时间无关,一次算完;循环里只剩 h 的部分。
        XR = X @ p["W_ir"].T + p["b_ir"]
        XZ = X @ p["W_iz"].T + p["b_iz"]
        XN = X @ p["W_in"].T + p["b_in"]

        h = np.zeros((B, G), dt) if h0 is None else h0.astype(dt)
        H = np.empty((B, L, G), dt)
        R = np.empty((B, L, G), dt)
        Z = np.empty((B, L, G), dt)
        N = np.empty((B, L, G), dt)
        HP = np.empty((B, L, G), dt)   # 每步的 h_{t-1},反向要用
        HH = np.empty((B, L, G), dt)   # 每步的 h_{t-1}W_hn^T+b_hn,反向要用

        for t in range(L):
            # h_prev 必须**在覆盖 h 之前**存下来:反向里 `dz = dh*(h_prev - n)` 和
            # 三处 `@ h_prev` 都要用它,拿更新后的 h 去算会得到"看着对、梯度错"的
            # 结果 —— 有限差分能抓到,但只看代码看不出来。
            h_prev = h
            hh = h_prev @ p["W_hn"].T + p["b_hn"]
            r = sigmoid(XR[:, t] + h_prev @ p["W_hr"].T + p["b_hr"])
            z = sigmoid(XZ[:, t] + h_prev @ p["W_hz"].T + p["b_hz"])
            n = np.tanh(XN[:, t] + r * hh)
            h = (1.0 - z) * n + z * h_prev
            HP[:, t], HH[:, t] = h_prev, hh
            R[:, t], Z[:, t], N[:, t] = r, z, n
            H[:, t] = h

        # cache=False 只是**不写**,不清旧的:上一次 cache=True 留下的那份还在,
        # 之后调 backward() 会静默按那个旧 batch 算梯度 —— 不报错,梯度是错的。
        if cache:
            self._cache = dict(X=X, H=H, R=R, Z=Z, N=N, HP=HP, HH=HH, h0=h0)
        return H, h

    # ------------------------------------------------------------------ 反向

    def backward(self, dH: np.ndarray) -> np.ndarray:
        """dH: (B, L, hidden) -> dX: (B, L, input_dim)。梯度累积到 self.g。"""
        c, p, g = self._cache, self.p, self.g
        if c is None:
            raise RuntimeError("backward() 之前必须先 forward(cache=True)")

        X, R, Z, N, HP = c["X"], c["R"], c["Z"], c["N"], c["HP"]
        HH = c["HH"]
        B, L, _ = X.shape
        dX = np.zeros_like(X)
        # 段起点就是截断点:dh_next 从零起,而且本函数只返回 dX —— 传进来的 h0
        # 那一侧拿不到梯度,这就是截断 BPTT 的"截断"。段长**以内**的梯度是完整的。
        dh_next = np.zeros((B, self.hidden), X.dtype)

        for t in range(L - 1, -1, -1):
            dh = dH[:, t] + dh_next
            r, z, n, h_prev, hh = R[:, t], Z[:, t], N[:, t], HP[:, t], HH[:, t]

            # h = (1-z)*n + z*h_prev
            dn = dh * (1.0 - z)
            dz = dh * (h_prev - n)
            dh_prev = dh * z

            dn_raw = dn * (1.0 - n * n)          # 过 tanh
            dz_raw = dz * z * (1.0 - z)          # 过 sigmoid
            dr = dn_raw * hh
            dr_raw = dr * r * (1.0 - r)
            dhh = dn_raw * r

            xt = X[:, t]
            g["W_in"] += dn_raw.T @ xt
            g["b_in"] += dn_raw.sum(axis=0)
            g["W_ir"] += dr_raw.T @ xt
            g["b_ir"] += dr_raw.sum(axis=0)
            g["W_iz"] += dz_raw.T @ xt
            g["b_iz"] += dz_raw.sum(axis=0)

            g["W_hn"] += dhh.T @ h_prev
            g["b_hn"] += dhh.sum(axis=0)
            g["W_hr"] += dr_raw.T @ h_prev
            g["b_hr"] += dr_raw.sum(axis=0)
            g["W_hz"] += dz_raw.T @ h_prev
            g["b_hz"] += dz_raw.sum(axis=0)

            dh_next = dh_prev + dhh @ p["W_hn"] + dr_raw @ p["W_hr"] + dz_raw @ p["W_hz"]
            dX[:, t] = dn_raw @ p["W_in"] + dr_raw @ p["W_ir"] + dz_raw @ p["W_iz"]

        return dX

    def zero_grad(self) -> None:
        for v in self.g.values():
            v.fill(0.0)
