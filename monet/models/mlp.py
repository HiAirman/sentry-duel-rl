"""纯 NumPy MLP —— 与 rl_v5 部署网络逐层同构。

结构(P0):
    x(428) -> fc1 -> LN -> GELU
           -> fc2 -> (+残差) -> LN -> GELU
           -> fc3 -> (+残差) -> LN -> GELU
           -> pi: fc4 -> 8 logits
           -> vf: fc5 -> 1   (仅训练使用;头文件里照导,部署侧不读)

之所以逐层对齐 rl_ai_v5.cpp 的 forward(),是为了让 export_cpp.py 生成的
rl_weights.h 能直接喂给那份推理壳(它读 kW0..kW3 / kB0..kB3 / kLN*)。

依赖:仅 numpy(erf 优先用 scipy,缺失时退回 1.5e-7 精度的近似式)。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:  # pragma: no cover - 取决于环境
    from scipy.special import erf as _erf
except ImportError:  # pragma: no cover
    def _erf(x):
        # Abramowitz & Stegun 7.1.26,|误差| < 1.5e-7
        sign = np.sign(x)
        ax = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * ax)
        y = 1.0 - (
            ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592
        ) * t * np.exp(-ax * ax)
        return sign * y

_INV_SQRT2 = 0.7071067811865476
_INV_SQRT_2PI = 0.3989422804014327
_EPS = 1e-5  # LayerNorm 方差下限,防止对常量输入除以 0

# C++ 侧符号名 ↔ 本模型参数名的对照表。**导出(export_cpp)与导入(pack)共用它**,
# 所以放在模型旁边而不是任一侧:两边各抄一份迟早会漂移,而漂移的表现是权重静默错位
# ——不会崩,只会让网络变成弱智。
# 张量形状不在这里:从 `MLP(...).p[param].shape` 取,那才是唯一真源。
TENSORS = [
    ("kW0", "W0"), ("kB0", "b0"),
    ("kLN1Gamma", "ln1_g"), ("kLN1Beta", "ln1_b"),
    ("kW1", "W1"), ("kB1", "b1"),
    ("kLN2Gamma", "ln2_g"), ("kLN2Beta", "ln2_b"),
    ("kW2", "W2"), ("kB2", "b2"),
    ("kLN3Gamma", "ln3_g"), ("kLN3Beta", "ln3_b"),
    ("kW3", "W3"), ("kB3", "b3"),
    ("kW4", "Wv"), ("kB4", "bv"),
]


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _erf(x * _INV_SQRT2))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(x * _INV_SQRT2)) + x * np.exp(-0.5 * x * x) * _INV_SQRT_2PI


class _Cache:
    """缓存前向中间量。

    注意 h* 是 GELU 的**输入**(LN 输出),a* 是 GELU 的**输出** ——
    反向必须用 gelu_grad(h) 而不是 gelu_grad(a)。
    """

    __slots__ = ("x", "ln1", "h1", "a1", "ln2", "h2", "a2", "ln3", "h3", "a3")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _ln_forward(x: np.ndarray, g: np.ndarray, b: np.ndarray):
    m = x.mean(axis=1, keepdims=True)
    v = x.var(axis=1, keepdims=True)
    inv = 1.0 / np.sqrt(v + _EPS)
    xh = (x - m) * inv
    return xh * g + b, (xh, inv, g)


def _ln_backward(dy: np.ndarray, cache) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # LN 的标准反向:dx 是"减去均值/方差那两项"的闭式解,xh/ inv /g 都来自前向缓存。
    xh, inv, g = cache
    dg = (dy * xh).sum(axis=0)
    db = dy.sum(axis=0)
    dxh = dy * g
    dx = inv * (
        dxh - dxh.mean(axis=1, keepdims=True) - xh * (dxh * xh).mean(axis=1, keepdims=True)
    )
    return dx, dg, db


class MLP:
    """策略/价值网络。参数命名与导出的 C++ 头文件一一对应。"""

    # 无记忆。训练/评测循环靠这一个标记决定要不要逐局传递隐状态,而不是
    # `isinstance` 判类型 —— 后者会让"再写一个带记忆的架构"变成到处改判断。
    is_recurrent = False

    def __init__(self, obs_dim: int = 428, hidden: int = 512, act_dim: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.obs_dim, self.hidden, self.act_dim = obs_dim, hidden, act_dim
        H, N, A = hidden, obs_dim, act_dim

        def lin(fan_in, fan_out):
            # 均匀初始化,界取 1/sqrt(fan_in):各层输出尺度不随 fan_in 变化。
            # 返回 (fan_out, fan_in),前向用 `X @ W.T` 消费。
            bound = 1.0 / np.sqrt(fan_in)
            return rng.uniform(-bound, bound, size=(fan_out, fan_in)).astype(np.float32)

        self.p: Dict[str, np.ndarray] = {
            "W0": lin(N, H), "b0": np.zeros(H, np.float32),
            "ln1_g": np.ones(H, np.float32), "ln1_b": np.zeros(H, np.float32),
            "W1": lin(H, H), "b1": np.zeros(H, np.float32),
            "ln2_g": np.ones(H, np.float32), "ln2_b": np.zeros(H, np.float32),
            "W2": lin(H, H), "b2": np.zeros(H, np.float32),
            "ln3_g": np.ones(H, np.float32), "ln3_b": np.zeros(H, np.float32),
            "W3": lin(H, A), "b3": np.zeros(A, np.float32),
            "Wv": lin(H, 1), "bv": np.zeros(1, np.float32),
        }
        self.g: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._cache = None

    # ------------------------------------------------------------------ 前向

    def encode(self, X: np.ndarray, cache: bool = True) -> np.ndarray:
        """只跑编码器,到 a3 为止,不出头。

        拆出来是给 RNNPolicy 用的:它要拿 a3 喂进循环层,再从**另一条路**回来。
        `forward()` 与它共用这一段,免得两处各写一遍前向而漂移。
        """
        p = self.p
        z0 = X @ p["W0"].T + p["b0"]
        h1, ln1 = _ln_forward(z0, p["ln1_g"], p["ln1_b"])
        a1 = gelu(h1)

        # 后两块是 pre-norm 残差:块的输入直连加到 fc 输出上。反向里 da*/dz* 表达式
        # 末尾那个 `+ dz*` 就是它,前向加了几处、反向就要补几处。
        z1 = a1 @ p["W1"].T + p["b1"] + a1
        h2, ln2 = _ln_forward(z1, p["ln2_g"], p["ln2_b"])
        a2 = gelu(h2)

        z2 = a2 @ p["W2"].T + p["b2"] + a2
        h3, ln3 = _ln_forward(z2, p["ln3_g"], p["ln3_b"])
        a3 = gelu(h3)

        # 同 GRU.forward:cache=False 不清旧缓存 —— 中间夹一次 logits() 之类的
        # 单步前向,后面的 backward() 就按旧 batch 算梯度了,而且不报错。
        if cache:
            self._cache = _Cache(x=X, ln1=ln1, h1=h1, a1=a1, ln2=ln2, h2=h2, a2=a2,
                                 ln3=ln3, h3=h3, a3=a3)
        return a3

    def forward(self, X: np.ndarray, cache: bool = True):
        """X: (B, obs_dim) float32 -> (logits (B,A), value (B,))."""
        p = self.p
        a3 = self.encode(X, cache=cache)
        logits = a3 @ p["W3"].T + p["b3"]
        value = (a3 @ p["Wv"].T + p["bv"]).reshape(-1)
        return logits, value

    # ------------------------------------------------------------------ 反向

    def backward(self, dlogits: np.ndarray, dvalue: np.ndarray) -> None:
        """累积梯度到 self.g(不归零,由调用方决定何时清零)。"""
        c, p, g = self._cache, self.p, self.g
        if c is None:
            raise RuntimeError("backward() 之前必须先 forward(cache=True)")

        g["W3"] += dlogits.T @ c.a3
        g["b3"] += dlogits.sum(axis=0)
        g["Wv"] += dvalue.reshape(1, -1) @ c.a3
        g["bv"] += dvalue.sum(axis=0, keepdims=True).reshape(-1)

        da3 = dlogits @ p["W3"] + dvalue.reshape(-1, 1) @ p["Wv"]
        self.backward_encoder(da3)

    def backward_encoder(self, da3: np.ndarray) -> None:
        """从 a3 的梯度往回累积,**不含** W3/b3/Wv/bv 三个头。

        调用方要么已经自己算过头(见 RNNPolicy),要么走 `backward()`。
        """
        c, p, g = self._cache, self.p, self.g
        if c is None:
            raise RuntimeError("backward_encoder() 之前必须先 encode(cache=True)")

        # fc3 分支:LN(z2) -> h3 -> GELU -> a3
        dz2, dln3_g, dln3_b = _ln_backward(da3 * gelu_grad(c.h3), c.ln3)
        g["ln3_g"] += dln3_g
        g["ln3_b"] += dln3_b
        da2 = dz2 @ p["W2"] + dz2  # 线性项 + 残差项
        g["W2"] += dz2.T @ c.a2
        g["b2"] += dz2.sum(axis=0)

        dz1, dln2_g, dln2_b = _ln_backward(da2 * gelu_grad(c.h2), c.ln2)
        g["ln2_g"] += dln2_g
        g["ln2_b"] += dln2_b
        da1 = dz1 @ p["W1"] + dz1
        g["W1"] += dz1.T @ c.a1
        g["b1"] += dz1.sum(axis=0)

        dz0, dln1_g, dln1_b = _ln_backward(da1 * gelu_grad(c.h1), c.ln1)
        g["ln1_g"] += dln1_g
        g["ln1_b"] += dln1_b
        g["W0"] += dz0.T @ c.x
        g["b0"] += dz0.sum(axis=0)

    def zero_grad(self) -> None:
        for v in self.g.values():
            v.fill(0.0)

    # ------------------------------------------------------------------ 推理

    def logits(self, obs: np.ndarray) -> np.ndarray:
        """单条观测(1-D)或一批 -> logits。"""
        single = obs.ndim == 1
        X = obs.reshape(1, -1).astype(np.float32) if single else obs.astype(np.float32)
        lg, _ = self.forward(X, cache=False)
        return lg[0] if single else lg

    # -------------------------------------------------------------- 参数读写

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.p.items()}

    def load_state_dict(self, sd: Dict[str, np.ndarray]) -> None:
        # 每个参数都拷贝一份:装载后模型与调用方的字典不共享内存,改一边不影响另一边。
        for k, v in sd.items():
            self.p[k] = np.asarray(v, dtype=np.float32).copy()
