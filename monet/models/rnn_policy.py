"""带记忆的策略:MLP 编码器 + GRU + 零初始化残差回注。

结构(每一步 t 独立地过编码器,GRU 把历史串起来):

    a_t   = Encoder(obs_t)                  # 与 rl_VG_v1.0 的编码器逐层同构
    h_t   = GRU(a_t, h_{t-1})               # 新增:记忆
    feat_t= a_t + P·h_t + bP                # P **零初始化**
    π_t   = W3·feat_t + b3
    V_t   = Wv·feat_t + bv

**为什么把记忆做成残差而不是直接替换编码器输出:**
`P` 和 `bP` 初始化成 0,于是训练第一步时 `feat_t ≡ a_t`,整个前向**逐位等于**
一个纯 MLP。把 v1.0 的编码器和三个头原样装进来,模型在初始化时就**逐位等于
rl_VG_v1.0** —— 记忆层最差也只能学成"不用它",不会一上来就把一个已经很强的
策略弄坏。这条性质在 tests/test_rnn_policy.py 里被逐位断言。

反过来说:**不能**把 GRU 的输出直接当 feat。那样三个头面对的是全新的输入分布,
v1.0 学到的策略头当场作废,要从随机头重新学起 —— 而这一轮的目标恰恰是"打赢
v1.0",从一个作废的策略出发是最差的开局。

维度取 gru_hidden=128 而不是 512:GRU 参数是 3*(I*G + G*G) 量级(I=512 固定),
G=128 时约 0.25M 个、G=512 时约 1.57M 个 —— 后者比前者**多约 5.3MB**(float32)。
交付包的权重(文本形式的 `rl_weights_part*.h`)已经 10.3MB,而评测服务有单文件
8MB 上限、靠分片绕开,总量仍要克制。7x7 棋盘、一局 60 个决策点,128 维够用。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .gru import GRU
from .gru import TENSORS as _GRU_TENSORS
from .mlp import MLP
from .mlp import TENSORS as _MLP_TENSORS

# 这些参数属于编码器/头,从 MLP 继承(也就是从 rl_VG_v1.0 的权重继承)
_INHERITED = ("W0", "b0", "ln1_g", "ln1_b", "W1", "b1", "ln2_g", "ln2_b",
              "W2", "b2", "ln3_g", "ln3_b", "W3", "b3", "Wv", "bv")

# 导出符号表 = 编码器/头的 16 个 + GRU 的 12 个 + 残差回注的 2 个,共 30 个张量。
# 前 16 个与纯 MLP 包的符号**完全同名**,所以部署侧那套 MLP 前向一行不用改,只是
# 后面多接了一段 GRU —— 这也是把记忆做成残差(而不是换掉编码器)的附带好处。
TENSORS = list(_MLP_TENSORS) + list(_GRU_TENSORS) + [
    ("kMemProj", "P"), ("kMemBias", "bP"),
]


class RNNPolicy:
    """编码器 + GRU 的策略网络。参数命名与 MLP 一致的地方就沿用,便于导出对齐。"""

    # 有记忆。训练/评测/对手三条路径都按它决定要不要逐局传递隐状态。
    is_recurrent = True

    def __init__(self, obs_dim: int = 428, hidden: int = 512, act_dim: int = 8,
                 gru_hidden: int = 128, seed: int = 0):
        self.obs_dim, self.hidden, self.act_dim = obs_dim, hidden, act_dim
        self.gru_hidden = gru_hidden

        self.enc = MLP(obs_dim=obs_dim, hidden=hidden, act_dim=act_dim, seed=seed)
        self.gru = GRU(input_dim=hidden, hidden=gru_hidden, seed=seed + 977)

        H, G = hidden, gru_hidden
        # P 与 bP **必须**是零。它们是不是零决定了"初始化 == v1.0"这条性质成不成立,
        # 所以不走 rng —— 一个手滑改成 randn 就能让整轮训练从作废的策略起步,
        # 而且不会报任何错。梯度缓冲一并建好,由 _relink() 挂进合并视图。
        self._P = np.zeros((H, G), np.float32)
        self._bP = np.zeros(H, np.float32)
        self._gP = np.zeros((H, G), np.float32)
        self._gbP = np.zeros(H, np.float32)

        self._relink()
        self._feat: Optional[np.ndarray] = None

    # ------------------------------------------------------------- 参数视图

    def _relink(self) -> None:
        """把三处拼成**同一个**字典(共享 ndarray 对象,不是拷贝)—— 参数和梯度各一次。

        `p`/`g` 必须是合并视图,PPO 的 Adam 才能一套循环走完所有参数;而共享对象意味着
        `p["W0"] -= ...` 会同时改到 `enc.p["W0"]`。

        **`g` 也必须合并,而且这一条比 `p` 更容易漏。** `backward_seq` 里 GRU 的梯度写进
        `self.gru.g`、编码器的梯度写进 `self.enc.g`,而 `_clip_and_step` 读的是 `net.g`
        —— 三者不共享时:Adam 看到的 W0..W2/LN/GRU 梯度**恒为零**(那些权重永远不动),
        `zero_grad()` 也清不掉它们(跨 minibatch 无限累加)。症状是"加了记忆层但完全没
        变强",不报错、不崩,曲线照常往上爬。tests/test_rnn_policy.py 逐键断言了共享。

        **注意:任何把 `enc.p[k]` 或 `gru.p[k]` 重新绑定的操作都会打断这个共享**
        (例如 `load_state_dict` 里的 `.copy()`),之后必须再调一次本函数 ——
        否则 Adam 更新的是新数组、而前向读的是旧数组,表现为"训练完全不动",
        且不报错。
        """
        self.p: Dict[str, np.ndarray] = {}
        for k, v in self.enc.p.items():
            self.p[k] = v
        for k, v in self.gru.p.items():
            self.p[k] = v
        self.p["P"] = self._P
        self.p["bP"] = self._bP

        self.g: Dict[str, np.ndarray] = {}
        for k, v in self.enc.g.items():
            self.g[k] = v
        for k, v in self.gru.g.items():
            self.g[k] = v
        self.g["P"] = self._gP
        self.g["bP"] = self._gbP

    # ---------------------------------------------------------------- 前向

    def new_hidden(self, batch: int = 1) -> np.ndarray:
        """一局的初始隐状态。每局**必须**从这里重来(见 selfplay 里 done 之后的 `_pick_opponent`)。"""
        return np.zeros((batch, self.gru_hidden), np.float32)

    def forward_seq(self, X: np.ndarray, h0: Optional[np.ndarray] = None,
                    cache: bool = True):
        """X: (B, L, obs_dim) -> (logits (B,L,A), values (B,L))。

        **一次只处理一段连续片段**,段内不做任何截断 —— 截断由调用方在**段的边界**
        上做(传进来一个 detached 的 h0)。段内再截断等于悄悄丢掉长程梯度,
        而它不会报错,只会让长记忆学不出来。
        """
        p = self.p
        B, L, N = X.shape
        a3 = self.enc.encode(X.reshape(B * L, N), cache=cache)
        a3 = a3.reshape(B, L, -1)
        H, _ = self.gru.forward(a3, h0, cache=cache)

        feat = a3 + H @ p["P"].T + p["bP"]
        logits = feat @ p["W3"].T + p["b3"]
        values = (feat @ p["Wv"].T + p["bv"]).reshape(B, L)
        if cache:
            self._feat = feat
        return logits, values

    def step(self, obs: np.ndarray, h: np.ndarray):
        """单步推理(采样用):(logits (A,), value float, h_new (1,G))。

        与 forward_seq 必须给出**同一个**结果 —— 两条路径各写一遍前向是这个项目
        反复踩的坑(训练强、部署弱),所以这条由 tests/test_rnn_policy.py 用
        "逐步走完一遍 == 一次性走完一遍"钉死。
        """
        X = obs.reshape(1, -1).astype(np.float32)
        a3 = self.enc.encode(X, cache=False)
        H, h_new = self.gru.forward(a3.reshape(1, 1, -1), h, cache=False)
        feat = a3 + H[:, 0] @ self.p["P"].T + self.p["bP"]
        logits = feat @ self.p["W3"].T + self.p["b3"]
        value = (feat @ self.p["Wv"].T + self.p["bv"]).reshape(-1)
        return logits[0], float(value[0]), h_new

    def logits(self, obs: np.ndarray) -> np.ndarray:
        """单次前向,隐状态取零 —— **只为与 `MLP` 保持同一个调用签名**。

        语义上它是"这一局刚开局、还没有任何历史"时网络会说的话,不是"这个观测
        单独看"的答案。带记忆的调用方(采样、评测、对手)一律该用 `step(obs, h)`
        把 h 传下去;只有那些**结构上拿不到 h** 的调用点才用这里(例如按名字装载
        一个包、只想知道它前向是否有限、或比对两个包是不是同一份权重)。

        少了这个方法,`LazyPackNet.logits()` / `test_pack` 的注册表检查会在遇到
        GRU 包时抛 AttributeError —— 也就是说"包能用"和"包能被注册"会变成两回事。
        """
        lg, _v, _h = self.step(obs, self.new_hidden())
        return lg

    # ---------------------------------------------------------------- 反向

    def backward_seq(self, dlogits: np.ndarray, dvalue: np.ndarray) -> None:
        """累积梯度到 self.g。dlogits: (B,L,A),dvalue: (B,L)。"""
        p, g = self.p, self.g
        # 这几份缓存(enc._cache / gru._cache / _feat)由 cache=True 的前向写入,而
        # `cache=False` 的调用(step() / logits())**不会**清掉它们。所以中间夹一次
        # 单步前向不会报错,梯度却会算在更早那个 batch 上 —— forward 与 backward
        # 必须成对紧挨着。
        if self._feat is None or self.gru._cache is None:
            raise RuntimeError("backward_seq() 之前必须先 forward_seq(cache=True)")
        B, L, A = dlogits.shape
        Hh = self.hidden
        feat = self._feat
        F = feat.reshape(-1, Hh)          # (B*L, H)

        g["W3"] += dlogits.reshape(-1, A).T @ F
        g["b3"] += dlogits.reshape(-1, A).sum(axis=0)
        g["Wv"] += dvalue.reshape(-1, 1).T @ F
        # dvalue 是 (B,L) 二维,sum(axis=0) 会给出 (L,) 而不是标量 —— 得全轴求和。
        g["bv"] += np.asarray(dvalue.sum()).reshape(1).astype(g["bv"].dtype)

        # feat 的梯度,两条来源:头的线性项 + 直连(残差那一项)
        dfeat = (dlogits.reshape(-1, A) @ p["W3"]
                 + dvalue.reshape(-1, 1) @ p["Wv"])          # (B*L, H)
        dfeat3 = dfeat.reshape(B, L, Hh)

        Hseq = self.gru._cache["H"]                              # (B,L,G)
        g["P"] += dfeat.T @ Hseq.reshape(-1, self.gru_hidden)
        g["bP"] += dfeat.sum(axis=0)

        # 记忆通路:dfeat -> h -> GRU -> a3;残差通路:dfeat -> a3 直连。
        # 两条都要,漏掉残差那条就是"记忆层学了、编码器不跟着调",训练会明显变慢。
        dH = dfeat3 @ p["P"]                                     # (B,L,G)
        da3 = dfeat3 + self.gru.backward(dH)                     # (B,L,H)

        self.enc.backward_encoder(da3.reshape(B * L, Hh))

    def zero_grad(self) -> None:
        for v in self.g.values():
            v.fill(0.0)

    # ------------------------------------------------------------ 参数读写

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.p.items()}

    def load_state_dict(self, sd: Dict[str, np.ndarray]) -> None:
        """只装载 sd 里**出现过的**键,其余保持原样。

        这条宽容是必要的:rl_VG_v1.0 的 checkpoint 里没有 GRU 和 P/bP —— 那些是
        这一版新加的。严格校验键集合会让"从 v1.0 权重起训"直接失败,而放宽到
        "缺的键保留初始化值"正好是我们要的语义(记得 P/bP 的初始化值是零)。

        同一条宽容也吞掉**反向**的错:sd 里多出来的键(装错架构、或导出改了名)会被
        静默跳过,一个都不装,调用方看到的是"加载成功"而网络停在初始化值。
        """
        for k, v in sd.items():
            if k not in self.p:
                continue
            arr = np.asarray(v, dtype=np.float32)
            sub = self.enc.p if k in self.enc.p else (self.gru.p if k in self.gru.p else None)
            if k == "P":
                self._P = arr.copy()
            elif k == "bP":
                self._bP = arr.copy()
            elif sub is not None:
                sub[k] = arr.copy()     # 重新绑定 —— 必须重新 relink
            else:
                continue
        # 重新绑定过 enc.p/gru.p,必须重新合并 p **和** g —— 只合并 p 会让 Adam
        # 读到的梯度停在旧数组上(见 _relink 的注释)。
        self._relink()
        self.zero_grad()

    @classmethod
    def from_mlp(cls, mlp: MLP, gru_hidden: int = 128, seed: int = 0) -> "RNNPolicy":
        """从一份已训好的 MLP 起步(编码器和三个头原样继承,GRU/P 全新)。

        连 `Wv/bv` 也继承:价值头面对的还是同一个 feat 空间(P=0 时 feat ≡ a3),
        扔掉它等于让价值估计从零重学,白白拖慢早期训练。
        """
        net = cls(obs_dim=mlp.obs_dim, hidden=mlp.hidden, act_dim=mlp.act_dim,
                  gru_hidden=gru_hidden, seed=seed)
        sd = {k: mlp.p[k] for k in _INHERITED if k in mlp.p}
        net.load_state_dict(sd)
        return net
