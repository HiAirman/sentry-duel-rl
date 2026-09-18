"""`RL_best040`(外部参赛 AI)当陪练:权重解析 + 对手封装。

两件事:

* `RLBestNet` / `LazyRLBestNet` —— 把 `../RL_best040-source/rl_weights.h` 里的
  `kW0/kB0/kW1/kB1/kW2/kB2` 解析成 NumPy 数组,跑 **412→256→256→8** 的前向。
  结构就是最朴素的 `y = relu(Wx+b)`:两层隐藏层各自带 bias、激活 ReLU,输出层线性,
  **没有 LayerNorm、没有残差、没有别的花样**。对应 `RL_best040-source/my_ai.cpp`
  里手写的 `matmul/forward`。
* `RLBestOpponent` —— 接进 `Opponent` 协议,用它**自己的 412 维观测**
  (`obs_rlbest.BeliefObs`,不能复用我们的 428 维)。

## 为什么观测要自己建、回合要写成生成器

`my_ai.cpp::rl_act()` 一次调用内部走完整回合(最多 3 个动作,每个动作的控制流都
依赖该动作返回的 `ActionObservation`),而且失败动作**不消耗额度**、要换次优重试。
本仓库里同一形态的对手(官方 Baseline / Hunter,见 `env/official.py`)都是把
`Opponent.turn()` 写成**生成器**:每个 `yield` 出去的动作由环境调 `game.apply` 执行、
结果 `send` 回来。这里照抄那条路 —— 只有这样,对手造成的击杀才会流经环境的奖励计算,
而且我们才能拿到每个动作的 `ActionResult` 去驱动自己的观测。

环境的 `ob`(428 维 `ObsBuilder`)在这里**被忽略**:412 和 428 是两套观测,共用一个
builder 会让观测和权重对不上,而那种错误不会崩,只会让对手变弱智。

## 与引擎的差异(移植保真度)

1. **蓝方回合 0 的补偿情报**(`obs_builder.h` 的 `else if (first_act_ && ...)`)。
   官方引擎会给蓝方补一份"红方回合末位置(visible=false)",我们的 `Game.view`
   只给公开 `intel`,开局 `intel` 是空的 ⇒ 这条分支恒不触发。影响面是蓝方**第一个**
   决策点的信念/情报少一格(其余标量不受影响)。要消掉它只能在引擎侧补情报,
   而那是信息泄漏面 —— 记为残留差异,别在 env 侧"顺手修"。
2. **`can_see(me, target, obstacles)`(utils.h)与我们的 `rules.can_see`** 假定等价
   (README §六.3 的第 3 条:_同格可见_的约定两边都成立)。`utils.h` 不在手边,
   无法逐位比对,只能接受。
3. **float32 累加顺序**。`my_ai.cpp` 的 `matmul` 是按输出行**顺序累加**的朴素循环,
   这里是 NumPy 的 BLAS `gemv`(可能用 FMA、分块)。两者差在 1e-6 量级,
   只在两个动作的 logits 几乎相等时才会翻转 argmax —— 无法消除,只能记录。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np

from . import obs_rlbest as O
from .obs_rlbest import BeliefObs
from .opponents import Opponent

# 权重头文件名(与 RL_best040-source 里那一份一致)
WEIGHT_HEADER = "rl_weights.h"

# 包目录名(与本仓库并列:`D:\WorkFile\Sentry_Dual\RL_best040-source`)
BEST040_DIRNAME = "RL_best040-source"

# 环境变量:指向别处的包目录
PACK_ENV_VAR = "MONET_RLBEST_DIR"

REPO_ROOT = Path(__file__).resolve().parents[2]  # monet/env/rlbest.py → 仓库根
DIR_ROOTS = (REPO_ROOT, REPO_ROOT.parent)

# 符号 → (参数名, 形状)。形状是**网络架构的真源**(412→256→256→8),
# 与 `kLayerIn/kLayerOut` 两处互相印证,不符就大声报错。
TENSORS = (
    ("kW0", "W0", (256, 412)),
    ("kB0", "b0", (256,)),
    ("kW1", "W1", (256, 256)),
    ("kB1", "b1", (256,)),
    ("kW2", "W2", (8, 256)),
    ("kB2", "b2", (8,)),
)

# `static const float kW0[105472] = { … };`
_DECL = r"static const float {sym}\[(\d+)\]\s*=\s*\{{(.*?)\}};"


class RLBestError(RuntimeError):
    """权重包缺失或规格不符 —— 消息里一定含试过的路径。"""


def resolve_best040_dir(pack_dir=None) -> Path:
    """解析包目录:显式参数 > `$MONET_RLBEST_DIR` > 与本仓库并列的默认位置。"""
    if pack_dir:
        return Path(pack_dir).expanduser().resolve()
    env = os.environ.get(PACK_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    cands = [(root / BEST040_DIRNAME).resolve() for root in DIR_ROOTS]
    for c in cands:
        if c.is_dir():
            return c
    return cands[0]


# --------------------------------------------------------------- 头文件解析


def _read_joined(header: Path) -> str:
    """读主头文件,并把 `#include "…"` 的分片正文接在后面(与 `pack.py` 同做法)。

    评测服务限制单文件 8MB,以后 best040 若重新导出成多分片,这里不用改。
    """
    try:
        text = header.read_text(encoding="utf-8")
    except OSError as exc:
        raise RLBestError(f"读不了权重头 {header}:{exc}") from exc
    for inc in re.findall(r'#include\s+"([^"]+)"', text):
        part = header.parent / inc
        if not part.exists():
            raise RLBestError(f'{header} 里 #include "{inc}",但 {part} 不存在')
        try:
            text += "\n" + part.read_text(encoding="utf-8")
        except OSError as exc:
            raise RLBestError(f"读不了分片 {part}:{exc}") from exc
    return text


def _parse_array(text: str, sym: str, n: int, where: Path) -> np.ndarray:
    """取出 `static const float <sym>[n] = { … };` 的正文并解析成 float32。

    声明的 `[N]` 与期望的 `n` 都要对:头文件若写成 `kW0[999999]`,只报"找不到"
    会把"符号名写错了"和"规格不符"两件事混在一起。
    """
    m = re.search(_DECL.format(sym=re.escape(sym)), text, re.S)
    if m is None:
        raise RLBestError(f"{where}:找不到 `static const float {sym}[…]` 的定义")
    declared = int(m.group(1))
    if declared != n:
        raise RLBestError(
            f"{where}:{sym} 声明 {declared} 个元素,本引擎期望 {n} 个"
            f" —— 权重多半是从别的观测/网络规格导出的"
        )
    body = m.group(2).replace("\n", "").replace(" ", "")
    raw = [t for t in body.split(",") if t]
    if len(raw) != n:
        raise RLBestError(f"{where}:{sym} 声明 {n} 个元素,实际解析出 {len(raw)} 个")
    try:
        return np.array(raw, dtype=np.float32)
    except ValueError as exc:
        raise RLBestError(f"{where}:{sym} 里有无解析成 float 的字面量({exc})") from exc


def _read_int_array(text: str, name: str, where: Path) -> list:
    m = re.search(rf"static const int {name}\[(\d+)\]\s*=\s*\{{([^}}]*)\}};", text)
    if m is None:
        raise RLBestError(f"{where}:缺少 `static const int {name}[…] = {{…}};`")
    return [int(t) for t in m.group(2).replace(" ", "").split(",") if t]


# --------------------------------------------------------------------- 网络


class RLBestNet:
    """412→256→256→8 的朴素 MLP(ReLU 隐藏层,线性输出),权重为只读 float32。"""

    def __init__(self, W0, b0, W1, b1, W2, b2):
        self.W0 = np.asarray(W0, dtype=np.float32)
        self.b0 = np.asarray(b0, dtype=np.float32)
        self.W1 = np.asarray(W1, dtype=np.float32)
        self.b1 = np.asarray(b1, dtype=np.float32)
        self.W2 = np.asarray(W2, dtype=np.float32)
        self.b2 = np.asarray(b2, dtype=np.float32)
        for sym, p, shape in TENSORS:
            got = getattr(self, p).shape
            if got != shape:
                raise RLBestError(f"{sym} 形状 {got} 与架构 {shape} 不符")

    obs_dim = O.OBS_DIM
    act_dim = O.ACTION_DIM

    def hidden(self, obs: np.ndarray) -> np.ndarray:
        """第二隐藏层激活(诊断用;与 `my_ai.cpp` 的 h2 同一份)。"""
        h1 = self.W0 @ _vec(obs) + self.b0
        np.maximum(h1, 0.0, out=h1)
        h2 = self.W1 @ h1 + self.b1
        np.maximum(h2, 0.0, out=h2)
        return h2

    def logits(self, obs: np.ndarray) -> np.ndarray:
        """8 个动作的 logits(未过掩码)。确定性 argmax 由调用方做。"""
        h = self.hidden(obs)
        return self.W2 @ h + self.b2

    def state_dict(self) -> dict:
        return {p: getattr(self, p) for _s, p, _sh in TENSORS}


def _vec(obs: np.ndarray) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float32).reshape(-1)
    if x.shape != (O.OBS_DIM,):
        raise RLBestError(f"观测长度 {x.shape} 与网络输入 {O.OBS_DIM} 不符")
    return x


def load_rlbest_net(pack_dir=None, name: str = "rl_best040") -> RLBestNet:
    """解析 `RL_best040-source/rl_weights.h` → `RLBestNet`。每个包目录只解析一次。"""
    return _load_cached(str(resolve_best040_dir(pack_dir)))


_CACHE: dict = {}


def _load_cached(pack_dir: str) -> RLBestNet:
    if pack_dir in _CACHE:
        return _CACHE[pack_dir]
    net = _parse_net(Path(pack_dir))
    _CACHE[pack_dir] = net
    return net


def _parse_net(d: Path) -> RLBestNet:
    header = d / WEIGHT_HEADER
    if not header.exists():
        raise RLBestError(
            f"参赛包里没有 {WEIGHT_HEADER}:试过 {header}"
            f"(用参数或环境变量 {PACK_ENV_VAR} 指定包目录)"
        )
    text = _read_joined(header)

    lin = _read_int_array(text, "kLayerIn", header)
    lout = _read_int_array(text, "kLayerOut", header)
    if len(lin) < 3 or lin[0] != O.OBS_DIM or lin[1] != 256 or lin[2] != 256 or lout[2] != O.ACTION_DIM:
        raise RLBestError(
            f"{header}:网络规格 kLayerIn={lin}/kLayerOut={lout},"
            f"本引擎要的是 412→256→256→8 —— 观测或动作空间不同,权重装了也不能用"
        )

    p = {}
    for sym, param, shape in TENSORS:
        got = _parse_array(text, sym, int(np.prod(shape)), header)
        if not np.isfinite(got).all():
            raise RLBestError(f"{header}:{sym} 里含 nan/inf —— 源权重本身坏了")
        p[param] = got.reshape(shape)
    # kW3/kB3(256→1)是价值头,`my_ai.cpp::forward` 不用它 —— 少一个张量不算错,
    # 但层数/规格变了上面的 kLayerIn 检查会先发难。
    return RLBestNet(**p)


class LazyRLBestNet:
    """延迟到第一次前向才解析权重头(2.4 MB 文本)。

    与 `pack.LazyPackNet` 同一个理由:**构造对手必须便宜** —— 注册表会在 import 期
    构造对手来探测 `is_official`,import 一个训练模块不该顺带解析几 MB 头文件。
    存在性检查留在 `__init__`:配置错误要在训练启动时报,而不是第一局中途才炸。
    """

    def __init__(self, pack_dir=None):
        self.pack_dir = resolve_best040_dir(pack_dir)
        header = self.pack_dir / WEIGHT_HEADER
        if not header.exists():
            raise RLBestError(
                f"参赛包里没有 {WEIGHT_HEADER}:试过 {header}"
                f"(用参数或环境变量 {PACK_ENV_VAR} 指定包目录)"
            )
        self._net: Optional[RLBestNet] = None

    def _resolve(self) -> RLBestNet:
        if self._net is None:
            self._net = _load_cached(str(self.pack_dir))
        return self._net

    def logits(self, obs: np.ndarray) -> np.ndarray:
        return self._resolve().logits(obs)

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(self._resolve(), item)


# ------------------------------------------------------------------- 对手


class RLBestOpponent(Opponent):
    """`RL_best040` 的推理壳:确定性 masked argmax + 被拒动作屏蔽重试。

    与 `my_ai.cpp::rl_act()` 一一对应:
      * 回合开头 `act_start`,每步 `encode()` → 前向 → masked argmax;
      * 选到 END(或没有可选动作)就结束本回合;
      * 引擎拒绝的动作**不消耗额度**,屏蔽它重挑次优(这也是环境那套
        "一次 step = 一次行动函数调用"的粒度存在的原因)。

    全确定性,没有采样、没有随机数;`seed` 只为对齐注册表签名。
    `is_official` 留 `False` —— 要不要按 `rl_v5` 那样算"外部强对手",由联盟侧决定。
    """

    name = "rl_best040"

    def __init__(self, net=None, seed: int = 0, name: str = None):
        self.net = net if net is not None else LazyRLBestNet()
        self.ob = BeliefObs()
        self.seed = seed
        self._started = False
        if name:
            self.name = name

    def reset(self) -> None:
        self.ob.reset()
        self._started = False

    def turn(self, view_fn, my_color: str, ob):
        """一整个行动阶段(生成器)。`ob` 是环境的 428 维 builder,这里**不用它**。"""
        board = view_fn()
        # C++ 侧:`if (board.turn == 0 || !g_game_started) { g_ob.reset(); … }`
        if board.turn == 0 or not self._started:
            self.ob.reset()
            self._started = True
        self.ob.act_start(board, my_color)

        banned = [False] * O.ACTION_DIM
        used = 0
        while used < O.ACTIONS_PER_TURN:
            logits = self.net.logits(self.ob.encode())
            mask = self.ob.action_mask()
            # masked argmax:按 a 升序、严格大于 —— 平局时取小动作号,与 C++ 同
            best, best_v = -1, -1e30
            for a in range(O.ACTION_DIM):
                if mask[a] <= 0.0 or banned[a]:
                    continue
                v = float(logits[a])
                if v > best_v:
                    best_v, best = v, a
            if best < 0 or best == O.END:
                return  # end
            res = yield best
            self.ob.on_observation(res.observation, res.consumed)
            if not res.success:
                banned[best] = True  # 被引擎拒绝(不消耗额度),换次优
                continue
            if res.consumed:
                used += 1
