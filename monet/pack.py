"""参赛包解析:把 C++ 源码包里的 `rl_weights.h` 读成一个装好权重的网络。

用途:把别处训练好的对手(如 `../rl_v5-source`)接进联盟与评测集,不必重新训练
也能拿它当陪练。这些包是**仓库外的目录**,所以路径要能改(见 `resolve_pack_dir`)。

## 为什么能直接装

同源导出器写出来的头文件,其符号名/张量布局与 `models/` 里的参数一一对应
(`mlp.TENSORS` / `rnn_policy.TENSORS` 是共用对照表,按包头的架构标记选)。C++ 侧
的 `linear()` 按 `W + o*in_dim`(行主序、out-major)寻址,与 `X @ W.T` 是同一套
约定,所以解析出来 reshape 成 `(out, in)` 就能直接前向。

## 为什么失败了要大声报错

**张量错位不会让任何东西崩** —— 它只会让对手变弱智,而一个弱智对手混在联盟里
几乎看不出来,只会让训练数据悄悄变样。所以这里所有异常路径都抛 `PackError`,
并且一定带上试过的绝对路径和期望/实际的大小。
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .env import obs as O
from .models.factory import make_net, tensor_table

# 权重头文件名(与 export_cpp.py 产出的一致)
WEIGHT_HEADER = "rl_weights.h"

# 环境变量:指向参赛包目录,便于换机器/换包而不改代码
PACK_ENV_VAR = "MONET_PACK_DIR"

# 已知的参赛包:注册名 → 目录名。加一个包 = 在这里加一行 + 在
# training/selfplay.py 注册,不用改别的。
KNOWN_PACKS = {
    "rl_v5": "rl_v5-source",  # 外部强 AI,与本仓库并列
    "m3_v1": "m3_v1",         # m3 的冻结导出,基线锚点
    "m3_v2": "m3_v2",         # m3 的另一次冻结导出,比 m3_v1 晚
    "rl_VG_v0.2": "rl_VG_v0.2",  # 仓库内;rl_VG_v1.0 的起点兼基线
    "rl_VG_v1.0": "rl_VG_v1.0",  # 仓库内;v2 要超越的对象 —— 已交付的冻结包
    "rl_VG_v2.0": "rl_VG_v2.0",  # 仓库内;v1 的编码器 + GRU 记忆层(kRecurrent=1)
    "rl_VG_v2.3": "rl_VG_v2.3",  # 仓库内;v2.0 续训 400k 步的冻结导出,已交付
}

# 包目录可能落在这两个根下:与本仓库并列(D:\WorkFile\Sentry_Dual\<目录名>),
# 或者就在仓库里(rl_monet_v1\<目录名>,rl_VG_* 这类自产包在这里 —— 它们和训它们
# 的代码一起进版本库)。按顺序找,先命中先用。
REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_ROOTS = (REPO_ROOT, REPO_ROOT.parent)

# `--pack-dir` / `$MONET_PACK_DIR` 只作用于这一个包:那是"我手头这份外部包挪地方了"
# 的场景,而仓库自己产出的包(m3_v1)位置是固定的,不该被一个全局口子改掉。
PRIMARY_PACK = "rl_v5"

# 默认位置。走 known_pack_dir 而不是自己拼一条路径:两条解析规则迟早会分叉,
# 而分叉的表现是"某个包突然找不到了"。
DEFAULT_PACK_DIR = None  # 见文件末尾(known_pack_dir 定义之后)

_OVERRIDE: Optional[Path] = None

_DECL = r"static const float {sym}\[(\d+)\]\s*=\s*\{{(.*?)\}};"


class PackError(RuntimeError):
    """参赛包缺失或结构不符 —— 消息里一定含试过的路径。"""


def set_pack_dir(pack_dir=None) -> None:
    """进程级覆盖(CLI 的 `--pack-dir` 用)。传 None 恢复默认解析链。"""
    global _OVERRIDE
    _OVERRIDE = Path(pack_dir).expanduser().resolve() if pack_dir else None


def resolve_pack_dir(pack_dir=None, name: str = PRIMARY_PACK) -> Path:
    """按注册名解析包目录,返回绝对路径。

    `name=PRIMARY_PACK`(默认)时优先级:显式参数 > `set_pack_dir` > 环境变量 >
    默认位置 —— 这几个口子都是给"外部包挪了地方"准备的。
    其它已知包按固定位置解析(见 `KNOWN_PACKS`),不受那几个口子影响。
    """
    if pack_dir:
        return Path(pack_dir).expanduser().resolve()
    known = known_pack_dir(name)
    if name != PRIMARY_PACK:
        return known
    if _OVERRIDE is not None:
        return _OVERRIDE
    env = os.environ.get(PACK_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    return known


def known_pack_dir(name: str) -> Path:
    """按注册名取包目录(不受 `--pack-dir`/环境变量影响)。

    在两个根下按序找 `PACK_ROOTS`,第一个存在的胜出。都不存在时返回**第一个**
    候选(与本仓库并列的那个),让报错路径每次都一样 —— 调用方要的是"找不到包"
    的清晰报错,而不是随环境变化的随机结果。
    """
    if name not in KNOWN_PACKS:
        raise PackError(f"未注册的参赛包 {name!r};已知:{sorted(KNOWN_PACKS)}")
    dirname = KNOWN_PACKS[name]
    cands = [(root / dirname).resolve() for root in PACK_ROOTS]
    for c in cands:
        if c.is_dir():
            return c
    return cands[0]


def all_pack_dirs() -> Dict[str, Path]:
    """所有已知包 → 目录。导出时的护栏要挨个挡住,见 export_cpp。"""
    return {n: known_pack_dir(n) for n in KNOWN_PACKS}


DEFAULT_PACK_DIR = known_pack_dir(PRIMARY_PACK)


# --------------------------------------------------------------- 头文件解析


def _read_joined(header: Path) -> str:
    """读主头文件,并把 `#include "…"` 的分片正文接在后面。

    评测服务限制单文件 8MB,所以大网络会被 export_cpp 拆成
    `rl_weights_part1.h` 之类。**主头只留 banner(那几个 int 标记)+ include 行**,
    数组正文全在分片里 —— 所以标记从主头读得到,权重必须拼完分片才找得到。
    """
    try:
        text = header.read_text(encoding="utf-8")
    except OSError as exc:
        raise PackError(f"读不了权重头 {header}:{exc}") from exc

    for inc in re.findall(r'#include\s+"([^"]+)"', text):
        part = header.parent / inc
        if not part.exists():
            raise PackError(f"{header} 里 #include \"{inc}\",但 {part} 不存在")
        try:
            text += "\n" + part.read_text(encoding="utf-8")
        except OSError as exc:
            raise PackError(f"读不了分片 {part}:{exc}") from exc
    return text


def _parse_array(text: str, sym: str, n: int, where: Path) -> np.ndarray:
    """取出 `static const float <sym>[n] = { … };` 的正文并解析成 float32。

    头文件里**声明的** `[N]` 与期望的 `n` 都要对。只用期望值去找的话,
    头文件若写成 `kW0[999999]`,报出来会是"找不到 `kW0[219136]` 的定义",
    把人往"符号名写错了"的方向带,而真正的问题是规格不符。
    """
    m = re.search(_DECL.format(sym=re.escape(sym)), text, re.S)
    if m is None:
        raise PackError(f"{where}:找不到 `static const float {sym}[…]` 的定义")
    declared = int(m.group(1))
    if declared != n:
        raise PackError(
            f"{where}:{sym} 声明 {declared} 个元素,本引擎期望 {n} 个"
            f" —— 权重多半是从别的观测/网络规格导出的"
        )
    # export_cpp 每行 8 个、逗号分隔、%.9g。去空白后按逗号切即可,不必上 np.loadtxt。
    body = m.group(2).replace("\n", "").replace(" ", "")
    raw = [t for t in body.split(",") if t]
    if len(raw) != n:
        raise PackError(f"{where}:{sym} 声明 {n} 个元素,实际解析出 {len(raw)} 个")
    try:
        return np.array(raw, dtype=np.float32)
    except ValueError as exc:
        raise PackError(f"{where}:{sym} 里有无解析成 float 的字面量({exc})") from exc


def _read_int(text: str, name: str, where: Path) -> int:
    m = re.search(rf"static const int {name}\s*=\s*(\d+);", text)
    if m is None:
        raise PackError(f"{where}:缺少 `static const int {name} = N;`")
    return int(m.group(1))


@lru_cache(maxsize=4)
def _load_weights(pack_dir: str, hidden: int) -> Dict[str, np.ndarray]:
    """解析参赛包 → 参数字典。**缓存的是字典,不是网络对象。**

    如果直接缓存并反复发出同一个网络,任何调用方就地对它做的修改
    (喂给 PPO 训练、`load_state_dict`、`forward(cache=True)`)都是**进程级持久**的:
    之后每次 `load_pack_net()` 拿到的都是被污染的网络。而它扮演的是"参考对手",
    坏了几乎不会被察觉 —— 每次发一份新副本的代价很小,值这个保险。

    返回的数组即缓存内容,调用方**不要再改**。

    架构与 `gru_hidden` 都从头部读(`kRecurrent` / `kGruHidden`),不额外传参:
    调用方通常只知道 `hidden`,让它再猜一个 `gru_hidden` 只会多一处对不上的可能。
    """
    d = Path(pack_dir)
    header = d / WEIGHT_HEADER
    if not header.exists():
        raise PackError(
            f"参赛包里没有 {WEIGHT_HEADER}:试过 {header}"
            f"(用 --pack-dir 或环境变量 {PACK_ENV_VAR} 指定包目录)"
        )

    text = _read_joined(header)
    obs_dim = _read_int(text, "kObsDim", header)
    hid = _read_int(text, "kHidden", header)
    act_dim = _read_int(text, "kActDim", header)
    # 老包(在 kRecurrent 出现之前导出的)没有这两行。缺省当纯 MLP —— 那正是它们
    # 的身份,所以这里不报错是正确的宽容,不是掩盖问题。
    rec = _read_int(text, "kRecurrent", header) if "kRecurrent" in text else 0
    gru = _read_int(text, "kGruHidden", header) if rec else 0

    if obs_dim != O.OBS_DIM or act_dim != O.ACTION_DIM:
        raise PackError(
            f"{header}:网络规格是 obs={obs_dim}/act={act_dim},"
            f"与本引擎的 obs={O.OBS_DIM}/act={O.ACTION_DIM} 不符 —— 观测或动作空间不同,"
            f"权重装了也不能用"
        )
    if hid != hidden:
        raise PackError(f"{header}:hidden={hid},但当前配置要 hidden={hidden}")

    # 先建一张随机网络:形状的真源就是它的参数表,不另抄一份形状表
    net = make_net(arch="gru" if rec else "mlp", obs_dim=obs_dim, hidden=hid,
                   act_dim=act_dim, gru_hidden=gru or 128, seed=0)
    p: Dict[str, np.ndarray] = {}
    for sym, param in tensor_table(net):
        want = net.p[param].shape
        got = _parse_array(text, sym, int(np.prod(want)), header)
        if not np.isfinite(got).all():
            raise PackError(f"{header}:{sym} 里含 nan/inf —— 源权重本身坏了")
        p[param] = got.reshape(want)

    # 逐键对齐:构造函数已经把 net.p 随机初始化过,漏掉的键会留着随机值,
    # 表现就是"网络一半是训练的、一半是噪声" —— 也是静默的错误。
    assert set(p) == set(net.p), "符号表与网络的参数集合不一致"
    return p


def load_pack_net(pack_dir=None, hidden: int = 512, name: str = PRIMARY_PACK):
    """解析参赛包 → 装好权重的网络(`MLP` 或 `RNNPolicy`,按包头的架构标记)。

    按目录缓存解析结果,但**每次发一张新网络**。
    """
    w = _load_weights(str(resolve_pack_dir(pack_dir, name)), hidden)
    # 架构由缓存里有没有 GRU 键决定 —— 与 _load_weights 读的是同一个标记,
    # 不重新解析一遍头部(两份判断迟早会不一致)。
    arch = "gru" if "W_ir" in w else "mlp"
    gru = int(w["P"].shape[1]) if arch == "gru" else 128
    net = make_net(arch=arch, obs_dim=O.OBS_DIM, hidden=hidden,
                   act_dim=O.ACTION_DIM, gru_hidden=gru, seed=0)
    for k, v in w.items():
        if net.p[k].shape != v.shape:
            raise PackError(f"缓存里的参数 {k} 形状 {v.shape} 与网络的 {net.p[k].shape} 不符")
    net.load_state_dict(w)
    return net


class LazyPackNet:
    """延迟到第一次前向才真正解析权重包(纯 MLP 约 10 MB 文本,GRU 约 14.6 MB)。

    存在的理由是**构造对手必须便宜**:注册表会在 import 期构造对手来探测
    `is_official`(见 `training/selfplay.py`),import 一个训练模块不该顺带解析
    十几 MB 头文件,更不该在包缺失时让整个引擎连 `export` 都跑不了。

    但**存在性**检查放在 `__init__`:配置错误要在训练启动时就报,而不是打到
    第一局中途才炸。`NeuralOpponent` 只用到 `.logits`,其余属性转发给真网络。
    """

    def __init__(self, pack_dir=None, name: str = PRIMARY_PACK):
        self.name = name
        self.pack_dir = resolve_pack_dir(pack_dir, name)
        header = self.pack_dir / WEIGHT_HEADER
        if not header.exists():
            raise PackError(
                f"参赛包 {name!r} 里没有 {WEIGHT_HEADER}:试过 {header}"
                f"(用 --pack-dir 或环境变量 {PACK_ENV_VAR} 指定包目录)"
            )
        self._net = None

    @property
    def is_recurrent(self) -> bool:
        """是否带记忆。**必须显式定义,不能靠 `__getattr__` 转发。**

        训练器和评测器构造对手时就要读这个标记来决定"要不要逐局传隐状态",而
        `__getattr__` 转发会顺带 `_resolve()` 整个包 —— 那正是本类存在要避免的
        10MB 解析,而且发生在 import 期探测官方集合的时候。标记就写在主头文件里
        (几百字节;数字全在分片里),直接读它即可。
        """
        if self._net is not None:
            return bool(getattr(self._net, "is_recurrent", False))
        text = (self.pack_dir / WEIGHT_HEADER).read_text(encoding="utf-8")
        m = re.search(r"static const int kRecurrent\s*=\s*(\d+);", text)
        # 没有这一行 = 老包 = 纯 MLP,与 pack 解析侧的缺省口径一致。
        return bool(int(m.group(1))) if m else False

    def _resolve(self):
        if self._net is None:
            self._net = load_pack_net(str(self.pack_dir))
        return self._net

    def logits(self, obs: np.ndarray) -> np.ndarray:
        return self._resolve().logits(obs)

    def __getattr__(self, item):
        # 只转发非下划线属性:否则 pickle/copy 探测 `__deepcopy__` 之类会顺着
        # `self._net` 把权重包提前拉起来,惰性就白做了。
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(self._resolve(), item)
