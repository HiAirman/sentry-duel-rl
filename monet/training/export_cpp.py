"""把 NumPy 权重导出成 C++ 头文件,直接喂给部署侧推理壳(如 rl_VG.cpp)。

符号表**不在这里写死**,而是问 `models/factory.py::tensor_table(net)` 要 —— 那边
和 `make_net` / `arch_of` 是同一份架构名真相。按架构分两种:

    mlp  kW0 kB0 kLN1Gamma kLN1Beta … kW3 kB3,加价值头 kW4 kB4
    gru  上面 16 个**同名照旧**(编码器与三个头没变),再接 12 个 GRU 张量
         (kGruWIr/kGruBIr/…,命名同 PyTorch 的 `chunk(3)`)与残差回注的
         kMemProj/kMemBias,共 30 个

价值头 kW4/kB4 推理侧不读,保留是为了两类架构复用同一套导出。维度另有
kObsDim/kHidden/kActDim,架构另有 kRecurrent/kGruHidden —— 部署侧靠这两个标记
区分,不靠"有没有某个符号"去猜。

评测服务对多文件源码包限制**单文件 ≤8MB**(api.md §9),而权重头光文本就有
~10MB(MLP 512 宽)/ ~14.6MB(GRU)。所以超过 `max_bytes` 时会自动按张量拆成
`<name>_part1.h` 等分片,主头只负责 `#include`。拆分是无损的 —— 不降精度,
推理代码除 include 外一字不改。
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..models.factory import arch_of, tensor_table
from ..models.mlp import MLP

# 生成的头部里那几行结构说明。**维度是写死的**,改 `hidden` / 观测维度时这里不会
# 报错,只会在头文件里写下一句错的描述 —— 数字以同文件里的 kObsDim/kHidden 为准。
_NET_STRUCT = {
    "mlp": "// 网络结构:428 → 512 → 512(+残差) → 512(+残差) → 8/1\n",
    "gru": (
        "// 网络结构:428 → 512 → 512(+残差) → 512(+残差) → a3(编码器输出, 512)\n"
        "//           h  = GRU_128(a3, h_prev)          ← 逐决策点推进,每局从零开始\n"
        "//           feat = a3 ⊕ (kMemProj·h + kMemBias)   ← 残差回注\n"
        "//           8/1 = k W3/kB3 · feat        (价值头 kW4/kB4 同源,推理不用)\n"
    ),
}

_HEADER = """// {fname} - 由 rl_monet_v1 的 monet/training/export_cpp.py 自动生成,请勿手改
// 来源检查点:{source}
#pragma once

{struct}
static const int kObsDim = {obs};
static const int kHidden = {hid};
static const int kActDim = {act};
// 架构标记。**显式写出来**而不是让部署侧靠"有没有 kGruHidden"去猜 —— 后者是个
// 隐式约定,少写一行就静默退化成纯 MLP:GRU 那段权重摆在头文件里没人读,
// 表现是"发出去的包比测出来的弱",而评测分数下降几乎不会被归因到这一行。
static const int kRecurrent = {rec};
static const int kGruHidden = {gru};
"""

_PART_HEADER = """// {fname} - 由 monet/training/export_cpp.py 自动生成,请勿手改
// {main} 的分片 {idx}/{total}(评测服务限制单文件 8MB,api.md §9)
#pragma once

"""

# 评测服务的硬限制是 8MB。留一点余量,免得贴着上限被拒。
DEFAULT_MAX_BYTES = 7 * 1024 * 1024

# 每行打印几个数。纯排版,但会改变文本长度,而分片是按字节装箱的。
_PER_LINE = 8


def _render_array(symbol: str, arr: np.ndarray) -> str:
    # `.9g`:float32 的十进制有效位上限就是 9 位,少一位就不再是逐位无损,
    # 而部署侧是同精度的 float32 —— 这是"拆分不降精度"那句话的依据。
    flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    buf = io.StringIO()
    buf.write(f"static const float {symbol}[{flat.size}] = {{\n")
    for i in range(0, flat.size, _PER_LINE):
        chunk = flat[i : i + _PER_LINE]
        buf.write("    " + ",".join(f"{float(v):.9g}" for v in chunk))
        buf.write(",\n" if i + _PER_LINE < flat.size else "\n")
    buf.write("};\n\n")
    return buf.getvalue()


def _split(chunks: List[str], overhead: int, max_bytes: int) -> List[List[str]]:
    """把若干段文本贪心装箱,每箱(含 overhead)不超过 max_bytes。"""
    parts: List[List[str]] = []
    cur: List[str] = []
    cur_size = overhead
    for text in chunks:
        if cur and cur_size + len(text) > max_bytes:
            parts.append(cur)
            cur, cur_size = [], overhead
        cur.append(text)
        cur_size += len(text)
    if cur:
        parts.append(cur)
    return parts


def _refuse_to_write_into_a_known_pack(out_path: Path) -> None:
    """拒绝把权重头写进任何**已知参赛包**目录。

    `pack.py` 把那些目录当成对手来**读**,而 `export` 会往里**写** —— 一次手滑
    (`--out ../m3_v1/rl_weights.h`)就把冻结的对手覆盖了。更隐蔽的是
    `_load_weights` 有 `lru_cache`:同一进程里刚读过的旧对象还在,当场看不出变化。

    只挡 `KNOWN_PACKS` 里那几个目录,不影响正常流程 —— 往自己的导出目录
    (如 `rl_VG_v0/`)里反复写是常态,那是**待提交的活包**,不是冻结的对手。
    """
    from ..pack import all_pack_dirs

    out_res = out_path.resolve()
    for name, pack_dir in all_pack_dirs().items():
        if out_res == pack_dir or pack_dir in out_res.parents:
            raise ValueError(
                f"拒绝写入参赛包目录 {pack_dir}({name}):那是 pack.py 读取对手的地方,"
                f"覆盖掉就没法恢复了(而且 lru_cache 可能让同进程内看不出变化)。"
                f"换个 --out 目录;确实要替换这个对手请手动先把它移走。"
            )


def export_weights_header(
    net: MLP,
    out_path,
    source: str = "checkpoint",
    max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
) -> Path:
    """写出权重头。超过 max_bytes 时按张量拆分成 `<stem>_partN.h` 分片。

    max_bytes=None 表示永不拆分(单文件提交或本地自用)。`net` 可以是 MLP,也可以
    是 RNNPolicy —— 符号表与架构标记都按 `arch_of(net)` 选,所以同一个调用点同时
    服务两种架构,不必在 CLI 里再分一次支。
    """
    out_path = Path(out_path)
    _refuse_to_write_into_a_known_pack(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arch = arch_of(net)
    banner = _HEADER.format(
        fname=out_path.name,
        source=source,
        struct=_NET_STRUCT[arch],
        obs=net.obs_dim,
        hid=net.hidden,
        act=net.act_dim,
        rec=1 if arch == "gru" else 0,
        gru=int(getattr(net, "gru_hidden", 0)),
    )
    chunks = [
        _render_array(symbol, net.p[param])
        for symbol, param in tensor_table(net)
        if param in net.p
    ]

    if max_bytes is None or len(banner) + sum(len(c) for c in chunks) <= max_bytes:
        out_path.write_text(banner + "".join(chunks), encoding="utf-8")
        return out_path

    # 分片:主头只放 banner + include,张量按上面的顺序切进各分片
    part_names = [f"{out_path.stem}_part{i}.h" for i in range(1, 64)]
    overhead = len(_PART_HEADER.format(fname="", main="", idx=1, total=1))
    parts = _split(chunks, overhead, max_bytes)
    if len(parts) > len(part_names):
        raise ValueError(f"权重需要 {len(parts)} 个分片,超过评测服务 64 文件上限")

    written = []
    for i, (name, body) in enumerate(zip(part_names, parts), start=1):
        p = out_path.parent / name
        p.write_text(
            _PART_HEADER.format(
                fname=name, main=out_path.name, idx=i, total=len(parts)
            )
            + "".join(body),
            encoding="utf-8",
        )
        written.append(p)

    # 清掉上次导出留下的、这次不再需要的分片,避免旧分片被一起打包
    for stale in out_path.parent.glob(f"{out_path.stem}_part*.h"):
        if stale not in written:
            stale.unlink()

    includes = "".join(f'#include "{p.name}"\n' for p in written)
    out_path.write_text(banner + "\n" + includes, encoding="utf-8")
    return out_path


def write_weights_binary(net: MLP, out_path) -> Path:
    """紧凑的 .npz 权重快照(训练侧续用,不参与 C++ 构建)。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **{k: v for k, v in net.p.items()})
    return out_path
