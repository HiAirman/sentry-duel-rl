"""源码包权重头的结构自检(不需要 C++ 编译器)。

校验拆分导出最容易出错的三处:include 链是否可解析、花括号是否配平、
符号是否重复或缺漏。逐位数值比对见 tests/test_train.py。

**这里的符号表是故意手抄一份、不从 `monet.models` import 的**:本脚本检查的是
"发出去的那个包",期望值必须独立于生成它的那份表 —— 从同一个 `TENSORS` 取,
导出器把符号名写错时两边会一起错,检查就成了空转。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# 纯 MLP 包(16 个张量)
MLP_EXPECT = [
    "kW0", "kB0", "kLN1Gamma", "kLN1Beta",
    "kW1", "kB1", "kLN2Gamma", "kLN2Beta",
    "kW2", "kB2", "kLN3Gamma", "kLN3Beta",
    "kW3", "kB3", "kW4", "kB4",
]

# 带 GRU 的包 = 上面 16 个(同名同序,编码器与三个头没变)+ GRU 的 12 个
# + 残差回注的 2 个,共 30 个。
GRU_EXPECT = [
    "kGruWIr", "kGruBIr", "kGruWHr", "kGruBHr",
    "kGruWIz", "kGruBIz", "kGruWHz", "kGruBHz",
    "kGruWIn", "kGruBIn", "kGruWHn", "kGruBHn",
    "kMemProj", "kMemBias",
]


def _int_of(text: str, name: str):
    m = re.search(rf"static const int {name}\s*=\s*(\d+);", text)
    return int(m.group(1)) if m else None


def check(pack_dir: str) -> None:
    d = Path(pack_dir)
    main = d / "rl_weights.h"
    assert main.exists(), f"{main} 不存在"
    text = main.read_text(encoding="utf-8")

    rec = _int_of(text, "kRecurrent") or 0
    EXPECT = MLP_EXPECT + (GRU_EXPECT if rec else [])
    print(f"=== 架构: {'GRU (kRecurrent=1)' if rec else '纯 MLP'} ===")

    print("=== include 链 ===")
    for inc in re.findall(r'#include "(.+?)"', text):
        exists = (d / inc).exists()
        print(f"  {inc:<26} 存在={exists}")
        assert exists, f"缺文件 {inc}"

    print()
    print("=== 分片结构 ===")
    pairs = []      # [(符号, 声明的元素个数)] —— 下面的维度自洽检查要用
    for p in sorted(d.glob("rl_weights*.h")):
        t = p.read_text(encoding="utf-8")
        decls = re.findall(r"static const float (\w+)\[(\d+)\]", t)
        if not decls:
            print(f"  {p.name:<26} 纯 include")
            continue
        opens, closes = t.count("{"), t.count("}")
        pragma = t.count("#pragma once")
        ok = opens == closes and pragma == 1
        print(
            f"  {p.name:<26} 数组 {len(decls):>2}  花括号 {opens}/{closes}  "
            f"pragma {pragma}  {'OK' if ok else '!! 异常'}"
        )
        assert ok, p.name
        pairs += decls

    syms = [s for s, _ in pairs]
    print()
    dup = sorted({s for s in syms if syms.count(s) > 1})
    missing = [s for s in EXPECT if s not in syms]
    extra = [s for s in syms if s not in EXPECT]
    print(f"重复符号: {dup or '无'}")
    print(f"缺失符号: {missing or '无'}")
    print(f"多余符号: {extra or '无'}")
    assert not dup and not missing and not extra, (dup, missing, extra)

    # 声明的维度与各张量的**实际元素个数**必须自洽。导出侧改了 gru_hidden 而漏改
    # 标记行的话,部署侧的 `linear(..., kG, kG)` 会按错的步长寻址 —— 越界或错位,
    # 而 C++ 数组是裸指针,不会报错。这里按行主序的 (out, in) 约定逐条核。
    print()
    sizes = {s: int(n) for s, n in pairs}
    obs = _int_of(text, "kObsDim")
    hid, act = _int_of(text, "kHidden"), _int_of(text, "kActDim")
    gru = _int_of(text, "kGruHidden")
    checks = {"kW0": obs * hid, "kW3": act * hid}
    if rec:
        checks.update({
            "kGruWIr": gru * hid, "kGruBIr": gru,
            "kGruWHr": gru * gru, "kGruBHr": gru,
            "kMemProj": hid * gru, "kMemBias": hid,
        })
    for sym, want in checks.items():
        got = sizes.get(sym)
        print(f"  {sym:<12} 声明 {got:>8}  应为 {want:>8}  {'OK' if got == want else '!! 不符'}")
        assert got == want, f"{sym} 声明 {got} 个元素,按 kHidden={hid}/kGruHidden={gru} 应为 {want}"

    print()
    print("结构检查通过(未做真实编译 —— 需要 C++ 工具链)")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "rl_VG_v0"
    if not os.path.isdir(target):
        target = os.path.join("rl_monet_v1", target)
    check(target)
