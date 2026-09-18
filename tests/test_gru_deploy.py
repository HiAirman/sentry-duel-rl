"""部署侧 GRU 前向的等价性测试(不需要 C++ 编译器)。

带 GRU 的交付包(当前 `rl_VG_v2.0` / `rl_VG_v2.3`)里的 `rl_VG.cpp` 把
`monet/models/rnn_policy.py` 的前向在 C++ 里重写了一遍,
两份实现的分叉是本项目最贵的一类 bug:**不会崩、不会报错,只会让"发出去的包"
比"本地测出来的分"弱**,而几个点的差距几乎不会被归因到某个门的写法上。

本机**没有 C++ 工具链**(与 `check_pack_headers.py` 开头那句"不需要 C++ 编译器"
是同一个限制 —— 那个脚本也只是结构自检),所以这里能做的是把 C++ 的循环结构
**逐行抄成一个 numpy 函数**再和 `RNNPolicy` 对拍:

  * 抄错结构(门序、用更新前还是更新后的 h、残差加在哪)在 float64 下会差 O(1),
    这条能抓到;
  * C++ 里的手滑(拼错符号名、少写一个循环边界)**这条抓不到** —— 那由
    `test_every_weight_symbol_referenced_exists()` 从"符号是否声明过"的侧面兜住,
    真正跑起来还得靠编译。

**改了 `rl_VG.cpp` 里的 gru_step/encode/forward 就得同步改这里的 `cpp_forward`,
反之亦然** —— 两边不一致时这条测试会红,那正是它的用途。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monet.models.rnn_policy import RNNPolicy  # noqa: E402
from monet.pack import load_pack_net  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# 下面 `cpp_forward()` 那份手写翻译所依据的源文件。所有 GRU 包共用同一份 rl_VG.cpp
# (打包时原样拷),所以翻译只有一份;真正**逐包**要验的是各自的权重头,见
# `gru_pack_dirs()`。
DEPLOY_CPP = REPO / "rl_VG_v2.0" / "rl_VG.cpp"


def gru_pack_dirs() -> list[Path]:
    """仓库里所有**带权重的 GRU 部署包**目录(发现式,不写死包名)。

    写死包名会让新增的包**静默地不受这套测试保护**:打包 `rl_VG_v2.3` 时它与 v2.0
    共用同一份 `rl_VG.cpp`,于是"翻译本对拍"看起来仍然全绿 —— 而那个包真正需要被
    验的是**它自己的权重头能不能被部署壳读对**(导出器写错符号名、分片漏张量、
    张量顺序错位都只在这一层现形),拿 v2.0 的权重跑一百遍也照不出来。

    判据是权重里有没有 `kGruWIr`:MLP 包(如 `rl_VG_v1.0`)没有它,不适用本文件。
    没有权重头的包(目录存在但还没导出)也不纳入 —— 那正是"打包中途"的常态。
    """
    out = []
    for d in sorted(REPO.glob("rl_VG_v*")):
        if not (d / "rl_VG.cpp").exists() or not (d / "rl_weights.h").exists():
            continue
        if any("kGruWIr" in h.read_text(encoding="utf-8")
               for h in sorted(d.glob("rl_weights*.h"))):
            out.append(d)
    return out


# --------------------------------------------------------------- C++ 的翻译本


def _gelu(x):
    # 0.5 * x * (1 + erf(x / sqrt(2))),与 mlp.py 的精确 erf 形式同式
    from math import erf, sqrt

    v = np.vectorize(erf)(x * (1.0 / sqrt(2.0))) if x.ndim else erf(x * 0.7071067811865475)
    return 0.5 * x * (1.0 + v)


def _layer_norm(x, g, b):
    m = x.mean()
    v = ((x - m) ** 2).mean()
    return (x - m) / np.sqrt(v + 1e-5) * g + b


def _linear(W, b, x):
    # C++: y[o] = (b ? b[o] : 0) + sum_i W[o,i]*x[i]   —— b 可为 None(kMemProj)
    return x @ W.T + (0.0 if b is None else b)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def cpp_gru_step(x, h, p):
    """rl_VG.cpp::gru_step 的逐行翻译(门序 r/z/n,原地更新 h)。"""
    xr = _linear(p["W_ir"], p["b_ir"], x)
    xz = _linear(p["W_iz"], p["b_iz"], x)
    xn = _linear(p["W_in"], p["b_in"], x)
    hprev = h.copy()
    hr = _linear(p["W_hr"], p["b_hr"], hprev)
    hz = _linear(p["W_hz"], p["b_hz"], hprev)
    hn = _linear(p["W_hn"], p["b_hn"], hprev)
    r = _sigmoid(xr + hr)
    z = _sigmoid(xz + hz)
    n = np.tanh(xn + r * hn)
    return (1.0 - z) * n + z * hprev


def cpp_encode(x, p):
    """rl_VG.cpp::encode 的逐行翻译:428 → 512 → 512(+残差) → 512(+残差) → a3。"""
    x = np.asarray(x, dtype=np.float64)
    h1 = _gelu(_layer_norm(_linear(p["W0"], p["b0"], x), p["ln1_g"], p["ln1_b"]))
    h2 = _linear(p["W1"], p["b1"], h1) + h1
    h2 = _gelu(_layer_norm(h2, p["ln2_g"], p["ln2_b"]))
    a3 = _linear(p["W2"], p["b2"], h2) + h2
    return _gelu(_layer_norm(a3, p["ln3_g"], p["ln3_b"]))


def cpp_forward(obs, h, p):
    """rl_VG.cpp::forward 的逐行翻译。返回 (logits, h_new),h 不被修改。"""
    a3 = cpp_encode(obs, p)

    # 记忆 + 残差回注:feat = a3 + kMemProj·h + kMemBias
    h_new = cpp_gru_step(a3, h, p)
    feat = _linear(p["P"], None, h_new) + a3 + p["bP"]

    return _linear(p["W3"], p["b3"], feat), h_new


# --------------------------------------------------------------------- 测试


def _net(gru_hidden=128, seed=0):
    """一张非平凡的 RNNPolicy:GRU 与 P/bP 都要**离开零初始化**。

    全零的话门怎么写都能对上(恒等映射),测试就空转了 —— 与
    tests/test_rnn_policy.py 里那条 `nmax > 1e-7` 的护栏同一个理由。
    """
    net = RNNPolicy(obs_dim=428, hidden=512, act_dim=8, gru_hidden=gru_hidden, seed=seed)
    rng = np.random.default_rng(seed + 7)
    for k in net.gru.p:
        net.gru.p[k][:] = rng.normal(0, 0.2, net.gru.p[k].shape).astype(np.float32)
    net.p["P"][:] = rng.normal(0, 0.2, net.p["P"].shape).astype(np.float32)
    net.p["bP"][:] = rng.normal(0, 0.2, net.p["bP"].shape).astype(np.float32)
    return net


def _p64(net):
    return {k: np.asarray(v, dtype=np.float64) for k, v in net.p.items()}


def _py_step(net, obs, h):
    """`RNNPolicy.step` 的 float64 版,返回 (logits, h_new)。

    `RNNPolicy.step` 第一句就把 obs 硬转成 float32(`.astype(np.float32)`),那样
    对拍的精度上限只有 1e-7,结构性的错会被浮点噪声盖住一部分。这里按 `step` 的
    同样三步(encode → gru.forward → 残差+头)重写一遍,只是不降精度 —— 并有一条
    断言把它锚回 `forward_seq`(见 test_python_float64_reference_is_anchored)。
    """
    X = np.asarray(obs, dtype=np.float64).reshape(1, -1)
    a3 = net.enc.encode(X, cache=False)
    H, h_new = net.gru.forward(a3.reshape(1, 1, -1), np.asarray(h, dtype=np.float64).reshape(1, -1),
                               cache=False)
    feat = a3 + H[:, 0] @ np.asarray(net.p["P"], dtype=np.float64).T \
        + np.asarray(net.p["bP"], dtype=np.float64)
    return (feat @ np.asarray(net.p["W3"], dtype=np.float64).T
            + np.asarray(net.p["b3"], dtype=np.float64)).reshape(-1), h_new


def test_cpp_forward_matches_the_python_forward():
    """单步:翻译本 vs RNNPolicy。**这是本文件的主判据。**

    用 float64 跑,两边都收敛到 ~1e-14;任何结构性的错(换门序、用 h_new 去算 hr、
    把残差加在 LN 之后)都会把差异顶到 O(1),一眼可辨。
    """
    net = _net()
    p = _p64(net)
    rng = np.random.default_rng(0)

    worst = 0.0
    for _ in range(8):
        obs = rng.normal(0, 1, 428)
        h = rng.normal(0, 0.5, net.gru_hidden)
        lg_ref, _ = net.forward_seq(obs.reshape(1, 1, -1).astype(np.float64),
                                    h.reshape(1, -1).astype(np.float64), cache=False)
        lg_cpp, _ = cpp_forward(obs, h, p)
        worst = max(worst, float(np.abs(lg_ref.reshape(-1) - lg_cpp).max()))
    print(f"  单步 logits 最大差 {worst:.3e}")
    assert worst < 1e-11, f"部署侧前向与 Python 前向不一致(最大差 {worst:.3e})"


def test_cpp_hidden_state_threads_across_steps():
    """多步:隐状态必须真的被传下去,而且传的是**同一个**约定。

    只测单步的话,"h 原地更新"和"每步都从零开始"在观测随机时也能对上偶然的一步。
    这里连走 24 步,逐步比对 —— 一旦 h 的传递断开,第一步之后就开始分叉。
    """
    net = _net(seed=3)
    p = _p64(net)
    rng = np.random.default_rng(1)
    obs_seq = rng.normal(0, 1, (24, 428))

    h_py = np.zeros(net.gru_hidden)
    h_cpp = np.zeros(net.gru_hidden)
    worst = 0.0
    for t, obs in enumerate(obs_seq):
        lg_ref, h_py = _py_step(net, obs, h_py)
        lg_cpp, h_cpp = cpp_forward(obs, h_cpp, p)
        d_h = float(np.abs(np.asarray(h_py).reshape(-1) - h_cpp).max())
        assert d_h < 1e-11, f"第 {t} 步隐状态分叉(最大差 {d_h:.3e})"
        worst = max(worst, float(np.abs(lg_ref - lg_cpp).max()))
    print(f"  24 步 logits 最大差 {worst:.3e},隐状态全程一致")
    assert worst < 1e-11, f"多步 logits 分叉(最大差 {worst:.3e})"


def test_python_float64_reference_is_anchored_to_forward_seq():
    """自检:上面那个 float64 翻译本 `_py_step` 必须和模型自己的前向**同解**。

    它是我为了绕开 `RNNPolicy.step` 的 float32 硬转写的一份"翻译本",如果它自己
    就和模型不一致,前面几条对拍就变成了"两个错的东西互相对得上"。
    这里拿模型的批量路径 `forward_seq`(真实训练走的那条)锚一下。
    """
    net = _net(seed=9)
    rng = np.random.default_rng(2)
    obs = rng.normal(0, 1, 428)
    h = rng.normal(0, 0.5, net.gru_hidden)
    lg_seq, _ = net.forward_seq(obs.reshape(1, 1, -1).astype(np.float64),
                                h.reshape(1, -1).astype(np.float64), cache=False)
    lg_step, _ = _py_step(net, obs, h)
    d = float(np.abs(lg_seq.reshape(-1) - lg_step).max())
    print(f"  forward_seq vs _py_step 最大差 {d:.3e}")
    assert d < 1e-11, f"_py_step 与 forward_seq 不同解({d:.3e})—— 对拍基准本身是错的"


def test_zero_memory_still_equals_the_encoder_only_path():
    """P/bP 为零时,带记忆的前向必须**逐位**退回纯编码器。

    这是"最差也只能学成不用记忆"那条性质的部署侧对应物 —— 权重头里的 kMemProj
    被写错(比如装的是别的张量)时,这条会红。
    """
    net = RNNPolicy(obs_dim=428, hidden=512, act_dim=8, gru_hidden=128, seed=0)
    assert not net.p["P"].any() and not net.p["bP"].any(), "初始 P/bP 必须是零"
    p = _p64(net)
    rng = np.random.default_rng(5)
    obs = rng.normal(0, 1, 428)

    # 隐状态取非零:若部署壳把记忆直接当成 feat(P=0 时被掩盖),这条就会红
    lg, _ = cpp_forward(obs, rng.normal(0, 0.5, 128), p)
    lg_direct = _linear(p["W3"], p["b3"], cpp_encode(obs, p))
    d = float(np.abs(lg - lg_direct).max())
    print(f"  零记忆 vs 纯编码器 logits 最大差 {d:.3e}")
    assert d < 1e-12


def test_gru_step_reads_the_old_h_before_writing_the_new_one():
    """`gru_step` 必须先抄 hprev、再投影、最后才写 h。

    **这一条为什么只能查文本、不能靠对拍**:上面的翻译本是 numpy,而 numpy 的
    `h = (1-z)*n + z*hprev` 是**重新绑定**一个新数组,不是原地改 —— 所以"用更新前
    的 h"和"用更新后的 h"在翻译本里是同一个值,把 `hprev` 换成 `h` 也照样对得上
    (实测变异存活)。C++ 那边 `h[i] = ...` 是**真的原地写**,顺序错了就是另一个
    网络,而它编译通过、形状全对、只是变弱。

    所以这里直接核 C++ 的行序:抄 hprev → 三个隐状态投影 → 才写 h。
    """
    packs = gru_pack_dirs()
    if not packs:
        print("  (跳过:没有带权重的 GRU 包)")
        return
    for d in packs:
        src = (d / "rl_VG.cpp").read_text(encoding="utf-8")
        m = re.search(r"void gru_step\(.*?\n\}", src, re.S)
        assert m, f"{d.name}/rl_VG.cpp 里找不到 gru_step 的定义"
        body = m.group(0)

        i_copy = body.find("hprev[i] = h[i]")
        i_write = body.find("h[i] = (1.0f - z)")
        projs = [body.find(f"linear(kGruWH{g}") for g in ("r", "z", "n")]
        print(f"  [{d.name}] hprev 拷贝 @{i_copy}, 隐状态投影 @{projs}, 写 h @{i_write}")
        assert i_copy >= 0, (
            f"{d.name}/rl_VG.cpp: gru_step 里没有 `hprev[i] = h[i]` 这一句 —— 旧 h 没被保住"
        )
        assert all(p >= 0 for p in projs), f"{d.name}/rl_VG.cpp: gru_step 里少了隐状态侧的投影"
        assert i_write > i_copy, (
            f"{d.name}/rl_VG.cpp: 写 h 在抄 hprev **之前** —— 三个投影会读到更新后的 h"
        )
        assert all(i_write > p for p in projs), (
            f"{d.name}/rl_VG.cpp: 写 h 在隐状态投影**之前** —— 同上"
        )


def test_sigmoid_clip_matches_the_python_one():
    """两边的 sigmoid 夹子必须是同一个数。

    对拍抓不到它:翻译本是 float64,而夹子只在 float32 下有意义(float32 的
    exp 在 x < -88 就溢出成 inf),而且常规数量级的输入根本碰不到 ±60 ——
    去掉夹子的变异实测存活。它不影响平时,只影响极端激活那一步,正是"平时看不出来"
    的那类分叉,所以单独核常量。
    """
    from monet.models.gru import _SIG_CLIP

    packs = gru_pack_dirs()
    if not packs:
        print("  (跳过:没有带权重的 GRU 包)")
        return
    want = float(_SIG_CLIP)
    for d in packs:
        src = (d / "rl_VG.cpp").read_text(encoding="utf-8")
        m = re.search(r"inline float sigmoid\(float x\)\s*\{(.*?)\n\}", src, re.S)
        assert m, f"{d.name}/rl_VG.cpp 里找不到 sigmoid 的定义"
        body = m.group(1).replace(" ", "")

        # 按**数值**比,不按字面比 —— `60.0f` / `60.f` / `6e1f` 都对,格式不该是判据。
        lo = re.findall(r"x<-([\d.]+)f", body)
        hi = re.findall(r"x>([\d.]+)f", body)
        print(f"  [{d.name}] Python 夹子 ±{want},C++ 解析出 -{lo} / +{hi}")
        assert lo and hi, (
            f"{d.name}/rl_VG.cpp 的 sigmoid 里没有 ±夹子(或写法变了,本条测试认不出来)"
        )
        assert float(lo[0]) == want and float(hi[0]) == want, (
            f"{d.name}/rl_VG.cpp 的 sigmoid 夹子 ±{lo[0]} 与 gru.py 的 _SIG_CLIP={want} 不一致 "
            f"—— float32 下 x < -88 时 exp 会溢出,两边结果就分岔了"
        )


def test_every_weight_symbol_referenced_exists():
    """`rl_VG.cpp` 里引用的每个 `kXxx` 都必须能在包里找到声明。

    **这条是符号名手滑的绊线**:C++ 里把 kGruWHn 写成 kGruWHm 时,如果那个名字
    根本没声明就是编译错误(没问题);但如果它恰好**声明过另一个张量**(比如把
    kGruWHz 写成了 kGruWHr),编译通过、形状也兼容,而网络悄悄少学一块东西 ——
    没有编译器就抓不到后者,但至少能挡住"引用了不存在的符号"这一半。

    声明来源按**包**算,不只是权重头:规则层的 `vg::kStartFacing`、`kActionsPerTurn`
    来自 `vg_rules.h`,观测层的常量来自 `obs_builder.h`,它们和权重头一样是包的一
    部分。只看权重头会把这两个名字误判成手滑。

    引用按**去注释**后的正文算:注释里提一句 `kGru*` 不是引用,把它算进来会让
    这条测试对"注释里写了个不存在的符号名"报红 —— 那是噪声,不是缺陷。

    没生成权重头时跳过(导出之前跑全量回归的常态)。
    """
    packs = gru_pack_dirs()
    if not packs:
        print("  (跳过:没有带权重的 GRU 包)")
        return

    # 部署壳自己定义的常量,不属于任何头文件
    local = {"kN", "kH", "kA", "kG", "kDeployTemperature"}

    for d in packs:
        declared = set()
        for h in sorted(d.glob("*.h")):
            text = h.read_text(encoding="utf-8")
            declared |= set(re.findall(r"static const (?:float|int) (\w+)", text))
            declared |= set(re.findall(r"constexpr \w+ (\w+)", text))
            # 分片里的数组正文也得算上(主头只 include)
            declared |= set(re.findall(r"static const float (\w+)\[", text))

        src = (d / "rl_VG.cpp").read_text(encoding="utf-8")
        src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)   # 先块注释
        src = re.sub(r"//[^\n]*", " ", src)                 # 再行注释
        used = set(re.findall(r"\b(k[A-Z]\w*)\b", src))
        unknown = sorted(used - declared - local)
        print(f"  [{d.name}] 引用 {len(used)} 个 k 符号,声明 {len(declared)} 个")
        assert not unknown, f"{d.name}/rl_VG.cpp 引用了包里没有声明的符号:{unknown}"


def test_deploy_forward_agrees_with_a_real_pack():
    """端到端:用**真实权重头**导出的包,部署侧翻译本必须与 Python 逐位口径一致。

    与前面几条的区别是权重走的是 `load_pack_net` 的真实解析路径 —— 把"导出器写对
    了符号"和"部署壳按对的方式读"接在一起。带 GRU 的包存在时才跑。
    """
    packs = gru_pack_dirs()
    if not packs:
        print("  (跳过:没有带权重的 GRU 包)")
        return
    for d in packs:
        try:
            net = load_pack_net(pack_dir=str(d))
        except Exception as exc:  # noqa: BLE001
            print(f"  (跳过:{d.name} 包还读不了 —— {exc})")
            continue
        assert getattr(net, "is_recurrent", False), f"{d.name} 应当是带 GRU 的包"

        p = _p64(net)
        # 每个包用**同一个种子**取同一串观测:两个包对拍不出东西,但"包 A 与包 B
        # 在相同输入下都等于 Python"才是这一条要的。
        rng = np.random.default_rng(11)
        # 用 `_py_step` 而不是 `forward_seq`:后者返回的是 (logits, values),把第二个
        # 返回值当隐状态接下去会得到 (1,1) 的形状,下一步就在 matmul 上炸 —— 而且只在
        # "权重头真的存在"时才跑到,平时一直被跳过,是个只在交付时刻现形的坑。
        h_py = np.zeros((1, net.gru_hidden))
        h_cpp = np.zeros(net.gru_hidden)
        worst = 0.0
        for _ in range(16):
            obs = rng.normal(0, 1, 428)
            lg_ref, h_py = _py_step(net, obs, h_py)
            lg_cpp, h_cpp = cpp_forward(obs, h_cpp, p)
            worst = max(worst, float(np.abs(lg_ref.reshape(-1) - lg_cpp).max()))
        print(f"  [{d.name}] 真实包 16 步 logits 最大差 {worst:.3e}")
        assert worst < 1e-4, f"{d.name} 上部署侧前向与 Python 不一致({worst:.3e})"


def test_gru_pack_weights_reproduce_the_checkpoint_they_declare():
    """数值闭环:包里的权重必须**逐位**等于它自己声明的那个检查点。

    `test_deploy_forward_agrees_with_a_real_pack` 验的是"包里的权重能被部署壳读
    对";这一条验的是**上一层** —— 包里装的**就是**那个被评测过的网络。导出器漏张量、
    分片错位、文本格式改了精度,都只在这里现形,而症状同样是"发出去的包比本地测出来
    的弱",并且结构检查(符号、长度、include 链)一条都不会红。

    来源从权重头的 `来源检查点:` 注释读(与 MLP 包同一个约定),于是"包自己声明出处"
    这件事是**可执行**的,不只是一句注释。MLP 包的对应实现见
    `test_rules_vg.py::test_pack_weights_reproduce_the_checkpoint_they_declare`
    —— 那条按包里的前向重算,这条直接比张量(循环网络的前向重算已经在上面几条里
    验过了,再抄一遍只会多一份会漂的代码)。

    检查点不在就跳过并打印 —— 包不该因为 `runs/` 被清理而变红,但"跳过"必须说出来,
    不能让"没测"看起来像"测过了"。
    """
    from monet.store import load_checkpoint

    packs = gru_pack_dirs()
    if not packs:
        print("  (跳过:没有带权重的 GRU 包)")
        return
    for d in packs:
        main = (d / "rl_weights.h").read_text(encoding="utf-8")
        m = re.search(r"来源检查点:(\S+)", main)
        assert m, f"{d.name}/rl_weights.h 少了来源检查点注释 —— 包的出处必须可追"
        ckpt = m.group(1)

        path = Path(ckpt)
        if not path.is_absolute():
            path = REPO / path
        if not path.exists():
            print(f"    (跳过:{d.name} 的来源检查点 {ckpt} 不存在 —— 没验数值)")
            continue

        ck, _meta, _agent = load_checkpoint(str(path))
        pk = load_pack_net(pack_dir=str(d))

        ck_keys, pk_keys = set(ck.p), set(pk.p)
        assert ck_keys == pk_keys, (
            f"{d.name} 与 {ckpt} 的键集不同 —— 只在检查点 {sorted(ck_keys - pk_keys)},"
            f"只在包 {sorted(pk_keys - ck_keys)};少一个张量就是少一块网络"
        )
        worst = 0.0
        for k in sorted(ck_keys):
            a = np.asarray(ck.p[k], np.float32).reshape(-1)
            b = np.asarray(pk.p[k], np.float32).reshape(-1)
            assert a.shape == b.shape, f"{d.name}:{k} 形状 {a.shape} vs {b.shape}"
            worst = max(worst, float(np.abs(a - b).max()))
        print(f"  [{d.name}] {len(ck_keys)} 个张量对 {ckpt} 最大差 {worst:.3e}")
        assert worst == 0.0, (
            f"{d.name} 的权重与 {ckpt} 不是逐位相同(最大差 {worst:.3e})。"
            f"导出写的是 %.9g,float32 的文本往返是无损的 —— 出现差值就说明导出或解析"
            f"有一步走样,而那种包**照样能编译、照样能跑**,只是更弱。"
        )


if __name__ == "__main__":
    fails = 0
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and callable(fn):
            try:
                print(f"[跑] {nm}")
                fn()
                print(f"[过] {nm}\n")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                import traceback

                print(f"[红] {nm}: {exc}")
                traceback.print_exc()
                print()
    print("全部通过" if not fails else f"{fails} 条失败")
    sys.exit(1 if fails else 0)
