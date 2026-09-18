"""参赛包解析(`monet/pack.py`)的测试。

## 为什么这个文件值得写这么多

权重装错了**不会抛任何异常**。它只会让 rl_v5 变成一个弱智,而弱智对手混在
联盟里几乎看不出来 —— 它照样能赢 random,照样打完一局,只会让训练数据悄悄
变样、让评测表上多一行难看的数字。所以判据必须一层层收紧,而且要**知道每一层
挡得住什么、挡不住什么**。

## 判据及其判别力

| 判据 | 挡什么 | 挡不住什么 |
|---|---|---|
| A 结构 | 缺张量、形状/规格不符、nan | 形状合法的错位(`W1`↔`W2` 都是 512×512) |
| B 字面 C++ 壳前向 | reshape 顺序、索引方向 | 只依赖共用对照表能发现的错 |
| C 行为(vs **camper**) | 张量顺序错位 | 数值微小偏移 |
| D 导出→解析往返 | 分片跟随、`%.9g` 浮点、尺寸校验 | 只依赖共用对照表能发现的错 |
| E 命名约定独立推导 | **对照表本身写错** | — |

**判据 C 必须打 camper,不能打 random** —— 这条与直觉相反:`random` 自己几乎不
得分(0.0~0.6 分/局),一个错位到近乎随机的网络照样能赢它满分,拿 random 当判据
等于没判。camper 会缩在得分区里不出来,逼出"能不能在有限回合内主动结束对局",
才分得开强弱。

判据 B/D 都绕不开 `mlp.TENSORS`(导出与解析共用它),共用的表错了两边一起错,
所以补一条 E 按命名约定独立推导一遍。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from math import erf
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monet import pack as P  # noqa: E402
from monet.env import obs as O  # noqa: E402
from monet.models.mlp import MLP, TENSORS  # noqa: E402
from monet.training.evaluate import evaluate  # noqa: E402
from monet.training.export_cpp import export_weights_header  # noqa: E402
from monet.training.selfplay import STATIC_OPPONENTS  # noqa: E402

try:
    import pytest
except ImportError:  # 仓库的测试也能当普通脚本跑(见文件末尾)
    pytest = None

ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = P.resolve_pack_dir()

# 参考值(40 局确定性,seed=100),用于以后对比漂移
CAMPER_WINRATE_OK = 0.90   # 正确 1.000;W1↔W2 互换 0.500;列主序 0.000
CAMPER_NETSCORE_OK = 15.0  # 正确 +20.50;W1↔W2 互换 +7.00
RANDOM_WINRATE_OK = 0.95   # 正确 1.000;列主序 0.400(判别力弱,只当补充)


class SkipTest(Exception):
    """包不在时跳过 —— 但必须**响亮**(调用方打印原因并计数),不许静默通过。"""


def _skip(reason: str):
    if os.environ.get("MONET_REQUIRE_PACK") == "1":
        raise AssertionError(f"[MONET_REQUIRE_PACK=1] 本该跑的测试被跳过了:{reason}")
    if pytest is not None:
        pytest.skip(reason)
    raise SkipTest(reason)


def _require_pack():
    """拿到真包,拿不到就响亮跳过。测试全绿但其实没测是最坏的结局。"""
    if not (PACK_DIR / P.WEIGHT_HEADER).exists():
        _skip(
            f"外部参赛包不在 {PACK_DIR / P.WEIGHT_HEADER} —— 这几条测试没法跑。"
            f"用 --pack-dir 或环境变量 {P.PACK_ENV_VAR} 指向包目录;"
            f"CI 里设 MONET_REQUIRE_PACK=1 可以把跳过变成失败。"
        )
    return P.load_pack_net()


# ------------------------------------------------ 判据 E:符号名独立推导(不读 TENSORS)


def _symbols_from_naming_convention():
    """只按 `rl_ai_v5.cpp` / `export_cpp.py` 的命名约定推导 符号→参数名。

    刻意**不读 `TENSORS`**:判据 B 与 D 都建立在共用对照表之上,表本身写错的话
    它们是发现不了的(两边一起错)。这条独立推导就是专门挡那个的。
    """
    out = []
    for n in range(3):  # 第 0/1/2 层,每层一个线性 + 一组 LayerNorm
        out += [(f"kW{n}", f"W{n}"), (f"kB{n}", f"b{n}")]
        out += [(f"kLN{n + 1}Gamma", f"ln{n + 1}_g"), (f"kLN{n + 1}Beta", f"ln{n + 1}_b")]
    out += [("kW3", "W3"), ("kB3", "b3")]  # 策略头
    out += [("kW4", "Wv"), ("kB4", "bv")]  # 价值头
    return out


def test_tensor_table_matches_the_naming_convention():
    assert dict(_symbols_from_naming_convention()) == dict(TENSORS), (
        "TENSORS 与命名约定推导的结果不一致 —— 导出与解析共用的这张表是全套判据的地基"
    )


def test_head_shapes_are_distinguishable():
    """策略头与价值头的形状必须不同,否则互换后形状校验抓不住。"""
    net = MLP()
    assert net.p["W3"].shape == (net.act_dim, net.hidden)
    assert net.p["Wv"].shape == (1, net.hidden)
    assert net.p["W3"].shape != net.p["Wv"].shape
    assert net.p["b3"].shape != net.p["bv"].shape


# --------------------------------------------------------- 判据 A:结构


def test_pack_tensors_are_complete_and_shaped_like_the_net():
    net = _require_pack()
    ref = MLP()
    assert set(net.p) == set(ref.p), "装出来的参数集合与 MLP 不一致"
    for sym, param in TENSORS:
        a = net.p[param]
        assert a.shape == ref.p[param].shape, (sym, param, a.shape, ref.p[param].shape)
        assert a.dtype == np.float32, (sym, param, a.dtype)
        assert np.isfinite(a).all(), (sym, param, "含 nan/inf —— 源权重本身坏了")


def test_pack_spec_matches_this_engine():
    """`kObsDim/kHidden/kActDim` 必须与本引擎一致 —— 不一致就该直接拒绝,不许尽力而为。"""
    net = _require_pack()
    assert net.obs_dim == O.OBS_DIM
    assert net.act_dim == O.ACTION_DIM
    assert net.hidden == 512


# ------------------------- 判据 B:照 C++ 逐字转写的前向(主判据,不需要 C++ 工具链)


def _cpp_linear(flat, b, x, in_dim, out_dim):
    """逐字对应 `rl_ai_v5.cpp` 的 `linear()`:

        const float* w_row = W + o * in_dim;
        float s = b[o];  for (i) s += w_row[i] * x[i];

    关键是**扁平寻址、永不 reshape** —— `net.p[param].ravel()` 拿回的就是头文件里
    的原始顺序,再套用 C++ 的 `o*in_dim + i` 规则。这和 `MLP.forward` 的 `X @ W.T`
    是两套独立实现:只要解析时reshape 顺序错了(比如 `order="F"`),ravel 出来的
    顺序就对不上,两边立刻分道扬镳。
    """
    y = np.empty(out_dim, dtype=np.float64)
    for o in range(out_dim):
        row = flat[o * in_dim:(o + 1) * in_dim]
        y[o] = b[o] + float(np.dot(row, x))
    return y


def _cpp_layer_norm(x, gamma, beta):
    """有偏方差 + `1/sqrt(var + 1e-5)`,与 `layer_norm_inplace` 一致。"""
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    return (x - mean) * (1.0 / np.sqrt(var + 1e-5)) * gamma + beta


_erf_vec = np.vectorize(erf, otypes=[np.float64])  # math.erf 只吃标量


def _cpp_gelu(x):
    return 0.5 * x * (1.0 + _erf_vec(x * 0.7071067811865475))


def _cpp_reference_logits(net, obs, ravel_order="C"):
    """照 `rl_ai_v5.cpp` 的 `forward()` 走一遍,返回 logits(float64)。"""
    by_param = {p: s for s, p in _symbols_from_naming_convention()}
    flat = {
        param: net.p[param].ravel(order=ravel_order).astype(np.float64)
        for param in by_param
    }
    # 第 1 层没有残差;第 2、3 层是 `linear(...)` 之后再 `h2[i] += h1[i]`
    h1 = _cpp_linear(flat["W0"], flat["b0"], obs, net.obs_dim, net.hidden)
    h1 = _cpp_gelu(_cpp_layer_norm(h1, flat["ln1_g"], flat["ln1_b"]))
    h2 = _cpp_linear(flat["W1"], flat["b1"], h1, net.hidden, net.hidden) + h1
    h2 = _cpp_gelu(_cpp_layer_norm(h2, flat["ln2_g"], flat["ln2_b"]))
    h3 = _cpp_linear(flat["W2"], flat["b2"], h2, net.hidden, net.hidden) + h2
    h3 = _cpp_gelu(_cpp_layer_norm(h3, flat["ln3_g"], flat["ln3_b"]))
    return _cpp_linear(flat["W3"], flat["b3"], h3, net.hidden, net.act_dim)


def test_pack_forward_matches_cpp_reference():
    """**主判据**:解析出来的权重喂进两条独立前向,结果必须一致。

    实测最大绝对差约 6e-07(float32 舍入)。阈值 1e-5 两头都留了几个数量级的余量:
    列主序误读的差是 ~7.0,差了七个数量级。
    """
    net = _require_pack()
    rng = np.random.default_rng(0)
    for _ in range(3):
        obs = rng.standard_normal(net.obs_dim).astype(np.float32)
        ref = _cpp_reference_logits(net, obs.astype(np.float64))
        got = np.asarray(net.logits(obs), dtype=np.float64).reshape(-1)
        assert got.shape == ref.shape
        assert np.abs(got - ref).max() < 1e-5, np.abs(got - ref).max()


def test_cpp_reference_would_notice_a_transposed_read():
    """反向验证判据 B 的牙:按列主序 ravel 出来的权重必须对不上。

    否则说明这条测试其实抓不住 reshape 顺序,绿了也没意义。
    """
    net = _require_pack()
    rng = np.random.default_rng(1)
    obs = rng.standard_normal(net.obs_dim).astype(np.float64)
    good = _cpp_reference_logits(net, obs)            # 文件原始顺序
    bad = _cpp_reference_logits(net, obs, "F")        # 模拟解析时 reshape 反了
    assert np.abs(good - bad).max() > 1e-3, "两种 ravel 顺序结果一样,判据 B 没有判别力"


# ------------------------------------------- 判据 C:行为(必须打 camper,不是 random)


def _score_vs(name: str, games: int = 40, deterministic: bool = True):
    net = _require_pack()
    opps = {name: STATIC_OPPONENTS[name](100)}
    res = evaluate(net, opps, games=games, seed=100, deterministic=deterministic)[name]
    return res["winrate"], res["net_score"]


def test_pack_opponent_beats_a_camper():
    """主行为判据。camper 缩在得分区里,逼出"能否主动结束对局" —— 强弱分得开。"""
    winrate, net_score = _score_vs("camper")
    assert winrate >= CAMPER_WINRATE_OK, (winrate, "参考值 1.000;错位网络约 0.500")
    assert net_score >= CAMPER_NETSCORE_OK, (net_score, "参考值 +20.50;错位网络约 +7.00")


def test_pack_opponent_beats_random():
    """补充判据:判别力弱(错位网络照样能赢 random),只用来挡"装成了废物"。"""
    winrate, _ = _score_vs("random")
    assert winrate >= RANDOM_WINRATE_OK, (winrate, "参考值 1.000;列主序误读约 0.400")


# ------------------------------------------------- 判据 D:导出 → 解析 往返


def _synth_header(d: Path, obs=428, hid=512, act=8, tensors=None, name=P.WEIGHT_HEADER):
    """写一个最小的、规格自洽的头文件。

    `tensors` 按**符号名**(`kW0`/`kB3`/`kLN2Gamma`,不是 `W0`/`b3`)覆盖单个张量的
    正文;值为 `None` 表示整个符号都不写。键必须是真的符号名 —— 写错了就抛,
    否则覆盖会**静默失效**,测试看起来还在测,其实什么都没改。
    """
    d = Path(d)
    known = {sym for sym, _ in TENSORS}
    unknown = set(tensors or ()) - known
    if unknown:
        raise ValueError(f"tensors 的键必须是符号名,不认识:{sorted(unknown)}(可选 {sorted(known)})")
    net = MLP(obs_dim=obs, hidden=hid, act_dim=act, seed=0)
    parts = [
        f"static const int kObsDim = {obs};",
        f"static const int kHidden = {hid};",
        f"static const int kActDim = {act};",
    ]
    for sym, param in TENSORS:
        want = net.p[param]
        if tensors is not None and sym in tensors:
            override = tensors[sym]
            if override is None:
                continue  # 故意整个符号都不写
            parts.append(override)
            continue
        vals = np.zeros(want.size, dtype=np.float32) if want.ndim else np.zeros(1, np.float32)
        body = ",".join(f"{float(v):.9g}" for v in vals.ravel())
        parts.append(f"static const float {sym}[{want.size}] = {{{body}}};")
    p = d / name
    p.write_text("\n".join(parts), encoding="utf-8")
    return p


def test_export_then_load_round_trips_bit_exact():
    """导出器与解析器互为逆运算 —— 顺带覆盖分片跟随。

    随机网络(不是真权重)才对得上:两边都是我们自己的代码,要求**逐位**相同,
    能同时验证分片 `#include`、`%.9g` 浮点解析和尺寸校验。`max_bytes` 压到很小
    强制走拆分路径。
    """
    src = MLP(obs_dim=428, hidden=16, act_dim=8, seed=3)
    with tempfile.TemporaryDirectory() as td:
        out = export_weights_header(src, Path(td) / P.WEIGHT_HEADER, max_bytes=1024)
        shards = sorted(Path(td).glob(f"{P.WEIGHT_HEADER[:-2]}_part*.h"))
        assert shards, "max_bytes 压这么小应该触发拆分,否则分片路径没被覆盖"

        got = P.load_pack_net(td, hidden=16)
        for sym, param in TENSORS:
            assert np.array_equal(got.p[param], src.p[param]), (sym, param, "往返不逐位相同")
        assert Path(out).exists()


def test_load_pack_net_hands_out_independent_copies():
    """同一进程里反复取的网络必须是各自独立的副本。

    缓存住 `MLP` 本身的话,任何调用方就地改它(喂给 PPO、`load_state_dict`)
    都会**进程级持久**地污染"参考对手",而且几乎没人会去查。
    """
    with tempfile.TemporaryDirectory() as td:
        _synth_header(td, obs=428, hid=8, act=8)
        a = P.load_pack_net(td, hidden=8)
        b = P.load_pack_net(td, hidden=8)
        assert a is not b
        a.p["W0"][:] = 0.0
        c = P.load_pack_net(td, hidden=8)
        assert np.array_equal(c.p["W0"], b.p["W0"]), "污染传到了新副本上"


# --------------------------------------------------------------- import 期无副作用


def test_importing_selfplay_does_not_parse_the_pack():
    """import 期必须零副作用 —— 否则包一缺失连 `export` 都跑不了。

    用子进程:本进程里别的测试早就把包解析过了,缓存状态不可信。环境变量指向
    一个**不存在**的目录,正是要守的那个场景。
    """
    code = (
        "import monet.training.selfplay as S\n"
        "import monet.pack as P\n"
        "assert P._load_weights.cache_info().misses == 0, 'import 期就解析了权重包'\n"
        "assert 'rl_v5' in S.STATIC_OPPONENTS\n"
        "print('ok')\n"
    )
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", P.PACK_ENV_VAR: str(ROOT / "不存在的包")}
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, f"包缺失时 import 训练模块失败了:\n{out.stderr}"
    assert "ok" in out.stdout


def test_lazy_pack_net_defers_parsing_but_checks_existence():
    """构造对手必须便宜(注册表会在 import 期构造),但存在性要立刻查。"""
    if not (PACK_DIR / P.WEIGHT_HEADER).exists():
        with tempfile.TemporaryDirectory() as td:
            _synth_header(td, obs=428, hid=8, act=8)
            before = P._load_weights.cache_info().misses
            P.LazyPackNet(td)
            assert P._load_weights.cache_info().misses == before, "构造时就解析了"
        try:
            P.LazyPackNet(str(ROOT / "不存在的包"))
        except P.PackError as exc:
            assert str(ROOT / "不存在的包") in str(exc), "报错没带上试过的路径"
        else:
            raise AssertionError("包目录不存在时构造对手没报错")
        return
    before = P._load_weights.cache_info().misses
    P.LazyPackNet()
    assert P._load_weights.cache_info().misses == before, "构造 LazyPackNet 就解析了 10MB"


# ------------------------------------------------------------------ 多包注册表


def test_registry_shape_and_unknown_name():
    """注册表 → 目录的映射:名字对得上、目录在已知根下、未注册的名字大声报错。

    包目录有**两个**合法根:与本仓库并列(外部包 rl_v5、冻结的 m3_vN),或者就在
    仓库里(`rl_VG_*` —— 它们和训它们的代码一起进版本库)。所以这里断言的是
    "解析结果落在某个根下、且那个候选目录确实存在",而不是死盯 ROOT.parent。
    """
    assert P.PRIMARY_PACK in P.KNOWN_PACKS
    dirs = P.all_pack_dirs()
    assert set(dirs) == set(P.KNOWN_PACKS), "all_pack_dirs 漏了或多了名字"
    for name, d in dirs.items():
        cands = [(root / P.KNOWN_PACKS[name]).resolve() for root in P.PACK_ROOTS]
        assert d in cands, (name, d, cands)
        assert d.is_dir(), f"{name} 解析到 {d},但那个目录不存在"
    assert len(set(dirs.values())) == len(dirs), f"两个注册名指向同一个目录:{dirs}"
    # 固定默认位置(不随 --pack-dir / 环境变量变),这正是 `PACK_DIR` 之外的第二个口径
    assert P.known_pack_dir(P.PRIMARY_PACK) == P.DEFAULT_PACK_DIR
    # 主包是外部包,不在仓库里 —— 钉住"两个根按序找"没有把 rl_v5 也搬进仓库
    assert P.DEFAULT_PACK_DIR == ROOT.parent / P.KNOWN_PACKS[P.PRIMARY_PACK]

    try:
        P.known_pack_dir("不存在的包")
    except P.PackError as exc:
        assert "不存在的包" in str(exc), str(exc)
        assert "m3_v1" in str(exc), "报错该列出已知的名字,否则没法知道能填什么"
    else:
        raise AssertionError("未注册的包名没有报错")


def test_pack_dir_override_only_moves_the_primary_pack():
    """`--pack-dir` / `$MONET_PACK_DIR` 是给"外部包挪了地方"准备的口子。

    它**不能**顺带改掉我们自己产出的包的位置:`--pack-dir D:/临时/rl_v5` 是
    很自然的用法,而 m3_v1 是我们冻结的对手,被一个全局口子悄悄换掉的话,
    "跟过去打平手"这个信号就没了 —— 而且是静默的。
    """
    before = P.resolve_pack_dir()  # 可能是默认位置,也可能是 $MONET_PACK_DIR 指的别处
    with tempfile.TemporaryDirectory() as td:
        try:
            P.set_pack_dir(td)
            assert P.resolve_pack_dir() == Path(td).resolve()
            assert P.resolve_pack_dir(name="rl_v5") == Path(td).resolve()
            assert P.resolve_pack_dir(name="m3_v1") == P.known_pack_dir("m3_v1")
        finally:
            P.set_pack_dir(None)
    assert P.resolve_pack_dir() == before, "set_pack_dir(None) 没把覆盖清干净"


def test_the_registered_packs_are_pairwise_different():
    """每个注册包都必须是**另一份权重**。

    注册表最容易出的错是把两个名字写成同一个目录(复制粘贴),那样两个对手
    其实是同一个,训练里"多了一个对手"是假的,而任何单包测试都发现不了 ——
    照着一份已有的包目录抄出新包时尤其容易踩。
    """
    # 用 resolve 过的目录判在不在:rl_v5 的位置是可以用 --pack-dir / 环境变量改的
    live = []
    for name in P.KNOWN_PACKS:
        d = P.resolve_pack_dir(name=name)
        if not (d / P.WEIGHT_HEADER).exists():
            _skip(f"注册包 {name!r} 不在 {d} —— 没法比对权重")
        live.append(name)

    obs = np.zeros(O.OBS_DIM, dtype=np.float32)
    out = {n: P.load_pack_net(name=n).logits(obs) for n in live}
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            assert np.abs(out[a] - out[b]).max() > 1e-3, (
                f"{a} 与 {b} 的前向输出一样 —— 多半指向了同一份权重"
            )


def test_every_default_roster_name_can_actually_be_built():
    """默认名单里的每个名字都必须真的构造得出对手 —— **光在两张表里同步是不够的**。

    名单里的名字没在 `pack.KNOWN_PACKS` 里注册的话,`LazyPackNet(名字)` 找不到
    目录,`League()` 会在**训练启动时**抛 PackError,eval 的默认名单也一样崩。
    只比对两张名单是否一致的测试走不到"构造"这一步,发现不了。

    构造本身是便宜的:`LazyPackNet.__init__` 只查文件在不在,不解析 10MB 权重。
    """
    from monet.training.selfplay import EVAL_DEFAULT, LEAGUE_DEFAULT, STATIC_OPPONENTS
    from monet.training.selfplay import League

    names = list(dict.fromkeys(list(LEAGUE_DEFAULT) + list(EVAL_DEFAULT)))
    unknown = [n for n in names if n not in STATIC_OPPONENTS]
    assert not unknown, f"名单里有没注册进 STATIC_OPPONENTS 的名字:{unknown}"

    try:
        league = League(names, seed=0)
    except P.PackError as exc:
        if "未注册的参赛包" in str(exc):
            # 这不是"外部包不在",是名单里有个包压根没进 KNOWN_PACKS —— 我们的 bug
            raise AssertionError(
                f"默认名单能构造,但注册表漏了包:{exc}"
                f"(加一个包 = KNOWN_PACKS 一行 + STATIC_OPPONENTS 一条,缺一不可)"
            ) from exc
        _skip(f"默认名单里有包不在本机:{exc}")
    assert league.names() == names, (league.names(), names)


def test_m3_v1_is_registered_as_an_eval_opponent():
    """口径:自己的导出**只进评测集**,权重 1.0,不算官方。

    m3_v* 与当前网络同源,当陪练的梯度价值低;留在评测集里是当"别退步"的锚。
    这条把两个名单一起钉住,防止它被挪回训练联盟。
    """
    from monet.training.selfplay import (
        EVAL_DEFAULT,
        LEAGUE_DEFAULT,
        LOW_WEIGHT,
        OFFICIAL_OPPONENTS,
        STATIC_OPPONENTS,
    )

    assert "m3_v1" in STATIC_OPPONENTS, "m3_v1 没注册进对手表"
    assert "m3_v1" in EVAL_DEFAULT, "m3_v1 是评测集的锚,不能被摘掉"
    assert "m3_v1" not in LEAGUE_DEFAULT, "m3_v1 已按口径移出训练联盟"
    assert "m3_v1" not in LOW_WEIGHT and "m3_v1" not in OFFICIAL_OPPONENTS


# ------------------------------------------------------------------ 失败路径


def test_export_refuses_to_clobber_the_reference_pack():
    """`export` 不能往**任何**参赛包目录里写 —— 那是参照对手,覆盖了没法重新生成。

    而且 `_load_weights` 有 lru_cache,同一进程里刚读过的旧对象还在,
    覆盖了当场都可能看不出来。子目录也一起挡。

    注意护栏只挡注册过的包:往 `rl_VG_v0/`(待提交的活包)反复导出是常态。
    """
    net = MLP(obs_dim=428, hidden=8, act_dim=8, seed=0)
    for name, pack_dir in P.all_pack_dirs().items():
        for target in (pack_dir / P.WEIGHT_HEADER, pack_dir / "sub" / P.WEIGHT_HEADER):
            try:
                export_weights_header(net, target)
            except ValueError as exc:
                assert "参赛包目录" in str(exc), str(exc)
                assert name in str(exc), f"报错没说是哪个包:{exc}"
            else:
                raise AssertionError(f"没拦住写入参赛包目录 {pack_dir}({name}):{target}")

    with tempfile.TemporaryDirectory() as td:  # 正常目录必须照写不误
        assert export_weights_header(net, Path(td) / P.WEIGHT_HEADER).exists()


def test_missing_pack_dir_raises_with_the_tried_path():
    missing = str(ROOT / "没有这个包")
    try:
        P.load_pack_net(missing)
    except P.PackError as exc:
        assert missing in str(exc), str(exc)
    else:
        raise AssertionError("包目录不存在时没有报错")


def test_header_missing_a_symbol_names_it():
    with tempfile.TemporaryDirectory() as td:
        _synth_header(td, obs=428, hid=8, act=8, tensors={"kW0": None})
        try:
            P.load_pack_net(td, hidden=8)
        except P.PackError as exc:
            assert "kW0" in str(exc), str(exc)
            assert td in str(exc), "报错没带上试过的路径"
        else:
            raise AssertionError("缺符号时没有报错")


def test_declared_size_mismatch_is_reported_as_a_spec_mismatch():
    """声明的 `[N]` 与期望不符时,不能报成"找不到符号" —— 那会把方向带偏。"""
    with tempfile.TemporaryDirectory() as td:
        _synth_header(td, obs=428, hid=8, act=8, tensors={"kB4": "static const float kB4[7] = {0,0,0,0,0,0,0};"})
        try:
            P.load_pack_net(td, hidden=8)
        except P.PackError as exc:
            assert "kB4" in str(exc) and "7" in str(exc), str(exc)
        else:
            raise AssertionError("声明尺寸不符时没有报错")


def test_truncated_tensor_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        _synth_header(td, obs=428, hid=8, act=8, tensors={"kB3": "static const float kB3[8] = {1,2,3};"})
        try:
            P.load_pack_net(td, hidden=8)
        except P.PackError as exc:
            assert "kB3" in str(exc), str(exc)
        else:
            raise AssertionError("张量被截断时没有报错")


def test_non_finite_weight_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        _synth_header(td, obs=428, hid=8, act=8, tensors={"kB4": "static const float kB4[1] = {nan};"})
        try:
            P.load_pack_net(td, hidden=8)
        except P.PackError as exc:
            assert "kB4" in str(exc), str(exc)
        else:
            raise AssertionError("权重含 nan 时没有报错")


def test_synth_header_rejects_a_misspelled_override_key():
    """守住上面那条:覆盖键写错必须炸,不能静默失效(否则测试看着在测其实没测)。"""
    with tempfile.TemporaryDirectory() as td:
        try:
            _synth_header(td, obs=428, hid=8, act=8, tensors={"b3": "static const float b3[8] = {};"})
        except ValueError as exc:
            assert "b3" in str(exc), str(exc)
        else:
            raise AssertionError("覆盖键写错时没有报错")


def test_wrong_hidden_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        _synth_header(td, obs=428, hid=8, act=8)
        try:
            P.load_pack_net(td, hidden=512)
        except P.PackError as exc:
            assert "hidden" in str(exc), str(exc)
        else:
            raise AssertionError("hidden 不符时没有报错")


def test_round_trip_of_a_synthetic_pack_is_exact():
    """小规格包走一遍完整链路,保证上面那些失败路径的"成功对照"是成立的。"""
    with tempfile.TemporaryDirectory() as td:
        _synth_header(td, obs=428, hid=8, act=8)
        net = P.load_pack_net(td, hidden=8)
        assert net.hidden == 8 and net.obs_dim == O.OBS_DIM
        assert np.isfinite(net.logits(np.zeros(O.OBS_DIM, dtype=np.float32))).all()


if __name__ == "__main__":
    # pytest 在场时 `_skip` 抛的是它自己的 Skipped,脚本模式也得认,否则跳过会变成崩溃
    _SKIP_EXC = (SkipTest, pytest.skip.Exception) if pytest is not None else (SkipTest,)
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok    {name}")
        except _SKIP_EXC as exc:
            skipped += 1
            print(f"  SKIP  {name}:{getattr(exc, 'msg', exc)}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    tag = f",{skipped} skipped" if skipped else ""
    print(f"\n{len(fns) - failed - skipped}/{len(fns)} passed{tag}")
    sys.exit(1 if failed else 0)
