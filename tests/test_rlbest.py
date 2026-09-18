"""`RL_best040` 参赛包移植层测试:412 维观测 + 权重前向 + 对手行为。

覆盖四件事(对应 `monet/env/obs_rlbest.py` 与 `monet/env/rlbest.py`):

1. **观测规格**:412 = 8x49 + 20,平面/标量与 `RL_best040-source/obs_builder.h`
   逐位对应,且**不是**本仓库那套 428 维。
2. **权重与推理**:`rl_weights.h` 解析出的张量形状,以及前向结果与
   `my_ai.cpp::matmul` 的字面转写(逐输出行、float32 顺序累加)一致。
3. **信念**:位掩码重写版与 C++ 三重循环的字面版**逐格等价**。
4. **对手**:掩码规则、被拒动作不消耗额度要重试、一整局能跑完不收尾。

行为层面的强弱不在这里断言(那是 `tests/diag_rlbest.py` 的事)—— 网络强不强是
权重的事,这里只保证"喂给权重的东西是对的"。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 同目录的 diag_rlbest

from monet.engine import rules as R  # noqa: E402
from monet.engine.game import END, FIRE, MOVE, SCAN  # noqa: E402
from monet.engine.types import ActionObservation, ActionResult  # noqa: E402
from monet.env import obs as O  # noqa: E402
from monet.env import obs_rlbest as B  # noqa: E402
from monet.env.obs import ObsBuilder  # noqa: E402
from monet.env.rlbest import (  # noqa: E402
    RLBestError,
    RLBestOpponent,
    TENSORS,
    load_rlbest_net,
    resolve_best040_dir,
)

_GAME = None


def _game():
    """一个刚开局、只用于取 `view` 的 Game(不推进)。"""
    from monet.engine.game import Game

    return Game()


def _start_view(color="R"):
    return _game().view(color)


# --------------------------------------------------------------------- 观测规格


def test_obs_dim_matches_deploy_header():
    # 必须与 RL_best040-source/obs_builder.h 的 kObsDim 一致
    assert B.OBS_DIM == 412
    assert B.PLANES * B.CELLS + B.SCALARS == 412
    assert B.ACTION_DIM == 8
    # 与部署侧同一份头文件里的常量对上
    assert (B.BOARD_SIZE, B.PLANES, B.SCALARS) == (7, 8, 20)


def test_two_obs_spaces_are_not_interchangeable():
    """412 和 428 是两套观测:长度必须不同,否则说明有一边改错了。"""
    assert O.OBS_DIM == 428
    assert B.OBS_DIM != O.OBS_DIM
    # 平面/棋盘是同一套,只有标量段不同
    assert (B.PLANES, B.CELLS) == (O.PLANES, O.CELLS)
    assert O.SCALARS == 36 and B.SCALARS == 20


def test_plane_layout_at_game_start():
    ob = B.BeliefObs()
    obs = B.encode_board(_start_view("R"), "R", ob)
    assert obs.shape == (412,) and obs.dtype == np.float32
    assert np.isfinite(obs).all()

    plane = lambda i: obs[i * 49 : (i + 1) * 49].reshape(7, 7)  # noqa: E731
    assert plane(0)[1, 1] == 1.0 and plane(0)[5, 5] == 1.0  # 障碍
    assert plane(0).sum() == 2.0
    assert plane(1).sum() == 5.0 and plane(1)[3, 3] == 1.0  # 中心十字得分区
    assert plane(2)[0, 0] == 1.0 and plane(2).sum() == 1.0  # 我方在 (0,0)
    assert plane(3)[6, 6] == 1.0 and plane(3).sum() == 1.0  # 敌方信念 = 出生点
    assert plane(4).sum() == 0.0  # 此刻没看见
    assert plane(5).sum() == 0.0  # 还没有情报
    assert plane(6)[0, 0] == 1.0 and plane(7)[6, 6] == 1.0  # 出生点常量

    s = obs[8 * 49 :]
    assert s[1] == 1.0  # 朝向 E
    assert s[8] == 1.0  # 敌方朝向未知
    assert s[9] == 0.0 and s[10] == 0.0
    assert s[11] == 0.0 and s[12] == 0.0 and s[13] == 0.0
    assert s[14] == 0.0 and s[15] == 0.0  # 红方
    assert s[16] == 0.0 and s[17] == 0.0
    assert s[18] == 1.0  # 无情报 → 1.0(与 C++ 一致,不是 0)
    assert s[19] == 0.0

    # 蓝方视角:镜像后自己仍在 (0,0),is_blue 置位
    obs_b = B.encode_board(_start_view("B"), "B", B.BeliefObs())
    assert obs_b[2 * 49 + 0] == 1.0
    assert obs_b[3 * 49 + 6 * 7 + 6] == 1.0  # 红方在其镜像视角的 (6,6)
    assert obs_b[8 * 49 + 15] == 1.0


def test_scalars_follow_the_cpp_index_order():
    ob = B.BeliefObs()
    ob.my_facing, ob.intel_facing = "S", "W"
    ob.fire_cd, ob.scan_cd = 2, 1
    ob.my_score, ob.opp_score = 7, 5
    ob.turn = 12
    ob.actions_used = 2
    ob.is_blue = True
    ob.opp_visible = True
    ob.opp_directly_visible = True
    ob.intel_pos, ob.intel_turn = (3, 3), 6
    ob.free_turn = True
    s = ob.encode()[8 * 49 :]
    assert list(s[:20]) == [
        0.0, 0.0, 1.0, 0.0,              # 我方朝向 S
        0.0, 0.0, 0.0, 1.0, 0.0,         # 敌方情报朝向 W
        2 / 3, 1 / 3, 7 / 20, 5 / 20, 12 / 24, 2 / 3,
        1.0, 1.0, 1.0,
        6 / 24,                          # (turn - intel_turn)/24
        1.0,
    ]


def test_encode_is_finite_and_planes_stay_binary():
    """跑完整对局,每个决策点的观测都合格(412 / finite / 单热平面 / 标量有界)。"""
    from monet.env.opponents import RandomOpponent

    from diag_rlbest import play_game

    checked = 0
    for g, color in enumerate(("R", "B")):
        # 红蓝各来一局:两个视角下镜像、is_blue、平面 6/7 都要过一遍
        res = play_game(
            RLBestOpponent(seed=0),
            RandomOpponent(seed=1 + g),
            ai_color=color,
            check_obs=True,
        )
        assert res.turns == 20, res.turns  # 一整局走满 20 回合
        assert not res.obs_violations, res.obs_violations[:3]
        checked += res.obs_checked
    assert checked >= 20, checked


def test_action_mask_rules():
    ob = B.BeliefObs()
    ob.act_start(_start_view("R"), "R")
    m = ob.action_mask()
    assert m.shape == (8,) and m.dtype == np.float32
    assert m[MOVE] == 1.0  # (0,0) 朝 E → (1,0) 可走
    assert m[2] == 0.0  # TURN_E = 当前朝向 → 掩掉
    assert m[1] == m[3] == m[4] == 1.0
    assert m[FIRE] == 1.0 and m[SCAN] == 1.0 and m[END] == 1.0

    # 额度用完 → 只能 END
    ob.actions_used = 3
    m = ob.action_mask()
    assert m[END] == 1.0 and m[:7].sum() == 0.0

    # 出界 / 障碍挡路
    ob.actions_used = 0
    ob.my_pos, ob.my_facing = (6, 0), "E"
    assert ob.action_mask()[MOVE] == 0.0
    ob.my_pos, ob.my_facing = (1, 0), "S"  # (1,1) 是障碍
    assert ob.action_mask()[MOVE] == 0.0

    # 直接看得见的敌人占着的格子不往里走(但隐形的敌人不掩,交给引擎拒)
    ob.my_pos, ob.my_facing, ob.intel_pos = (3, 3), "E", (4, 3)
    ob.opp_directly_visible = False
    assert ob.action_mask()[MOVE] == 1.0
    ob.opp_directly_visible = True
    assert ob.action_mask()[MOVE] == 0.0


# --------------------------------------------------------------------- 权重与推理


def test_weight_pack_shapes():
    net = load_rlbest_net()
    for sym, param, shape in TENSORS:
        got = getattr(net, param)
        assert got.shape == shape, (sym, got.shape, shape)
        assert got.dtype == np.float32
        assert np.isfinite(got).all(), sym
    assert net.W0.shape == (256, 412)  # 输入维度必须与 412 对齐
    assert resolve_best040_dir().is_dir()


def _literal_matmul(w, b, x, nin, nout, relu):
    """`my_ai.cpp::matmul` 的字面转写:`wr = w + o*nin`,float32 顺序累加。"""
    w = np.ascontiguousarray(w, dtype=np.float32).reshape(nout, nin)
    b = np.asarray(b, dtype=np.float32).reshape(nout)
    x = np.asarray(x, dtype=np.float32).reshape(nin)
    y = np.zeros(nout, dtype=np.float32)
    for o in range(nout):
        s = np.float32(b[o])
        for i in range(nin):
            s = np.float32(s + np.float32(w[o, i] * x[i]))
        y[o] = np.float32(s if (relu and s > 0.0) else (0.0 if relu else s))
    return y


def test_forward_matches_literal_cpp_matmul():
    net = load_rlbest_net()
    rng = np.random.default_rng(0)
    samples = [
        rng.standard_normal(412).astype(np.float32),
        rng.random(412).astype(np.float32),
        np.asarray(B.encode_board(_start_view("B"), "B", B.BeliefObs()), dtype=np.float32),
    ]
    for x in samples:
        h1 = _literal_matmul(net.W0, net.b0, x, 412, 256, True)
        h2 = _literal_matmul(net.W1, net.b1, h1, 256, 256, True)
        want = _literal_matmul(net.W2, net.b2, h2, 256, 8, False)
        got = net.logits(x)
        assert int(np.argmax(got)) == int(np.argmax(want)), (got, want)
        assert float(np.max(np.abs(got - want))) < 1e-3


def test_forward_is_deterministic_and_obs_len_checked():
    net = load_rlbest_net()
    x = B.encode_board(_start_view("R"), "R", B.BeliefObs())
    assert np.array_equal(net.logits(x), net.logits(x))
    try:
        net.logits(np.zeros(428, dtype=np.float32))
    except RLBestError:
        pass
    else:  # 428 维喂进来必须炸,不能静默对着错位的输入算
        raise AssertionError("428 维观测没有被拒绝")


# ------------------------------------------------------------------------ 信念


def _slow_dilate(belief, my_pos, obstacles):
    """`obs_builder.h::dilate_belief` 的字面转写(逐格三重循环 + 跨轮累加)。"""
    nxt = list(belief)
    for _ in range(3):
        cur = list(nxt)
        for y in range(7):
            for x in range(7):
                if cur[y * 7 + x] <= 0.0:
                    continue
                for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < 7 and 0 <= ny < 7):
                        continue
                    if (nx, ny) in obstacles:
                        continue
                    if (nx, ny) == my_pos:
                        continue
                    nxt[ny * 7 + nx] = 1.0
    return nxt


def _slow_subtract(belief, my_pos, my_facing, obstacles):
    """`obs_builder.h::subtract_visible_cells` 的字面转写(逐格调 can_see)。"""
    out = list(belief)
    for y in range(7):
        for x in range(7):
            if out[y * 7 + x] > 0.0 and R.can_see(my_pos, my_facing, (x, y), obstacles):
                out[y * 7 + x] = 0.0
    return out


def test_belief_rewrite_matches_the_literal_cpp_reference():
    """位掩码版与字面版逐格等价 —— 蒙 256 组随机状态。"""
    rng = np.random.default_rng(7)
    obstacles = [(1, 1), (5, 5)]
    for _ in range(256):
        belief = [float(v) for v in rng.integers(0, 2, size=49)]
        my_pos = (int(rng.integers(0, 7)), int(rng.integers(0, 7)))
        my_facing = ["N", "E", "S", "W"][int(rng.integers(0, 4))]

        ob = B.BeliefObs()
        ob.obstacles = list(obstacles)
        ob._obs_frozen = frozenset(obstacles)
        ob.my_pos, ob.my_facing = my_pos, my_facing
        ob.belief = list(belief)
        ob._dilate_belief()
        want = _slow_dilate(belief, my_pos, set(obstacles))
        assert all((a > 0.0) == (b > 0.0) for a, b in zip(ob.belief, want)), (
            my_pos,
            my_facing,
            belief,
            ob.belief,
            want,
        )

        ob.belief = list(want)
        ob._subtract_visible_cells()
        want2 = _slow_subtract(want, my_pos, my_facing, set(obstacles))
        assert ob.belief == want2, (my_pos, my_facing, ob.belief, want2)


def test_dilate_respects_obstacles_and_my_cell():
    ob = B.BeliefObs()
    ob.obstacles = [(1, 1), (5, 5)]
    ob._obs_frozen = frozenset(ob.obstacles)
    ob.my_pos, ob.my_facing = (3, 3), "N"
    ob.belief = [0.0] * 49
    ob.belief[3 * 7 + 3] = 1.0  # 假设敌人就在我脚下(不可能,但用来测边界)
    ob._dilate_belief()
    assert ob.belief[3 * 7 + 3] == 1.0  # 起点保留(扩张是并集)
    assert ob.belief[2 * 7 + 3] == 1.0  # N
    assert ob.belief[3 * 7 + 4] == 1.0  # E
    assert sum(ob.belief) >= 5.0


# ------------------------------------------------------------------------ 对手


def _drive(opp, color="R", sends=(), obs_arg=None):
    """手动驱动 `Opponent.turn` 生成器:先要一个动作,再逐个把结果发回去。"""
    gen = opp.turn(lambda: _start_view(color), color, obs_arg)
    acts = []
    try:
        a = gen.send(None)
        for res in sends:
            acts.append(a)
            a = gen.send(res)
        acts.append(a)
    except StopIteration:
        pass
    finally:
        gen.close()
    return acts


def test_opponent_ignores_the_envs_428_obs_builder():
    """对手必须用自己的 412 维观测 —— 传进来的 428 维 builder 应当被无视。"""
    opp = RLBestOpponent(seed=0)
    a_none = _drive(opp, "R", obs_arg=None)
    opp2 = RLBestOpponent(seed=0)
    a_env = _drive(opp2, "R", obs_arg=ObsBuilder())
    assert a_none == a_env and a_none, (a_none, a_env)


def test_opponent_retries_rejected_actions_without_spending_budget():
    opp = RLBestOpponent(seed=0)
    first = _drive(opp, "R")[0]
    # 第一次行动被引擎拒绝(success=False, consumed=False)→ 必须换一个动作重试。
    # 观测按引擎真实会返回的样子填(被拒时是"当前状态",朝向/位置一定合法)。
    rej = ActionResult(
        success=False,
        consumed=False,
        observation=ActionObservation(
            my_pos=(0, 0), my_facing="E", opp_last_known_pos=(-1, -1), opp_last_known_facing="?"
        ),
    )
    acts = _drive(RLBestOpponent(seed=0), "R", sends=[rej, rej, rej])
    assert acts and acts[0] == first
    assert acts[1] != acts[0], acts  # 被拒的动作被屏蔽,不会原地重试


def test_opponent_stops_after_three_consumed_actions():
    opp = RLBestOpponent(seed=0)
    ok = ActionResult(
        success=True,
        consumed=True,
        observation=ActionObservation(my_pos=(1, 0), my_facing="E"),
    )
    acts = _drive(opp, "R", sends=[ok, ok, ok])
    assert len(acts) == 3, acts  # 3 个额度用完就收手,不会多要第 4 个


def test_opponent_plays_a_full_game_within_horizon():
    from monet.env.opponents import RandomOpponent

    from diag_rlbest import play_game

    for color in ("R", "B"):
        ai = RLBestOpponent(seed=0)
        res = play_game(ai, RandomOpponent(seed=3), ai_color=color, check_obs=True)
        assert res.turns <= 25, res.turns  # 20 回合上限(+ 加时)
        assert res.actions <= 3 * res.turns, (res.actions, res.turns)
        assert res.illegal == 0
        assert not res.obs_violations


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
