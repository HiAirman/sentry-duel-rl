"""规则状态机的单元测试 —— 每条断言都对应 rules.md / api.md 的一条规则。

可以直接 `python tests/test_engine.py` 跑,也可以交给 pytest。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monet.engine import rules as R  # noqa: E402
from monet.engine.game import (  # noqa: E402
    END,
    FIRE,
    MOVE,
    SCAN,
    TURN_E,
    TURN_N,
    TURN_S,
    TURN_W,
    Game,
)


def make(obstacles=(), zones=(), **kw) -> Game:
    return Game(obstacles=obstacles, score_zones=zones, **kw)


def put(g: Game, color: str, pos, facing=None):
    g.s[color].pos = tuple(pos)
    if facing:
        g.s[color].facing = facing


def end_turn(g: Game):
    """把当前回合走完(R 阶段 + B 阶段)。"""
    if g.phase == "R":
        g.end_phase("R")
    g.end_phase("B")


# --------------------------------------------------------------- 初始状态


def test_initial_state():
    g = make()
    assert g.s["R"].pos == (0, 0) and g.s["R"].facing == "E"
    assert g.s["B"].pos == (6, 6) and g.s["B"].facing == "W"
    assert g.turn == 0 and g.phase == "R" and g.score == (0, 0)
    # 首回合不提供任何敌方情报(api.md §7)
    v = g.view("R")
    assert v.red.last_known_pos == (0, 0) and v.red.last_known_facing == "E"
    assert v.blue.last_known_pos == (-1, -1) and not v.blue.visible
    # 双方 CD 不公开
    assert v.blue.fire_cd == -1 and v.blue.scan_cd == -1


def test_blue_view_is_mirrored():
    g = make()
    v = g.view("B")
    assert v.blue.last_known_pos == (0, 0) and v.blue.last_known_facing == "E"
    assert v.red.last_known_pos == (-1, -1)
    # 镜像后的障碍 / 得分区
    assert set(map(tuple, v.obstacles)) == {R.mirror_pos(p) for p in g.obstacles}
    assert set(map(tuple, v.score_zones)) == {R.mirror_pos(p) for p in g.score_zones}


def test_blue_actions_are_mirrored():
    g = make(obstacles=(), zones=())
    g.end_phase("R")  # 红方先手,蓝方才能行动
    g.apply("B", MOVE)  # 蓝方视角朝 E 前进 → 绝对坐标 (5,6)
    assert g.s["B"].pos == (5, 6)
    assert g.view("B").blue.last_known_pos == (1, 0)
    g2 = make(obstacles=(), zones=())
    g2.end_phase("R")
    g2.apply("B", TURN_N)  # 视角 N → 绝对 S
    assert g2.s["B"].facing == "S"


# --------------------------------------------------------------- 行动与额度


def test_move_forward_only():
    g = make(obstacles=(), zones=())
    g.apply("R", MOVE)
    assert g.s["R"].pos == (1, 0)
    g.apply("R", TURN_S)
    g.apply("R", MOVE)
    assert g.s["R"].pos == (1, 1)


def test_move_rejected_by_wall_and_opponent_and_edge():
    g = make(obstacles=((1, 0),), zones=())
    res, _ = g.apply("R", MOVE)  # 撞障碍
    assert not res.success and not res.consumed
    assert g.s["R"].pos == (0, 0)

    g = make(obstacles=(), zones=())
    put(g, "B", (1, 0))
    res, _ = g.apply("R", MOVE)  # 撞对方
    assert not res.success

    g = make(obstacles=(), zones=())
    put(g, "R", (0, 0), "W")
    res, _ = g.apply("R", MOVE)  # 出界
    assert not res.success


def test_action_budget_three():
    g = make(obstacles=(), zones=())
    for _ in range(3):
        res, _ = g.apply("R", MOVE)
        assert res.success and res.consumed
    assert g.phase_over()
    res, _ = g.apply("R", MOVE)
    assert not res.success and not res.consumed  # §7.1 失败不消耗,但额度已满


def test_end_action_is_free_and_neutral():
    g = make(obstacles=(), zones=())
    res, ev = g.apply("R", END)
    assert res.success and not res.consumed and ev.get("end")
    assert g.s["R"].pos == (0, 0) and not g.phase_over()


# --------------------------------------------------------------- 免费 TURN


def test_free_turn_at_spawn():
    g = make(obstacles=(), zones=())
    res, _ = g.apply("R", TURN_S)  # 开局仍在出生点 → 免费
    assert res.success and not res.consumed
    assert g.s["R"].facing == "S"
    res, _ = g.apply("R", TURN_W)  # 免费资格已用掉
    assert res.success and res.consumed


def test_free_turn_lost_after_leaving_spawn():
    g = make(obstacles=(), zones=())
    g.apply("R", MOVE)  # 离开出生点
    res, _ = g.apply("R", TURN_S)
    assert res.success and res.consumed


def test_free_turn_after_respawn():
    g = make(obstacles=(), zones=())
    put(g, "R", (0, 0), "E")
    put(g, "B", (2, 0), "W")
    g.apply("R", FIRE)  # 击杀 B
    assert g.s["B"].pos == (6, 6) and g.s["B"].facing == "W"
    g.end_phase("R")
    res, _ = g.apply("B", TURN_N)  # 复活后仍在出生点 → 免费
    assert res.success and not res.consumed


def test_free_turn_expires_at_end_of_phase():
    g = make(obstacles=(), zones=())
    g.end_phase("R")  # 红方没使用免费 TURN
    g.end_phase("B")
    assert not g.free_turn["R"]


# --------------------------------------------------------------- FIRE


def test_fire_cooldown_two_turns():
    g = make(obstacles=(), zones=())
    res, _ = g.apply("R", FIRE)
    assert res.success and res.consumed and g.s["R"].fire_cd == 2
    end_turn(g)  # 回合结束统一递减
    assert g.s["R"].fire_cd == 1
    res, _ = g.apply("R", FIRE)
    assert not res.success  # 下一回合不可用
    end_turn(g)
    assert g.s["R"].fire_cd == 0
    res, _ = g.apply("R", FIRE)
    assert res.success  # 再下一回合恢复


def test_fire_hits_3x3_cone():
    for target in [(1, 0), (2, 0), (3, 0), (1, 1), (3, 1), (1, -1), (3, -1)]:
        if not R.in_bounds(*target):
            continue
        g = make(obstacles=(), zones=())
        put(g, "B", target)
        _, ev = g.apply("R", FIRE)
        assert ev.get("kill"), f"target {target} 应被命中"


def test_fire_misses_outside_cone():
    for target in [(4, 0), (0, 0), (0, 1), (0, -1), (2, 2)]:
        if not R.in_bounds(*target):
            continue
        g = make(obstacles=(), zones=())
        put(g, "B", target)
        _, ev = g.apply("R", FIRE)
        assert not ev.get("kill"), f"target {target} 不该被命中"


def test_fire_blocked_by_obstacle_in_same_channel():
    # 障碍在 (2,1):挡住同通道的 (3,1),但不影响 (1,1) 与中间通道
    g = make(obstacles=((2, 1),), zones=())
    put(g, "B", (3, 1))
    _, ev = g.apply("R", FIRE)
    assert not ev.get("kill")

    g = make(obstacles=((2, 1),), zones=())
    put(g, "B", (1, 1))
    _, ev = g.apply("R", FIRE)
    assert ev.get("kill")

    g = make(obstacles=((1, 0),), zones=())
    put(g, "B", (3, 0))
    _, ev = g.apply("R", FIRE)  # 中间通道被 (1,0) 挡住
    assert not ev.get("kill")


def test_kill_scores_and_respawn_keeps_cd():
    g = make(obstacles=(), zones=())
    put(g, "B", (2, 0))
    g.s["B"].scan_cd = 1
    g.apply("R", SCAN)  # 顺便烧掉一个红方额度
    g.actions_used = 0
    _, ev = g.apply("R", FIRE)
    assert ev.get("kill")
    assert g.score == (2, 0)
    assert g.s["B"].pos == (6, 6) and g.s["B"].facing == "W"
    assert g.s["B"].scan_cd == 1  # 复活不清 CD(§6.2)


def test_wall_blocks_fire_and_does_not_hit_shooter():
    g = make(obstacles=((1, 0),), zones=())
    put(g, "B", (2, 0))
    _, ev = g.apply("R", FIRE)
    assert not ev.get("kill")


# --------------------------------------------------------------- SCAN


def test_scan_cd_and_reveal():
    g = make(obstacles=(), zones=())
    put(g, "B", (5, 5), "N")
    res, ev = g.apply("R", SCAN)
    assert res.success and res.consumed and ev.get("scan")
    assert g.s["R"].scan_cd == 3
    assert res.observation.opp_visible  # 本次 act 内实时可见
    assert res.observation.opp_last_known_pos == (5, 5)
    assert res.observation.opp_last_known_facing == "N"
    g.end_phase("R")
    g.end_phase("B")
    assert g.s["R"].scan_cd == 2
    assert not g.scan_reveal["R"]  # 下一回合临时视野消失


def test_scan_rejected_on_cooldown():
    g = make(obstacles=(), zones=())
    g.apply("R", SCAN)
    g.end_phase("R")
    g.end_phase("B")
    res, _ = g.apply("R", SCAN)
    assert not res.success and not res.consumed


# --------------------------------------------------------------- 视野


def test_vision_t_shape():
    g = make(obstacles=(), zones=())
    put(g, "R", (3, 3), "E")
    vis = g.visible_cells("R")
    assert vis == {(4, 3), (5, 2), (5, 3), (5, 4)}


def test_vision_occlusion_is_per_cell():
    # R 在 (2,0) 朝 S;障碍 (1,1) 挡左斜线,(2,1) 畅通
    g = make(obstacles=((1, 1),), zones=())
    put(g, "R", (2, 0), "S")
    vis = g.visible_cells("R")
    assert (2, 1) in vis  # 正前方 1 格
    assert (2, 2) in vis  # 中心远端:挡格是 (2,1),畅通
    assert (3, 2) in vis  # 右斜远端:挡格是 (3,1),畅通
    assert (1, 2) not in vis  # 左斜远端:挡格是 (1,1),是障碍

    # 正前方被挡 → 中心远端也不可见,但左右仍各自判定
    g = make(obstacles=((2, 1),), zones=())
    put(g, "R", (2, 0), "S")
    vis = g.visible_cells("R")
    assert (2, 1) not in vis
    assert (2, 2) not in vis
    assert (1, 2) in vis and (3, 2) in vis


def test_vision_no_rear():
    g = make(obstacles=(), zones=())
    put(g, "R", (3, 3), "E")
    assert (2, 3) not in g.visible_cells("R")


def test_opponent_visible_only_in_vision():
    g = make(obstacles=(), zones=())
    put(g, "R", (0, 0), "E")
    put(g, "B", (2, 0))
    assert g.directly_visible("R")
    put(g, "B", (0, 2))  # 侧后方
    assert not g.directly_visible("R")
    assert not g.view("R").blue.visible


# --------------------------------------------------------------- 计分 / 胜负


def test_zone_scoring_at_end_of_phase():
    zones = ((3, 3),)
    g = make(obstacles=(), zones=zones)
    put(g, "R", (3, 3))
    put(g, "B", (3, 3))  # 蓝方先占上再挪走:这条只验红方单独占点的结算
    g.s["B"].pos = (0, 0)
    g.end_phase("R")
    assert g.score == (1, 0)
    put(g, "B", (3, 3))
    g.end_phase("B")
    assert g.score == (1, 1)  # 分别结算,互不排斥


def test_game_ends_after_20_turns():
    # 0:0 平分 → 常规 20 回合后进入加时,5 个加时回合仍不分胜负 → 平局(§6.3)
    g = make(obstacles=(), zones=())
    guard = 0
    while not g.done and guard < 100:
        end_turn(g)
        guard += 1
    assert g.done and g.turn == 25 and g.winner is None


def test_overtime_then_decision():
    zones = ((3, 3),)
    g = make(obstacles=(), zones=zones, max_turns=1, max_overtime=3)
    put(g, "R", (3, 3))
    put(g, "B", (0, 0), "E")
    end_turn(g)  # R +1 → 1:0
    assert g.done and g.winner == "R" and g.turn == 1


def test_overtime_capped_at_five():
    g = make(obstacles=(), zones=(), max_turns=1, max_overtime=2)
    for _ in range(4):
        if g.done:
            break
        end_turn(g)
    assert g.done and g.winner is None and g.turn == 3  # 1 + 2 个加时


def test_timeout_gives_opponent_a_point():
    g = make(obstacles=(), zones=())
    g.force_timeout("R")
    assert g.score == (0, 1)
    assert g.phase == "B"


# --------------------------------------------------------------- 状态机


def test_phase_order_and_rejection():
    g = make(obstacles=(), zones=())
    assert g.phase == "R"
    try:
        g.apply("B", MOVE)
        raise AssertionError("蓝方不该能在红方阶段行动")
    except RuntimeError:
        pass
    g.end_phase("R")
    assert g.phase == "B"


def test_same_facing_turn_is_legal_and_succeeds():
    """§3.4:转向到当前朝向也是成功行动,legal_actions 不能漏掉它。"""
    g = make()
    assert TURN_E in g.legal_actions("R"), "红方朝 E,同朝向 TURN 必须合法"
    res, _ = g.apply("R", TURN_E)
    assert res.success, "同朝向 TURN 必须判成功"
    assert g.actions_used == 0, "出生点上的首个 TURN 免费,不消耗额度(§4.2)"
    assert not g.free_turn["R"], "免费资格用掉即失效"

    # 免费额度用完后,同朝向 TURN 正常消耗
    g2 = make()
    g2.apply("R", TURN_N)
    before = g2.actions_used
    res2, _ = g2.apply("R", TURN_N)
    assert res2.success and g2.actions_used == before + 1


def test_legal_actions_respects_budget():
    """额度用尽后只剩 END —— 不能再给出 apply 会拒绝的行动。"""
    g = make()
    g.apply("R", TURN_E)  # 免费,不消耗
    for _ in range(3):
        g.apply("R", MOVE)  # (1,0) (2,0) (3,0),各消耗 1
    assert g.actions_used == 3, g.actions_used
    assert g.legal_actions("R") == [END]


def test_legal_actions_contract_matches_apply():
    """压测契约:legal_actions 给出的每个行动,apply 都必须接受。"""
    import copy
    import random

    rnd = random.Random(7)
    g = make()
    checked = 0
    guard = 0
    while not g.done and guard < 3000:
        guard += 1
        legal = g.legal_actions(g.phase)
        assert legal, "行动阶段至少要给出 END"
        for a in legal:
            probe = copy.deepcopy(g)
            _, ev = probe.apply(probe.phase, a)
            assert "rejected" not in ev, (
                f"legal_actions 给出了 apply 拒绝的行动 {a} "
                f"(phase={g.phase} used={g.actions_used}): {ev}"
            )
            checked += 1
        a = rnd.choice(legal)
        g.apply(g.phase, a)
        if g.phase_over() or a == END:
            g.end_phase(g.phase)
    assert checked > 200, f"覆盖样本太少({checked}),契约没被真正压到"


def test_killer_can_share_spawn_with_victim():
    """边界:击杀者站在对方出生点上时,复活会把两人叠在同一格。

    规则没写这种情况,引擎按 §6.2 字面执行(无条件回出生点)。
    后果:同格时 `directly_visible` 必须判为可见(否则隐形的敌人),
    但两人都无法用 FIRE 打到对方 —— 3x3 锥形不含自己所在格。
    """
    g = make()
    g.s["R"].pos, g.s["R"].facing = (6, 6), "W"
    g.s["B"].pos, g.s["B"].facing = (5, 6), "W"
    _, ev = g.apply("R", FIRE)
    assert ev.get("kill"), "红方朝 W 应命中 (5,6) 的蓝方"
    assert g.s["B"].pos == g.s["R"].pos == (6, 6), "两人叠在蓝方出生点"

    assert R.can_see((6, 6), "W", (6, 6), set()), "同格必须可见"
    assert g.directly_visible("R"), "同格时红方必须能看到蓝方"

    # 锥形不含自己所在格 → 叠在一起时谁都打不到谁,只能走开
    assert not R.fire_hits((6, 6), "W", (6, 6), set()), "不能击中自己所在格"
    g.end_phase("R")
    g.apply("B", MOVE)  # 蓝方朝 W,(5,6) 空着,可以脱身
    assert g.s["B"].pos == (5, 6), "蓝方必须能走出同格死锁"


def test_full_random_game_terminates():
    import random

    rnd = random.Random(0)
    g = make()
    guard = 0
    while not g.done and guard < 5000:
        guard += 1
        acts = g.legal_actions(g.phase)
        a = rnd.choice(acts)
        g.apply(g.phase, a)
        if g.phase_over() or a == END:
            g.end_phase(g.phase)
    assert g.done, "随机对局必须能在 20(+5) 回合内结束"
    assert g.turn <= 25


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
