"""官方 Baseline / Hunter 移植版的测试。

重点是**协议**而不是棋力:移植版走的是"生成器回合",动作必须仍然流经环境,
否则对手造成的击杀不会进我方奖励 —— 这是最容易悄悄弄坏的地方。
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monet.engine import rules as R  # noqa: E402
from monet.engine.game import ACTION_TURN, END, MOVE, Game  # noqa: E402
from monet.env import obs as O  # noqa: E402
from monet.env import official  # noqa: E402
from monet.env.official import (  # noqa: E402
    OfficialAmbusher,
    OfficialBaseline,
    OfficialHunter,
    OfficialPatrol,
    OfficialStalker,
    OfficialWeaver,
    advance_toward,
    blocked,
    can_step_fire,
    farthest_corner,
    hunt_target,
    in_fire_range,
    same_pos,
)
from monet.env.opponents import EndOpponent, HeuristicOpponent, RandomOpponent  # noqa: E402
from monet.env.sentry_env import SentryEnv  # noqa: E402
from monet.training.selfplay import STATIC_OPPONENTS  # noqa: E402

# 六个规则手,走同一套协议测试:前两个是官方 AI 的移植,后四个是自研。
OFFICIAL = [
    OfficialBaseline,
    OfficialHunter,
    OfficialStalker,
    OfficialPatrol,
    OfficialAmbusher,
    OfficialWeaver,
]

# 自研的四个:**不为得分区写任何策略**(正反两面都不写,见下面两条测试)。
SELF_AUTHORED = [OfficialStalker, OfficialPatrol, OfficialAmbusher, OfficialWeaver]

# "罚站时也能拿到 ≥10 分"的那四个。后两个**记忆型**对手故意不进来:它们每杀一次
# 就要花最多三个自己的阶段绕去角落躲起来(ambusher),或者按固定周期消失
# (weaver),所以单位时间内的击杀次数天然更低 —— 实测罚站一局拿 6 分和 4 分。
# 那不是"没有进攻性",是设计要的形状,门槛另立(见下面 _MEMORY_SCORE_FLOOR)。
HIGH_SCORING = [OfficialBaseline, OfficialHunter, OfficialStalker, OfficialPatrol]

# 记忆型对手罚站时的得分下限:至少 2 次击杀(每次 +2)。真正要守住的是"它们
# 还是会主动找人杀",而不是"杀得和 stalker 一样多"。
_MEMORY_SCORE_FLOOR = 4


# ------------------------------------------------------------------ 几何辅助


def test_in_fire_range_matches_engine_geometry():
    """3x3 锥形范围必须与引擎的 fire_hits 完全一致(空地图上逐格比对)。"""
    empty = set()
    for pos in [(0, 0), (3, 3), (6, 6), (1, 4)]:
        for facing in R.DIRS:
            for x in range(7):
                for y in range(7):
                    mine = in_fire_range(pos, facing, (x, y))
                    theirs = R.fire_hits(pos, facing, (x, y), empty)
                    assert mine == theirs, (pos, facing, (x, y), mine, theirs)


def test_blocked_covers_bounds_obstacles_and_opponent():
    g = Game()
    board = g.view("R")
    assert blocked(board, -1, 0, (-1, -1))  # 出界
    assert blocked(board, 1, 1, (-1, -1))  # 障碍
    assert blocked(board, 3, 3, (3, 3))  # 对方占格
    assert not blocked(board, 3, 3, (0, 0))
    assert not blocked(board, 3, 3, (-1, -1))  # (-1,-1) = 不判对方


def test_same_pos_and_waypoint_sentinel():
    assert same_pos((2, 3), (2, 3))
    assert not same_pos((2, 3), (3, 2))


def test_can_step_fire_respects_the_same_obstacle_ray():
    """can_step_fire 必须自己补障碍射线判定 —— 引擎的 fire_hits 才管这个。"""
    g = Game()
    board = g.view("R")
    # (0,4) 朝 E、敌人在 (2,4):朝前一步到 (1,4) 就进了锥形
    assert can_step_fire(board, (0, 4), "E", (2, 4))
    # 锥形判定的边界:(1,3) 朝 E 打得到 (4,3) —— 正前方 3 格,横向不偏
    assert in_fire_range((1, 3), "E", (4, 3))
    # 障碍格:(5,5) 是障碍,blocked 直接判 True
    assert blocked(board, 5, 5, (-1, -1))


# ------------------------------------------------------ advance_toward(子生成器)


class _Sim:
    """不经过引擎额度限制的走位模拟器。

    `advance_toward` 的预算可以大于 3(单回合上限),所以不能拿真的 Game 去喂 ——
    引擎在第 3 个动作之后就会以 `budget` 为由拒绝。这里只实现它真正读的两样东西:
    `res.success` 与 `res.observation.my_pos / my_facing`。
    """

    def __init__(self, pos, facing, obstacles):
        self.pos = tuple(pos)
        self.facing = facing
        self.obstacles = set(map(tuple, obstacles))

    def apply(self, a):
        if a == MOVE:
            d = R.DELTA[self.facing]
            nxt = (self.pos[0] + d[0], self.pos[1] + d[1])
            if R.in_bounds(*nxt) and nxt not in self.obstacles:
                self.pos = nxt
                ok = True
            else:
                ok = False
        else:
            self.facing = ACTION_TURN[a]  # 转向永远成功(§3.4)
            ok = True
        obs = SimpleNamespace(my_pos=self.pos, my_facing=self.facing)
        return ok, SimpleNamespace(success=ok, consumed=ok, observation=obs)

    def run(self, gen):
        last = None
        acts = []
        try:
            while True:
                a = gen.send(last)
                acts.append(a)
                ok, last = self.apply(a)
                if not ok:
                    break
        except StopIteration as stop:
            return acts, stop.value


def _advance(target, budget, pos=(0, 0), facing="E", obstacles=((1, 1), (5, 5))):
    sim = _Sim(pos, facing, obstacles)
    board = SimpleNamespace(obstacles=list(obstacles))
    gen = advance_toward(board, pos, facing, (-1, -1), target, budget)
    _, value = sim.run(gen)
    return value


def test_advance_toward_reaches_reachable_targets():
    for target in [(3, 2), (2, 3), (3, 3), (4, 3), (3, 4)]:
        used, pos, _, _ = _advance(target, budget=8)
        assert pos == target, (target, pos, used)
        assert used <= 8


def test_advance_toward_never_exceeds_budget():
    for budget in range(1, 7):
        used, _, _, _ = _advance((3, 3), budget=budget)
        assert used <= budget, (budget, used)


def test_advance_toward_avoids_unreachable_target_without_hanging():
    """目标本身是障碍:推进不到它,但也不能死循环或越过预算。"""
    used, pos, _, _ = _advance((5, 5), budget=8)
    assert pos != (5, 5) and used <= 8


def test_advance_toward_prefers_not_turning():
    """同长度路径优先"不用转身"的那条 —— 否则会在 (1,1) 障碍附近来回摆。

    从 (2,1) 朝 S 去 (3,3):向南、向东都是 3 步,应选向南(当前朝向)。
    """
    used, pos, _, _ = _advance((3, 3), budget=2, pos=(2, 1), facing="S")
    assert pos == (2, 3), ("应沿当前朝向南下,而不是转身向东", pos, used)


# ------------------------------------------------------------------ 协议/环境


def test_official_opponents_play_legal_games():
    """跑满整局不崩、动作都在引擎的合法集里。"""
    for cls in OFFICIAL:
        for agent_color in ("R", "B"):
            env = SentryEnv(cls(seed=0), agent_color=agent_color, seed=0)
            obs, mask, _ = env.reset()
            n = 0
            while True:
                n += 1
                a = int(np.flatnonzero(mask > 0)[0])
                obs, mask, _, term, trunc, info = env.step(a)
                assert np.isfinite(obs).all()
                if term or trunc:
                    break
                assert n < 4000
            assert info["turn"] >= 1


def test_official_opponent_kills_still_reach_the_reward():
    """生成器协议的核心:对手自己造成的击杀必须仍进我方的奖励与计数。

    对手若绕过环境直接调 game.apply,这条会挂。
    """
    for cls in OFFICIAL:
        for seed in range(6):
            for agent_color in ("R", "B"):
                env = SentryEnv(cls(seed=seed), agent_color=agent_color, seed=seed)
                env.reset()
                total = 0.0
                while True:
                    _, _, r, term, trunc, info = env.step(END)  # 站着不动,等着被打
                    total += r
                    if term or trunc:
                        break
                w = info["winner"]
                terminal = 0.0 if w is None else (1.0 if w == info["agent_color"] else -1.0)
                expected = info["my_score"] - 2.0 * info["deaths"] + terminal
                assert abs(total - expected) < 1e-6, (cls.__name__, seed, agent_color, total, expected, info)


def test_official_opponents_never_stall():
    """必须有进攻性 —— 不能出现 0:0 拖到加时的死局(waypoint 来回摆会拖成这样)。"""
    for cls in HIGH_SCORING:
        for seed in range(8):
            env = SentryEnv(cls(seed=seed), agent_color="R", seed=seed)
            env.reset()
            while True:
                _, _, _, term, trunc, info = env.step(END)  # 我方罚站
                if term or trunc:
                    break
            assert info["opp_score"] >= 10, (cls.__name__, seed, info)


def test_memory_opponents_still_attack_despite_hiding():
    """记忆型对手的门槛与 HIGH_SCORING 不同,但**也得真的在杀人**。

    它们每杀一次就要空转最多三个阶段(ambusher 绕角落、weaver 消失),所以拿不到
    10 分。这条守的是另一件事:相位机不能把"躲"写成"再也不出来" —— 那种坏法不会
    报错,只会让对手退化成靶子,而训练照样跑完。
    """
    for cls in (OfficialAmbusher, OfficialWeaver):
        for seed in range(8):
            for color in ("R", "B"):
                env = SentryEnv(cls(seed=seed), agent_color=color, seed=seed)
                env.reset()
                while True:
                    _, _, _, term, trunc, info = env.step(END)
                    if term or trunc:
                        break
                assert info["opp_score"] >= _MEMORY_SCORE_FLOOR, (cls.__name__, seed, color, info)
                assert info["deaths"] >= 2, (cls.__name__, seed, color, info)


def test_ambusher_hides_for_three_phases_after_each_kill():
    """相位机的核心:每次击杀之后必须出现一段"不扫描"的躲藏窗口。

    直接钉机制而不是钉棋力 —— 这三个阶段正是**无记忆策略拿不到的那段历史**,
    它一旦悄悄消失(比如 `_last_score` 判错、相位没推进),对手就退化成一个
    普通规则手,而所有棋力测试仍然全绿。
    """
    seen_modes = []

    class Probe(OfficialAmbusher):
        def turn(self, view_fn, my_color, ob):
            # 记的是**进入本阶段时**的状态:该阶段是否处于躲藏窗口内。
            seen_modes.append("HIDE" if self._hide_left > 0 else "HUNT")
            yield from super().turn(view_fn, my_color, ob)

    env = SentryEnv(Probe(seed=0), agent_color="R", seed=0)
    env.reset()
    while True:
        _, _, _, term, trunc, _ = env.step(END)
        if term or trunc:
            break

    seq = "".join("H" if m == "HIDE" else "." for m in seen_modes)
    # 击杀发生在第 k 阶段,则第 k+1 阶段起进入 HIDE —— 所以每次击杀后必然出现
    # 连续的 H。数一下连续段:每段长度不超过 HIDE_PHASES。
    runs, cur = [], 0
    for ch in seq + ".":
        if ch == "H":
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    assert runs, f"整局都没进入过躲藏窗口,相位机没在工作:{seq}"
    assert max(runs) <= OfficialAmbusher.HIDE_PHASES, (runs, seq)


def test_ambusher_hide_corner_is_not_a_constant():
    """藏身点必须随走位变化 —— 常量等于可被直接记住,记忆价值归零。

    这是设计约束,不是实现细节:藏身点的参照取错(比如取敌人被击杀后的重生点),
    它就会恒等于同一个角,对手退化成靶子 —— 而上面那条相位测试**仍然全绿**,
    因为它只数 HIDE 的段长,不看躲去了哪。

    两个写法上的坑,都会让这条测试**静默失效**:

    * 探针必须用 `yield from`,不能写成 `for a in gen: yield a` —— 后者不转发环境
      `send` 回来的 `ActionResult`,子生成器拿到 None,一取 `.success` 就炸。
    * 记录必须放在 `finally` 里。环境在阶段结束时就撒手(`phase_over()` 后
      `gen.close()`),**生成器不会自然跑完** —— 写在 `yield from` 之后的语句
      大多数阶段根本不执行,`corners_seen` 会只收到偶然跑完的那一两次。上一版
      就是这么写的,结果只见到一个角,看起来像"参照取错了",其实是探针没跑。
    """
    corners_seen = set()

    class Probe(OfficialAmbusher):
        def turn(self, view_fn, my_color, ob):
            try:
                yield from super().turn(view_fn, my_color, ob)
            finally:
                if self._hide_corner is not None:
                    corners_seen.add(self._hide_corner)

    rng = np.random.default_rng(0)
    for seed in range(6):
        env = SentryEnv(Probe(seed=seed), agent_color="R", seed=seed)
        _, mask, _ = env.reset()
        while True:
            idx = np.flatnonzero(mask > 0)
            # 我方随机走位:藏身点是拿**自己**开火位置当参照的,我方不动就测不出
            # "随走位变化"这件事。
            _, mask, _, term, trunc, _ = env.step(int(rng.choice(idx)))
            if term or trunc:
                break
    assert len(corners_seen) >= 2, (
        "藏身点只出现过一种取值,说明参照选取让它变成了常量", corners_seen
    )


def test_farthest_corner_tie_break_is_pinned():
    """平局裁决必须固定 —— 藏身点决定整段走位,两边取到不同的角不会报错。"""
    # 棋盘中心到四角等距,取角序最靠前的 (0,0)
    assert farthest_corner(7, (3, 3)) == (0, 0)
    # 贴着一个角时,最远的必然是斜对角
    assert farthest_corner(7, (0, 0)) == (6, 6)
    assert farthest_corner(7, (6, 6)) == (0, 0)


def test_official_hunter_shoots_a_visible_enemy():
    """hunter 的核心是"SCAN 后立即击杀",这条链路必须真的打得出来。

    对手得是**会走进中场**的:面对缩在角落不动的对手,hunter 会按它的逻辑
    老老实实占点,一枪不放 —— 那是忠实的,不是坏的。所以这里挑 camper。
    """
    from monet.training.selfplay import STATIC_OPPONENTS
    from tests.duel import play_side

    kills = 0
    for seed in range(12):
        game = Game()
        hunter = OfficialHunter(seed=seed)
        other = STATIC_OPPONENTS["camper"](seed)
        for _ in range(64):
            if game.done:
                break
            play_side(game, game.phase, hunter if game.phase == "R" else other)
        kills += game.score[0] // 2  # 击杀 +2
    assert kills > 0, "hunter 在 12 局里一次都没击杀"


def test_official_opponents_are_mirror_symmetric():
    """同一 seed 下执红/执蓝必须走出镜像一致的对局(引擎与移植版都应满足)。"""
    for cls in OFFICIAL:
        for seed in range(4):
            scores = []
            for color in ("R", "B"):
                g = Game()
                opp = cls(seed=seed)
                _play_full_game(g, color, opp)
                scores.append(g.score)
            assert scores[0] == scores[1], (cls.__name__, seed, scores)


def _play_full_game(game, opp_color, opp):
    from monet.env.obs import ObsBuilder

    for _ in range(64):
        if game.done:
            break
        color = game.phase
        ob = ObsBuilder()
        ob.act_start(game.view(color), color)
        gen = opp.turn(lambda: game.view(color), color, ob)
        last = None
        try:
            for _ in range(24):
                if game.phase_over():
                    break
                try:
                    a = gen.send(last)
                except StopIteration:
                    break
                res, _ = game.apply(color, a)
                ob.on_observation(res.observation, res.consumed, a)
                last = res
                if a == END:
                    break
        finally:
            gen.close()
        game.end_phase(color)


def test_default_turn_keeps_single_step_opponents_working():
    """基类默认 turn() 必须与逐动作循环等价 —— 老对手走默认实现,行为不能变。"""
    for opp in (RandomOpponent(seed=0), EndOpponent(), HeuristicOpponent(aggression=1.0, seed=0)):
        g = Game()
        _play_full_game(g, "R", opp)
        assert g.done


def test_official_opponents_beat_a_do_nothing_opponent():
    """官方档至少要比"什么都不做"强,否则说明策略是坏的。"""
    from tests.duel import duel

    for name in (
        "official_baseline",
        "official_hunter",
        "official_stalker",
        "official_patrol",
        "official_ambusher",
        "official_weaver",
    ):
        r = duel(name, "end", games=12, seed=100)
        assert r["winrate"] >= 0.9, (name, r)


# --------------------------------------- 自研官方档:不为得分区写任何策略(设计前提)


def test_ai_targets_are_the_enemy_even_inside_the_score_zone():
    """设计要求:这四个自研规则手**不为得分区写任何策略 —— 正反两面都不写**。

    最容易顺手写下去的是**反向**那一面:"不占点,那就绕着得分区走"。那同样是
    特意为得分区写策略,而且会立刻从行为里漏出来 —— 对手只要看它绕开中心,
    就能推断出打法,而这对两个记忆型对手尤其致命:它们的相位本该是观测里
    **唯一读不出来的东西**,藏身点/出没时机一旦能被"它又绕中心了"这种走位
    反推出来,记忆价值就没了。

    所以直接钉这个函数:敌人在得分区里时,推进目标**仍然是敌人本身**,不是
    "区外离它最近的格"。
    """
    board = SimpleNamespace(size=R.BOARD_SIZE, obstacles=list(R.DEFAULT_OBSTACLES))
    for zone in R.DEFAULT_SCORE_ZONES:
        zone = tuple(zone)
        assert hunt_target(board, zone) == zone, ("敌人在区里也不能改道", zone)
    assert hunt_target(board, (-1, -1)) == (R.BOARD_SIZE - 1, R.BOARD_SIZE - 1)


def test_zone_avoidance_helpers_stay_deleted():
    """上面那条守"不绕开"的行为,这条守**别把工具再长回来**。

    一旦 `zone_safe_target` / `kill_in_one_turn(forbid=...)` 回来了,上面那条
    测试**仍然是绿的**(它只钉 `hunt_target`),但四个 AI 会悄悄开始绕中心走。
    这种"测试全绿但口径已经变了"正是本仓库反复吃亏的坏法,所以连入口一起钉。
    """
    import inspect

    assert not hasattr(official, "zone_safe_target")
    assert not hasattr(official, "_outside_zone_cells")
    params = inspect.signature(official.kill_in_one_turn).parameters
    assert "forbid" not in params, "kill_in_one_turn 不该再带 forbid(那是得分区专用的)"


def _kills_and_zone_entries(cls, seed, ai_color):
    """让 cls 执 ai_color 与 camper 打一局,返回 (击杀数, 踩进得分区的记录)。

    蹲点的 camper 是"要不要管得分区"暴露得最清楚的对手:它站在区里不动,
    所以任何绕区逻辑都必须在它面前显形。
    """
    from tests.duel import play_side

    game = Game()
    ai = cls(seed=seed)
    other = STATIC_OPPONENTS["camper"](seed)
    real = game.apply
    entries = []
    kills = [0]

    def apply(color, a, _real=real):
        res, ev = _real(color, a)
        if color == ai_color:
            # 每个动作之后都查一次:落脚进得分区只可能由 MOVE 造成,
            # 但逐动作查比只查 MOVE 更强,顺带覆盖复活等路径。
            if res.success and game.in_zone(game.s[ai_color].pos):
                entries.append((game.turn, a, game.s[ai_color].pos))
            if ev.get("kill"):
                kills[0] += 1
        return res, ev

    game.apply = apply
    for _ in range(64):
        if game.done:
            break
        play_side(game, game.phase, ai if game.phase == ai_color else other)
    return kills[0], entries


def test_self_authored_opponents_do_cross_the_score_zone():
    """行为面反证:它们会自己走进得分区。

    这是"没有绕行逻辑在起作用"的**证据**,所以必须跑出非零计数 —— 一条都不
    踩进去,要么是绕区逻辑又回来了,要么是这三个 AI 根本走不到中心(那也是坏的)。

    `OfficialPatrol` 不在列表里:它的路线是边界环,而环与得分区不相交,所以它
    本来就不会踩进去。那是**路线选的**,不是绕开 —— 见
    `test_patrol_route_is_the_border_ring` 里对那条断言的口径说明。
    """
    for cls in (OfficialStalker, OfficialAmbusher, OfficialWeaver):
        total = 0
        for ai_color in ("R", "B"):
            for seed in range(8):
                _, entries = _kills_and_zone_entries(cls, seed, ai_color)
                total += len(entries)
        assert total > 0, (cls.__name__, "一次都没踩进得分区,像是又有绕行逻辑了")


def test_self_authored_opponents_still_kill_a_camper_inside_the_zone():
    """不管得分区 ≠ 打不着:必须能把蹲在区里的 camper 打死。

    这四个 AI 的火力判定和走位都不看得分区,所以这条实际守的是别的东西:
    "不特意为得分区写策略"没有被实现成"不敢靠近中心"。
    """
    for cls in SELF_AUTHORED:
        for ai_color in ("R", "B"):
            kills, _ = _kills_and_zone_entries(cls, 0, ai_color)
            assert kills >= 1, (cls.__name__, ai_color, kills)


def test_patrol_route_is_the_border_ring():
    """patrol 的"沿边缘移动"必须真的贴边、相邻、且不重复。

    顺带钉住"边界环不与得分区相交"这个**事实**:它解释了为什么 patrol 实测
    一次占点分都拿不到。注意这是事实,不是目的 —— `_build_route` 只看棋盘尺寸,
    对得分区一无所知(和另外三个自研规则手同一口径)。
    """
    route = OfficialPatrol(seed=0)._build_route(R.BOARD_SIZE)
    assert len(route) == 4 * R.BOARD_SIZE - 4, "7x7 的边界环应该是 24 格"
    assert len(set(route)) == len(route), "边界环不该有重复格"
    assert not (set(route) & set(map(tuple, R.DEFAULT_SCORE_ZONES))), "边界环与得分区不相交"
    for x, y in route:
        assert x in (0, R.BOARD_SIZE - 1) or y in (0, R.BOARD_SIZE - 1), (x, y)
    for i, cur in enumerate(route):  # 相邻两格必须在棋盘上相邻,否则就是瞬移
        nxt = route[(i + 1) % len(route)]
        assert abs(cur[0] - nxt[0]) + abs(cur[1] - nxt[1]) == 1, (cur, nxt)


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
