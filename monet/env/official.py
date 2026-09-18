"""官方档对手:Baseline / Hunter 的 Python 移植 + 四个自研规则手。

## 两大类,别混

| 类 | 来源 | 说明 |
|---|---|---|
| `OfficialBaseline` `OfficialHunter` | `baseline_ai.cpp` / `hunter_ai.cpp` | **移植**。决策逻辑 1:1 照搬,下面的保真度说明只针对这两个 |
| `OfficialStalker` `OfficialPatrol` | 无 | **自研**(见文件末尾)。按需求新写的规则手,不移植任何 C++,因此没有"保真度"问题 |
| `OfficialAmbusher` `OfficialWeaver` | 无 | **自研,记忆型**(见文件末尾)。行为由观测里读不到的**内部相位**驱动,专门用来给"带记忆的策略"提供无记忆策略拿不到的梯度 |

**六个类上都写了 `is_official = True`,但那不决定任何事** —— 没有任何代码读这个
类属性。真正决定"算不算官方档"的是 `training/selfplay.py` 里**注册工厂**上的标记
(`OFFICIAL_OPPONENTS` 是拿 `getattr(工厂, "is_official", False)` 筛的,而工厂是
那些 lambda)。实际后果:

| 类 | 在 `OFFICIAL_OPPONENTS` 里? |
|---|---|
| `OfficialBaseline` `OfficialHunter` | **在**(工厂套了 `_official`) |
| `OfficialStalker` `OfficialPatrol` `OfficialAmbusher` `OfficialWeaver` | **不在**(工厂没套) |

也就是说:把 stalker / patrol / ambusher / weaver 放进评测名单**不会**让它们进
`guard_official_winrate`(官方保护闸)或快照闸门。成员名单钉在
`tests/test_train.py::test_official_opponents_are_registered_and_flagged`,
要改就先回那里和 README §五 对口径。

后两个记忆型对手还默认不进 `LEAGUE_DEFAULT` / `EVAL_DEFAULT` —— 名单是用户手改的
口径,要用它们陪练得显式传 `--league`。

来源:`baseline_ai.cpp`(占点+攻击 ⭐⭐⭐☆☆)、`hunter_ai.cpp`(扫描压制)。

## 移植保真度(重要)

**决策逻辑是 1:1 照搬的**,包括优先级、分支顺序、动作计数,以及那些看起来奇怪
的细节(比如 baseline 在必杀分支里 turn→fire→scan 不检查额度)。

但移植来的这两个 AI 依赖 `utils.h` / `navigation.h`,而这两份源码不在手边,所以:

| 函数 | 状态 |
|---|---|
| `same_pos` `direction_delta` `blocked` `in_score_zone` | 语义明确,按规则重写 |
| `best_turn_to_face` `in_fire_range` | 语义明确(与 `monet/engine/rules.py` 同源) |
| `advance_toward` | ⚠️ **寻路核心,是本仓库自研的实现**(BFS 最短路 + 先转后走),不是官方原版 |

官方的 `advance_toward` 还带可选参数(hunter 传了敌人位置与朝向,疑似"边压边找"),
本实现忽略这些参数。**官方版在 (1,1) 障碍附近的路线可能与这里不同** —— hunter 的
路径点 `(2,1)`、`(1,2)` 正好贴着那个障碍。

**结论:这里给出的是「官方决策逻辑 + 我方寻路」,不是官方 AI 本身。**
拿它当训练陪练没问题;拿它评测出来的数字**不能当打榜预测**。要准确评测请走官方
API(api.md §10,榜单本来就含官方 Baseline / Hunter)。

## 为什么是生成器

官方 `act()` 一次调用内部就走完整回合(最多 3 个动作,且控制流依赖每个动作返回的
观测)。所以这里的 `turn()` 写成生成器:每个 `yield` 出去的动作由环境用 `game.apply`
执行、结果 `send` 回来。这样击杀/复活等事件仍然流经环境,奖励不会漏 —— 如果让对手
自己调 `game.apply`,它击杀我方时我方就拿不到阵亡惩罚。
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from ..engine import rules as R
from . import obs as O
from .opponents import Opponent

Action = int


# --------------------------------------------------------------- 几何辅助(明确)


def same_pos(a, b) -> bool:
    return a[0] == b[0] and a[1] == b[1]


def blocked(board, x: int, y: int, opponent) -> bool:
    """出界 / 障碍 / 被对方占住。opponent 传 (-1,-1) 表示"不判对方"。"""
    if not R.in_bounds(x, y):
        return True
    if (x, y) in set(map(tuple, board.obstacles)):
        return True
    return same_pos((x, y), opponent)


def in_score_zone(pos, zones) -> bool:
    return pos in set(map(tuple, zones))


def in_fire_range(frm, facing: str, to) -> bool:
    """纯几何:to 是否在 frm 朝 facing 的 3x3 火力范围内(**不判障碍**)。

    障碍判定由调用方另外做 —— hunter 的 can_step_fire 就是自己补的射线检查。
    """
    fx, fy = R.DELTA[facing]
    px, py = R.PERP[facing]
    dx, dy = to[0] - frm[0], to[1] - frm[1]
    forward = dx * fx + dy * fy
    lateral = dx * px + dy * py
    return 1 <= forward <= R.FIRE_RANGE and -1 <= lateral <= 1


def can_step_fire(board, frm, facing: str, opponent) -> bool:
    """前进一步后是否落进火力范围,且中间没有障碍(对应 hunter_ai.cpp 的同名函数)。"""
    dx, dy = R.DELTA[facing]
    nxt = (frm[0] + dx, frm[1] + dy)
    if blocked(board, nxt[0], nxt[1], opponent):
        return False
    if not in_fire_range(nxt, facing, opponent):
        return False
    forward = abs(opponent[0] - nxt[0]) if facing in ("E", "W") else abs(opponent[1] - nxt[1])
    for distance in range(1, forward):
        if facing == "E":
            x, y = nxt[0] + distance, opponent[1]
        elif facing == "W":
            x, y = nxt[0] - distance, opponent[1]
        elif facing == "S":
            x, y = opponent[0], nxt[1] + distance
        else:
            x, y = opponent[0], nxt[1] - distance
        if blocked(board, x, y, (-1, -1)):
            return False
    return True


# ------------------------------------------------------- advance_toward(我方实现)


def _bfs_dist_to(board, target, avoid):
    """从 target 反向 BFS,返回每格的剩余步数(不可达的不在表里)。"""
    obstacles = set(map(tuple, board.obstacles))
    dist = {tuple(target): 0}
    q = deque([tuple(target)])
    while q:
        cur = q.popleft()
        for d in R.DIRS:
            nxt = (cur[0] + R.DELTA[d][0], cur[1] + R.DELTA[d][1])
            if not R.in_bounds(*nxt) or nxt in obstacles or nxt in dist:
                continue
            if nxt in avoid and nxt != tuple(target):
                continue
            dist[nxt] = dist[cur] + 1
            q.append(nxt)
    return dist


def _next_step(board, pos, facing, target, avoid):
    """最短路上的下一格;无路可走返回 None。

    **同长度路径优先选"不用转身"的那条**:转身本身要吃掉一个动作,而
    `advance_toward` 的预算只有 1~3。否则会出现"这一步转身向东、下一步又转身
    向西"的来回摆动(hunter 的路径点 (2,1) 紧贴障碍 (1,1),最容易踩到)。
    """
    if same_pos(pos, target):
        return None
    dist = _bfs_dist_to(board, target, avoid)
    if tuple(pos) not in dist:
        return None
    best = None
    for d in R.DIRS:
        nxt = (pos[0] + R.DELTA[d][0], pos[1] + R.DELTA[d][1])
        if not R.in_bounds(*nxt) or dist.get(nxt, 1 << 30) != dist[tuple(pos)] - 1:
            continue
        if nxt in avoid and nxt != tuple(target):
            continue
        if d == facing:
            return nxt  # 同长度里优先"不用转身"
        if best is None:
            best = nxt
    return best


def advance_toward(board, pos, facing, enemy, target, budget, avoid_extra=()):
    """朝 target 推进,最多花 budget 个动作。

    ⚠️ **本仓库自研,非官方 navigation.h**。BFS 最短路 + 先转后走,绕开障碍与对方占格。

    子生成器:yield 动作,`return (actions, pos, facing, last_observation)`。
    最后一个元素对应 C++ 侧的 `ActionObservation*` 出参:没有动作时是 None
    (等价于那边的零初始化结构体 —— `opp_visible` 为假,分支自然跳过)。
    """
    actions = 0
    last = None
    if target is None or target[0] < 0:
        return actions, pos, facing, last

    avoid = set()
    if enemy is not None and enemy[0] >= 0:
        avoid.add(tuple(enemy))
    avoid |= {tuple(c) for c in avoid_extra}
    avoid.discard(tuple(pos))

    while actions < budget and not same_pos(pos, target):
        nxt = _next_step(board, pos, facing, target, avoid)
        if nxt is None:
            break
        want = R.best_turn_to_face(pos, nxt)
        if facing != want:
            res = yield O.TURN_ACTION[want]
            if not res.success:
                break
            actions += 1
            facing = want
            last = res.observation
            continue
        res = yield O.MOVE
        if not res.success:
            break
        actions += 1
        pos = res.observation.my_pos
        facing = res.observation.my_facing
        last = res.observation

    return actions, pos, facing, last


# ------------------------------------------------------------------- 状态/RNG


class _XorShift32:
    """对应 C++ 侧的 xorshift32(种子在那边取时钟,这里取可复现的种子)。"""

    def __init__(self, seed: int = 1):
        self.s = (int(seed) or 1) & 0xFFFFFFFF

    def next(self) -> int:
        x = self.s
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.s = x & 0xFFFFFFFF
        return self.s

    def coin_flip(self) -> bool:
        return (self.next() & 1) != 0


# ------------------------------------------------------------------ 官方 Baseline


class OfficialBaseline(Opponent):
    """baseline_ai.cpp:在得分区就蹲着打,不在就朝中心入口走,沿途开火。"""

    name = "official_baseline"
    is_official = True

    def __init__(self, seed: int = 0):
        self.rng = _XorShift32(seed + 1)
        self.route_target: Optional[tuple] = None
        self.previous_pos = (-1, -1)
        self.route_side = 0
        self.route_initialized = False
        self.spawn_turn_pending = False

    def reset(self) -> None:
        self.route_target = None
        self.previous_pos = (-1, -1)
        self.spawn_turn_pending = False

    def _route_entry(self):
        if self.route_target is None:
            self.route_target = (3, 2) if self.route_side == 0 else (2, 3)
        return self.route_target

    def turn(self, view_fn, my_color, ob):
        board = view_fn()
        me, opp = board.me_opp(my_color)
        actions = 0

        # 复活检测:(0,0) 重新出现 → 换一条入口路线
        if same_pos(me.last_known_pos, (0, 0)) and not same_pos(self.previous_pos, me.last_known_pos):
            if not self.route_initialized:
                self.route_side = 1 if self.rng.coin_flip() else 0
                self.route_initialized = True
            else:
                self.route_side = 1 - self.route_side
            self.route_target = None
            self.spawn_turn_pending = True
        self.previous_pos = me.last_known_pos

        # 视野中的敌人必定处于 3x3 火力范围 → 立即斩杀
        if opp.visible and opp.last_known_pos[0] >= 0 and me.fire_cd == 0:
            fd = R.best_turn_to_face(me.last_known_pos, opp.last_known_pos)
            if me.last_known_facing != fd:
                yield O.TURN_ACTION[fd]
                actions += 1
            yield O.FIRE
            actions += 1
            if me.scan_cd == 0:
                yield O.SCAN
                actions += 1
            return

        # SCAN 后若能打到,立即 turn+fire
        if me.scan_cd == 0:
            res = yield O.SCAN
            if res.success:
                actions += 1
            fd = R.best_turn_to_face(me.last_known_pos, res.observation.opp_last_known_pos)
            if me.fire_cd == 0 and in_fire_range(
                me.last_known_pos, fd, res.observation.opp_last_known_pos
            ):
                if me.last_known_facing != fd:
                    yield O.TURN_ACTION[fd]
                    actions += 1
                yield O.FIRE
                actions += 1
                return

        position = me.last_known_pos
        facing = me.last_known_facing
        if in_score_zone(position, board.score_zones):
            self.route_target = None
            return

        observation = None
        target = self._route_entry()
        if self.spawn_turn_pending:
            route_direction = "E" if self.route_side == 0 else "S"
            if facing != route_direction and actions < 3:
                turned = yield O.TURN_ACTION[route_direction]
                if not turned.success:
                    return
                actions += 1 if turned.consumed else 0
                facing = route_direction
            if actions < 3:
                first_move = yield O.MOVE
                if not first_move.success:
                    return
                actions += 1 if first_move.consumed else 0
                position = first_move.observation.my_pos
                observation = first_move.observation
            self.spawn_turn_pending = False

        used, position, facing, observation = yield from advance_toward(
            board, position, facing, opp.last_known_pos, target, 3 - actions
        )
        actions += used

        # 走完拿到实时视野 → 用剩余动作反击
        if (
            observation is not None
            and observation.opp_visible
            and observation.fire_cd == 0
            and actions < 3
        ):
            fd = R.best_turn_to_face(position, observation.opp_last_known_pos)
            if facing != fd and actions < 2:
                turned = yield O.TURN_ACTION[fd]
                if turned.success:
                    actions += 1
                    facing = fd
            if actions < 3 and facing == fd:
                fired = yield O.FIRE
                if fired.success:
                    actions += 1


# -------------------------------------------------------------------- 官方 Hunter


class OfficialHunter(Opponent):
    """hunter_ai.cpp:SCAN 后立即击杀 > 一步前压击杀 > 进中心得分区。"""

    name = "official_hunter"
    is_official = True

    def __init__(self, seed: int = 0):
        self.rng = _XorShift32(seed + 1)
        self.route_target: Optional[tuple] = None
        self.route_waypoint: Optional[tuple] = None
        self.previous_pos = (-1, -1)
        self.route_kind = 0
        self.route_initialized = False
        self.spawn_turn_pending = False
        self.skip_scan_once = False

    def reset(self) -> None:
        self.route_target = None
        self.route_waypoint = None
        self.previous_pos = (-1, -1)
        self.spawn_turn_pending = False
        self.skip_scan_once = False

    def _route_destination(self, board, position):
        if self.route_target is None:
            self.route_target = (board.size // 2, board.size // 2)
        if self.route_waypoint is not None and not same_pos(position, self.route_waypoint):
            return self.route_waypoint
        return self.route_target

    def turn(self, view_fn, my_color, ob):
        board = view_fn()
        me, opp = board.me_opp(my_color)
        position = me.last_known_pos
        facing = me.last_known_facing
        enemy = opp.last_known_pos
        enemy_facing = opp.last_known_facing
        enemy_known = opp.last_known_pos[0] >= 0 and opp.last_known_pos[1] >= 0
        holding = in_score_zone(position, board.score_zones)
        actions = 0
        fired_this_turn = False

        # 复活 → 换一条中心入口,并跳过这次的 SCAN
        if same_pos(position, (0, 0)) and not same_pos(self.previous_pos, position):
            if not self.route_initialized:
                self.route_kind = self.rng.next() % 4
                self.route_initialized = True
            else:
                self.route_kind = (self.route_kind + 1 + self.rng.next() % 3) % 4
            self.route_target = None
            self.route_waypoint = (
                (2, 1) if self.route_kind == 1 else ((1, 2) if self.route_kind == 3 else None)
            )
            self.spawn_turn_pending = True
            self.skip_scan_once = True
        self.previous_pos = position

        # SCAN 是超视距攻击的入口
        if not self.skip_scan_once and me.fire_cd == 0 and me.scan_cd == 0:
            res = yield O.SCAN
            if res.success:
                actions += 1 if res.consumed else 0
                enemy = res.observation.opp_last_known_pos
                enemy_facing = res.observation.opp_last_known_facing
                enemy_known = True
        self.skip_scan_once = False

        if enemy_known and me.fire_cd == 0:
            fd = R.best_turn_to_face(position, enemy)

            # 已在火力范围:SCAN 后 TURN+FIRE 立即击杀
            if in_fire_range(position, fd, enemy):
                if facing != fd:
                    turned = yield O.TURN_ACTION[fd]
                    if not turned.success:
                        return
                    actions += 1 if turned.consumed else 0
                    facing = fd
                fired = yield O.FIRE
                if fired.success:
                    actions += 1
                    fired_this_turn = True
                    enemy_known = False

            # 超视距前压:一步 MOVE 后进火力范围 → MOVE+FIRE
            if (
                not fired_this_turn
                and not holding
                and actions <= 1
                and can_step_fire(board, position, facing, enemy)
            ):
                moved = yield O.MOVE
                if not moved.success:
                    return
                actions += 1 if moved.consumed else 0
                position = moved.observation.my_pos
                fired = yield O.FIRE
                if fired.success:
                    actions += 1
                    fired_this_turn = True
                    enemy_known = False

        if holding:
            self.route_target = None
            self.route_waypoint = None
            return

        observation = None
        target = self._route_destination(board, position)
        if self.spawn_turn_pending:
            route_direction = "E" if self.route_kind < 2 else "S"
            if facing != route_direction and actions < 3:
                turned = yield O.TURN_ACTION[route_direction]
                if not turned.success:
                    return
                actions += 1 if turned.consumed else 0
                facing = route_direction
            if actions < 3:
                first_move = yield O.MOVE
                if not first_move.success:
                    return
                actions += 1 if first_move.consumed else 0
                position = first_move.observation.my_pos
                observation = first_move.observation
            self.spawn_turn_pending = False

        used, position, facing, observation = yield from advance_toward(
            board,
            position,
            facing,
            enemy if enemy_known else (-1, -1),
            target,
            3 - actions,
        )
        actions += used

        if (
            observation is not None
            and observation.opp_visible
            and observation.fire_cd == 0
            and actions < 3
        ):
            fd = R.best_turn_to_face(position, observation.opp_last_known_pos)
            if facing != fd and actions < 2:
                turned = yield O.TURN_ACTION[fd]
                if turned.success:
                    actions += 1 if turned.consumed else 0
                    facing = fd
            # 对应官方 fire_if_in_range:射程内才开火(只判几何,不判障碍)
            if actions < 3 and facing == fd and in_fire_range(
                position, facing, observation.opp_last_known_pos
            ):
                fired = yield O.FIRE
                if fired.success:
                    actions += 1


# ------------------------------------------- 自研官方档规则手(非移植,见模块说明)


def kill_in_one_turn(board, position, facing, enemy, actions, obstacles, limit=3):
    """尝试在一个回合内击杀 enemy。子生成器:`return (actions, position, facing, fired)`。

    火力判定用 `R.fire_hits`(**含障碍阻挡**),而不是官方的纯几何
    `in_fire_range` —— 被障碍挡住的一枪等于白费一个动作,而开火冷却要隔一回合,
    代价比官方那两家大。够不着时先走一步再打(`can_step_fire` 同语义)。

    调用方必须保证 `enemy` 是有效坐标(>=0);没有敌情时不要调进来。
    """
    fired = False
    fd = R.best_turn_to_face(position, enemy)
    if facing != fd and actions < limit:
        res = yield O.TURN_ACTION[fd]
        if not res.success:
            return actions, position, facing, fired
        actions += 1 if res.consumed else 0
        facing = fd
        position = res.observation.my_pos
    if facing != fd:
        return actions, position, facing, fired

    if actions < limit and R.fire_hits(position, facing, enemy, obstacles):
        res = yield O.FIRE
        if res.success:
            actions += 1
            fired = True
        return actions, position, facing, fired

    # 够不着:沿当前朝向走一步后若能开火,就 MOVE + FIRE
    if actions < limit - 1 and can_step_fire(board, position, facing, enemy):
        res = yield O.MOVE
        if not res.success:
            return actions, position, facing, fired
        actions += 1 if res.consumed else 0
        position = res.observation.my_pos
        facing = res.observation.my_facing
        if actions < limit and R.fire_hits(position, facing, enemy, obstacles):
            res = yield O.FIRE
            if res.success:
                actions += 1
                fired = True
    return actions, position, facing, fired


def hunt_target(board, enemy):
    """四个自研规则手共用的推进目标:**有情报就直扑,没有就朝敌方出生角搜**。

    注意这里**没有任何得分区逻辑**。这不是疏漏,是设计要求:得分区对这些 AI
    不该存在 —— 它们既不为占点做规划,也不绕着它走。走到区里就去,站在区里
    就站着。任何"绕开得分区""占区外最近的格"的写法都是在给它们补一套
    **反向的占点策略**,那同样是"特意为得分区写策略",而且会立刻泄漏到行为里:
    对手只要看它绕开中心就能推断出打法,记忆型的这两个尤其不能这样
    (它们的相位本来就该是观测里唯一读不出来的东西)。

    由此带来的一个副作用要记住:它们**会**拿占点分。所以"分数差 >= 2 就是击杀"
    这个判据成立的前提是"一个阶段最多只结算 1 分占点分"(`Game.end_phase`),
    而不是"它们不占点"—— 见 `OfficialAmbusher.turn`。
    """
    if enemy[0] < 0:
        return (board.size - 1, board.size - 1)
    return tuple(enemy)


def corners(size: int):
    """棋盘四角,固定顺序 —— 记忆型对手的藏身点按这个顺序取,平局裁决也用它。"""
    n = size - 1
    return ((0, 0), (n, 0), (0, n), (n, n))


def farthest_corner(size: int, ref):
    """离 `ref` 曼哈顿距离最远的角;同距取 `corners()` 里靠前的那个。

    平局裁决必须显式写死:藏身点决定整段 HIDE 的走位,两边实现取到不同的角
    不会报错,只会让棋力悄悄变样。
    """
    best, best_key = corners(size)[0], None
    for i, c in enumerate(corners(size)):
        d = abs(c[0] - ref[0]) + abs(c[1] - ref[1])
        key = (-d, i)  # 距离大的优先,再按角序
        if best_key is None or key < best_key:
            best_key, best = key, c
    return best


class OfficialStalker(Opponent):
    """自研规则手:靠 SCAN 和视野找人击杀。**完全不理会得分区**。

    "不理会"是**双向**的:既不为占点做任何规划,也不绕着得分区走 —— 推进目标
    永远是敌人本身(`hunt_target`),路怎么短怎么走,穿过中心十字就穿过。给它们
    补一套"绕开得分区"的走位是很容易顺手写下去的事,但那是**反向的占点策略**,
    同样是特意为得分区写策略,而且会立刻从行为里漏出来(对手只要看它绕开中心
    就能推断打法)。要去掉得分区在这个文件里的最后一点影响,`kill_in_one_turn`
    也一样不带 `forbid`。

    行动优先级:
      1. 视野内(直接视野或本回合 SCAN)有敌人、且开火冷却好 → 当回合杀掉;
      2. SCAN 冷却好 → 扫描。视野只有 T 形(正前 1 格 + 距离 2 的横向 3 格),
         而火力范围是前方 3x3,所以不扫描就拿不到大部分开火机会;
      3. 扫描后发现击杀机会 → 杀掉;
      4. 否则朝敌人推进;没有情报就朝敌方出生角搜索。

    策略全确定性,`seed` 只为对齐注册表签名。
    """

    name = "official_stalker"
    is_official = True

    def __init__(self, seed: int = 0):
        pass

    def turn(self, view_fn, my_color, ob):
        board = view_fn()
        me, opp = board.me_opp(my_color)
        position = me.last_known_pos
        facing = me.last_known_facing
        obstacles = board.obstacles
        actions = 0

        # 1) 视野内有人 → 当回合击杀(SCAN 过的话 scan_reveal 也让 opp.visible 为真)
        if opp.visible and opp.last_known_pos[0] >= 0 and me.fire_cd == 0:
            actions, position, facing, _ = yield from kill_in_one_turn(
                board, position, facing, opp.last_known_pos, actions, obstacles
            )

        # 2) SCAN:它找人的主要手段
        if me.scan_cd == 0 and actions < 3:
            res = yield O.SCAN
            if res.success:
                actions += 1 if res.consumed else 0
                # 3) 扫描结果直接给出实时位置 → 还能打就补刀
                if res.observation.opp_visible and me.fire_cd == 0 and actions < 3:
                    actions, position, facing, _ = yield from kill_in_one_turn(
                        board,
                        position,
                        facing,
                        res.observation.opp_last_known_pos,
                        actions,
                        obstacles,
                    )

        if actions >= 3:
            return

        # 4) 没有击杀机会:直扑敌人;没有情报就朝敌方出生角搜
        enemy = opp.last_known_pos if opp.last_known_pos[0] >= 0 else (-1, -1)
        used, position, facing, observation = yield from advance_toward(
            board,
            position,
            facing,
            enemy,
            hunt_target(board, enemy),
            3 - actions,
        )
        actions += used

        # 推进途中拿到实时视野 → 用剩余动作补一枪
        if (
            observation is not None
            and observation.opp_visible
            and observation.fire_cd == 0
            and actions < 3
        ):
            yield from kill_in_one_turn(
                board,
                position,
                facing,
                observation.opp_last_known_pos,
                actions,
                obstacles,
            )


class OfficialPatrol(Opponent):
    """自研规则手:沿棋盘**边缘**巡逻,**不间断 SCAN**,敌人落进"一个回合内
    打得死"的范围就直接杀掉。

    巡逻路线是边界环(7x7 → 24 格):从出生点 (0,0) 沿 y=0 向东,绕一圈回来。
    边界环恰好不与 5 个得分格相交,所以它实际上很少拿到占点分 —— 但那是**路线
    选的**结果,不是它的目的:`_build_route` 只看棋盘尺寸,对得分区一无所知,
    和其它三个自研规则手一样(见 `hunt_target`)。

    与 `OfficialStalker` 的关键区别是**它不追击** —— 只有"当回合就能杀"
    (够得着,或走一步就够得着)才动手,否则回巡逻线。

    巡逻路线是静态的,目标格只由"当前站在环上哪一格"决定,所以复活、跨局复用
    都能自然接上,不需要额外的重置逻辑。策略全确定性,`seed` 只为对齐注册表签名。
    """

    name = "official_patrol"
    is_official = True

    def __init__(self, seed: int = 0):
        self.route = []

    def reset(self) -> None:
        self.route = []

    def _build_route(self, size: int):
        """边界环,按顺时针排列(视图坐标系)。"""
        n = size - 1
        return (
            [(x, 0) for x in range(size)]
            + [(n, y) for y in range(1, size)]
            + [(x, n) for x in range(n - 1, -1, -1)]
            + [(0, y) for y in range(n - 1, 0, -1)]
        )

    def _next_target(self, position):
        """环上的下一格。万一不在环上(理论上不会),就近回到环上。"""
        route = self.route
        try:
            idx = route.index(tuple(position))
        except ValueError:
            return min(
                route, key=lambda c: abs(c[0] - position[0]) + abs(c[1] - position[1])
            )
        return route[(idx + 1) % len(route)]

    def turn(self, view_fn, my_color, ob):
        board = view_fn()
        me, opp = board.me_opp(my_color)
        position = me.last_known_pos
        facing = me.last_known_facing
        obstacles = board.obstacles
        if not self.route:
            self.route = self._build_route(board.size)
        actions = 0

        # 1) 视野内(含本回合 SCAN)有人 → 当回合击杀
        if opp.visible and opp.last_known_pos[0] >= 0 and me.fire_cd == 0:
            actions, position, facing, _ = yield from kill_in_one_turn(
                board, position, facing, opp.last_known_pos, actions, obstacles
            )

        # 2) 不断 SCAN
        if me.scan_cd == 0 and actions < 3:
            res = yield O.SCAN
            if res.success:
                actions += 1 if res.consumed else 0
                if res.observation.opp_visible and me.fire_cd == 0 and actions < 3:
                    actions, position, facing, _ = yield from kill_in_one_turn(
                        board,
                        position,
                        facing,
                        res.observation.opp_last_known_pos,
                        actions,
                        obstacles,
                    )

        if actions >= 3:
            return

        # 3) 沿边界环推进(不追人;把已知敌人列为回避格,免得撞上去白费动作)
        enemy = opp.last_known_pos if opp.last_known_pos[0] >= 0 else (-1, -1)
        used, position, facing, observation = yield from advance_toward(
            board, position, facing, enemy, self._next_target(position), 3 - actions
        )
        actions += used

        # 巡逻途中撞见人 → 补一枪
        if (
            observation is not None
            and observation.opp_visible
            and observation.fire_cd == 0
            and actions < 3
        ):
            yield from kill_in_one_turn(
                board,
                position,
                facing,
                observation.opp_last_known_pos,
                actions,
                obstacles,
            )


# ------------------------------------------------- 自研:记忆型规则手(设计前提)


class _MemoryOpponent(Opponent):
    """两个记忆型自研规则手的共同骨架(**非移植**,没有保真度问题)。

    ## 为什么这两个对手"吃记忆"

    它们的行动由一个**内部相位**决定,而这个相位同时满足两条:

    1. **观测里读不到**。428 维观测只有即时量:位置、朝向、CD、比分、回合数。
       没有任何一个字段在说"对手现在处于什么模式"。
    2. **由事件推进,而不是由棋盘上可读的量推进**。凡是"位置 / 回合数的函数"的
       行为,无记忆的网络查一张足够大的表就能逼近 —— 那些量本来就在观测里。
       而"某件事发生之后又过了几个阶段",单帧里根本不存在,只能靠**差分一个
       可观测量**得到。

    差分正是循环结构最擅长、MLP 最不擅长的事:MLP 只能把它压成一张巨大的
    (比分 × 回合 × 位置) 查表,换个起点权重就失效。

    ## 共同约定

    * **不理会得分区**(与 `OfficialStalker` / `OfficialPatrol` 同一口径,见
      `hunt_target`):不占点、也不绕开它。得分只来自击杀,占点分是顺路捡的。
    * 相位按**自己的行动阶段**推进:环境每个阶段恰好调一次 `turn()`。
    * 全确定性;`seed` 只为对齐注册表签名。
    """

    def _hunt(self, board, me, opp, position, facing, obstacles, actions):
        """生成器:走完一个 HUNT 阶段。与 `OfficialStalker` 同款。"""
        if opp.visible and opp.last_known_pos[0] >= 0 and me.fire_cd == 0:
            actions, position, facing, _ = yield from kill_in_one_turn(
                board, position, facing, opp.last_known_pos, actions, obstacles
            )
        if me.scan_cd == 0 and actions < 3:
            res = yield O.SCAN
            if res.success:
                actions += 1 if res.consumed else 0
                if res.observation.opp_visible and me.fire_cd == 0 and actions < 3:
                    actions, position, facing, _ = yield from kill_in_one_turn(
                        board,
                        position,
                        facing,
                        res.observation.opp_last_known_pos,
                        actions,
                        obstacles,
                    )
        if actions >= 3:
            return
        enemy = opp.last_known_pos if opp.last_known_pos[0] >= 0 else (-1, -1)
        used, position, facing, observation = yield from advance_toward(
            board, position, facing, enemy, hunt_target(board, enemy), 3 - actions
        )
        actions += used
        if (
            observation is not None
            and observation.opp_visible
            and observation.fire_cd == 0
            and actions < 3
        ):
            yield from kill_in_one_turn(
                board,
                position,
                facing,
                observation.opp_last_known_pos,
                actions,
                obstacles,
            )

    def _retreat(self, board, me, opp, position, facing, obstacles, corner, actions):
        """生成器:HIDE / GHOST 阶段。不 SCAN、不追击,朝 `corner` 走,**返回收尾位置**。

        **允许开火**是有意的:躲藏期间被撞见还一枪不放,这个对手就只是个活靶子,
        教出来的策略是"无视它"。禁的是**为了找人而主动暴露**(SCAN),不是还击。
        """
        if opp.visible and opp.last_known_pos[0] >= 0 and me.fire_cd == 0:
            actions, position, facing, _ = yield from kill_in_one_turn(
                board, position, facing, opp.last_known_pos, actions, obstacles
            )
        if actions < 3 and corner is not None:
            enemy = opp.last_known_pos if opp.last_known_pos[0] >= 0 else (-1, -1)
            _used, position, _f, _obs = yield from advance_toward(
                board, position, facing, enemy, corner, 3 - actions
            )
        return position


class OfficialAmbusher(_MemoryOpponent):
    """自研记忆型规则手:**每击杀一次,就在三个自己的行动阶段内绕到一个角落躲起来**,
    躲满再出来找人杀。

    相位机只有两级:

      * `HUNT` —— 能杀就杀 → SCAN 找人 → 推进追击;
      * 判定到**自己刚击杀**(`me.score` 比上一阶段 +2)→ 把**离自己开火位置最远的
        那个角**定为藏身点,转入 `HIDE`;
      * `HIDE` 最多持续 `HIDE_PHASES`(=3)个自己的阶段:不 SCAN、不追击,朝藏身点走;
        **提前走到就提前结束**,不把剩下的阶段蹲满;被撞见时照常开火(见 `_retreat`);
      * 回到 `HUNT`。

    藏身点取"自己开火位置的最远角",有两个理由,都跟记忆有关:

    * **不能是常量**。若固定躲同一个角(或用一个与我方无关的参照),这个角会被
      直接记成事实,对手退化成普通靶子。用开火位置做参照,藏身点随它每局的走位变化。
    * **不能只靠当前帧**。藏身点由"三个阶段前那次击杀发生在哪"决定,而那个位置
      在需要它的时候早已不在画面里 —— 想用上就必须把这段历史留住。

    相应地,训练侧真正可学的是那个**窗口**:"它刚杀完人,接下来两三个阶段里它
    既不扫描也不追人,这段时间去占点最划算"。无记忆的策略拿不到这个窗口的开合时刻。
    """

    name = "official_ambusher"
    is_official = True
    HIDE_PHASES = 3

    def __init__(self, seed: int = 0):
        self.reset()

    def reset(self) -> None:
        self._last_score = None
        self._hide_left = 0
        self._hide_corner = None

    def turn(self, view_fn, my_color, ob):
        board = view_fn()
        me, opp = board.me_opp(my_color)
        position = me.last_known_pos
        facing = me.last_known_facing
        obstacles = board.obstacles

        # --- 相位推进:先判"上一阶段是不是击杀了对方" ---
        # 判据是比分差 >= 2,靠的是**结算频率**而不是"它不占点":击杀 +2 是一次性的,
        # 而占点只在 `Game.end_phase` 结算一次、一次 +1,所以单个阶段里 >= 2 的分差
        # 只可能含击杀(杀+占同阶段是 +3,照样 >= 2)。这条**不随得分区口径变化** ——
        # 这四个规则手现在会路过得分区并顺路拿 1 分,不能改回"它们不占点所以 ..."。
        # `_last_score is None` 是局中第一个阶段,没有上一阶段可比 —— 不当成击杀,
        # 否则开局它就先白躲三个阶段。
        killed = self._last_score is not None and me.score - self._last_score >= 2
        self._last_score = me.score
        if killed:
            self._hide_left = self.HIDE_PHASES
            # 参照取**自己**开火时所在的格(就是这一刻自己站的地方),不取敌人的
            # 最后已知位置:敌人被击杀后在它自己的重生点复活,拿它当参照会让藏身点
            # 恒等于同一个角 —— 常量一旦能被记住,这个对手就退化成普通靶子。
            self._hide_corner = farthest_corner(board.size, tuple(position))

        if self._hide_left > 0:
            self._hide_left -= 1
            position = yield from self._retreat(
                board, me, opp, position, facing, obstacles, self._hide_corner, 0
            )
            # 提前到了就不再耗剩下的阶段 —— 口径是"三个回合**内**绕到角落,然后再
            # 找人杀",而不是"绕到角落之后还要蹲满三个回合"。蹲满会把近一半的阶段
            # 花在原地不动,对手的威胁随时间塌成常数,记忆也就没东西可记了。
            if position == self._hide_corner:
                self._hide_left = 0
            return

        yield from self._hunt(board, me, opp, position, facing, obstacles, 0)


class OfficialWeaver(_MemoryOpponent):
    """自研记忆型规则手:**固定周期地出没** —— 主动找 `HUNT_PHASES`(=4)个阶段,
    然后消失 `GHOST_PHASES`(=3)个阶段,往复。

    与 `OfficialAmbusher` 的关键区别在**驱动源**:ambusher 的躲藏由"刚杀过人"
    这个事件触发,weaver 的由它自己的阶段计数器触发,与击杀完全无关。于是两者
    "看不见的时刻"在时间上的分布不一样 —— 在 ambusher 身上学到的那套记忆,
    拿到 weaver 身上不成立。

    每次消失去**轮换的角**:第 k 次消失去 `corners()[k % 4]`。轮换是有意的 ——
    固定躲同一个角会被"记住那个角"破解,轮换则要求策略记住"这是第几次消失",
    而不只是"它正在消失"。

    周期取 4+3=7:与 `HIDE_PHASES`=3 不同频,两种对手的相位不会长期对齐。
    """

    name = "official_weaver"
    is_official = True
    HUNT_PHASES = 4
    GHOST_PHASES = 3

    def __init__(self, seed: int = 0):
        self.reset()

    def reset(self) -> None:
        self._phase = 0

    def turn(self, view_fn, my_color, ob):
        board = view_fn()
        me, opp = board.me_opp(my_color)
        position = me.last_known_pos
        facing = me.last_known_facing
        obstacles = board.obstacles

        period = self.HUNT_PHASES + self.GHOST_PHASES
        idx = self._phase
        self._phase += 1
        if idx % period >= self.HUNT_PHASES:
            angle = corners(board.size)
            yield from self._retreat(
                board,
                me,
                opp,
                position,
                facing,
                obstacles,
                angle[(idx // period) % len(angle)],
                0,
            )
            return

        yield from self._hunt(board, me, opp, position, facing, obstacles, 0)
