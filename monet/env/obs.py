"""观测构建 + 敌方信念追踪 —— rl_v5-source/obs_builder.h 的逐行 Python 移植。

移植目的:让 rl_monet_v1 训练出来的策略可以直接导出成 C++ .so
(rl_ai_v5.cpp 的推理壳),训练与部署共用同一份观测定义,杜绝观测漂移。

约束(与部署侧一致):只使用选手可得的公开信息 —— act() 收到的 Board 视图、
行动返回的 ActionObservation、utils.h 语义的几何函数。
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..engine import rules as R
from ..engine.types import ActionObservation, Board, Pos, Sentry

BOARD_SIZE = 7
CELLS = BOARD_SIZE * BOARD_SIZE
PLANES = 8
SCALARS = 36
OBS_DIM = PLANES * CELLS + SCALARS  # 428

MOVE, TURN_N, TURN_E, TURN_S, TURN_W, FIRE, SCAN, END = range(8)
ACTION_DIM = 8
_TURN_DIRS = ("N", "E", "S", "W")
TURN_ACTION = {"N": TURN_N, "E": TURN_E, "S": TURN_S, "W": TURN_W}


def _cell(x: int, y: int) -> int:
    return y * BOARD_SIZE + x


# 格号 → (x, y),与 _cell 严格互逆。信念扩张/视野扣除都按这个顺序逐格扫。
_XY = tuple((i % BOARD_SIZE, i // BOARD_SIZE) for i in range(CELLS))


@lru_cache(maxsize=8)
def _neighbor_masks(obstacles: frozenset) -> tuple:
    """每格的「可走进的邻居」位掩码(位 i 对应格 i)。

    扩张用的图只由障碍决定 —— 棋盘 7×7 是固定的,障碍一局之内也不变,所以能按
    障碍集合缓存。

    注意掩码只排除障碍,不排除 my_pos:自己的格子能不能当**起点**取决于信念,
    由调用方按 my_pos 单独处理。
    """
    masks = []
    for x, y in _XY:
        m = 0
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if R.in_bounds(nx, ny) and (nx, ny) not in obstacles:
                m |= 1 << _cell(nx, ny)
        masks.append(m)
    return tuple(masks)


class ObsBuilder:
    """与 obs_builder.h::rl::ObsBuilder 一一对应。"""

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------ reset

    def reset(self) -> None:
        self.belief = [0.0] * CELLS
        self.belief[_cell(6, 6)] = 1.0  # 对方出生在镜像视角的 (6,6)
        self.intel_pos: Pos = (-1, -1)
        self.intel_facing = "?"
        self.intel_turn = -100
        self.my_pos: Pos = (0, 0)
        self.my_facing = "E"
        self.my_score = 0
        self.opp_score = 0
        self.fire_cd = 0
        self.scan_cd = 0
        self.turn = 0
        self.actions_used = 0
        self.is_blue = False
        self.opp_visible = False
        self.opp_directly_visible = False
        self.free_turn = False
        self.first_act = True
        self.prev_act_end_pos: Pos = (0, 0)
        self.seen_now: Pos = (-1, -1)
        self.obstacles: list = []
        self.zones: list = []
        # obstacles 的 frozenset 副本:视野、信念、掩码每步都要查成员。必须与
        # self.obstacles 同步更新 —— 只改一个不会报错,但掩码会与视野对不上。
        self._obs_frozen: frozenset = frozenset()
        # 统计/指纹
        self.recent_deaths = 0
        self.total_deaths = 0
        self.steps_since_move = 0
        self.kills = 0
        self.my_zone_turns_ago = 99
        self.opp_zone_turns_ago = 99
        self.my_turn_zone_streak = 0
        self.last_was_fire = False
        self.last_action_id = -1
        self.start_pos: Pos = (0, 0)
        self.last_turn_pos: Pos = (0, 0)
        self.path_anchor_x = 0
        self.path_anchor_y = 0
        self.recent_displacement = 0
        self.recent_unique_cells = 0
        self.consecutive_same_pos_turns = 0
        self.death_count_short_window = 0

    # -------------------------------------------------------------- act_start

    def act_start(self, view: Board, my_color: str) -> None:
        me, opp = view.me_opp(my_color)
        self.is_blue = my_color == "B"
        self.turn = view.turn
        self.obstacles = list(view.obstacles)
        self._obs_frozen = frozenset(self.obstacles)
        self.zones = list(view.score_zones)

        # --- 击杀/死亡侦查(分数差是公开信息)---
        my_delta = me.score - self.my_score
        opp_delta = opp.score - self.opp_score
        # 死亡判定,两个来源取或:
        #   ① 对方分数 +2 —— 公开的击杀计分,直接可读;
        #   ② 我出现在出生点 (0,0),而上一阶段结束时不在那里 —— 复活位移。
        # **这是观测语义,不只是统计**:下面的 s[19] free_turn 直接取它,
        # s[20]/s[21]/s[35] 与 recent_deaths 也都吃它。改口径 = 废掉现有全部权重。
        i_died = opp_delta >= 2 or (
            self._same(me.last_known_pos, (0, 0))
            and not self._same(me.last_known_pos, self.prev_act_end_pos)
            and not self.first_act
        )
        if my_delta >= 2:
            # 我击杀了对方:对方回出生点(公开规则)
            self.belief = [0.0] * CELLS
            self.belief[_cell(6, 6)] = 1.0
            self.intel_pos = (6, 6)
            self.intel_facing = "W"
            self.intel_turn = self.turn
        # **缺口**:这里只反映"上一阶段我死过",引擎在开局给的那次免费转身不在内 ——
        # 第一阶段的 s[19] 因此是 False,而引擎那边是 True。不去修它(改口径 = 重新
        # 定义 428 维观测、废掉全部已训权重),消费方得自己补判据:见
        # `monet/env/rules_vg.py::free_turn_available`。信了 s[19] 不会报错,只会
        # 让"开局转身"这类动作被判成不可行,表现为静默变弱。
        self.free_turn = i_died

        self.my_score = me.score
        self.opp_score = opp.score

        if i_died:
            self.total_deaths += 1
            self.recent_deaths += 1
        if my_delta >= 2:
            self.kills += 1

        if not self.first_act:
            if not self._same(me.last_known_pos, self.prev_act_end_pos):
                self.steps_since_move = 0
            else:
                self.steps_since_move += 1

        me_in_zone = self._in_zone(me.last_known_pos, self.zones)
        opp_in_zone = self._in_zone(opp.last_known_pos, self.zones)
        if me_in_zone:
            self.my_turn_zone_streak += 1
            self.my_zone_turns_ago = 0
        else:
            self.my_zone_turns_ago += 1
        if opp_in_zone:
            self.opp_zone_turns_ago = 0
        else:
            self.opp_zone_turns_ago += 1

        if self.recent_deaths > 0 and self.turn > 0:
            self.recent_deaths -= 1

        # --- 路径指纹 ---
        self.start_pos = me.last_known_pos
        dx_start = abs(me.last_known_pos[0] - self.last_turn_pos[0])
        dy_start = abs(me.last_known_pos[1] - self.last_turn_pos[1])
        if dx_start + dy_start > 0:
            self.recent_displacement += 1
            # 这里读到的 `actions_used` 是**上一阶段结束时**的额度 —— 它在函数末尾
            # (`self.actions_used = 0`)才清零。把这个清零挪到前面,`recent_unique_cells`
            # 会静默变成"每回合都 +1",不报错,只是这个特征失去区分度。
            if self.actions_used == 0 and self.turn % 2 == 0:
                self.recent_unique_cells += 1
            self.consecutive_same_pos_turns = 0
        elif not self.first_act:
            self.consecutive_same_pos_turns += 1
        self.last_turn_pos = me.last_known_pos
        self.death_count_short_window = min(self.recent_deaths, 3)
        if self.turn % 4 == 0:
            self.path_anchor_x, self.path_anchor_y = me.last_known_pos

        self.my_pos = me.last_known_pos
        self.my_facing = me.last_known_facing
        self.fire_cd = me.fire_cd
        self.scan_cd = me.scan_cd
        self.actions_used = 0

        # --- 信念时间推进:敌方自上一来我方 act 起行动过一个阶段 ---
        if not self.first_act:
            self._dilate_belief()

        # --- 视野证伪 ---
        self.opp_visible = opp.visible
        self.opp_directly_visible = opp.visible and self._can_see_me(opp.last_known_pos)
        if opp.visible and opp.last_known_pos[0] >= 0:
            self.belief = [0.0] * CELLS
            self.belief[_cell(*opp.last_known_pos)] = 1.0
            self.intel_pos = opp.last_known_pos
            self.intel_facing = opp.last_known_facing
            self.intel_turn = self.turn
        if not self.opp_directly_visible:
            self._subtract_visible_cells()

        self.seen_now = opp.last_known_pos if self.opp_directly_visible else (-1, -1)
        self.first_act = False

    # -------------------------------------------------------- on_observation

    def on_observation(self, o: ActionObservation, consumed: bool, action_id: int = -1) -> None:
        self.last_action_id = action_id
        # 朝向变了但额度未消耗 = 复活免费 TURN 被用掉
        if self.free_turn and not consumed and o.my_facing != self.my_facing:
            self.free_turn = False
        self.my_pos = o.my_pos
        self.my_facing = o.my_facing
        self.fire_cd = o.fire_cd
        self.scan_cd = o.scan_cd
        if consumed:
            self.actions_used += 1
            self.last_was_fire = action_id == FIRE
        self.opp_visible = o.opp_visible
        self.opp_directly_visible = o.opp_directly_visible
        if o.opp_visible and o.opp_last_known_pos[0] >= 0:
            self.belief = [0.0] * CELLS
            self.belief[_cell(*o.opp_last_known_pos)] = 1.0
            self.intel_pos = o.opp_last_known_pos
            self.intel_facing = o.opp_last_known_facing
            self.intel_turn = self.turn
        if not self.opp_directly_visible:
            self._subtract_visible_cells()
        self.seen_now = o.opp_last_known_pos if self.opp_directly_visible else (-1, -1)
        self.prev_act_end_pos = self.my_pos

    # ------------------------------------------------------------------ encode

    def encode(self, out: np.ndarray) -> np.ndarray:
        # 8 个 7×7 平面:0 障碍 / 1 得分区 / 2 我方位置 / 3 敌方信念分布 /
        # 4 本步直接看到 / 5 最后已知情报 / 6、7 双方出生点常量。
        out.fill(0.0)
        for o in self.obstacles:
            out[0 * CELLS + _cell(o[0], o[1])] = 1.0
        for z in self.zones:
            out[1 * CELLS + _cell(z[0], z[1])] = 1.0
        out[2 * CELLS + _cell(*self.my_pos)] = 1.0
        out[3 * CELLS : 4 * CELLS] = self.belief
        if self.seen_now[0] >= 0:
            out[4 * CELLS + _cell(*self.seen_now)] = 1.0
        if self.intel_pos[0] >= 0:
            out[5 * CELLS + _cell(*self.intel_pos)] = 1.0
        out[6 * CELLS + _cell(0, 0)] = 1.0
        out[7 * CELLS + _cell(6, 6)] = 1.0

        s = out[PLANES * CELLS :]
        # 36 个标量。除朝向 one-hot 外都已归一化,分母取各自的上界:冷却与单阶段
        # 动作数除以 3,比分除以 20,回合数除以 24,棋盘坐标除以 6(坐标范围 0..6)。
        #   s[0..3] 我方朝向;s[4..7] 敌人朝向;s[8] 敌人朝向未知;
        #   s[9..19] 冷却/比分/回合/额度/阵营/可见性/情报新旧/免费转身;
        #   s[20..27] 死亡/连动/击杀/得分区统计;
        #   s[28..35] 路径指纹(锚点、位移、去重格、原地不动)加一个短期死亡数。
        if self.my_facing in _TURN_DIRS:
            s[_TURN_DIRS.index(self.my_facing)] = 1.0
        if self.intel_facing in _TURN_DIRS:
            s[4 + _TURN_DIRS.index(self.intel_facing)] = 1.0
        else:
            s[8] = 1.0
        s[9] = self.fire_cd / 3.0
        s[10] = self.scan_cd / 3.0
        s[11] = self.my_score / 20.0
        s[12] = self.opp_score / 20.0
        s[13] = self.turn / 24.0
        s[14] = self.actions_used / 3.0
        s[15] = 1.0 if self.is_blue else 0.0
        s[16] = 1.0 if self.opp_visible else 0.0
        s[17] = 1.0 if self.opp_directly_visible else 0.0
        s[18] = (self.turn - self.intel_turn) / 24.0 if self.intel_pos[0] >= 0 else 1.0
        s[19] = 1.0 if self.free_turn else 0.0
        s[20] = min(self.recent_deaths, 4) / 4.0
        s[21] = min(self.total_deaths, 10) / 10.0
        s[22] = min(self.steps_since_move, 10) / 10.0
        s[23] = min(self.kills, 5) / 5.0
        s[24] = min(self.my_zone_turns_ago, 10) / 10.0
        s[25] = min(self.opp_zone_turns_ago, 10) / 10.0
        s[26] = min(self.my_turn_zone_streak, 6) / 6.0
        s[27] = 1.0 if self.last_was_fire else 0.0
        s[28] = self.path_anchor_x / 6.0
        s[29] = self.path_anchor_y / 6.0
        s[30] = (self.my_pos[0] - self.start_pos[0]) / 6.0
        s[31] = (self.my_pos[1] - self.start_pos[1]) / 6.0
        s[32] = min(self.recent_displacement, 6) / 6.0
        s[33] = min(self.recent_unique_cells, 7) / 7.0
        s[34] = min(self.consecutive_same_pos_turns, 4) / 4.0
        s[35] = min(self.death_count_short_window, 3) / 3.0
        return out

    # ------------------------------------------------------------- action_mask

    def action_mask(self, mask: np.ndarray) -> np.ndarray:
        """只用公开信息构造掩码;隐形的敌人占据格不掩,交给引擎拒绝(失败不消耗)。"""
        mask.fill(0.0)
        if self.actions_used >= 3:
            mask[END] = 1.0
            return mask
        dx, dy = R.DELTA.get(self.my_facing, (0, 0))
        nx, ny = self.my_pos[0] + dx, self.my_pos[1] + dy
        blocked_by_opp = self.opp_directly_visible and (nx, ny) == self.intel_pos
        if R.in_bounds(nx, ny) and (nx, ny) not in self._obs_frozen and not blocked_by_opp:
            mask[MOVE] = 1.0
        for i, d in enumerate(_TURN_DIRS):
            if self.my_facing != d:
                mask[1 + i] = 1.0
        if self.fire_cd == 0:
            mask[FIRE] = 1.0
        if self.scan_cd == 0:
            mask[SCAN] = 1.0
        mask[END] = 1.0
        return mask

    # ------------------------------------------------------------------ 工具

    @staticmethod
    def _same(a, b) -> bool:
        return a[0] == b[0] and a[1] == b[1]

    @staticmethod
    def _in_zone(p, zones) -> bool:
        return any(z[0] == p[0] and z[1] == p[1] for z in zones)

    def _can_see_me(self, target) -> bool:
        return R.can_see(self.my_pos, self.my_facing, target, self._obs_frozen)

    def _dilate_belief(self) -> None:
        """敌方一个行动阶段最多移动 3 格:信念按 BFS≤3(绕障碍、不占我方格)扩张。

        这里把信念当 49 位掩码算,邻居关系由 `_neighbor_masks` 预计算。掩码表示
        无损:belief 的取值只在 {0.0, 1.0},由 `reset`、`encode` 与本法共同保证。
        """
        nbm = _neighbor_masks(self._obs_frozen)
        bit = 0
        for i, v in enumerate(self.belief):
            if v > 0.0:
                bit |= 1 << i
        # 我方所在格可以当扩张的**起点**,但永不作为目标加进来
        my_bit = 1 << _cell(*self.my_pos)
        for _ in range(3):
            add = 0
            m = bit
            while m:
                low = m & -m
                add |= nbm[low.bit_length() - 1]
                m ^= low
            bit |= add & ~my_bit
        self.belief = [1.0 if (bit >> i) & 1 else 0.0 for i in range(CELLS)]

    def _subtract_visible_cells(self) -> None:
        """把「我方现在看得见」的格子从信念里清掉。

        视野只依赖 (my_pos, my_facing, obstacles) —— 本次调用内是常量,所以先算一次
        可见集,再逐格查表。

        等价性:`visible_cells` 从不返回障碍格,所以对任意 target
            can_see(t) ≡ (t == my_pos) or (t in visible_cells(...))
        —— 把 my_pos 并进可见集就是复刻 `can_see` 的「同格可见」约定(README §六.3),
        连同它多清掉一格信念的行为一起原样保留。
        """
        vis = set(R.visible_cells(self.my_pos, self.my_facing, self._obs_frozen))
        vis.add(self.my_pos)
        belief = self.belief
        for i, p in enumerate(_XY):
            if belief[i] > 0.0 and p in vis:
                belief[i] = 0.0


def encode_board(view: Board, my_color: str, ob: ObsBuilder) -> np.ndarray:
    """便捷入口:act_start + encode。"""
    ob.act_start(view, my_color)
    out = np.zeros(OBS_DIM, dtype=np.float32)
    return ob.encode(out)
