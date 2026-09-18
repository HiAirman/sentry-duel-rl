"""观测构建 + 敌方信念追踪 —— `RL_best040-source/obs_builder.h` 的逐行 Python 移植。

与 `monet/env/obs.py` 的关系
---------------------------
`obs.py` 是 **rl_v5-source/obs_builder.h** 的移植(428 = 8x49 + **36** 个标量);
本文件是 **RL_best040-source/obs_builder.h** 的移植(412 = 8x49 + **20** 个标量)。
两份的差别只有三处:

1. `kScalars` 36 → 20:s[20..35] 那 16 个"死亡/得分区/路径指纹"标量在 best040 里
   不存在,输入维度因此不同 —— **两套权重不能互换**,`BeliefObs.encode()` 的输出
   只能喂 best040 的网络(412→256→256→8,见 `env/rlbest.py`)。
2. `on_observation` 少了 `action_id` 出参(它只喂给 s[27] `last_was_fire`,而那个
   标量在 20 个里没有)。
3. `act_start` 的视野分支多了一条 `else if (first_act_ && opp.last_known_pos.x >= 0)`:
   官方引擎在蓝方回合 0 会**额外**补一份红方回合末位置(`visible=false`),best040
   用它把信念/情报一次性初始化掉。我们的 `Game.view` 目前不提供这份补偿情报
   (它只给公开的 `intel`,而开局 `intel` 是空的),所以这条分支在本引擎里恒不触发 ——
   按原文保留,引擎哪天补上就自动生效。见 `env/rlbest.py` 顶部的"与引擎的差异"。

除了这三处,其余(8 个平面、20 个标量的取值、掩码规则、信念扩张/视野扣除)与
`obs.py` 逐字相同。
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..engine import rules as R
from ..engine.types import ActionObservation, Board, Pos

BOARD_SIZE = 7
CELLS = BOARD_SIZE * BOARD_SIZE
PLANES = 8
SCALARS = 20
OBS_DIM = PLANES * CELLS + SCALARS  # 412

MOVE, TURN_N, TURN_E, TURN_S, TURN_W, FIRE, SCAN, END = range(8)
ACTION_DIM = 8
ACTIONS_PER_TURN = 3
_TURN_DIRS = ("N", "E", "S", "W")


def _cell(x: int, y: int) -> int:
    return y * BOARD_SIZE + x


# 格号 → (x, y),与 _cell 严格互逆。信念扩张/视野扣除都按这个顺序逐格扫。
_XY = tuple((i % BOARD_SIZE, i // BOARD_SIZE) for i in range(CELLS))

# 四邻(与 C++ `kDx/kDy` 同序:N/E/S/W)
_NEIGHBORS = ((0, -1), (1, 0), (0, 1), (-1, 0))


@lru_cache(maxsize=8)
def _neighbor_masks(obstacles: frozenset) -> tuple:
    """每格的「可走进的邻居」位掩码(位 i 对应格 i)。与 `obs.py` 同一套做法。

    扩张用的图只由障碍决定 —— 棋盘 7×7 固定,障碍一局之内也不变。掩码只排除障碍,
    不排除 `my_pos`:自己的格子能不能当**起点**取决于信念,由调用方按 `my_pos` 处理。
    """
    masks = []
    for x, y in _XY:
        m = 0
        for dx, dy in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            if R.in_bounds(nx, ny) and (nx, ny) not in obstacles:
                m |= 1 << _cell(nx, ny)
        masks.append(m)
    return tuple(masks)


class BeliefObs:
    """与 `RL_best040-source/obs_builder.h::rl::ObsBuilder` 一一对应(412 维)。

    生命周期与选手 `act()` 一致:每局 `reset()`,每次行动阶段开始 `act_start(view)`,
    每个动作返回后 `on_observation(res.observation, res.consumed)`,
    决策前 `encode()` / `action_mask()`。
    """

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
        # 障碍的 frozenset 副本:视野、信念、掩码每步都要查成员,列表线性扫描和
        # `visible_cells` 内部的 set(obstacles) 重建都是白花的。与 obstacles 同步更新。
        self._obs_frozen: frozenset = frozenset()

    # -------------------------------------------------------------- act_start

    def act_start(self, view: Board, my_color: str) -> None:
        me, opp = view.me_opp(my_color)
        self.is_blue = my_color == "B"
        self.turn = view.turn
        self.obstacles = list(view.obstacles)
        self._obs_frozen = frozenset(self.obstacles)
        self.zones = list(view.score_zones)

        # --- 击杀/死亡侦查(分数差是公开信息)---
        # 对方 +2 的唯一来源是击杀我(占点只有 +1),比位置判断更可靠
        my_delta = me.score - self.my_score
        opp_delta = opp.score - self.opp_score
        i_died = opp_delta >= 2 or (
            self._same(me.last_known_pos, (0, 0))
            and not self._same(me.last_known_pos, self.prev_act_end_pos)
            and not self.first_act
        )
        if my_delta >= 2:
            # 我击杀了对方:对方回其出生点(公开规则),情报立刻更新
            self.belief = [0.0] * CELLS
            self.belief[_cell(6, 6)] = 1.0
            self.intel_pos = (6, 6)
            self.intel_facing = "W"
            self.intel_turn = self.turn
        # 免费 TURN 仅在复活后第一个 act 内有效;本 act 未被杀则收回
        self.free_turn = i_died

        self.my_score = me.score
        self.opp_score = opp.score
        self.my_pos = me.last_known_pos
        self.my_facing = me.last_known_facing
        self.fire_cd = me.fire_cd
        self.scan_cd = me.scan_cd
        self.actions_used = 0

        # --- 信念时间推进:从我方上一次 act 到现在,敌方行动过一个阶段 ---
        if not self.first_act:
            self._dilate_belief()

        # --- 视野证伪:当前 T 形视野内的格子若有人我必看到 ---
        self.opp_visible = opp.visible
        self.opp_directly_visible = opp.visible and self._can_see_me(opp.last_known_pos)
        if opp.visible and opp.last_known_pos[0] >= 0:
            self.belief = [0.0] * CELLS
            self.belief[_cell(*opp.last_known_pos)] = 1.0
            self.intel_pos = opp.last_known_pos
            self.intel_facing = opp.last_known_facing
            self.intel_turn = self.turn
        elif self.first_act and opp.last_known_pos[0] >= 0:
            # 蓝方回合 0 的先手补偿情报:引擎直接给出红方回合末位置(visible=false)。
            # 我们的 Game.view 不提供它,所以本分支恒不触发(见模块说明第 3 条)。
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

    def on_observation(self, o: ActionObservation, consumed: bool) -> None:
        # 朝向变了但额度未消耗 = 复活免费 TURN 被用掉
        if self.free_turn and not consumed and o.my_facing != self.my_facing:
            self.free_turn = False
        self.my_pos = o.my_pos
        self.my_facing = o.my_facing
        self.fire_cd = o.fire_cd
        self.scan_cd = o.scan_cd
        if consumed:
            self.actions_used += 1
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

    def encode(self, out: np.ndarray | None = None) -> np.ndarray:
        """编码成 412 维 float32。`out` 省略时新分配一份(对手侧用)。"""
        if out is None:
            out = np.zeros(OBS_DIM, dtype=np.float32)
        else:
            out.fill(0.0)

        # 平面 0:障碍;1:得分区
        for o in self.obstacles:
            out[_cell(o[0], o[1])] = 1.0
        for z in self.zones:
            out[CELLS + _cell(z[0], z[1])] = 1.0
        # 平面 2:我方位置
        out[2 * CELLS + _cell(*self.my_pos)] = 1.0
        # 平面 3:敌方信念
        out[3 * CELLS : 4 * CELLS] = self.belief
        # 平面 4:当前直接看到
        if self.seen_now[0] >= 0:
            out[4 * CELLS + _cell(*self.seen_now)] = 1.0
        # 平面 5:最后已知情报
        if self.intel_pos[0] >= 0:
            out[5 * CELLS + _cell(*self.intel_pos)] = 1.0
        # 平面 6/7:出生点常量
        out[6 * CELLS + _cell(0, 0)] = 1.0
        out[7 * CELLS + _cell(6, 6)] = 1.0

        s = out[PLANES * CELLS :]
        if self.my_facing in _TURN_DIRS:
            s[_TURN_DIRS.index(self.my_facing)] = 1.0
        if self.intel_facing in _TURN_DIRS:
            s[4 + _TURN_DIRS.index(self.intel_facing)] = 1.0
        else:
            s[8] = 1.0  # 未知
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
        return out

    # ------------------------------------------------------------- action_mask

    def action_mask(self, mask: np.ndarray | None = None) -> np.ndarray:
        """只用公开信息构造掩码;隐形的敌人占据格不掩,交给引擎拒绝(失败不消耗)。"""
        if mask is None:
            mask = np.zeros(ACTION_DIM, dtype=np.float32)
        else:
            mask.fill(0.0)
        if self.actions_used >= ACTIONS_PER_TURN:
            mask[END] = 1.0  # 只能 end
            return mask
        # move
        dx, dy = R.DELTA.get(self.my_facing, (0, 0))
        nx, ny = self.my_pos[0] + dx, self.my_pos[1] + dy
        blocked_by_opp = self.opp_directly_visible and (nx, ny) == self.intel_pos
        if R.in_bounds(nx, ny) and (nx, ny) not in self._obs_frozen and not blocked_by_opp:
            mask[MOVE] = 1.0
        # turn 1..4 = N/E/S/W
        for i, d in enumerate(_TURN_DIRS):
            if self.my_facing != d:
                mask[1 + i] = 1.0
        if self.fire_cd == 0:
            mask[FIRE] = 1.0
        if self.scan_cd == 0:
            mask[SCAN] = 1.0
        mask[END] = 1.0
        return mask

    # -------------------------------------------------------------- 内部工具

    @staticmethod
    def _same(a, b) -> bool:
        return a[0] == b[0] and a[1] == b[1]

    def _can_see_me(self, target) -> bool:
        """C++ `can_see_me`:把 Sentry me 摊成 (my_pos, my_facing) 再调 can_see。"""
        return R.can_see(self.my_pos, self.my_facing, target, self._obs_frozen)

    def _dilate_belief(self) -> None:
        """敌方一个行动阶段最多移动 3 格:信念按 BFS≤3(绕障碍、不占我方格)扩张。

        信念当 49 位掩码算,邻居关系由 `_neighbor_masks` 预计算。掩码表示无损 ——
        belief 只在 {0.0, 1.0} 里取值,凡是写它的地方(`reset` / `act_start` /
        `on_observation` / 本方法 / `_subtract_visible_cells`)都只写这两个值。
        `encode` 只读不写。

        C++ 那份的 `next` 一开始等于 `belief_`,且每轮 `cur = next`(**跨轮累加**),
        所以语义是"3 步可达集";本版每轮 `bit |= add` 与之相同。`my_pos` 只被排除在
        **目标**一侧(它可以当扩张起点),由 `& ~my_bit` 复刻。差分基准见
        `tests/test_rlbest.py::test_belief_rewrite_matches_the_literal_cpp_reference`。
        """
        nbm = _neighbor_masks(self._obs_frozen)
        bit = 0
        for i, v in enumerate(self.belief):
            if v > 0.0:
                bit |= 1 << i
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
        """当前视野内确认无人的格子从信念剔除。

        视野只依赖 (my_pos, my_facing, obstacles) —— 本次调用内是常量。先算一次
        可见集再逐格查表,与 C++ 逐格调 `can_see` 等价:`visible_cells` 从不返回
        障碍格,所以 `can_see(t) ≡ (t == my_pos) or (t in visible_cells(...))`。
        把 `my_pos` 并进可见集就是复刻 `can_see` 的「同格可见」约定(README §六.3)。
        """
        vis = set(R.visible_cells(self.my_pos, self.my_facing, self._obs_frozen))
        vis.add(self.my_pos)
        belief = self.belief
        for i, p in enumerate(_XY):
            if belief[i] > 0.0 and p in vis:
                belief[i] = 0.0


def encode_board(view: Board, my_color: str, ob: BeliefObs) -> np.ndarray:
    """便捷入口:act_start + encode。"""
    ob.act_start(view, my_color)
    return ob.encode()
