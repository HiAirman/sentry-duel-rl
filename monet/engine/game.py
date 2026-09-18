"""哨兵大战规则状态机(sentry_duel V4.4)。

绝对坐标下的权威状态;对外通过 view(color) 提供"已镜像"的 Board 快照,
与选手 act() 收到的视图一致(双方都认为自己出生在 (0,0) 且朝 E)。

设计要点
--------
* 分数、CD、回合结构、加时、免费 TURN 全部按 rules.md/api.md 实现。
* 视野 / 火力用 rules.py 的纯函数,保证与部署侧 utils.h 同语义。
* 失败行动不消耗额度(§7.1),由调用方决定重试还是结束。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from . import rules as R
from .types import ActionObservation, ActionResult, Board, Pos, Sentry

# 行动编号(与 rl_v5 deploy 侧完全一致)
MOVE, TURN_N, TURN_E, TURN_S, TURN_W, FIRE, SCAN, END = range(8)
ACTION_DIM = 8
ACTION_TURN = {TURN_N: "N", TURN_E: "E", TURN_S: "S", TURN_W: "W"}
TURN_ACTION = {v: k for k, v in ACTION_TURN.items()}

ACTIONS_PER_TURN = 3


class _Side:
    """一方的绝对坐标状态。"""

    __slots__ = ("pos", "facing", "fire_cd", "scan_cd", "score")

    def __init__(self, color: str):
        self.pos: Pos = R.SPAWN[color]
        self.facing: str = R.START_FACING[color]
        self.fire_cd: int = 0
        self.scan_cd: int = 0
        self.score: int = 0

    def respawn(self, color: str) -> None:
        """被击杀:回出生点、朝向重置,CD 保持不变(§6.2)。"""
        self.pos = R.SPAWN[color]
        self.facing = R.START_FACING[color]


class Game:
    """一局 1v1。所有坐标为绝对坐标。"""

    def __init__(
        self,
        obstacles=R.DEFAULT_OBSTACLES,
        score_zones=R.DEFAULT_SCORE_ZONES,
        max_turns: int = 20,
        max_overtime: int = 5,
        budget: int = ACTIONS_PER_TURN,
    ):
        self.obstacles = set(map(tuple, obstacles))
        self.score_zones = set(map(tuple, score_zones))
        self.max_turns = max_turns
        self.max_overtime = max_overtime
        self.budget = budget

        self.s: Dict[str, _Side] = {"R": _Side("R"), "B": _Side("B")}
        # 对方最后已知情报(绝对坐标)。`seen` 与 `pos[0] >= 0` 是同一个事实的两种
        # 写法:**读的人一律用后者**(`view()` 与 `_result()` 里各一处),`seen` 只写
        # 不读。新增读取点时沿用 `pos[0] >= 0` —— 两条判据并存,迟早会有一条忘了更新。
        self.intel: Dict[str, Dict] = {
            "R": {"pos": (-1, -1), "facing": "?", "seen": False},
            "B": {"pos": (-1, -1), "facing": "?", "seen": False},
        }
        self.turn = 0
        self.phase = "R"
        self.actions_used = 0
        self.free_turn = {"R": True, "B": True}
        self.scan_reveal = {"R": False, "B": False}
        self.overtime = 0
        self.done = False
        self.winner: Optional[str] = None  # 'R' / 'B' / None(平局)
        self.history = []

        self.begin_phase("R")

    # ------------------------------------------------------------------ 查询

    @property
    def score(self) -> Tuple[int, int]:
        return self.s["R"].score, self.s["B"].score

    def opp(self, color: str) -> str:
        return R.OTHER[color]

    def phase_over(self) -> bool:
        return self.actions_used >= self.budget

    def directly_visible(self, color: str) -> bool:
        """对方此刻是否在我方 T 形直接视野内(绝对坐标判定)。"""
        me, op = self.s[color], self.s[self.opp(color)]
        return R.can_see(me.pos, me.facing, op.pos, self.obstacles)

    def opp_visible(self, color: str) -> bool:
        """对方是否对我方可见 = 直接视野 ∪ 本回合 SCAN 临时视野。"""
        return self.directly_visible(color) or self.scan_reveal[color]

    def visible_cells(self, color: str) -> set:
        me = self.s[color]
        return set(R.visible_cells(me.pos, me.facing, self.obstacles))

    def in_zone(self, pos: Pos) -> bool:
        return pos in self.score_zones

    # -------------------------------------------------------------- 视图构建

    def view(self, color: str) -> Board:
        """构造 color 视角下的 Board 快照(蓝方已做 180 度镜像)。"""
        me = self.s[color]
        op = self.s[self.opp(color)]
        intel = self.intel[color]
        vis = self.opp_visible(color)

        opp_pos_abs = op.pos if vis else intel["pos"]
        opp_face_abs = op.facing if vis else intel["facing"]

        me_v = Sentry(
            last_known_pos=R.to_view(color, me.pos),
            last_known_facing=R.view_dir(color, me.facing),
            visible=True,
            fire_cd=me.fire_cd,
            scan_cd=me.scan_cd,
            score=me.score,
        )
        opp_v = Sentry(
            last_known_pos=R.to_view(color, opp_pos_abs) if opp_pos_abs[0] >= 0 else (-1, -1),
            last_known_facing=R.view_dir(color, opp_face_abs),
            visible=vis,
            fire_cd=-1,  # 对方 CD 不公开(api.md §3)
            scan_cd=-1,
            score=op.score,
        )
        if color == "R":
            red, blue = me_v, opp_v
            obs = sorted(self.obstacles)
            zones = sorted(self.score_zones)
        else:
            red, blue = opp_v, me_v
            obs = sorted(R.mirror_pos(p) for p in self.obstacles)
            zones = sorted(R.mirror_pos(p) for p in self.score_zones)

        return Board(
            red=red,
            blue=blue,
            turn=self.turn,
            size=R.BOARD_SIZE,
            obstacles=obs,
            score_zones=zones,
        )

    # ---------------------------------------------------------------- 生命周期

    def begin_phase(self, color: str) -> None:
        self.phase = color
        self.actions_used = 0
        self._refresh_intel(color)

    def _refresh_intel(self, color: str) -> None:
        """把"此刻确实看得见"的敌方信息写入 color 的情报。"""
        if self.opp_visible(color):
            op = self.s[self.opp(color)]
            self.intel[color] = {"pos": op.pos, "facing": op.facing, "seen": True}

    def end_phase(self, color: str) -> None:
        """该方行动阶段结束:结算占点分 → 移交行动权 / 结束本回合。"""
        if self.s[color].pos in self.score_zones:
            self.s[color].score += 1

        self.free_turn[color] = False  # 免费 TURN 只在"复活/开局后的那一次 act"内有效
        self.scan_reveal[color] = False  # SCAN 临时视野只持续到本次 act 结束
        self.history.append((self.turn, color, self.s["R"].score, self.s["B"].score))

        if color == "R":
            self.begin_phase("B")
            return

        # 回合结束:双方 CD 统一递减(§3.2)
        for side in self.s.values():
            side.fire_cd = max(0, side.fire_cd - 1)
            side.scan_cd = max(0, side.scan_cd - 1)
        self.turn += 1
        self._check_terminal()
        if not self.done:
            self.begin_phase("R")

    def _check_terminal(self) -> None:
        if self.turn < self.max_turns:
            return
        a, b = self.score
        if a != b:
            self.done = True
            self.winner = "R" if a > b else "B"
            return
        # 平分 → 加时,最多 max_overtime 个完整回合
        self.overtime = self.turn - self.max_turns
        if self.overtime >= self.max_overtime:
            self.done = True
            self.winner = None  # 平局

    def force_timeout(self, color: str) -> None:
        """§7.2 超时:该方本回合放弃,对方 +1,已结算的行动不回滚。"""
        self.s[self.opp(color)].score += 1
        self.end_phase(color)

    # ------------------------------------------------------------------ 行动

    def legal_actions(self, color: str) -> list:
        """从引擎角度(真实状态)判断的合法行动。用于对手策略/自检。

        契约:`apply(color, a)` 对 a ∈ legal_actions(color) 不会返回
        `{"rejected": ...}`。因此这里必须和 apply 逐条对齐 ——
        同朝向 TURN 也算成功行动(§3.4),额度用尽后只剩 END。
        """
        if self.done:
            return []
        me = self.s[color]
        op = self.s[self.opp(color)]
        legal = [END]  # END 永远被接受(§3.4)
        if self.actions_used >= self.budget:
            return legal
        dx, dy = R.DELTA[me.facing]
        nx, ny = me.pos[0] + dx, me.pos[1] + dy
        if R.in_bounds(nx, ny) and (nx, ny) not in self.obstacles and (nx, ny) != op.pos:
            legal.append(MOVE)
        legal.extend(ACTION_TURN)  # 含当前朝向:转向到当前朝向是成功行动(§3.4)
        if me.fire_cd == 0:
            legal.append(FIRE)
        if me.scan_cd == 0:
            legal.append(SCAN)
        return legal

    def apply(self, color: str, action: int):
        """执行一个行动(在 color 的视角坐标系下给出)。

        返回 (ActionResult, events)。events 供 RL 奖励使用:
          kill / death / scan / moved / turned / fired
        失败的行动 success=False 且不消耗额度(§7.1)。
        """
        if self.done:
            return ActionResult(), {}
        if color != self.phase:
            raise RuntimeError(f"not {color}'s phase (phase={self.phase})")

        me = self.s[color]
        op_color = self.opp(color)
        op = self.s[op_color]
        events = {}

        if action == END:
            return self._result(color, True, False), {"end": True}

        if self.actions_used >= self.budget:
            return self._result(color, False, False), {"rejected": "budget"}

        success = False
        consumed = False

        if action == MOVE:
            dx, dy = R.DELTA[me.facing]
            nx, ny = me.pos[0] + dx, me.pos[1] + dy
            if R.in_bounds(nx, ny) and (nx, ny) not in self.obstacles and (nx, ny) != op.pos:
                me.pos = (nx, ny)
                success = consumed = True
                events["moved"] = True
                if me.pos != R.SPAWN[color]:
                    self.free_turn[color] = False  # 离开出生点 → 免费资格失效
            else:
                events["rejected"] = "move"

        elif action in ACTION_TURN:
            want = R.abs_dir(color, ACTION_TURN[action])  # 视角朝向 → 绝对朝向
            success = True
            events["turned"] = want
            if self.free_turn[color] and me.pos == R.SPAWN[color]:
                # 开局/复活后仍在出生点时的首个成功 TURN 免费(§4.2)
                self.free_turn[color] = False
                consumed = False
            else:
                consumed = True
            me.facing = want

        elif action == FIRE:
            if me.fire_cd == 0:
                success = consumed = True
                me.fire_cd = 2  # 回合末 -1 → 下回合 1(不可用)、再下回合 0
                events["fired"] = True
                if R.fire_hits(me.pos, me.facing, op.pos, self.obstacles):
                    op.respawn(op_color)
                    me.score += 2
                    self.free_turn[op_color] = True  # 复活后重新获得免费 TURN 资格
                    # 击杀是公开信息:击杀者立刻知道对方回到出生点
                    self.intel[color] = {"pos": op.pos, "facing": op.facing, "seen": True}
                    events["kill"] = True
                    events["death_color"] = op_color
            else:
                events["rejected"] = "fire_cd"

        elif action == SCAN:
            if me.scan_cd == 0:
                success = consumed = True
                me.scan_cd = 3
                self.scan_reveal[color] = True
                events["scan"] = True
                # SCAN 立刻返回实时位置与朝向(§4.4)
                self.intel[color] = {"pos": op.pos, "facing": op.facing, "seen": True}
            else:
                events["rejected"] = "scan_cd"
        else:
            raise ValueError(f"unknown action id {action}")

        if consumed:
            self.actions_used += 1

        self._refresh_intel(color)
        return self._result(color, success, consumed), events

    # ------------------------------------------------------------------ 内部

    def _result(self, color: str, success: bool, consumed: bool) -> ActionResult:
        me = self.s[color]
        op = self.s[self.opp(color)]
        intel = self.intel[color]
        vis = self.opp_visible(color)
        dir_vis = self.directly_visible(color)

        opp_pos_abs = op.pos if vis else intel["pos"]
        opp_face_abs = op.facing if vis else intel["facing"]

        obs = ActionObservation(
            my_pos=R.to_view(color, me.pos),
            my_facing=R.view_dir(color, me.facing),
            opp_last_known_pos=(
                R.to_view(color, opp_pos_abs) if opp_pos_abs[0] >= 0 else (-1, -1)
            ),
            opp_last_known_facing=R.view_dir(color, opp_face_abs),
            opp_visible=vis,
            opp_directly_visible=dir_vis,
            fire_cd=me.fire_cd,
            scan_cd=me.scan_cd,
        )
        return ActionResult(success=success, consumed=consumed, observation=obs)
