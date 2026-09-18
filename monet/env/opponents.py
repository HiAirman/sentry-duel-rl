"""陪练对手:随机 / 启发式 / 冻结的神经网络快照。

所有对手都在"视图坐标系"(己方出生在 (0,0)、朝 E)下工作 —— 引擎已经替
调用方做了镜像,所以这里不需要关心红蓝。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..engine import rules as R
from ..engine.types import Board
from . import obs as O

ZONE_CENTER = (3, 3)  # 视图坐标系下的得分区中心


class Opponent:
    name = "base"
    # ⚠️ **这个类属性没有任何代码读它**。真正决定"算不算官方档"的是
    # `training/selfplay.py` 里注册工厂上的标记 —— `OFFICIAL_OPPONENTS` 是拿
    # `getattr(工厂, "is_official", False)` 筛出来的,而工厂是那些 lambda,
    # 不是这些类。所以 `env/official.py` 里六个类上写的 `is_official = True`
    # 只是**意图说明**,不构成成员资格:stalker / patrol / ambusher / weaver
    # 四个自研的都不在 `OFFICIAL_OPPONENTS` 里(现有成员见 tests/test_train.py
    # 钉住的那份名单)。
    #
    # 留着它是因为删掉会让"这个对手是不是官方档"在类上完全看不出来;但**要改
    # 成员资格,改的是 selfplay.py 的工厂包装(`_official(...)`),不是这里。**
    is_official = False

    def reset(self) -> None:  # pragma: no cover - 默认无状态
        pass

    def act(self, view: Board, my_color: str, ob: "O.ObsBuilder") -> int:
        raise NotImplementedError

    def turn(self, view_fn, my_color: str, ob: "O.ObsBuilder"):
        """一个完整的行动阶段,写成**生成器**。

        默认实现:反复调用 `act()`,直到它返回 END 或环境判定阶段结束。
        每步都重新调 `view_fn()` 取视图 —— 每个动作必须基于该动作执行后的最新
        状态,不能拿本回合开头那一份。

        整回合型对手(官方 Baseline / Hunter,见 `official.py`)覆写本方法:
        用 `res = yield action` 拿回每个动作的 `ActionResult`,从而 1:1 移植官方
        `act()` 里"边打边看观测"的控制流。

        动作必须 yield 给环境执行 —— 由环境调 `game.apply`,对手自己不能碰
        `game`,否则它造成的击杀不会流经奖励计算。
        """
        while True:
            a = self.act(view_fn(), my_color, ob)
            yield a
            if a == O.END:
                return


class RandomOpponent(Opponent):
    """按公开掩码均匀采样(掩码里可能有被隐形敌人占住的格子,交给引擎拒绝)。"""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, view, my_color, ob):
        mask = ob.action_mask(np.zeros(O.ACTION_DIM, dtype=np.float32))
        idx = np.flatnonzero(mask > 0)
        return int(self.rng.choice(idx))


class EndOpponent(Opponent):
    """什么都不做 —— 用于冒烟测试和"送分"基线。"""

    name = "end"

    def act(self, view, my_color, ob):
        return O.END


class HeuristicOpponent(Opponent):
    """规则手:能开火就开火,否则进得分区蹲点。

    `aggression` 是 0~1 的概率闸:越大越倾向追击已知敌人,0 = 永远蹲点
    (注册表里的 `camper`),1.0 = 冷却中也不蹲。
    """

    name = "heuristic"

    def __init__(
        self,
        aggression: float = 0.5,
        use_scan: bool = True,
        seed: int = 0,
        name: str = "heuristic",
    ):
        self.aggression = aggression
        self.use_scan = use_scan
        self.rng = np.random.default_rng(seed)
        # 同一个类的三个配置(baseline/hunter/camper)在日志与联盟占比里要能分开看
        self.name = name

    def act(self, view: Board, my_color, ob):
        me, opp = view.me_opp(my_color)
        obstacles = view.obstacles
        my_pos = me.last_known_pos
        facing = me.last_known_facing
        in_zone = ob._in_zone(my_pos, view.score_zones)

        # ---- 有敌情:能打就打,打不到就摆正朝向 ----
        if opp.visible and opp.last_known_pos[0] >= 0:
            tgt = opp.last_known_pos
            if me.fire_cd == 0 and R.fire_hits(my_pos, facing, tgt, obstacles):
                return O.FIRE
            want = R.best_turn_to_face(my_pos, tgt)
            if facing != want:
                return O.TURN_ACTION[want]
            if me.fire_cd == 0:
                return self._approach(me, tgt, obstacles)  # 朝向对但不在火力范围
            # 冷却中:占点优先;激进时追击
            if in_zone and self.rng.random() > self.aggression:
                return O.END
            return self._approach(me, tgt, obstacles)

        # ---- 无敌情 ----
        if self.use_scan and me.scan_cd == 0 and not in_zone and self.rng.random() < 0.3:
            return O.SCAN
        if in_zone:
            # 已在得分区:蹲着拿分,朝向摆到敌方出生角 —— 视图坐标下恒定是 (6, 6)
            want = R.best_turn_to_face(my_pos, (6, 6))
            if facing != want:
                return O.TURN_ACTION[want]
            return O.END
        return self._approach(me, ZONE_CENTER, obstacles)

    def _approach(self, me, target, obstacles) -> int:
        """先转后走,朝目标推进一格;主轴被挡则换另一轴。"""
        my_pos = me.last_known_pos
        facing = me.last_known_facing
        if my_pos == target:
            return O.END
        dx = target[0] - my_pos[0]
        dy = target[1] - my_pos[1]
        if abs(dx) >= abs(dy):
            order = [("E" if dx > 0 else "W") if dx else None,
                     ("S" if dy > 0 else "N") if dy else None]
        else:
            order = [("S" if dy > 0 else "N") if dy else None,
                     ("E" if dx > 0 else "W") if dx else None]
        blocked = set(map(tuple, obstacles))
        for d in order:
            if d is None:
                continue
            step = R.DELTA[d]
            nx, ny = my_pos[0] + step[0], my_pos[1] + step[1]
            if not R.in_bounds(nx, ny) or (nx, ny) in blocked:
                continue
            if facing != d:
                return O.TURN_ACTION[d]
            return O.MOVE
        for d in R.DIRS:  # 全被挡:随便转一个不等于当前的朝向
            if d != facing:
                return O.TURN_ACTION[d]
        return O.END


class NeuralOpponent(Opponent):
    """冻结的网络快照。ob 由环境负责 act_start / on_observation。

    `rules` 非空时是 rl_VG_v1.0 的混合策略(规则强制 + 网络残差)。**默认 None**,
    只有 rl_VG_v1.0 才传:其余权重都不是在规则下训出来的,挂上规则会改写棋力,
    与既有评测数字不再可比。
    """

    def __init__(self, net, temperature: float = 1.0, seed: int = 0, name: str = "nn", rules=None):
        self.net = net
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.name = name
        self.rules = rules
        self._buf = np.zeros(O.OBS_DIM, dtype=np.float32)
        self._mask = np.zeros(O.ACTION_DIM, dtype=np.float32)
        # 带记忆的对手(自己的 GRU 快照)必须在一局之内传递隐状态。
        # 引擎每局都会调 `opponent.reset()`(sentry_env.py 的 reset),所以
        # "逐局归零"是免费的,不用额外挂钩子。
        self.recurrent = getattr(net, "is_recurrent", False)
        self.h = None

    def reset(self):
        if self.rules is not None:
            self.rules.reset()
        if self.recurrent:
            self.h = self.net.new_hidden()

    def _logits(self, obs):
        if not self.recurrent:
            return self.net.logits(obs)
        if self.h is None:
            self.h = self.net.new_hidden()
        lg, _v, self.h = self.net.step(obs, self.h)
        return lg

    def act(self, view, my_color, ob):
        obs = ob.encode(self._buf)
        mask = ob.action_mask(self._mask)
        # 与训练侧同一个入口(act_mask),两条路径不会分叉。
        # 注意 action_mask 把结果写进 self._mask,所以这里必须**接收返回值**,
        # 后面的 logits 屏蔽用的是收窄/关过 SCAN 的那一份。
        if self.rules is not None:
            mask = self.rules.act_mask(ob, mask)[0]
        lg = self._logits(obs)
        lg = np.where(mask > 0, lg, -1e9)
        if self.temperature <= 1e-3:
            return int(np.argmax(lg))
        z = (lg - lg.max()) / self.temperature
        p = np.exp(z)
        p /= p.sum()
        return int(self.rng.choice(len(p), p=p))
