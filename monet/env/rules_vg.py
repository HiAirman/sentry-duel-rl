"""规则层:击杀搜索 + 失明计数。

策略是三层优先级:
  1. 本回合剩余额度内能击杀 → 击杀;
  2. 否则,敌方连续失明 `MISS_THRESHOLD` 个阶段且 SCAN 可用 → SCAN;
  3. 否则交给 RL 网络。

两条规则都只用**公开信息**:观测里的自己的位置/朝向/CD,以及 `opp_visible` 为真时
的 `intel_pos`。`game.py` 里唯一往 intel 写字的是 `_refresh_intel()`(由
`begin_phase()` 调用),它只在 `opp_visible` 为真时把 `op.pos` 原样写入;`view()`
只是把 intel 读出来。所以 `opp_visible` 为真时 `intel_pos` 是敌人的**实时**位置,
不是估计。而敌人只在我方阶段之外行动(`game.py` 的分阶段状态机),所以一次搜索
之内敌方格是静止的 —— 这是**证明**,不是预测。

本模块是规则的唯一真相来源,交付包里各有一份逐行移植的 C++(`vg_rules.h`)。
改这里就必须同步改**每一份**(tests/test_rules_vg.py 有差分测试盯着,但它只能抓
规格漂移,抓不到 C++ 侧的编译/编码错误)。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..engine import rules as R
from ..engine.game import ACTIONS_PER_TURN, FIRE, MOVE, SCAN, TURN_ACTION

Pos = Tuple[int, int]

# 失明几个**完整阶段**才扫。语义:阈值 2 ⇒ 第 3 个失明阶段才扫(两个阶段都"走完了"
# 才计数),阈值 4 ⇒ 第 5 个。想让它在第 2 个失明阶段就扫,把这里改成 1 即可
# (C++ 侧 `kMissThreshold` 必须同步)。
#
# 取值偏大是刻意的:扫得越勤越掉分。机制是 SCAN 把 `opp_visible` 翻成 True,网络的
# "看得见敌人"分支会离开得分区去追,而一个阶段站在得分区里 END 就是 +1 分。这是在
# **没带规则训过的网络**上量出来的代价,不是规则训过之后的稳态;复现脚本见
# tests/diag_vg_rules.py。
MISS_THRESHOLD = 4

# 一个阶段最多 ACTIONS_PER_TURN 个行动,搜索深度不会超过它。
MAX_SEARCH_DEPTH = ACTIONS_PER_TURN

# 网络**不能**自己选 SCAN —— SCAN 完全归规则 2 管:该扫的时候规则会强制扫,不该扫
# 的时候一次都不许浪费(SCAN 花掉的是一个行动额度,而不扫的代价只是情报旧一点)。
# 关法是把掩码里的 SCAN 位置零(见 `policy_mask`),不是事后覆盖动作 —— 后者会让存下来
# 的 logp 变成网络对那个动作的概率,PPO 的 ratio 就废了。
# C++ 侧 `vg_rules.h` 的 `kNetworkScan` 必须同步。
NETWORK_SCAN = False


def free_turn_available(ob) -> bool:
    """此刻还能不能"免费转身"。

    引擎只在"复活/开局后的那一次 act 内"给免费 TURN:`end_phase` 清掉它,`apply` 里
    只有"免费资格为真 **且** 人还在出生点"时才走免费分支(不消耗额度)。

    `ob.free_turn` 覆盖了"还没用过",但**开局第一个阶段是例外**:obs.py 的 `act_start`
    把它设成 `i_died`,而那时还没人死过,于是 s[19] 是 False,尽管引擎开局时给的是
    True。这个观测缺口不能去修 —— 改它等于重新定义 428 维观测、废掉所有已训权重 ——
    只能在规则里绕过去:`turn==0` + 一个行动都没消耗 + 朝向还是初始的 E,三者同时
    成立就等价于"开局阶段的第一次决策"。(免费转身**不消耗**额度,所以转身之后
    `actions_used` 仍是 0,必须靠朝向把自己和"还没动过"区分开。)
    """
    if ob.my_pos != (0, 0):
        return False
    if ob.free_turn:
        return True
    return ob.turn == 0 and ob.actions_used == 0 and ob.my_facing == R.START_FACING["R"]


def kill_plan(ob, budget: Optional[int] = None) -> Optional[int]:
    """本回合内能击杀就返回**最短方案的第一个动作**,否则 None。

    只回答"能不能",不返回整条路径:调用方照做第一步,下一个决策点重新调用即可。
    每步都从当前观测重推,所以中途 SCAN 揭示、走位被挡之类的意外都能立刻修正,也
    不需要跨行动维护计划状态。

    全程在**视角坐标**里算。180 度旋转对 `fire_hits` 是把障碍集映到自身、把
    前方映到前方的等距变换,所以在视角坐标里判定命中等价于引擎在绝对坐标里判定。
    """
    if ob.fire_cd != 0:
        return None
    if not ob.opp_visible:
        # 只凭乐观信念开枪不算"能击杀"(opp_visible 为真 ⟹ intel_pos 是实时位置)。
        return None
    tx, ty = ob.intel_pos
    if tx < 0 or ty < 0:
        return None

    if budget is None:
        budget = ACTIONS_PER_TURN - ob.actions_used
    if budget <= 0:
        return None

    start = ob.my_pos
    target = (tx, ty)
    # 开火前最多走 budget-1 步,而一发子弹最远打到曼哈顿距离 FIRE_RANGE+1
    # (前向 3 格、横向最多偏 1 格)。够不着就直接退出 —— 绝大多数决策点在这里
    # 出局,省掉整棵 BFS。
    if R.manhattan(start, target) > (budget - 1) + R.FIRE_RANGE + 1:
        return None

    obstacles = ob._obs_frozen  # ob.obstacles 的 frozenset 副本,在 obs.py 的 act_start 里同步

    # 根节点。顺序就是平局裁决:C++ 侧必须用同一套顺序,否则等价性测试会红。
    #   先"不转身";再按 N/E/S/W 展开免费转身(它们代价 0,first 就是那次转身)。
    roots: List[Tuple[Pos, str, Optional[int]]] = [(start, ob.my_facing, None)]
    if free_turn_available(ob):
        for d in R.DIRS:
            if d != ob.my_facing:
                roots.append((start, d, TURN_ACTION[d]))

    frontier = roots
    seen = {(p, f) for p, f, _ in roots}

    # spent = 开火**之前**花掉的行动数;命中需要再花 1 步开火,所以 spent <= budget-1。
    # 按 spent 递增扫描,第一次命中就是最短方案;同一层内按 frontier 顺序取,即平局裁决。
    for _ in range(min(budget, MAX_SEARCH_DEPTH)):
        for pos, facing, first in frontier:
            if R.fire_hits(pos, facing, target, obstacles):
                return FIRE if first is None else first

        nxt: List[Tuple[Pos, str, Optional[int]]] = []
        for pos, facing, first in frontier:
            # 子节点顺序:TURN 先(N/E/S/W),再 MOVE。与掩码的构造顺序一致。
            for d in R.DIRS:
                if d == facing:
                    continue  # 掩码不允许转向当前朝向,转了也是白花一步
                key = (pos, d)
                if key in seen:
                    continue
                seen.add(key)
                nxt.append((pos, d, TURN_ACTION[d] if first is None else first))

            dx, dy = R.DELTA[facing]
            step = (pos[0] + dx, pos[1] + dy)
            key = (step, facing)
            # 与 action_mask 同口径:出界 / 障碍 / 敌人所在格都不能进。引擎的 MOVE
            # 分支用 `(nx, ny) != op.pos` 拒绝对手所在格;这里比掩码更严 —— 只要有
            # opp_visible 就不进,而掩码只在 opp_**directly**_visible 时才挡 ——
            # 严格的那一侧才不会让规则给出被引擎拒绝的动作。
            if (key in seen or step == target or step in obstacles
                    or not R.in_bounds(step[0], step[1])):
                continue
            seen.add(key)
            nxt.append((step, facing, MOVE if first is None else first))

        if not nxt:
            break
        frontier = nxt
    return None


def narrow(mask: np.ndarray, k: int) -> np.ndarray:
    """把掩码收窄成"只有 k 合法"(返回**新数组**,不原地改)。

    收窄之后再采样,存进轨迹的 `logp` 恰好是 0.0、`ratio` 恰好是 1.0、策略梯度恰好
    是 0(python 侧由 tests/test_rules_vg.py 钉住)。熵那一项要 ppo.py 用**逐行**熵才
    干净 —— 批量均值版本会顺着广播往强制行里漏梯度。
    """
    out = np.zeros_like(mask)
    out[k] = 1.0
    return out


def policy_mask(mask: np.ndarray, k: Optional[int] = None) -> np.ndarray:
    """把环境给的掩码整成"交给网络采样"的掩码(总是**新数组**,不原地改)。

    规则命中 → 收窄成只有它合法;
    没命中   → 照抄,但按 `NETWORK_SCAN` 关掉 SCAN。
    """
    if k is not None:
        return narrow(mask, k)
    out = mask.copy()
    if not NETWORK_SCAN:
        out[SCAN] = 0.0
    return out


class VGRules:
    """规则 2 的阶段计数器 + 两条规则的决策点钩子。

    每个决策点调用**恰好一次** `forced()`;`_note()` 借它推进阶段计数,幂等。
    """

    # 与模块常量同名的别名。**改这个属性不会有任何效果** —— `__init__` 读的是模块
    # 那个常量(以及参数 `miss_threshold`),类属性只是照着模块常量抄了一份放在这里。
    MISS_THRESHOLD = MISS_THRESHOLD

    def __init__(
        self,
        enable_kill: bool = True,
        enable_scan: bool = True,
        miss_threshold: Optional[int] = None,
        scan_outside_zone_only: bool = False,
    ) -> None:
        # 两条规则的独立开关。**只用于诊断**(量单条规则各自的贡献:一起开会把
        # "哪条在帮倒忙"混成一个数),交付形态两条都开。
        self.enable_kill = enable_kill
        self.enable_scan = enable_scan
        # 失明几个阶段才扫。默认取模块常量;单独传是为了能扫参数量出代价曲线。
        self.miss_threshold = MISS_THRESHOLD if miss_threshold is None else miss_threshold
        # 只在**不在得分区**时扫:站在得分区里一个阶段 END 就是 +1 分,而强制 SCAN
        # 会把这份进账换成一个动作、还把 `opp_visible` 翻成 True,网络的"看得见敌人"
        # 分支随即离开中心。这个开关就是"别拿正在进账的动作去换情报"。
        #
        # **没有任何调用方传 True**(CLI 与 selfplay 都不暴露它),要用得自己构造
        # `VGRules(scan_outside_zone_only=True)` —— 别去找对应的命令行开关。
        self.scan_outside_zone_only = scan_outside_zone_only
        self.reset()
        # 统计:强制了多少步、其中击杀/扫描各多少。训练日志用它区分"规则没生效"
        # 和"规则生效了但熵本来就降"。
        self.forced_count = 0
        self.kill_count = 0
        self.scan_count = 0

    def reset(self) -> None:
        self._turn: Optional[int] = None
        self._seen = False
        self._miss = 0

    def _note(self, ob) -> None:
        turn = ob.turn
        if self._turn is None or turn < self._turn:
            # 新的一局从 turn=0 重来。少了这一句,上一局残留的 turn 会把 0 当成
            # "又一个阶段结束",把连败跨局累加下去。
            self._turn, self._seen, self._miss = turn, False, 0
        elif turn != self._turn:
            # 上一个阶段走完了:整个过程都没见到敌人就算一次失明。
            self._miss = 0 if self._seen else self._miss + 1
            self._turn, self._seen = turn, False
        if ob.opp_visible:
            # 直接视野或 SCAN 揭示都算"看到"(口径已定),看到就清零。
            self._seen = True
            self._miss = 0

    def blind_phases(self) -> int:
        return self._miss

    def _scan_worth_it(self, ob) -> bool:
        """规则 2 的触发条件(不含掩码可不可用 —— 那个由调用方查 `mask[SCAN]`)。"""
        if self._miss < self.miss_threshold or ob.scan_cd != 0:
            return False
        if self.scan_outside_zone_only and ob._in_zone(ob.my_pos, ob.zones):
            # 站在得分区里,每个阶段 END 就是 +1 分 —— 这一格的动作比情报值钱。
            return False
        return True

    def forced(self, ob, mask: np.ndarray) -> Optional[int]:
        """返回规则强制执行的合法动作 id;两条规则都没命中则 None。"""
        self._note(ob)

        k = kill_plan(ob) if self.enable_kill else None
        # mask[k] > 0 是安全网:被引擎拒绝的动作不消耗额度,规则若反复给出同一个
        # 被拒的动作就会卡死(env 的 max_attempts 会兜住,但那是报错不是策略)。
        if k is not None and mask[k] > 0:
            self.forced_count += 1
            self.kill_count += 1
            return k

        if self.enable_scan and self._scan_worth_it(ob) and mask[SCAN] > 0:
            self.forced_count += 1
            self.scan_count += 1
            return SCAN
        return None

    def act_mask(self, ob, mask: np.ndarray):
        """一个决策点的完整规则处理:返回 `(net_mask, forced_k)`。

        训练与部署两条路径都只调这一个入口,免得各写一遍而分叉(分叉的表现是
        "训练时很强、部署后变弱",而且不会报错)。`mask` 是环境给的**原始**掩码 ——
        规则要看到它才能判断 SCAN 可不可用;收窄/关 SCAN 只发生在返回值里。
        """
        k = self.forced(ob, mask)
        return policy_mask(mask, k), k
