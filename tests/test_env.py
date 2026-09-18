"""观测 / 环境层测试。"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monet.engine import rules as R  # noqa: E402
from monet.engine.game import END, MOVE, SCAN  # noqa: E402
from monet.env import obs as O  # noqa: E402
from monet.env.opponents import EndOpponent, HeuristicOpponent, RandomOpponent  # noqa: E402
from monet.env.sentry_env import SentryEnv  # noqa: E402


def test_obs_dim_matches_deploy_header():
    # 必须与 rl_v5-source/rl_weights.h 的 kObsDim 一致,否则导出的权重喂不进去
    assert O.OBS_DIM == 428
    assert O.ACTION_DIM == 8
    assert O.PLANES * O.CELLS + O.SCALARS == 428


def test_obs_layout_tiny():
    env = SentryEnv(RandomOpponent(seed=0), agent_color="R", seed=0)
    obs, mask, info = env.reset()
    assert obs.shape == (428,) and obs.dtype == np.float32
    assert mask.shape == (8,) and mask[END] == 1.0

    plane = lambda i: obs[i * 49 : (i + 1) * 49].reshape(7, 7)
    assert plane(2)[0, 0] == 1.0  # 我方在 (0,0)
    assert plane(6)[0, 0] == 1.0 and plane(7)[6, 6] == 1.0  # 双方出生点常量
    # 障碍平面
    assert plane(0)[1, 1] == 1.0 and plane(0)[5, 5] == 1.0
    # 得分区平面(中心十字 5 格)
    assert plane(1).sum() == 5.0
    assert plane(1)[3, 3] == 1.0 and plane(1)[2, 3] == 1.0
    # 标量段:t15 = 是否蓝方;t13 = turn
    s = obs[392:]
    assert s[15] == 0.0 and s[13] == 0.0


def test_mask_allows_end_always():
    env = SentryEnv(RandomOpponent(seed=1), agent_color="R", seed=1)
    _, mask, _ = env.reset()
    for _ in range(30):
        _, mask, _, term, trunc, _ = env.step(END)
        assert mask[END] == 1.0
        if term or trunc:
            break


def test_masked_actions_are_engine_legal_or_recoverable():
    """掩码里允许的动作,要么被引擎接受,要么因"隐形敌人占格"被拒(然后是重试)。"""
    env = SentryEnv(HeuristicOpponent(aggression=1.0, seed=2), agent_color="R", seed=2)
    obs, mask, _ = env.reset()
    rng = np.random.default_rng(0)
    for _ in range(300):
        legal = np.flatnonzero(mask > 0)
        a = int(rng.choice(legal))
        obs, mask, _, term, trunc, _ = env.step(a)
        if term or trunc:
            obs, mask, _ = env.reset()
    assert True


def test_episode_reward_matches_score_difference():
    """事件奖励必须与"我方的计分事件"对得上:

        return = 我方得分 - 2 x 我方阵亡 + 终局奖励

    对手的占点分不进我方奖励,所以不能拿比分差去比。
    """
    env = SentryEnv(HeuristicOpponent(aggression=1.0, seed=3), agent_color="R", seed=3)
    env.reset()
    total = 0.0
    while True:
        _, _, r, term, trunc, info = env.step(END)
        total += r
        if term or trunc:
            break
    w = info["winner"]
    terminal = 0.0 if w is None else (1.0 if w == info["agent_color"] else -1.0)
    expected = info["my_score"] - 2.0 * info["deaths"] + terminal
    assert abs(total - expected) < 1e-6, (total, expected, info)


def test_scan_reveal_reward_is_opt_in_and_counts_only_reveals():
    """scan_reveal 默认 0(回报仍等于比分差);打开后只奖"扫出之前看不见的敌人"。

    对手罚站不动、我方脚本化地"能扫就扫",所以整局是确定性的:两次跑的差异
    必须**正好**等于 bonus x 扫到人的次数。
    """
    from monet.env.sentry_env import RewardConfig

    def play(bonus):
        env = SentryEnv(
            EndOpponent(),
            agent_color="R",
            reward=RewardConfig(scan_reveal=bonus),
            seed=7,
        )
        _, mask, _ = env.reset()
        total = 0.0
        reveals = 0
        while True:
            a = SCAN if mask[SCAN] > 0 else END
            was_visible = env.ag_ob.opp_visible
            _, mask, r, term, trunc, info = env.step(a)
            total += r
            if a == SCAN and not was_visible and env.ag_ob.opp_visible:
                reveals += 1
            if term or trunc:
                break
        return total, reveals, info

    base, reveals, info = play(0.0)
    assert reveals > 0, "这局一次都没扫到人,测试没意义"
    # 默认关闭:回报仍然只由计分事件构成
    w = info["winner"]
    terminal = 0.0 if w is None else (1.0 if w == info["agent_color"] else -1.0)
    expected = info["my_score"] - 2.0 * info["deaths"] + terminal
    assert abs(base - expected) < 1e-6, (base, expected, info)

    boosted, reveals2, _ = play(0.5)
    assert reveals2 == reveals
    assert abs((boosted - base) - 0.5 * reveals) < 1e-6, (boosted, base, reveals)


def test_blue_agent_gets_first_move_from_opponent():
    env = SentryEnv(RandomOpponent(seed=4), agent_color="B", seed=4)
    obs, mask, info = env.reset()
    assert info["agent_color"] == "B"
    # 蓝方视角:我方仍在 (0,0)
    assert obs[2 * 49 : 3 * 49].reshape(7, 7)[0, 0] == 1.0


def test_obs_builder_belief_tracks_scan():
    env = SentryEnv(EndOpponent(), agent_color="R", seed=5)
    env.reset()
    ob = env.ag_ob
    assert ob.scan_cd == 0
    _, mask, _, _, _, _ = env.step(SCAN)
    assert ob.intel_pos[0] >= 0  # 扫描后拿到实时位置
    assert ob.scan_cd == 3


def test_env_terminates_within_expected_horizon():
    for seed in range(5):
        env = SentryEnv(HeuristicOpponent(seed=seed), agent_color=None, seed=seed)
        env.reset()
        steps = 0
        while True:
            steps += 1
            _, _, _, term, trunc, _ = env.step(END)
            if term or trunc:
                break
            assert steps < 4000
        assert steps < 4000


def test_obs_dtype_and_belief_normalization():
    env = SentryEnv(RandomOpponent(seed=6), agent_color="R", seed=6)
    obs, _, _ = env.reset()
    assert np.isfinite(obs).all()
    assert obs[392:].max() <= 1.0 + 1e-6
    assert obs[392:].min() >= -1e-6


# --------------------------------------------------------------- obs 等价重写
#
# `obs.py` 的 `_dilate_belief` / `_subtract_visible_cells` 是"位掩码 + 预计算邻居表"
# 的快版(见这两个方法的 docstring)。它们处在**训练与部署共用的观测定义**上,
# 一旦算错不会崩,只会让训练出来的策略和导出的 C++ 推理壳看到不同的世界 —— 典型的
# 静默失效。所以这里把逐格慢版抄进来当差分基准,真实对局上两版都跑、逐格比对;
# 这是 README §六 那条"不能去修 obs.py"约束的可执行版本。
#
# 基准版**故意保持又慢又笨**(逐格 49x4 次 Python 调用):它是规格,不是实现。


def _ref_is_obstacle(self, p) -> bool:
    return any(o[0] == p[0] and o[1] == p[1] for o in self.obstacles)


def _ref_dilate_belief(self) -> None:
    for _ in range(3):
        nxt = list(self.belief)
        for y in range(O.BOARD_SIZE):
            for x in range(O.BOARD_SIZE):
                if self.belief[O._cell(x, y)] <= 0.0:
                    continue
                for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                    nx, ny = x + dx, y + dy
                    if not R.in_bounds(nx, ny):
                        continue
                    if _ref_is_obstacle(self, (nx, ny)):
                        continue
                    if (nx, ny) == self.my_pos:
                        continue
                    nxt[O._cell(nx, ny)] = 1.0
        self.belief = nxt


def _ref_subtract_visible_cells(self) -> None:
    for y in range(O.BOARD_SIZE):
        for x in range(O.BOARD_SIZE):
            if self.belief[O._cell(x, y)] > 0.0 and R.can_see(
                self.my_pos, self.my_facing, (x, y), self.obstacles
            ):
                self.belief[O._cell(x, y)] = 0.0


class _BeliefDiffer:
    """临时把两个信念方法换成"快版跑一遍、还原、慢版再跑一遍、逐格比对"。"""

    def __init__(self):
        self.n_dilate = 0
        self.n_subtract = 0
        self._fast = (
            O.ObsBuilder._dilate_belief,
            O.ObsBuilder._subtract_visible_cells,
        )

    def _wrap(self, name, ref, fast, bump):
        xy = [(i % O.BOARD_SIZE, i // O.BOARD_SIZE) for i in range(O.CELLS)]

        def wrapper(ob):
            keep = list(ob.belief)
            fast(ob)
            got = list(ob.belief)
            ob.belief = list(keep)
            ref(ob)
            want = list(ob.belief)
            if got != want:
                diff = [xy[i] for i in range(O.CELLS) if got[i] != want[i]]
                raise AssertionError(
                    f"{name} 与重写前不等价:my_pos={ob.my_pos} facing={ob.my_facing} "
                    f"turn={ob.turn} obstacles={ob.obstacles} "
                    f"start={ob.start_pos} 差异格={diff}"
                )
            ob.belief = got
            bump()

        wrapper.__name__ = name
        return wrapper

    def __enter__(self):
        O.ObsBuilder._dilate_belief = self._wrap(
            "_dilate_belief", _ref_dilate_belief, self._fast[0],
            lambda: setattr(self, "n_dilate", self.n_dilate + 1),
        )
        O.ObsBuilder._subtract_visible_cells = self._wrap(
            "_subtract_visible_cells", _ref_subtract_visible_cells, self._fast[1],
            lambda: setattr(self, "n_subtract", self.n_subtract + 1),
        )
        return self

    def __exit__(self, *exc):
        O.ObsBuilder._dilate_belief, O.ObsBuilder._subtract_visible_cells = self._fast
        return False


def test_belief_rewrite_is_bitwise_equivalent_to_the_slow_version():
    """信念扩张 / 视野扣除的快版必须与逐格原版**逐位相同**。

    跑真实对局而不是构造状态:扩张与扣除的边界情形(刚复活、被击杀回出生点、
    贴障碍、扫描后视野突变)都是引擎按规则推出来的,手搓状态覆盖不到。
    """
    with _BeliefDiffer() as d:
        for seed in range(4):
            opps = (
                (RandomOpponent(seed=seed), HeuristicOpponent(aggression=1.0, seed=seed))
                if seed < 2
                else (RandomOpponent(seed=seed), HeuristicOpponent(aggression=0.0, seed=seed))
            )
            for opp in opps:
                env = SentryEnv(opp, agent_color="R", seed=seed)
                _, mask, _ = env.reset()
                rng = np.random.default_rng(seed)
                for _ in range(300):
                    a = int(rng.choice(np.flatnonzero(mask > 0)))
                    _, mask, _, term, trunc, _ = env.step(a)
                    if term or trunc:
                        break

    # 两个方法都得真被调用过,否则这条测试是空的
    assert d.n_dilate > 100, f"只比对了 {d.n_dilate} 次扩张,覆盖不足"
    assert d.n_subtract > 100, f"只比对了 {d.n_subtract} 次扣除,覆盖不足"


def test_belief_stays_binary():
    """快版把信念当 49 位掩码用,这个表示要成立,信念只能取 {0.0, 1.0}。"""
    env = SentryEnv(HeuristicOpponent(aggression=0.5, seed=11), agent_color="R", seed=11)
    _, mask, _ = env.reset()
    rng = np.random.default_rng(0)
    for _ in range(300):
        b = env.ag_ob.belief
        assert len(b) == O.CELLS
        assert all(v in (0.0, 1.0) for v in b), sorted(set(b))
        a = int(rng.choice(np.flatnonzero(mask > 0)))
        _, mask, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break


def test_can_see_equals_visible_cells_membership():
    """`_subtract_visible_cells` 的等价基础:视野集从不含障碍格,所以

        can_see(pos, facing, target)  ==  (target == pos) or (target in visible_cells(...))

    对**任意** target 成立。这条不成立的话,"算一次可见集再查表"就与原版不同。
    """
    obstacles = [(1, 1), (5, 5)]
    pos = (3, 3)
    for facing in ("N", "E", "S", "W"):
        vis = set(R.visible_cells(pos, facing, obstacles))
        assert not (vis & set(obstacles)), "可见集里出现了障碍格"
        for y in range(O.BOARD_SIZE):
            for x in range(O.BOARD_SIZE):
                want = (x, y) == pos or (x, y) in vis
                got = R.can_see(pos, facing, (x, y), obstacles)
                assert got == want, (pos, facing, (x, y), got, want)


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
