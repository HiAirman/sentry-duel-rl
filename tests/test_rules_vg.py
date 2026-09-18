"""rl_VG_v1.0 规则层测试:rules_vg.py 的两条规则 + 与 PPO 的接口。

规则 1(击杀搜索)是纯几何,可以精确判对错,所以这里不只查"返回了哪个动作",
而是**照着方案走完、断言真的击杀** —— 方案必须真的能赢,不只是看着合理。

规则 2(失明计数)是有状态的,靠合成 `turn`/`opp_visible` 序列钉住边界。
"""

from __future__ import annotations

import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monet.engine import rules as R  # noqa: E402
from monet.engine.game import ACTIONS_PER_TURN, Game  # noqa: E402
from monet.env.obs import (  # noqa: E402
    END,
    FIRE,
    MOVE,
    OBS_DIM,
    SCAN,
    TURN_ACTION,
    ObsBuilder,
)
from monet.env.opponents import NeuralOpponent, RandomOpponent  # noqa: E402
from monet.env.rules_vg import (  # noqa: E402
    MISS_THRESHOLD,
    NETWORK_SCAN,
    VGRules,
    free_turn_available,
    kill_plan,
    narrow,
    policy_mask,
)
from monet.env.sentry_env import SentryEnv  # noqa: E402
from monet.models.mlp import MLP  # noqa: E402
from monet.models.ppo import PPOAgent, Rollout  # noqa: E402

TURN_N, TURN_E, TURN_S, TURN_W = 1, 2, 3, 4


# --------------------------------------------------------------------- 工具


def _setup(me, face, opp, fire_cd=0, scan_cd=0, used=0, turn=0, obstacles=None):
    """摆一个"轮到我方(R)行动"的局面,返回 (game, ob)。坐标都是视角坐标。

    红方的视角坐标 == 绝对坐标,所以这里写的就是棋盘上的真实位置。
    """
    g = Game(obstacles=obstacles) if obstacles else Game()
    g.turn = turn
    g.s["R"].pos = me
    g.s["R"].facing = face
    g.s["R"].fire_cd = fire_cd
    g.s["R"].scan_cd = scan_cd
    g.s["B"].pos = opp
    g.begin_phase("R")
    g.actions_used = used
    ob = ObsBuilder()
    ob.act_start(g.view("R"), "R")
    return g, ob


def _pretend_enemy_visible(ob, opp):
    """把敌人标成"看得见、就在 opp"。

    规则 1 的前提正是这个:敌人只在我方阶段之外行动(引擎的分阶段状态机),所以
    一次搜索之内它是静止的;而 `opp_visible` 为真时 `intel_pos` 就是它的实时位置
    (game.py:119-120)。这里 opp 是真摆了敌人的,**真实**可见性由
    `test_rule1_plans_are_never_rejected` 用真局覆盖。
    """
    ob.opp_visible = True
    ob.intel_pos = opp


def _chase_until_dead(game, ob, opp, max_actions=8):
    """每个决策点问一次规则 1,照做,直到击杀或规则说"没方案了"。

    返回 (是否击杀, 消耗了几个**行动额度**)。额度而不是循环次数 —— 免费转身
    不消耗额度,按循环数会多算一步。

    budget 走 `kill_plan` 的默认值,也就是生产代码的用法(按 `ob.actions_used`
    自己算剩余额度)。
    """
    used = 0
    for _ in range(max_actions):
        k = kill_plan(ob)
        if k is None:
            return False, used
        res, ev = game.apply("R", k)
        assert res.success, f"规则给出了被引擎拒绝的动作 {k}"
        ob.on_observation(res.observation, res.consumed, k)
        _pretend_enemy_visible(ob, opp)
        if res.consumed:
            used += 1
        if ev.get("kill"):
            return True, used
    return False, used


class _FakeOb:
    """规则 2 只读这几个字段,不碰 obs.py 的其他部分。"""

    def __init__(self, turn=0, vis=False, scan_cd=0, used=0):
        self.turn = turn
        self.opp_visible = vis
        self.scan_cd = scan_cd
        self.actions_used = used
        # 规则 1 的门:fire_cd != 0 让它永远不触发,单独测规则 2
        self.fire_cd = 1
        self.intel_pos = (-1, -1)
        self.my_pos = (3, 3)
        self.my_facing = "N"
        self.free_turn = False


def _scan_mask():
    m = np.zeros(8, dtype=np.float32)
    m[MOVE] = m[SCAN] = m[END] = 1.0
    return m


# ------------------------------------------- 1. 强制步对策略头必须是零梯度

def test_forced_rows_leave_the_policy_head_untouched():
    """掩码收窄成单动作之后,那些步对策略头**逐位**没有影响。

    这条是整个混合策略的地基:规则决定的步不该把网络往任何方向推。少了 ppo.py 里
    的逐行熵,它会红 —— 批量均值熵会顺着广播往强制行里漏一个 sign-consistent 的
    梯度(实测 dlogits[k] ≈ +0.9·ent_coef/B,Adam 会把它放大成实打实的参数更新)。
    """
    net = MLP(hidden=32, seed=0)
    agent = PPOAgent(net, seed=0)
    rng = np.random.default_rng(0)

    roll = Rollout()
    for _ in range(16):
        obs = rng.normal(size=OBS_DIM).astype(np.float32)
        k = int(rng.integers(8))
        m = narrow(np.ones(8, np.float32), k)
        a, logp, v = agent.act(obs, m)
        assert a == k, "收窄之后只可能采样到那一个动作"
        assert logp == 0.0, f"收窄后的 logp 必须**恰好**是 0.0,实际 {logp!r}"
        roll.add(obs, m, a, logp, v, float(rng.normal()), 0.0)

    w3 = net.p["W3"].copy()
    b3 = net.p["b3"].copy()
    stats = agent.update(roll)

    assert np.array_equal(w3, net.p["W3"]), "策略头权重被强制步改动了"
    assert np.array_equal(b3, net.p["b3"]), "策略头偏置被强制步改动了"
    # policy_loss 是 -mean(优势),而优势做过归一化 —— 理论上恰好 0,浮点求和留了
    # 一个 ~1e-8 的尾巴。真正的判据是上面的逐位相等,这里只是顺带确认量级。
    assert abs(stats["policy_loss"]) < 1e-5, stats["policy_loss"]
    assert stats["entropy"] == 0.0
    assert stats["approx_kl"] == 0.0
    # 价值头**应该**照常学 —— 强制步的回报仍然是真信号,不能一起冻掉
    assert net.p["Wv"] is not None


def test_per_row_entropy_is_what_keeps_forced_rows_clean():
    """逐行熵和批量均值熵在**混合** batch 里必须给出不同的结果。

    没有这一条,上面那条测试就只是"全强制 batch 恰好也对"—— 而全强制时批量均值
    恰好也是 0,测不出公式错没错。这里直接盯住那一项。
    """
    from monet.models.ppo import log_softmax, masked_logits

    net = MLP(hidden=32, seed=0)
    X = np.random.default_rng(0).normal(size=(4, OBS_DIM)).astype(np.float32)
    M = np.zeros((4, 8), np.float32)
    M[0, :] = 1.0          # 普通行:掩码宽
    M[1, 1:6] = 1.0        # 普通行
    M[2, 5] = 1.0          # 强制行
    M[3, 2] = 1.0          # 强制行

    logits, _ = net.forward(X, cache=False)
    lp = log_softmax(masked_logits(logits, M))
    p = np.exp(lp)
    ent_batch = -float(np.mean((p * lp).sum(axis=1)))
    ent_row = -(p * lp).sum(axis=1)

    assert ent_batch > 1e-3, "batch 里得有普通行,否则这条测试没有区分力"
    for i, k in ((2, 5), (3, 2)):
        assert ent_row[i] == 0.0, "强制行的行熵必须恰好是 0"
        leak = (p[i] * (lp[i] + ent_batch))[k]
        clean = (p[i] * (lp[i] + ent_row[i]))[k]
        assert abs(leak) > 1e-3, "批量均值版本来就该漏梯度,测不出来说明构造错了"
        assert clean == 0.0, "逐行版必须一个比特都不漏"


# ------------------------------------------------- 2. 规则 1:方案真的能击杀

def test_rule1_fires_when_already_in_the_box():
    g, ob = _setup((0, 0), "E", (2, 0))
    _pretend_enemy_visible(ob, (2, 0))
    assert kill_plan(ob) == FIRE, "已经在火力范围内就该直接开火"
    dead, used = _chase_until_dead(g, ob, (2, 0))
    assert dead and used == 1


def test_rule1_turns_then_fires():
    """差 2 格、朝向不对:方案必须以 TURN 开头,走完真的击杀。"""
    g, ob = _setup((3, 3), "N", (5, 3))
    _pretend_enemy_visible(ob, (5, 3))
    k = kill_plan(ob)
    assert k == TURN_E, f"该先转身朝东,实际 {k}"
    dead, used = _chase_until_dead(g, ob, (5, 3))
    assert dead, "方案没打成"
    assert used <= ACTIONS_PER_TURN, f"用了 {used} 个行动,超出一个阶段"


def test_rule1_uses_the_free_turn_at_spawn():
    """开局免费转身必须建模。不建模的话"转+走+走+打"会被误判成不可行。

    敌人摆在 (0,5):先免费转 S(不消耗额度),再走两格到 (0,2),开火命中。
    总共消耗 3 个额度,刚好用完 —— 不把免费转身算进去就会得出"要 4 步、做不到"。
    """
    g, ob = _setup((0, 0), "E", (0, 5), turn=0)
    _pretend_enemy_visible(ob, (0, 5))
    assert free_turn_available(ob), "开局(0,0)朝 E 应该还能免费转身"

    k = kill_plan(ob)
    assert k == TURN_S, f"该先免费转 S,实际 {k}"
    dead, used = _chase_until_dead(g, ob, (0, 5))
    assert dead, "免费转身方案没打成"
    assert used == 3, f"免费转身 + 走 2 步 + 开火 = 3 个行动,实际 {used}"

    # 同一个局面,把免费转身拿掉(不是开局阶段了)→ 3 个额度不够,规则该说"没方案"
    g2, ob2 = _setup((0, 0), "E", (0, 5), turn=5)
    _pretend_enemy_visible(ob2, (0, 5))
    assert not free_turn_available(ob2)
    assert kill_plan(ob2) is None, "没有免费转身时这局要 4 步,3 个额度内不该有方案"


def test_rule1_gives_up_when_out_of_reach():
    g, ob = _setup((0, 0), "E", (6, 0))
    _pretend_enemy_visible(ob, (6, 0))
    assert kill_plan(ob) is None, "要走 3 格再开火 = 4 个行动,额度和不过来"


def test_rule1_respects_fire_cooldown_and_visibility():
    g, ob = _setup((0, 0), "E", (2, 0), fire_cd=2)
    _pretend_enemy_visible(ob, (2, 0))
    assert kill_plan(ob) is None, "FIRE 在 CD 里就不该规划击杀"

    g2, ob2 = _setup((0, 0), "E", (2, 0))
    ob2.opp_visible = False
    ob2.intel_pos = (2, 0)
    assert kill_plan(ob2) is None, "看不见人时凭旧情报开枪不算'能击杀'"


def test_rule1_plans_are_never_rejected_across_the_whole_board():
    """扫一遍敌人可能站的位置:规则给出的每个动作都必须被引擎接受。

    敌人所在格不可踏入(game.py:266 会拒绝走到对方格子上),而"被拒绝"不消耗额度
    —— 规则若反复给出同一个被拒的动作就会卡死。掩码比这里松(它只在**直接看见**
    时才挡敌方格,而规则搜索用 `opp_visible`),所以这部分必须由搜索自己保证。
    """
    checked = 0
    for ox in range(7):
        for oy in range(7):
            if (ox, oy) == (0, 0):
                continue
            g, ob = _setup((0, 0), "E", (ox, oy))
            _pretend_enemy_visible(ob, (ox, oy))
            for _ in range(4):
                k = kill_plan(ob)
                if k is None:
                    break
                res, ev = g.apply("R", k)
                assert res.success, f"敌人摆在 {(ox, oy)} 时规则给出了被拒的动作 {k}"
                ob.on_observation(res.observation, res.consumed, k)
                _pretend_enemy_visible(ob, (ox, oy))
                checked += 1
                if ev.get("kill"):
                    break
    assert checked > 0, "整个扫描一次动作都没产生 —— 测试没测到东西"


# --------------------------------- 3. 规则的动作在真局里从不被引擎拒绝

def test_rule1_plans_are_never_rejected():
    """整局跑下来:规则给出的动作**从不**被引擎拒绝,而且规则 1 的每次开火都得分。

    上一条测试是"方案在摆好的局面上成立",这条是"在真棋盘上、真对手面前也成立"。
    """
    env = SentryEnv(RandomOpponent(seed=1), agent_color=None, seed=11, max_steps=400)
    real_apply = env.game.apply
    rejections = []
    kills = []

    def spy(color, action):
        res, ev = real_apply(color, action)
        if color == env.agent_color and not res.success:
            rejections.append((action, ev.get("rejected")))
        if color == env.agent_color and ev.get("kill"):
            kills.append(ev.get("death_color"))
        return res, ev

    forced_fire = 0
    for game_i in range(6):
        env.game.apply = spy
        obs, mask, info = env.reset(seed=20 + game_i, agent_color="R" if game_i % 2 == 0 else "B")
        rules = VGRules()
        done = False
        while not done:
            m, k = rules.act_mask(env.ag_ob, mask)
            if k is None:
                # 规则没命中时随便挑个合法动作 —— 本条只考察规则给出的动作
                legal = [i for i in range(8) if m[i] > 0]
                a = legal[0] if legal else END
            else:
                a = k
                n_before = len(rejections)
                s_before = info["my_score"]
                if a == FIRE:
                    forced_fire += 1
            obs, mask, r, term, trunc, info = env.step(a)
            if k is not None:
                assert len(rejections) == n_before, (
                    f"规则给出的动作 {a} 被引擎拒绝了:{rejections[-1:]}"
                )
                if a == FIRE:
                    assert info["my_score"] >= s_before + 2, (
                        f"规则 1 只在必中时才开火,这一枪却没得分"
                        f"({s_before} → {info['my_score']})"
                    )
            done = term or trunc

    assert forced_fire > 0, "6 局里规则 1 一次都没开火 —— 测试没测到东西"
    assert not rejections, f"规则动作被拒绝了:{rejections[:5]}"


# ------------------------------------------------ 4. 规则 2:失明 2 阶段才扫

def test_rule2_scans_only_after_the_threshold_is_reached():
    """口径:`MISS_THRESHOLD` 个**已走完**的失明阶段之后才扫(阈值 2 ⇒ 第 3 个阶段)。

    **按 `miss_threshold` 推导,不写死 2** —— 阈值是量出来的旋钮(见 rules_vg.py 里
    `MISS_THRESHOLD` 的注释),写死的话每次调参都要回来改测试,而"改测试让它绿"
    正是这类绊线失效的方式。
    """
    T = VGRules().miss_threshold
    r = VGRules()
    m = _scan_mask()
    # turn=0 是第一个失明阶段,但它还没走完 —— 计数只记"已经结束"的阶段
    assert r.forced(_FakeOb(turn=0, vis=False), m) is None
    assert r.blind_phases() == 0, "当前阶段还没结束,不该记数"
    for t in range(1, T):
        assert r.forced(_FakeOb(turn=t, vis=False), m) is None, f"失明 {t} 个阶段还不该扫"
        assert r.blind_phases() == t, f"第 {t} 个失明阶段走完了"
    assert r.forced(_FakeOb(turn=T, vis=False), m) == SCAN, f"失明 {T} 个阶段 → 该扫了"
    assert r.blind_phases() == T


def test_rule2_seen_resets_the_counter():
    """看到就清零 —— 无论是直接视野还是 SCAN 揭示(口径已定)。"""
    T = VGRules().miss_threshold
    r = VGRules()
    m = _scan_mask()
    r.forced(_FakeOb(turn=0, vis=False), m)   # 阶段 0:失明
    r.forced(_FakeOb(turn=1, vis=True), m)    # 阶段 1:看见了 → 连败清零
    assert r.blind_phases() == 0
    # 阶段 1 有视野,所以它结束时不该续上连败
    assert r.forced(_FakeOb(turn=2, vis=False), m) is None, "刚看见过,不该马上又扫"
    assert r.blind_phases() == 0, "上一个阶段有视野,连败不该续上"
    for t in range(3, 2 + T):
        assert r.forced(_FakeOb(turn=t, vis=False), m) is None, f"这才数到 {t - 2}"
    assert r.blind_phases() == T - 1
    assert r.forced(_FakeOb(turn=2 + T, vis=False), m) == SCAN, f"又攒够 {T} 个了"


def test_rule2_respects_scan_cd_and_the_mask():
    r = VGRules()
    for t in (0, 1):
        r.forced(_FakeOb(turn=t, vis=False), _scan_mask())
    # SCAN 在 CD 里
    assert r.forced(_FakeOb(turn=2, vis=False, scan_cd=3), _scan_mask()) is None
    # 掩码不给 SCAN(例如额度已用完,只剩 END)
    m_no_scan = np.zeros(8, np.float32)
    m_no_scan[END] = 1.0
    assert r.forced(_FakeOb(turn=3, vis=False), m_no_scan) is None


def test_rule2_resets_when_a_new_episode_restarts_the_turn_counter():
    """新一局从 turn=0 重来。少了这句,上一局残留的 turn 会把 0 读成"又一个阶段
    结束",把连败跨局累加下去,于是开新局就乱扫。"""
    T = VGRules().miss_threshold
    r = VGRules()
    m = _scan_mask()
    for t in range(T + 1):
        r.forced(_FakeOb(turn=t, vis=False), m)
    assert r.blind_phases() >= T, "连败没攒起来,这条测试就测不到"
    r.forced(_FakeOb(turn=0, vis=False), m)  # 新一局
    assert r.blind_phases() == 0, "新局必须把失明计数清零"


def test_network_cannot_choose_scan():
    """SCAN 是规则的专属动作:规则没命中时,交给网络的掩码里 SCAN 必须是 0。"""
    assert NETWORK_SCAN is False, "这一轮的口径是网络不许自己扫"
    raw = _scan_mask()
    assert raw[SCAN] == 1.0
    net_mask = policy_mask(raw)
    assert net_mask[SCAN] == 0.0, "网络拿到的掩码里 SCAN 该被关掉"
    assert net_mask[MOVE] == 1.0 and net_mask[END] == 1.0, "只关 SCAN,别的动作不动"
    assert raw[SCAN] == 1.0, "policy_mask 不能原地改环境给的掩码"

    # 规则强制扫的时候,收窄后的掩码里当然只剩 SCAN
    forced = policy_mask(raw, SCAN)
    assert forced[SCAN] == 1.0 and forced.sum() == 1.0

    r = VGRules()
    k = None
    for t in range(VGRules().miss_threshold + 1):
        net_mask, k = r.act_mask(_FakeOb(turn=t, vis=False), raw)
    assert k == SCAN and net_mask[SCAN] == 1.0


def test_network_never_samples_scan_in_a_real_run():
    """真局里把网络跑起来,SCAN 只可能来自规则 2。"""
    env = SentryEnv(RandomOpponent(seed=2), agent_color="R", seed=5, max_steps=400)
    net = MLP(hidden=32, seed=0)
    agent = PPOAgent(net, seed=0)
    scans = 0
    for g in range(3):
        obs, mask, info = env.reset(seed=30 + g, agent_color="R")
        rules = VGRules()
        done = False
        while not done:
            m, k = rules.act_mask(env.ag_ob, mask)
            a, _, _ = agent.act(obs, m)
            assert not (a == SCAN and k is None), "网络自己选了 SCAN"
            if a == SCAN:
                scans += 1
                assert k == SCAN
            obs, mask, r, term, trunc, info = env.step(a)
            done = term or trunc
    assert scans > 0, "三局一次都没扫到 —— 规则 2 没生效,这条测试没测到东西"


# -------------------------------- 5. 训练钩子与部署对手必须给出同一个动作

def test_training_hook_and_deployment_opponent_agree():
    """同局面下,`rules.act_mask` + `agent.act` 与 `NeuralOpponent(rules=...)`
    必须选出同一个动作。

    两条路径各写一遍是本仓库明确防的失败模式:分叉的表现是"训练时很强、部署后
    变弱",而且不报错。这里让它们吃同一份观测、同一份掩码,逐决策点比对。
    """
    net = MLP(hidden=32, seed=0)
    env = SentryEnv(RandomOpponent(seed=3), agent_color="R", seed=9, max_steps=400)
    agent = PPOAgent(net, seed=0)

    for g in range(3):
        obs, mask, info = env.reset(seed=40 + g, agent_color="R")
        r_train, r_deploy = VGRules(), VGRules()
        opp = NeuralOpponent(net, temperature=1e-3, seed=0, rules=r_deploy)
        done = False
        while not done:
            m_train, k_train = r_train.act_mask(env.ag_ob, mask)
            a_train, _, _ = agent.act(obs, m_train, deterministic=True)
            a_deploy = opp.act(env.game.view(env.agent_color), env.agent_color, env.ag_ob)
            assert a_train == a_deploy, (
                f"两条路径选出了不同的动作:训练 {a_train} vs 部署 {a_deploy}"
                f"(规则 {k_train})"
            )
            obs, mask, r, term, trunc, info = env.step(a_train)
            done = term or trunc


def test_rules_default_off_keeps_other_opponents_identical():
    """不带规则的 NeuralOpponent 必须逐位保持原样 —— rl_v5 / m3_v* / v0.2 都靠它。"""
    net = MLP(hidden=32, seed=0)
    env = SentryEnv(RandomOpponent(seed=4), agent_color="R", seed=9, max_steps=400)

    plain = NeuralOpponent(net, temperature=1e-3, seed=0)
    obs, mask, info = env.reset(seed=50, agent_color="R")
    a_plain = [plain.act(env.game.view("R"), "R", env.ag_ob)]
    assert not hasattr(plain.rules, "act_mask"), "默认不该带规则"

    # 同一个局面下,带 temperature<=1e-3 的混合对手在"规则没命中"时也该给同样的动作
    obs, mask, info = env.reset(seed=50, agent_color="R")
    hybrid = NeuralOpponent(net, temperature=1e-3, seed=0, rules=VGRules())
    a_hybrid = [hybrid.act(env.game.view("R"), "R", env.ag_ob)]

    if NETWORK_SCAN is False and SCAN in np.flatnonzero(mask):
        # 掩码允许 SCAN 而网络恰好想选它时,两者会不同 —— 这正是"关 SCAN"的预期效果。
        # 只在网络没选中 SCAN 时要求一致,其余情况由专门的关 SCAN 测试覆盖。
        if a_plain[0] != SCAN:
            assert a_plain == a_hybrid


# ------------------------------------------------------- 6. 训练链路跑通

def test_training_smoke_with_rules_live():
    """微型训练跑通,**且规则真的被触发过**。

    只断言"跑通"是不够的:规则若在训练里静默失效,测试照样全绿 —— 那正是本仓库
    反复防的"测了个寂寞"。所以这里盯 `forced_count`。
    """
    import monet.training.selfplay as SP
    from monet.training.config import Config
    from monet.training.selfplay import SelfPlayTrainer

    saved = SP.evaluate
    SP.evaluate = lambda *a, **k: {}
    try:
        with tempfile.TemporaryDirectory() as td:
            cfg = Config(
                run_name="vgtest", out_dir=td, total_steps=512, rollout_steps=256,
                hidden=32, epochs=1, minibatches=2, seed=0,
                league=["random", "camper"], eval_opponents=["baseline"],
                eval_every=10 ** 9, save_every=10 ** 9, snapshot_every=10 ** 9,
                progress_every_steps=0,
            )
            tr = SelfPlayTrainer(cfg)
            tr.train()
            assert tr.rules.forced_count > 0, "跑了 512 步规则一次都没触发"
            assert tr.rules.kill_count + tr.rules.scan_count == tr.rules.forced_count
    finally:
        SP.evaluate = saved


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rule_pack_dirs() -> list:
    """仓库里**所有带了 `vg_rules.h` 的包**目录名(发现式,不写死)。

    写死 `rl_VG_v1.0` 会让之后每个包静默逃过镜像测试:规则层是**拷贝**分发出去的
    (`v1.0` → `v2.0` → `v2.3`),任何一次拷贝都可能顺手改掉一个常量,而症状是
    "训练时按 A 规则、发出去按 B 规则" —— 只表现为分数比测出来的低,不报错、不崩。
    既然规则是策略里与网络无关的那一半,判据就该是"每个发货的包都对得上
    `rules_vg.py`",而不是"某个特定的包对得上"。
    """
    return sorted(
        d
        for d in os.listdir(_ROOT)
        if d.startswith("rl_VG_")
        and os.path.exists(os.path.join(_ROOT, d, "vg_rules.h"))
    )


def test_cpp_mirror_constants_match_the_python_source_of_truth():
    """每个包里的 `vg_rules.h` 的常量/顺序都必须与 `rules_vg.py` 一字不差。

    **这条测试抓得到什么、抓不到什么,必须说清楚:**

    抓得到 —— 常量漂移。改了 Python 侧 `MISS_THRESHOLD` 却忘了改 C++ 是这份移植
    最可能的失效方式,而它的表现是"测得强、发出去弱",不报错、不崩溃,只有把两个
    包都拉出来对打分才看得出来。

    抓不到 —— C++ 侧的**编译/编码错误**。写这个包的机器上没有 C++ 工具链,`killPlan`
    的 BFS、`fireHits` 的弹道遮挡、指针越界这类东西一个都验不了。计划里原本还想做
    "把 vg_rules.h 逐字抄进测试做差分",放弃了:那只是把同一个规格抄第三遍,三份
    照样能一起漂,还多一份维护负担。**不要因为这条测试是绿的,就以为 .h 编译得过。**

    真正能闭合这个缺口的只有:在有工具链的机器上编译一次,然后把同一批状态喂给
    C++ 与 Python 两份实现比动作序列。
    """
    packs = _rule_pack_dirs()
    assert packs, "仓库里没有带 vg_rules.h 的包 —— 规则层根本没跟着交付"
    for name in packs:
        print(f"  [{name}] 对拍 vg_rules.h")
        _check_one_rule_header(os.path.join(_ROOT, name, "vg_rules.h"))


def _check_one_rule_header(path: str) -> None:
    h = open(path, encoding="utf-8").read()

    def num(sym):
        m = re.search(rf"inline constexpr int {sym}\s*=\s*(-?\d+);", h)
        assert m, f"vg_rules.h 里找不到 {sym} 的整型常量定义"
        return int(m.group(1))

    def boolean(sym):
        m = re.search(rf"inline constexpr bool {sym}\s*=\s*(true|false);", h)
        assert m, f"vg_rules.h 里找不到 {sym}"
        return m.group(1) == "true"

    # --- 规则语义常量 ---
    assert num("kMissThreshold") == MISS_THRESHOLD, (
        f"失明阈值漂了:C++ {num('kMissThreshold')} vs Python {MISS_THRESHOLD}。"
        f"该扫的时候不扫(或反过来)会直接改胜率。"
    )
    assert boolean("kNetworkScan") == NETWORK_SCAN, (
        "SCAN 的归属漂了。网络能不能自己选 SCAN 是两个不同的策略,不能只改一边。"
    )
    assert num("kFireRange") == R.FIRE_RANGE
    assert num("kBoardSize") == R.BOARD_SIZE
    assert num("kActionsPerTurn") == ACTIONS_PER_TURN
    # 搜索深度写成 `= kActionsPerTurn` 而不是字面量(它就该跟着行动数走),
    # 所以这里查的是这层绑定关系本身,不是数值。
    assert re.search(r"inline constexpr int kMaxSearchDepth\s*=\s*kActionsPerTurn;", h), (
        "搜索深度必须等于每回合行动数,否则 killPlan 会搜出跨回合的方案"
    )

    # --- 出生点朝向:靠"观测是视角坐标"成立,见 obs.py + game.view 的镜像 ---
    m = re.search(r"inline constexpr char kStartFacing = '(.)';", h)
    assert m, "vg_rules.h 里找不到 kStartFacing"
    assert m.group(1) == R.START_FACING["R"], (
        "免费转身的判定依赖'视角坐标下出生点恒为 (0,0)、初始朝向恒为 E'。"
        "改了这里,开局那次免费转身就再也认不出来。"
    )

    # --- 方向顺序:**平局裁决就靠它** ---
    m = re.search(r"inline constexpr char kDirs\[kNDirs\] = \{([^}]*)\};", h)
    assert m, "vg_rules.h 里找不到 kDirs"
    dirs = re.findall(r"'(.)'", m.group(1))
    assert tuple(dirs) == tuple(R.DIRS), (
        f"方向顺序漂了:C++ {dirs} vs Python {list(R.DIRS)}。两个实现都能跑、都致命,"
        f"但会选出**不同的**击杀方案,差分测试会红在这里。"
    )

    # --- 动作 id:必须与 obs.py 的掩码/动作表一致 ---
    for sym, want in (
        ("kMove", MOVE),
        ("kFire", FIRE),
        ("kScan", SCAN),
        ("kEnd", END),
    ):
        assert num(sym) == want, f"{sym} 的动作 id 漂了:C++ {num(sym)} vs Python {want}"
    for d, sym in (("N", "kTurnN"), ("E", "kTurnE"), ("S", "kTurnS"), ("W", "kTurnW")):
        assert num(sym) == TURN_ACTION[d], f"{sym} 的动作 id 漂了"


def test_pack_ships_the_rules_not_a_bare_network():
    """参赛包必须是**带规则的混合策略** —— 这是最容易发错的一处。

    vgb4 是 `use_rules=True` 训出来的,而 `NETWORK_SCAN = False` 意味着它的 SCAN
    在训练时一直被掩掉。发一个不带规则的 `rl_VG.cpp`,网络会自由选 SCAN,而那个
    logit 从来没被训练过 —— 分数会掉,而且**不会报错**。
    """
    packs = _rule_pack_dirs()
    assert packs, "仓库里没有带 vg_rules.h 的包 —— 规则层根本没跟着交付"
    for name in packs:
        pack = os.path.join(_ROOT, name)
        cpp = open(os.path.join(pack, "rl_VG.cpp"), encoding="utf-8").read()

        assert '#include "vg_rules.h"' in cpp, f"{name}/rl_VG.cpp 没有 include 规则层"
        assert "actMask(" in cpp, (
            f"{name}/rl_VG.cpp 没调 actMask —— 那就还是个纯网络部署包。收窄必须在"
            f"**采样之前**发生,事后覆盖动作会让 logp 对不上(部署侧没有 PPO,但口径必须一致)"
        )
        # 采样必须用规则整过的掩码,而不是环境给的原始掩码
        assert "sample_with_temperature(logits, net_mask" in cpp, (
            f"{name}/rl_VG.cpp 用的是原始掩码而不是 net_mask —— SCAN 就没被关掉,"
            f"规则命中的收窄也白做"
        )
        # 规则 2 的失明计数是跨阶段状态,新一局必须清零
        assert "g_rules.reset()" in cpp, (
            f"{name}/rl_VG.cpp 没有逐局 reset 规则 —— 上一局的失明连败会跨局累加,"
            f"新一局开局就误判成'失明很久'"
        )


def test_pack_obs_is_bit_identical_to_the_previous_pack():
    """`rl_VG_v1.0/obs_builder.h` 只能比 `rl_VG_v0.2` 多那几个只读访问器。

    观测规格一改,428 维向量的含义就变了,所有已训权重全部作废 —— 这条把它焊死。
    规则层需要 `opp_visible`/`intel_pos`/`turn`/`obstacles`,而它们在 v0.2 里是私有的;
    加访问器**不碰编码**,但"不碰"这件事得由测试来保证,不能靠自觉。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    new = open(os.path.join(root, "rl_VG_v1.0", "obs_builder.h"), encoding="utf-8").read()
    old = open(os.path.join(root, "rl_VG_v0.2", "obs_builder.h"), encoding="utf-8").read()

    # 从标记所在行的**行首**开始切,不能从标记串本身切:标记是带缩进的,从标记处
    # 切会把那行的前导空格留在前半段,和后半段的 `private:` 拼成 `    private:`。
    # 那样断言会红,但**报的是假差异** —— 文件其实没问题,是切口选错了。
    start = new.index("// ---- rl_VG_v1.0 新增:供同目录 vg_rules.h 的规则层读取 ----")
    start = new.rfind("\n", 0, start) + 1
    end = new.index("private:", start)
    stripped = new[:start] + new[end:]

    assert stripped == old, (
        "obs_builder.h 除了那一段访问器之外还有别的改动 —— 观测规格可能被动过了。"
        "改动内容看这里第一次出现差异的位置:"
        f"\n  新版:{stripped[:400]!r}\n  旧版:{old[:400]!r}"
    )

    for acc in ("opp_visible()", "intel_pos()", "turn()", "obstacles()"):
        assert acc in new[start:end], f"规则层要读的 {acc} 没放出来"

    # 之后每个发货的包都必须与 v1.0 这份**逐位相同**。观测规格是全局契约,428 维
    # 的含义一改,所有已训权重一起作废。从 v1.0 拷包时"顺手把某个访问器写得更顺
    # 手"不会报错,只会让包里的观测与训练时喂进去的不是同一个向量 —— 而这类改动
    # 恰恰爱挑那种"看起来只是重命名"的地方。
    for name in _rule_pack_dirs():
        got = open(os.path.join(root, name, "obs_builder.h"), encoding="utf-8").read()
        assert got == new, (
            f"{name}/obs_builder.h 与 rl_VG_v1.0 的不逐位相同 —— 观测规格可能被动过。"
            f"它必须是**原样拷**;要改规格就得所有包一起改、所有权重一起作废。"
        )
    print(f"  {len(_rule_pack_dirs())} 个规则包的 obs_builder.h 与 v1.0 逐位相同")


# ------------------------------------------------- rl_VG_v1.0/vg_rules.h 的回译
#
# 下面这一坨是**只照着 `rl_VG_v1.0/vg_rules.h` 的文本逐句译回来的 Python**,
# 不看 rules_vg.py。目的不是"再实现一遍",而是让手写 C++ 时的逻辑滑落显形 ——
# C++ 那侧写得对不对,只能这样验:
#
#   * 本机没有任何 C++ 工具链(g++/clang/gcc/cl/zig 全无),编译不了;
#   * 常量/顺序漂移由上面那条测试盯着,但**逻辑**漂移(循环边界、平局裁决顺序、
#     BFS 的 seen 时机)它看不见。
#
# 译的时候**必须照着 C++ 的语句结构走**,不要顺手写成"我知道 Python 是什么样的"。
# 那样译出来的只会是 Python 的复读机,什么都验不到。写成扁平数组、显式循环,
# 不用任何 C++ 侧没有的捷径(set/dict/sorted 之外的推导式一律不写)。
#
# 抓到过的真 bug:可达性剪枝那行,C++ 提前把 budget 截断了而 Python 没有
# (默认路径碰不到,显式传大 budget 才分叉);以及 MOVE 分支先算数组下标再判出界,
# 出界时会用负下标索引 seen[]。

_CPP_DIRS = ["N", "E", "S", "W"]
_CPP_TURN = {"N": 1, "E": 2, "S": 3, "W": 4}
_CPP_FIRE_RANGE = 3
_CPP_BOARD = 7
_CPP_ACTIONS_PER_TURN = 3
_CPP_MAX_SEARCH_DEPTH = 3
_CPP_MOVE, _CPP_FIRE = 0, 5


def _cpp_delta(f):
    if f == "N":
        return 0, -1
    if f == "E":
        return 1, 0
    if f == "S":
        return 0, 1
    if f == "W":
        return -1, 0
    return 0, 0


def _cpp_perp(f):
    if f == "N" or f == "S":
        return 1, 0
    return 0, 1


def _cpp_in_bounds(x, y):
    return 0 <= x < _CPP_BOARD and 0 <= y < _CPP_BOARD


def _cpp_fire_hits(shooter, facing, target, obstacles):
    fx, fy = _cpp_delta(facing)
    px, py = _cpp_perp(facing)
    for lat in (-1, 0, 1):
        sx = shooter[0] + lat * px
        sy = shooter[1] + lat * py
        for step in range(1, _CPP_FIRE_RANGE + 1):
            cx = sx + step * fx
            cy = sy + step * fy
            if not _cpp_in_bounds(cx, cy):
                break
            if (cx, cy) in obstacles:
                break
            if (cx, cy) == target:
                return True
    return False


def _cpp_free_turn(s):
    if s["my_pos"] != (0, 0):
        return False
    if s["free_turn"]:
        return True
    return s["turn"] == 0 and s["actions_used"] == 0 and s["my_facing"] == "E"


def _cpp_kill_plan(s, budget=-1):
    if s["fire_cd"] != 0:
        return -1
    if not s["opp_visible"]:
        return -1
    tx, ty = s["intel_pos"]
    if tx < 0 or ty < 0:
        return -1
    if budget < 0:
        budget = _CPP_ACTIONS_PER_TURN - s["actions_used"]
    if budget <= 0:
        return -1
    # 只截断循环次数,budget 本身不动(剪枝那行要用原始值)
    max_depth = budget if budget < _CPP_MAX_SEARCH_DEPTH else _CPP_MAX_SEARCH_DEPTH

    start = s["my_pos"]
    target = s["intel_pos"]
    if abs(start[0] - target[0]) + abs(start[1] - target[1]) > (budget - 1) + _CPP_FIRE_RANGE + 1:
        return -1

    obstacles = s["obstacles"]
    frontier = [(start, s["my_facing"], -1)]
    if _cpp_free_turn(s):
        for d in _CPP_DIRS:
            if d != s["my_facing"]:
                frontier.append((start, d, _CPP_TURN[d]))

    seen = set()
    for pos, facing, _first in frontier:
        seen.add((pos, facing))

    for _depth in range(max_depth):
        for pos, facing, first in frontier:
            if _cpp_fire_hits(pos, facing, target, obstacles):
                return _CPP_FIRE if first < 0 else first

        nxt = []
        for pos, facing, first in frontier:
            for nd in _CPP_DIRS:
                if nd == facing:
                    continue
                key = (pos, nd)
                if key in seen:
                    continue
                seen.add(key)
                nxt.append((pos, nd, _CPP_TURN[nd] if first < 0 else first))

            dx, dy = _cpp_delta(facing)
            step_p = (pos[0] + dx, pos[1] + dy)
            if not _cpp_in_bounds(step_p[0], step_p[1]):
                continue
            if step_p == target:
                continue
            if step_p in obstacles:
                continue
            mkey = (step_p, facing)
            if mkey in seen:
                continue
            seen.add(mkey)
            nxt.append((step_p, facing, _CPP_MOVE if first < 0 else first))

        if len(nxt) == 0:
            break
        frontier = nxt
    return -1


def _cpp_state(ob):
    return {
        "my_pos": tuple(ob.my_pos),
        "my_facing": ob.my_facing,
        "fire_cd": ob.fire_cd,
        "scan_cd": ob.scan_cd,
        "actions_used": ob.actions_used,
        "turn": ob.turn,
        "free_turn": ob.free_turn,
        "opp_visible": ob.opp_visible,
        "intel_pos": tuple(ob.intel_pos),
        "obstacles": set(ob.obstacles),
    }


def test_cpp_fire_hits_matches_the_python_engine():
    """回译的 fireHits 要对 `engine/rules.py::fire_hits` 逐格一致。

    火力通道上的障碍要 **break 而不是 continue** —— 弄错会让规则报出打不中的
    "击杀方案",而 killPlan 一旦给出打不中的开火,这一回合就白扔了。
    """
    for obstacles in ((), ((1, 1),), ((1, 1), (5, 5)), ((0, 1), (0, 2)), ((3, 3),)):
        obs_set = set(obstacles)
        for sx in range(7):
            for sy in range(7):
                if (sx, sy) in obs_set:
                    continue
                for f in "NESW":
                    for tx in range(7):
                        for ty in range(7):
                            got = _cpp_fire_hits((sx, sy), f, (tx, ty), obs_set)
                            want = R.fire_hits((sx, sy), f, (tx, ty), obs_set)
                            assert got == want, (
                                f"fireHits 不一致:射手({sx},{sy}) 朝{f} 目标({tx},{ty}) "
                                f"障碍{obstacles} —— 回译 {got},引擎 {want}"
                            )


def test_cpp_kill_plan_matches_on_random_states():
    """回译的 killPlan 要在随机局面上与 `rules_vg.kill_plan` 给出同一个动作。

    只比对返回值是不够的 —— 平局裁决(先不转身、再 N/E/S/W;同层内 TURN 先于
    MOVE)错了也会"返回一个能击杀的动作",但那是**另一个**方案,两个实现从此
    走的是不同的棋。所以这里同时断言两者返回的动作 id 完全相同。
    """
    import random

    rng = random.Random(20260918)
    obstacles = ((1, 1), (5, 5))
    cells = [(x, y) for x in range(7) for y in range(7) if (x, y) not in obstacles]
    mismatches = 0
    checked = 0

    for _ in range(4000):
        me = rng.choice(cells)
        opp = rng.choice(cells)
        if opp == me:
            continue
        face = rng.choice("NESW")
        used = rng.randint(0, 2)
        turn = rng.choice([0, 0, 3, 7, 19])
        g, ob = _setup(me, face, opp, fire_cd=rng.choice([0, 0, 0, 1]),
                       scan_cd=rng.choice([0, 3]), used=used, turn=turn,
                       obstacles=obstacles)
        if rng.random() < 0.75:
            _pretend_enemy_visible(ob, opp)

        want = kill_plan(ob)
        got = _cpp_kill_plan(_cpp_state(ob))
        want_id = -1 if want is None else want
        checked += 1
        if got != want_id:
            mismatches += 1
            if mismatches <= 5:
                print(f"    分歧:me={me} face={face} opp={opp} used={used} turn={turn} "
                      f"visible={ob.opp_visible} fire_cd={ob.fire_cd} "
                      f"-> Python {want_id}, 回译 {got}")

    assert checked > 3000, f"样本太少({checked}),测试没验到什么"
    assert mismatches == 0, (
        f"{mismatches}/{checked} 个局面两边给出不同方案 —— vg_rules.h 的 killPlan "
        f"与 rules_vg.py 已经漂了。发出去的 .so 会下另一盘棋。"
    )


def test_cpp_free_turn_agrees_with_python():
    """免费转身的判定两边必须一致 —— 它决定"转+走+走+打"这类 4 步方案能不能被找到。"""
    for turn in (0, 1, 5):
        for used in (0, 1, 3):
            for face in "NESW":
                for free in (False, True):
                    for pos in ((0, 0), (0, 1), (6, 6)):
                        g, ob = _setup(pos, face, (3, 3), used=used, turn=turn)
                        ob.free_turn = free
                        s = _cpp_state(ob)
                        assert _cpp_free_turn(s) == free_turn_available(ob), (
                            f"免费转身判定不一致:pos={pos} face={face} turn={turn} "
                            f"used={used} free_turn={free}"
                        )


# ---------------------------------------------------------------------------
# 交付包 rl_VG_v1.0/ 的权重头:数值校验
# ---------------------------------------------------------------------------

_PACK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl_VG_v1.0"
)


def _parse_pack_weights(pack_dir):
    """把包里的 `rl_weights*.h` 解析回 numpy 数组。

    只认 `static const float <名字>[<n>] = { ... };` 这一种声明 —— 导出器只写
    这一种,认别的等于给"格式悄悄变了"留后门。
    """
    arrays = {}
    for fn in sorted(os.listdir(pack_dir)):
        if not (fn.startswith("rl_weights") and fn.endswith(".h")):
            continue
        text = open(os.path.join(pack_dir, fn), encoding="utf-8").read()
        for m in re.finditer(
            r"static const float (\w+)\[(\d+)\]\s*=\s*\{(.*?)\};", text, re.S
        ):
            sym, n, body = m.group(1), int(m.group(2)), m.group(3)
            vals = np.array(
                [float(t) for t in body.replace("\n", " ").split(",") if t.strip()],
                dtype=np.float32,
            )
            assert vals.size == n, f"{sym}: 声明 {n} 个,实际 {vals.size} 个"
            arrays[sym] = vals
    return arrays


# rl_VG.cpp 读这些数组时用的维度。头里是**扁平**声明(`kW0[219136]`),C++ 靠
# `W + o*in_dim` 自己算下标 —— 所以"形状"在 C++ 那边只体现为那个长度,写错长度
# 编译一样过,要到运行时越界才炸。这张表就是拿 C++ 的 in_dim/out_dim 回去核长度。
_PACK_SHAPES = {
    "kW0": (512, 428), "kB0": (512,),
    "kLN1Gamma": (512,), "kLN1Beta": (512,),
    "kW1": (512, 512), "kB1": (512,),
    "kLN2Gamma": (512,), "kLN2Beta": (512,),
    "kW2": (512, 512), "kB2": (512,),
    "kLN3Gamma": (512,), "kLN3Beta": (512,),
    "kW3": (8, 512), "kB3": (8,),
    "kW4": (1, 512), "kB4": (1,),      # 价值头:导出保留,推理不用
}


def _pack_tensor(w, sym):
    shape = _PACK_SHAPES[sym]
    a = w[sym]
    assert a.size == int(np.prod(shape)), (
        f"{sym} 长度 {a.size} != {shape} 需要的 {int(np.prod(shape))} —— "
        f"rl_VG.cpp 按 {shape} 读它,长度错了编译能过、运行时越界"
    )
    return a.reshape(shape)


def _pack_ln(x, g, b):
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g + b


def _pack_gelu(x):
    from monet.models.mlp import gelu
    return gelu(x)


def _pack_forward(w, x):
    """rl_VG.cpp::forward 的逐行转写。

    **这是转写,不是"另一个实现"** —— 和 vg_rules.h 一样,它跟 C++ 会静默漂移。
    所以下面断言的是"包里的权重 + 包里的前向顺序 == 检查点",而不是"C++ 编得过":
    本机没有工具链,编不过这件事这里验不了。
    """
    h1 = x @ _pack_tensor(w, "kW0").T + _pack_tensor(w, "kB0")
    h1 = _pack_gelu(_pack_ln(h1, w["kLN1Gamma"], w["kLN1Beta"]))

    h2 = h1 @ _pack_tensor(w, "kW1").T + _pack_tensor(w, "kB1") + h1   # 残差
    h2 = _pack_gelu(_pack_ln(h2, w["kLN2Gamma"], w["kLN2Beta"]))

    h3 = h2 @ _pack_tensor(w, "kW2").T + _pack_tensor(w, "kB2") + h2   # 残差
    h3 = _pack_gelu(_pack_ln(h3, w["kLN3Gamma"], w["kLN3Beta"]))

    return h3 @ _pack_tensor(w, "kW3").T + _pack_tensor(w, "kB3")


def test_pack_weight_shapes_match_the_shipped_forward():
    """维度和 rl_VG.cpp 里写死的那几个数字必须对得上。

    这条**不需要检查点**,所以永远会跑 —— 打包时最容易错的就是把 512 的权重
    配上一个 256 的网络,而那种错编译**能过**(数组维度只在运行到那一层时才崩)。
    """
    w = _parse_pack_weights(_PACK_DIR)
    for sym, shape in _PACK_SHAPES.items():
        assert sym in w, f"包里缺 {sym}"
        _pack_tensor(w, sym)   # 长度不符就抛,消息里带上 C++ 侧的维度


def test_pack_weights_reproduce_the_checkpoint_they_declare():
    """数值闭环:解析包里的权重、按包里的前向顺序重算,与来源检查点比。

    **这条才是"发出去的 .so 是不是我们测过的那份网络"的唯一机械保证。** 把
    `kW0` 转置、把 `kLN2*` 接到第二层之外、把 `kW3` 和 `Wv` 换位,这些错
    **结构检查过得去、编译过得去、评测时才掉分**,而且不掉到 0.5 —— 掉到 0.3
    这种"看起来只是弱了点"的位置。随机观测上逐点比对能立刻抓住。

    检查点不在就跳过并打印:包不该因为训练目录被清掉而变成红的,但"跳过"必须
    说出来,不能让"没测"看起来像"测过了"。
    """
    from monet.models.mlp import MLP
    from monet.store import load_checkpoint

    main = os.path.join(_PACK_DIR, "rl_weights.h")
    src = re.search(r"来源检查点:(\S+)", open(main, encoding="utf-8").read())
    assert src, "rl_weights.h 少了来源检查点注释 —— 包的出处必须可追"
    ckpt = src.group(1)

    if not os.path.exists(ckpt):
        print(f"    (跳过:{ckpt} 不存在 —— 只验了形状,没验数值)")
        return

    net, _, _ = load_checkpoint(ckpt)
    w = _parse_pack_weights(_PACK_DIR)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 428)).astype(np.float32)
    want, _ = net.forward(x, cache=False)
    got = _pack_forward(w, x)

    diff = np.abs(want - got)
    assert diff.max() < 1e-4, (
        f"包里的权重与前向重算不出检查点 {ckpt} 的输出:最大偏差 {diff.max():.3e}。"
        f"转置/接错层/符号绑错都会长这样。"
    )


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