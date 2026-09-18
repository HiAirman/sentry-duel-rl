"""训练链路集成测试:小规模跑通 采数据 → PPO 更新 → 评测 → 落盘 → 导出。"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monet.models.rnn_policy import RNNPolicy  # noqa: E402
from monet.store import load_checkpoint, save_checkpoint  # noqa: E402
from monet.training.config import Config  # noqa: E402
from monet.training.export_cpp import export_weights_header  # noqa: E402
from monet.training.selfplay import League, SelfPlayTrainer  # noqa: E402


def _tiny_cfg(out_dir: str) -> Config:
    return Config(
        run_name="t",
        out_dir=out_dir,
        total_steps=300,
        rollout_steps=96,
        hidden=32,
        epochs=2,
        minibatches=2,
        seed=0,
        league=["random", "camper"],
        # 显式钉住评测集:默认值里有 rl_v5,而它要读仓库外的权重包 ——
        # 不然这几条本该自足的集成测试会变成"别人机器上没包就红"。
        eval_opponents=["baseline", "camper"],
        snapshot_every=2,
        eval_every=2,
        eval_games=2,
        save_every=2,
        # 闸门默认 0.40,而这些小跑几百步根本打不过官方 AI → 永远不加快照。
        # 这里测的是"加快照"本身,所以关掉闸门;闸门另有专门的测试。
        snapshot_min_official_winrate=0.0,
    )


def test_training_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        cfg = _tiny_cfg(td)
        trainer = SelfPlayTrainer(cfg)
        net = trainer.train()
        out = Path(td) / "t"
        assert (out / "final.npz").exists(), "必须落盘 final.npz"
        assert (out / "last.npz").exists(), "必须周期存档"
        assert (out / "metrics.csv").exists(), "必须写评测日志"
        assert trainer.history, "评测历史不该为空"
        for v in net.p.values():
            assert np.isfinite(v).all(), "训练后权重出现 NaN/Inf"


def test_league_adds_snapshots():
    with tempfile.TemporaryDirectory() as td:
        cfg = _tiny_cfg(td)
        cfg.total_steps = 200
        trainer = SelfPlayTrainer(cfg)
        trainer.train()
        assert trainer.league.snapshots, "训练中应加入历史快照对手"
        assert len(trainer.league.snapshots) <= cfg.max_snapshots


def test_snapshot_names_stay_unique_after_the_cap_is_reached():
    """快照被 `max_snapshots` 裁掉之后,存活的那两个**不能同名**。

    编号原本写成 `f"snap{len(self.snapshots)}"`,而它在 `append` **之前**求值:
    列表一旦被裁到定长,编号就卡住 —— 第 4 次冻结起两个存活快照都叫 `snap2`,
    联盟分布打出来是 `snap2=13.9% snap2=13.9%`,看着像混进了重复项。名字只用于
    日志(`_opp_bucket` 按对象身份认快照),所以数据没坏;但"名字标识第几代"这件事
    本身得成立,不然下次谁按名字查就踩雷。所以冻结到超过上限,再验唯一性。
    """
    from monet.models.mlp import MLP

    league = League(["random"], max_snapshots=2, seed=0)
    net = MLP(obs_dim=8, hidden=8, act_dim=4)
    for _ in range(6):
        league.add_snapshot(net)
    names = league.names()
    assert len(league.snapshots) == 2, "快照数应被裁到 max_snapshots"
    assert len(names) == len(set(names)), f"快照重名: {names}"


def test_snapshot_gate_waits_for_official_winrate():
    """闸门:官方 AI 胜率没到线就一个快照都不加,过线才开始加。"""
    with tempfile.TemporaryDirectory() as td:
        cfg = _tiny_cfg(td)
        cfg.snapshot_min_official_winrate = 0.40
        # 闸门测的是"打官方 AI 的胜率",评测集里必须**有**官方对手,否则
        # `_gate_measurable` 为假、闸门直接放行(那是另一条测试)。
        # 这里挑 official_baseline:它是纯 Python 的移植版,不碰外部权重包。
        cfg.eval_opponents = ["official_baseline", "camper"]
        trainer = SelfPlayTrainer(cfg)
        assert trainer._gate_measurable

        # 还没评测过 → 关着
        trainer.last_official_winrate = None
        assert not trainer._snapshot_gate_open(), "还没评测过就该关着"
        # 打过 40% 但没超过 → 仍然关着(条件是"大于")
        trainer.last_official_winrate = 0.40
        assert not trainer._snapshot_gate_open(), "正好等于线不该放行"
        trainer.last_official_winrate = 0.39
        assert not trainer._snapshot_gate_open(), "低于线不该放行"
        # 超过 → 放行
        trainer.last_official_winrate = 0.41
        assert trainer._snapshot_gate_open(), "过线就该放行"

        # 闸门关着的整轮训练:一个快照都不该出现
        cfg.total_steps = 300
        trainer.last_official_winrate = 0.0
        trainer.train()
        assert not trainer.league.snapshots, "闸门关着却加了快照"


def test_snapshot_gate_opens_without_official_opponents_in_eval():
    """评测集里没有官方对手时,闸门测不出来 —— 必须放行而不是静默卡死。"""
    with tempfile.TemporaryDirectory() as td:
        cfg = _tiny_cfg(td)
        cfg.snapshot_min_official_winrate = 0.40
        cfg.eval_opponents = ["baseline", "camper"]
        cfg.total_steps = 200
        trainer = SelfPlayTrainer(cfg)
        assert not trainer._gate_measurable
        assert trainer._snapshot_gate_open()
        trainer.train()
        assert trainer.league.snapshots, "闸门测不出来时应按不设闸门处理"


def test_official_opponents_are_registered_and_flagged():
    """闸门靠 is_official 认官方对手,注册名与标记必须对得上。

    按**工厂上的标记**断言,不靠实例化探测 —— 注册表里有 rl_v5(要读仓库外的
    权重包),实例化它在没包的机器上会抛 PackError,这条测试不该因此变红。
    """
    from monet.training.selfplay import OFFICIAL_OPPONENTS, STATIC_OPPONENTS

    # 这份名单是**有意为之**,不是漏改:rl_v5 是联盟里唯一一份原样的外部强 AI
    # (官方四家是移植版,寻路还是我们自己写的),而闸门问的正是"能不能打赢手上
    # 最难的对手"。代价是闸门语义 = 官方四家 + 外部包的平均得分率。
    #
    # ⚠️ README §五 的说法与此相反(「官方档 4 家」且「`rl_v5` 刻意不算官方」),
    # 代码与文档口径不一致。改这个集合之前先回 README 对一次。见 selfplay.py:66-76。
    assert set(OFFICIAL_OPPONENTS) == {
        "official_baseline",
        "official_hunter",
        "rl_v5",
    }, "官方档名单变了:加/减官方对手都要显式改这里"
    assert set(OFFICIAL_OPPONENTS) <= set(STATIC_OPPONENTS)
    for n, make in STATIC_OPPONENTS.items():
        flagged = bool(getattr(make, "is_official", False))
        assert flagged == (n in OFFICIAL_OPPONENTS), (n, flagged)
    # 但 m3_v* 不能进:那是我们自己的过去,算进去等于自己给自己发合格证。
    for n in ("m3_v1", "m3_v2"):
        assert n not in OFFICIAL_OPPONENTS, f"{n} 是自家导出,不该进官方档"


def test_default_rosters_do_not_drift():
    """`Config` 的默认名单与 selfplay 的 LEAGUE_DEFAULT/EVAL_DEFAULT 是手抄两份。

    两边不能互相 import(会循环),只能靠这条测试焊死漂移。不焊的话,直接
    `Config()` 构造的路径(测试、程序化调用)会悄悄漏掉新对手,而 CLI 路径
    (`cmd_train` 传的是 `",".join(LEAGUE_DEFAULT)`)反而正常 —— 最难发现的那种不一致。
    """
    from monet.training.selfplay import (
        DEFAULT_STATIC_WEIGHT,
        EVAL_DEFAULT,
        LEAGUE_DEFAULT,
        LOW_WEIGHT,
        STATIC_WEIGHTS,
    )

    cfg = Config()
    assert cfg.league == LEAGUE_DEFAULT
    assert cfg.eval_opponents == EVAL_DEFAULT

    # 口径:**训练联盟** = 官方四家 + 外部包 rl_v5;**评测集**再加上两个 m3 导出。
    # 两个 m3 导出只评测不陪练 —— 它们与当前网络同源,当陪练的梯度价值低,
    # 但当"别退步"的锚还有用。
    assert "rl_v5" in LEAGUE_DEFAULT and "rl_v5" in EVAL_DEFAULT
    for n in ("m3_v1", "m3_v2"):
        assert n not in LEAGUE_DEFAULT, f"{n} 已按口径移出训练联盟(仍在评测集里)"
        assert n in EVAL_DEFAULT, n

    # 采样权重分三档:
    #   rl_v5 2.0 —— 三个外部包里唯一还没打服的(评测得分率 ~0.68,还有两成负场),
    #               额度最紧,多给它采样才有梯度。
    #   rl_VG_v0.2 / rl_best040 1.0 —— 已接近全胜(~0.98 / 1.00),对着它们练几乎
    #               不产生梯度,给默认档即可。注意这两个是**显式登记**在
    #               STATIC_WEIGHTS 里的,和 m3_v* 的"不在表里所以取默认值"不是一回事,
    #               所以下面用 `in STATIC_WEIGHTS` 把它们和 m3_v* 区分开。
    #   手工四家 + 官方两家移植 + 自研四家 = 10 家 0.3 —— 已打饱和,留一点额度
    #               只为"别把它们打回去"。**"自研"不等于"强"**:两个记忆型规则手
    #               结构上是记忆型,棋力上仍是规则手(diag 300 局对 v1.0/v2.0 全胜),
    #               所以和 stalker/patrol 同档,不要因为名字里有"记忆"就往上升档。
    assert STATIC_WEIGHTS["rl_v5"] > DEFAULT_STATIC_WEIGHT, "rl_v5 应当高于默认档"
    for n in ("rl_v5", "rl_VG_v0.2", "rl_best040"):
        assert n not in LOW_WEIGHT, f"{n} 一旦进了 LOW_WEIGHT 会被压到 0.3"
        assert n in STATIC_WEIGHTS, f"{n} 应当在权重表里显式登记"
        assert STATIC_WEIGHTS[n] >= DEFAULT_STATIC_WEIGHT, (
            f"{n} 是外部强敌,不该被压到默认档以下"
        )
    # m3_v1/m3_v2 **只评测不陪练**(口径见上面那段),所以不进权重表 ——
    # 取默认 1.0,而它们根本不在联盟里,所以这个值不影响任何训练。
    for n in ("m3_v1", "m3_v2"):
        assert n not in LOW_WEIGHT, f"{n} 一旦进了 LOW_WEIGHT 会被压到 0.3"
        assert n not in STATIC_WEIGHTS, f"{n} 不该进权重表(它们不陪练)"
    for n in (
        "random",
        "official_baseline",
        "official_stalker",
        "official_patrol",
        "official_ambusher",
        "official_weaver",
    ):
        assert STATIC_WEIGHTS[n] < DEFAULT_STATIC_WEIGHT, f"{n} 应该在低权重档里"

    # 闸门看的是 OFFICIAL_OPPONENTS 的平均得分率。m3_v* 是自家导出,算进去等于
    # 自己给自己发合格证;rl_v5 进官方档则是**有意的**(见
    # test_official_opponents_are_registered_and_flagged 的说明)。
    from monet.training.selfplay import OFFICIAL_OPPONENTS

    for n in ("m3_v1", "m3_v2"):
        assert n not in OFFICIAL_OPPONENTS, f"{n} 不该进官方档,否则闸门会自证"


def test_cli_train_defaults_come_from_config():
    """`cli.py` 里 train/eval 的 argparse 默认值必须与 `Config` 字段逐一对上。

    两边一旦不一致,赢的总是 argparse:**在 config.py 里调参等于没调,而且不报错**
    —— 跑起来一切正常,只是用的不是你以为的那组数。比"名单漏了一个对手"更难发现。
    """
    from monet.cli import build_parser

    a = build_parser().parse_args(["train"])
    c = Config()
    # `use_rules` 故意不在这张表里:它在 CLI 那侧是 `--no-rules`(store_true)的
    # **取反**,名字和极性都对不上。硬塞进来只会让"名实相符"变成一句空话 ——
    # 真正该钉的是下面这一条。
    assert a.no_rules is (not c.use_rules), "训练默认不挂规则,与 --no-rules 的极性不符"
    for flag, fld in (
        ("run_name", "run_name"),
        ("out_dir", "out_dir"),
        ("steps", "total_steps"),
        ("rollout", "rollout_steps"),
        ("hidden", "hidden"),
        ("lr", "lr"),
        ("seed", "seed"),
        ("scan_reveal", "scan_reveal"),
        ("snapshot_min_official_winrate", "snapshot_min_official_winrate"),
        ("eval_every", "eval_every"),
        ("eval_games", "eval_games"),
        ("init_pack", "init_pack"),
        ("stop_winrate", "stop_winrate"),
        ("stop_confirm_games", "stop_confirm_games"),
        ("stop_screen_margin", "stop_screen_margin"),
        ("guard_official_winrate", "guard_official_winrate"),
        ("stop_metric_slack", "stop_metric_slack"),
        ("league_weight_mode", "league_weight_mode"),
        ("league_weight_cap", "league_weight_cap"),
    ):
        assert getattr(a, flag) == getattr(c, fld), (
            f"--{flag.replace('_', '-')} 默认 {getattr(a, flag)!r},"
            f"Config.{fld} 是 {getattr(c, fld)!r} —— CLI 会静默盖掉 config.py"
        )
    assert a.league == ",".join(c.league)
    # 训练期的评测名单与联盟是**两张表**,各有各的默认值,所以各钉一条 ——
    # 抄错了会让"拿它陪练"和"量它"变成两件事(`--eval-opponents` 就是为此加的)。
    assert a.eval_opponents == ",".join(c.eval_opponents)
    # `stop_opponents` 在 CLI 那侧是**逗号串**、Config 那侧是 list,和
    # `eval_opponents` 同一个形状,所以同样单独钉,不能塞进上面那张逐字段相等表。
    assert a.stop_opponents == ",".join(c.stop_opponents)

    e = build_parser().parse_args(["eval", "--ckpt", "x"])
    assert e.opponents == ",".join(c.eval_opponents)


def test_resume_keeps_previous_eval_rows():
    """续训**不能**把 `metrics.csv` 前面的评测行抹掉。

    `metrics.csv` 是整表重写,重写的原料是 `self.history`;而续训时 `history` 从空
    开始,于是第一条新数据就把整张表覆盖了。这是纯静默失效:`train_metrics.csv` 走
    追加所以毫发无损,于是 `curves.png` 上半张(评测)只剩最后两个点、下半张(训练)
    却完整 —— 看起来只像"评测点少",不像丢了数据。
    """
    import csv

    def rows(p):
        with p.open("r", newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    with tempfile.TemporaryDirectory() as td:
        cfg = _tiny_cfg(td)
        cfg.total_steps = 200
        SelfPlayTrainer(cfg).train()
        out = Path(td) / "t"
        before = rows(out / "metrics.csv")
        assert before, "第一次跑就该落下评测行"

        net, meta, _ = load_checkpoint(out / "final.npz")
        cfg2 = _tiny_cfg(td)
        cfg2.total_steps = 500
        SelfPlayTrainer(cfg2, init_net=net).train(resume_meta=meta)

        after = rows(out / "metrics.csv")
        assert len(after) > len(before), "续训后应该多出评测行"
        # 按**数值**比,不按文本比:读回来的旧行会统一成 float,`192` 会被重写成
        # `192.0`。那是列的格式归一(原本就是 int/float 混着的),不是数据丢了 ——
        # 钉文本会把这条测试变成"谁改格式谁红",而不是"谁丢数据谁红"。
        got = [(float(r["step"]), float(r["eval_metric"])) for r in after[: len(before)]]
        want = [(float(r["step"]), float(r["eval_metric"])) for r in before]
        assert got == want, f"续训把旧评测行弄丢了:\n  之前 {want}\n  现在 {got}"


def _stop_probe(cfg):
    """装一个假的 `evaluate` 和一个"进入下一个评测点"的助手,供早停测试用。

    返回 `(trainer, point, calls, restore)`。`point(读数, 真值)` 模拟一次评测:
    `读数` 写进 `last_eval`(即 `eval_games` 局那次的读数),`真值` 是那次高局数
    测量会看到的数 —— 两者故意可以不同,早停测试要的正是这个差。
    """
    import monet.training.selfplay as sp

    tr = SelfPlayTrainer(cfg)
    calls: list = []
    true_wr: dict = {}
    real = sp.evaluate

    def _fake(net, opps, games=40, seed=0, rules=None):
        calls.append({"games": games, "seed": seed, "opps": sorted(opps)})
        return {n: {"winrate": true_wr[n], "games": games} for n in opps}

    sp.evaluate = _fake

    def _point(reading, truth=None):
        true_wr.update(reading if truth is None else truth)
        tr.last_eval = {n: {"winrate": v} for n, v in reading.items()}
        tr.last_eval_update = (tr.last_eval_update or 0) + 1
        tr._stop_checked = None  # 新评测点 = 判据可以再判一次

    return tr, _point, calls, lambda: setattr(sp, "evaluate", real)


def test_early_stop_requires_every_opponent_to_reach_the_line():
    """早停是"全部达标",不是"任一达标"。

    "任一达标"会在策略**最偏科的那一刻**停下:对强敌 0.92、对弱敌 0.28 也会触发,
    而那正是最不该收工的时候。
    """
    with tempfile.TemporaryDirectory() as td:
        cfg = _tiny_cfg(td)
        cfg.stop_opponents = ["camper", "baseline", "end"]
        cfg.eval_opponents = ["camper", "baseline", "end"]
        cfg.stop_winrate = 0.70
        cfg.stop_confirm_games = 200
        cfg.stop_screen_margin = 0.10
        tr, point, _, restore = _stop_probe(cfg)
        try:
            assert tr._stop_reason() is None, "还没评测过就不该早停"

            # 三个都在名单里,但只有 camper 到线 —— 这正是"任一"会误停的位置
            point({"camper": 0.92, "baseline": 0.28, "end": 0.57})
            assert tr._stop_reason() is None, "只有一个到线就停 = 偏科也能过关"

            # 三个都到线 → 停
            point({"camper": 0.75, "baseline": 0.72, "end": 0.71})
            reason = tr._stop_reason()
            assert reason is not None and "0.70" in reason, reason

            # 名字不在评测名单里(拼错)→ 整个判据失效。**不能**静默当成"没到线",
            # 否则一个笔误会让早停永远不触发,而跑的人以为一切正常。
            cfg.stop_opponents = ["camper", "typo_opponent"]
            point({"camper": 0.90})
            assert tr._stop_reason() is None
        finally:
            restore()


def test_early_stop_is_decided_by_a_dedicated_high_game_measurement():
    """达标与否由**另起的一次高局数测量**决定,不是 `eval_games` 那次的读数。

    两个方向都必须挡住,而"低局数先判、达标了再复核"的写法只挡得住第一个:

      · 假阳性 —— 60 局读 0.75、200 局实为 0.62 → 不许停(vgb3 的 update 120);
      · 假阴性 —— 60 局读 0.62、200 局实为 0.75 → **必须停**(vgb3 的 update 84)。

    vgb3 两个方向同时错了:该停在 84 却没停(读到 0.617,实为 0.700),不该停在
    120 却停了(读到 0.725,实为 0.647)。整轮跑的命运由 ±0.065 的噪声决定。
    """
    with tempfile.TemporaryDirectory() as td:
        cfg = _tiny_cfg(td)
        cfg.stop_opponents = ["camper", "baseline"]
        cfg.eval_opponents = ["camper", "baseline"]
        cfg.stop_winrate = 0.70
        cfg.stop_confirm_games = 200
        cfg.stop_screen_margin = 0.10
        tr, point, calls, restore = _stop_probe(cfg)
        try:
            # --- 假阳性:60 局读到达标,高局数一测并不达标 → 不许停 ---
            point({"camper": 0.75, "baseline": 0.75},
                  truth={"camper": 0.62, "baseline": 0.62})
            assert tr._stop_reason() is None, "高局数实测没到线,不许收工"
            assert calls and calls[0]["games"] == cfg.stop_confirm_games, (
                f"决定性的那次该用 {cfg.stop_confirm_games} 局,实际 {calls[0]['games']}"
            )
            assert calls[0]["opps"] == ["baseline", "camper"], "只测 stop_opponents,不必重测全部"
            # 必须换种子:沿用正式评测的 `cfg.seed + update` 等于把同一批局再跑一遍,
            # 那测的还是噪声,只是多花了几分钟。
            assert calls[0]["seed"] > cfg.seed + 10000, calls[0]["seed"]

            # --- 假阴性:60 局读得偏低,高局数实测其实达标 → **必须停** ---
            # "先判低局数、达标了再复核"的写法在这里会直接返回,永远发现不了。
            # vgb3 的 update 84 就是这个形状:读到 0.617,实为 0.700。
            calls.clear()
            point({"camper": 0.62, "baseline": 0.62},
                  truth={"camper": 0.75, "baseline": 0.75})
            reason = tr._stop_reason()
            assert reason is not None, "60 局读低了不等于没到线 —— 这正是 vgb3 update 84 的形状"
            assert "0.750" in reason, reason

            # --- 便宜筛子:明显没到就别花那几分钟 ---
            calls.clear()
            point({"camper": 0.40, "baseline": 0.75})
            assert tr._stop_reason() is None
            assert calls == [], "有对手明显没到线时就该跳过那次高局数测量"

            # --- 同一个评测点只判一次(判据每个 update 都被调用,测量很贵)---
            calls.clear()
            point({"camper": 0.75, "baseline": 0.75},
                  truth={"camper": 0.60, "baseline": 0.60})
            assert tr._stop_reason() is None
            assert len(calls) == 1, "第一次该真的去测"
            assert tr._stop_reason() is None
            assert len(calls) == 1, "同一个评测点不该把那次高局数测量重跑一遍"

            # --- `stop_confirm_games = 0` = 退回旧行为:直接认 60 局那次读数 ---
            calls.clear()
            cfg.stop_confirm_games = 0
            point({"camper": 0.75, "baseline": 0.75})
            reason = tr._stop_reason()
            assert reason is not None and "未另测" in reason, reason
            assert calls == [], "退回旧行为就不该再跑评测"
        finally:
            restore()


def test_early_stop_holds_back_when_officials_are_pushed_back():
    """保护官方 AI:官方档掉到地板以下时,即使 stop_opponents 全达标也不收工。

    这正是 vgb3 缺的那一条 —— 策略可以一边把 rl_v5 从 0.617 刷到 0.725,一边把
    官方 hunter 从 0.929 打到 0.667、把 m3_v1/m3_v2 打到 0.53/0.50,而旧判据照样
    收工。官方档是"没练偏"的锚,掉了就说明这一轮的达标是拿偏科换的。
    """
    with tempfile.TemporaryDirectory() as td:
        cfg = _tiny_cfg(td)
        cfg.stop_opponents = ["camper"]
        cfg.eval_opponents = ["camper", "official_baseline"]
        cfg.stop_winrate = 0.70
        cfg.stop_confirm_games = 200
        cfg.guard_official_winrate = 0.60
        tr, point, _, restore = _stop_probe(cfg)
        try:
            point({"camper": 0.90, "official_baseline": 0.42})
            assert tr._stop_reason() is None, "官方被打回去了,不许拿它换偏科"

            # 回到地板上方就放行(闸是地板,不是目标)
            point({"camper": 0.90, "official_baseline": 0.65})
            assert tr._stop_reason() is not None, "官方在地板上方就该正常判"

            # 0 = 关闸
            cfg.guard_official_winrate = 0.0
            point({"camper": 0.90, "official_baseline": 0.10})
            assert tr._stop_reason() is not None, "闸关掉之后不该再拦"
        finally:
            restore()


def test_early_stop_refuses_to_fire_below_the_runs_own_best():
    """不许在"自己都已经不如自己最好"那一刻收工 —— vgb3 就栽在这。

    那次 `best.npz` 的判据(综合得分率更大才更新)早在 update 84 就否决了
    update 120(0.8556 → 0.8046),早停判据却批准了它。两个判据打架,而早停赢了。
    这条让早停服从 best.npz 的判断。
    """
    with tempfile.TemporaryDirectory() as td:
        cfg = _tiny_cfg(td)
        cfg.stop_opponents = ["camper"]
        cfg.eval_opponents = ["camper"]
        cfg.stop_winrate = 0.70
        cfg.stop_confirm_games = 200
        cfg.stop_metric_slack = 0.02
        tr, point, _, restore = _stop_probe(cfg)
        try:
            tr.best_metric = 0.95
            point({"camper": 0.90})  # 综合 0.90,比本轮最优低 0.05
            assert tr._stop_reason() is None, "退步的点不该收工"

            # 与最优持平(容差之内)→ 放行
            tr.best_metric = 0.91  # 0.90 ≥ 0.91 - 0.02
            point({"camper": 0.90})
            assert tr._stop_reason() is not None, "持平就该正常判"

            # 容差是给"两个点其实一样好"留的余地,给 0 就该卡死
            cfg.stop_metric_slack = 0.0
            tr.best_metric = 0.91
            point({"camper": 0.90})
            assert tr._stop_reason() is None, "容差 0 时低于最优一律不许收工"
        finally:
            restore()


def test_room_weighting_moves_samples_toward_whoever_has_room():
    """动态配重:打服的让出额度,还有空间的、以及被打回去的多拿。

    vgb3 的死区数据:官方 stalker / patrol 在 48 个 update 里恒为 1.000,五个打服
    的对手合计吃掉 44.8% 采样却不产生梯度;同期官方 hunter 从 0.929 掉到 0.667,
    权重纹丝不动,没人救它。这里验的就是这两件事由同一个动作解决。
    """
    from monet.training.selfplay import League

    lg = League(["baseline", "camper", "hunter"], max_snapshots=2, weight_mode="room")
    total = sum(lg.static_w)
    lg.update_weights({"baseline": 1.00, "camper": 0.50, "hunter": 0.90})
    w = {o.name: x for o, x in zip(lg.static, lg.static_w)}
    assert w["camper"] > w["hunter"] > w["baseline"], (
        f"该按剩余空间排序,实际 {w} —— 打服的(1.00)必须最少"
    )
    assert abs(sum(w.values()) - total) < 1e-9, (
        "总额度不能变,否则会连带改掉静态档与快照档的配比(max_snapshots 对自对弈的封顶就失效了)"
    )


def test_room_weighting_caps_a_single_opponent():
    """只剩一家有空间时不许把额度全压给它 —— 那正是过拟合的形状。"""
    from monet.training.selfplay import League

    lg = League(["baseline", "camper", "hunter"], max_snapshots=2,
                weight_mode="room", weight_cap=0.35)
    total = sum(lg.static_w)
    lg.update_weights({"baseline": 1.0, "camper": 0.0, "hunter": 1.0})
    assert max(lg.static_w) <= 0.35 * total + 1e-9, (
        f"单个对手超过封顶:{lg.static_w}"
    )
    w = {o.name: x for o, x in zip(lg.static, lg.static_w)}
    assert w["camper"] > w["baseline"], "封顶之后camper 仍该是最多的"


def test_room_weighting_smooths_and_survives_a_missing_reading():
    """权重跟趋势不跟噪声;某次评测没测到的对手保留上次读数,不当成归零。"""
    from monet.training.selfplay import League

    lg = League(["baseline", "camper"], max_snapshots=2,
                weight_mode="room", weight_ema=0.25)
    lg.update_weights({"baseline": 0.0, "camper": 1.0})  # 第一次没得平滑,直接采用
    assert lg._wr["baseline"] == 0.0 and lg._wr["camper"] == 1.0
    lg.update_weights({"baseline": 1.0, "camper": 1.0})
    assert abs(lg._wr["baseline"] - 0.25) < 1e-9, "第二次只该走 25% 的路"
    assert abs(lg._wr["camper"] - 1.0) < 1e-9, "读数没变就不该动"

    lg.update_weights({})  # 这次评测没测它们
    assert abs(lg._wr["baseline"] - 0.25) < 1e-9, "没测到 ≠ 归零 —— 归零会被读成'已经打服了'"


def test_static_weighting_is_untouched_when_asked_for():
    """`league_weight_mode="static"` 必须逐字保留旧行为,否则没有退路。"""
    from monet.training.selfplay import League

    lg = League(["baseline", "camper", "hunter"], max_snapshots=2, weight_mode="static")
    before = list(lg.static_w)
    lg.update_weights({"baseline": 0.0, "camper": 1.0, "hunter": 0.5})
    assert lg.static_w == before, "static 模式下 update_weights 不该动权重"


def test_room_weighting_is_wired_into_training():
    """端到端:训练真的会拿评测读数去动权重(否则上面那些测的是死代码)。"""
    with tempfile.TemporaryDirectory() as td:
        cfg = _tiny_cfg(td)
        cfg.total_steps = 600
        cfg.eval_every = 2
        cfg.eval_games = 4
        trainer = SelfPlayTrainer(cfg)
        assert trainer.league.weight_mode == "room", "默认该是动态配重"
        trainer.train()
        assert trainer.league._wr, "跑完之后该有胜率读数进来了"



def test_train_csv_records_per_opponent_winrate():
    """`train_metrics.csv` 必须**逐对手**记胜率。

    只有总胜率答不了"对谁在退步":对强敌 0.92、对弱敌 0.28 也能凑出一个挺好看
    的总数。而且这些局本来就要打,记下来是免费的 —— 评测点则是 `eval_every` 个
    update 才一个,稀疏到看不出趋势。
    """
    import csv

    with tempfile.TemporaryDirectory() as td:
        cfg = _tiny_cfg(td)
        cfg.total_steps = 200
        SelfPlayTrainer(cfg).train()
        with (Path(td) / "t" / "train_metrics.csv").open("r", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows, "train_metrics.csv 应该有行"
        # 静态对手各一列 + 快照合并成的一列
        for name in ("random", "camper", "snapshot"):
            for pre in ("ep_wr_", "ep_n_"):
                assert f"{pre}{name}" in rows[0], f"缺列 {pre}{name}"
        played = sum(float(r["ep_n_random"]) + float(r["ep_n_camper"]) for r in rows)
        assert played > 0, "整轮跑下来一局都没记到,列是空的"
        # 没抽到的对手记 nan 而不是 0.0 —— 0.0 会被读成"全输了"
        for r in rows:
            for name in ("random", "camper", "snapshot"):
                if float(r[f"ep_n_{name}"]) == 0:
                    assert r[f"ep_wr_{name}"].strip().lower() == "nan", (
                        f"没抽到 {name} 时应该记 nan,记成了 {r[f'ep_wr_{name}']!r}"
                    )


def test_checkpoint_roundtrip_preserves_outputs():
    with tempfile.TemporaryDirectory() as td:
        cfg = _tiny_cfg(td)
        trainer = SelfPlayTrainer(cfg)
        net = trainer.net
        p = Path(td) / "ck.npz"
        save_checkpoint(p, net, None, {"step": 7})

        obs = np.random.default_rng(0).normal(size=428).astype(np.float32)
        before = net.logits(obs).copy()

        net2, meta, _ = load_checkpoint(p)
        assert meta["step"] == 7
        assert np.allclose(before, net2.logits(obs), atol=1e-6), "存取后前向应完全一致"


def test_init_pack_gru_to_gru_copies_the_memory_too():
    """从**带记忆的包**起训时,GRU 与 P/bP 也必须整份继承 —— 不能只继承编码器。

    这条挡的是一个不报错的错:`load_init_net` 原来无条件走 `RNNPolicy.from_mlp`,
    而 `from_mlp` 只搬 `_INHERITED` 那 16 个张量。拿 v2.0 的包当起点时它会把训好
    的 GRU/P/bP 丢掉、重新随机初始化,打印出来的话术却是"初始前向逐位等于
    rl_VG_v2.0"。表现就是"续训了但没变强",而所有形状都对得上、没有一处报错。

    用真实包跑:它同时覆盖 `load_pack_net` 解析 GRU 包这一段。
    """
    import monet.pack as P
    from monet.training.selfplay import load_init_net

    if not P.known_pack_dir("rl_VG_v2.0").is_dir():
        print("  (跳过:rl_VG_v2.0 不在 —— 没法测 gru→gru 的继承)")
        return
    cfg = replace(Config(), init_pack="rl_VG_v2.0", arch="gru", gru_hidden=128)
    net = load_init_net(cfg)
    assert getattr(net, "is_recurrent", False)

    n = net.gru_hidden
    for _ in range(12):   # 推进若干步,让隐状态离开零,这样 GRU 是否真被装载才有区别
        net.step(np.zeros(428, np.float32), np.full((1, n), 0.3, np.float32))
    src = RNNPolicy(obs_dim=428, hidden=cfg.hidden, act_dim=8, gru_hidden=n, seed=1)
    assert not np.array_equal(net.p["P"], src.p["P"]), "P 应当是从包里继承来的"
    assert np.abs(net.p["bP"]).max() > 0, "bP 应当是从包里继承来的(不是零初始化)"
    assert not np.array_equal(net.p["W_ir"], src.p["W_ir"]), "GRU 应当是从包里继承来的"

    packed = P.load_pack_net(name="rl_VG_v2.0")
    obs = np.random.default_rng(5).normal(size=428).astype(np.float32)
    h1 = np.zeros((1, n), np.float32)
    h2 = np.zeros((1, n), np.float32)
    for _ in range(20):
        a, _v1, h1 = net.step(obs, h1)
        b, _v2, h2 = packed.step(obs, h2)
    assert np.abs(a - b).max() == 0.0, "起训网络的前向应当逐位等于起点包"


def test_render_only_stamps_sentries_on_their_own_row():
    """观战棋盘:哨兵只能出现在自己所在的那一行。"""
    from monet.cli import render
    from monet.engine.game import Game

    g = Game()
    g.s["R"].pos, g.s["R"].facing = (2, 3), "E"
    g.s["B"].pos, g.s["B"].facing = (4, 5), "W"
    rows = render(g).splitlines()[1:8]  # 跳过表头,7 行棋盘
    assert len(rows) == 7
    for y, row in enumerate(rows):
        cells = row.split()[1:]  # 跳过行号
        assert len(cells) == 7
        assert (cells[2] == "R") == (y == 3), f"第 {y} 行 R 位置错了: {row}"
        assert (cells[4] == "B") == (y == 5), f"第 {y} 行 B 位置错了: {row}"
    # 障碍与得分区也要画对
    assert rows[1].split()[2] == "#", "障碍 (1,1)"
    assert rows[5].split()[6] == "#", "障碍 (5,5)"
    assert rows[3].split()[4] == "*", "得分区 (3,3)"


def test_export_header_shape_and_parse():
    with tempfile.TemporaryDirectory() as td:
        cfg = _tiny_cfg(td)
        trainer = SelfPlayTrainer(cfg)
        net = trainer.net
        header = Path(td) / "rl_weights.h"
        export_weights_header(net, header, source="unit-test")
        text = header.read_text(encoding="utf-8")
        assert "static const int kObsDim = 428;" in text
        assert "static const int kHidden = 32;" in text
        assert "static const int kActDim = 8;" in text
        # 数组大小必须与网络形状一致(C++ 侧按这个维度读)
        assert f"kW0[{32 * 428}]" in text
        assert f"kW3[{8 * 32}]" in text
        for sym in ("kW0", "kB0", "kLN1Gamma", "kLN1Beta", "kW3", "kB3", "kW4", "kB4"):
            assert f"static const float {sym}[" in text, sym


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
