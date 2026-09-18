"""命令行入口。

    python -m monet.cli train  --run-name m1 --steps 200000
    python -m monet.cli eval   --ckpt runs/m1/best.npz --games 200
    python -m monet.cli export --ckpt runs/m1/best.npz --out rl_weights_m1.h
    python -m monet.cli play   --ckpt runs/m1/best.npz --opponent hunter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine.game import ACTION_TURN, END, FIRE, MOVE, SCAN, Game
from .env.opponents import RandomOpponent
from .env.rules_vg import VGRules
from .env.sentry_env import SentryEnv
from .pack import KNOWN_PACKS, set_pack_dir
from .plot import maybe_plot, plot_run
from .store import load_checkpoint
from .training.config import Config
from .training.evaluate import NetPolicy, evaluate, play_match, summary_line
from .training.export_cpp import export_weights_header
from .training.selfplay import (
    STATIC_OPPONENTS,
    SelfPlayTrainer,
    load_init_net,
)


# --------------------------------------------------------------------- train


def cmd_train(args) -> int:
    cfg = Config(
        run_name=args.run_name,
        out_dir=args.out_dir,
        seed=args.seed,
        total_steps=args.steps,
        rollout_steps=args.rollout,
        hidden=args.hidden,
        arch=args.arch,
        gru_hidden=args.gru_hidden,
        seg_len=args.seg_len,
        segs_per_mb=args.segs_per_mb,
        lr=args.lr,
        league=[s for s in args.league.split(",") if s],
        eval_opponents=[s for s in args.eval_opponents.split(",") if s],
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        scan_reveal=args.scan_reveal,
        snapshot_min_official_winrate=args.snapshot_min_official_winrate,
        pack_dir=args.pack_dir or "",
        init_pack=args.init_pack,
        stop_opponents=[s for s in args.stop_opponents.split(",") if s],
        stop_winrate=args.stop_winrate,
        stop_confirm_games=args.stop_confirm_games,
        stop_screen_margin=args.stop_screen_margin,
        guard_official_winrate=args.guard_official_winrate,
        stop_metric_slack=args.stop_metric_slack,
        league_weight_mode=args.league_weight_mode,
        league_weight_cap=args.league_weight_cap,
        use_rules=not args.no_rules,
    )
    set_pack_dir(cfg.pack_dir or None)
    init_net, meta = None, None
    if args.resume:
        init_net, meta, _ = load_checkpoint(args.resume)
        print(f"[monet] 续训自 {args.resume}(step={meta.get('step')})")
    else:
        # 与 selfplay.train() 走同一条装载路径,免得两处各写一份而分叉。
        init_net = load_init_net(cfg)
    trainer = SelfPlayTrainer(cfg, init_net=init_net)
    trainer.train(resume_meta=meta)
    # 训练收尾自动出曲线。内部 try/except —— 画图失败只警告,绝不让一次训练跑废。
    maybe_plot(trainer.out_dir)
    return 0


# ---------------------------------------------------------------------- eval


def cmd_eval(args) -> int:
    set_pack_dir(args.pack_dir or None)
    net, meta, _ = load_checkpoint(args.ckpt)
    print(f"[monet] 载入 {args.ckpt}(step={meta.get('step')},update={meta.get('update')})")
    # 规则开着 = 复合策略(rl_VG_v1.0 的形态),关掉 = 纯网络。
    # **两种模式的数字不可直接比较**(不只是"规则有没有触发":纯网络模式下连 SCAN
    # 的屏蔽都撤了,完全是另一套策略)。所以每次评测都把当前用的是哪种印出来,
    # 免得事后拿着两列不同的数做减法。
    rules = None if args.no_rules else VGRules()
    print(f"[monet] 规则:{'关(纯网络)' if rules is None else '开(规则 + 网络,复合策略)'}")
    seed = 1000  # 写死:同一份权重跑两次要给出同一组数,种子不能随命令变
    # 认不出的名字在这里被**静默丢掉**,不报错。打错一个字不会失败,只会少测一个
    # 对手 —— 而 evaluate 的采样流是按名单顺序走的,少一个对手会让**其余每个**
    # 读数一起变(README §五)。数不对劲时先数一数上面印出来几行。
    opps = {
        n: STATIC_OPPONENTS[n](seed) for n in args.opponents.split(",") if n in STATIC_OPPONENTS
    }
    res = evaluate(
        net,
        opps,
        games=args.games,
        seed=seed,
        deterministic=args.deterministic,
        temperature=args.temperature,
        rules=rules,
    )
    for name, r in res.items():
        print("  " + summary_line(name, r))
    metric = sum(r["winrate"] for r in res.values()) / max(1, len(res))
    print(f"  综合得分率 {metric:.4f}")
    return 0


# ---------------------------------------------------------------------- plot


def cmd_plot(args) -> int:
    out = plot_run(Path(args.out_dir) / args.run, out=args.out)
    print(f"[monet] 曲线已写入 {out}")
    return 0


# -------------------------------------------------------------------- export


def cmd_export(args) -> int:
    net, meta, _ = load_checkpoint(args.ckpt)
    max_bytes = None if args.max_file_mb <= 0 else int(args.max_file_mb * 1024 * 1024)
    out = export_weights_header(
        net, args.out, source=args.source or str(args.ckpt), max_bytes=max_bytes
    )
    files = [out] + sorted(out.parent.glob(f"{out.stem}_part*.h"))
    print(f"[monet] 已导出 {out}:")
    total = 0.0
    for f in files:
        mb = f.stat().st_size / 1024 / 1024
        total += mb
        print(f"         {f.name:<26} {mb:6.2f} MB")
    print(f"         {'合计':<24} {total:6.2f} MB,{len(files)} 个文件")
    if len(files) > 1:
        print(f"         (超单文件上限,已按张量无损拆分;{out.name} 负责 include 各分片)")
    print("        用法:把它们一起放进 AI 源码包,重新编译即可。")
    return 0


# ---------------------------------------------------------------------- play


ARROW = {"N": "^", "E": ">", "S": "v", "W": "<"}
ACTION_NAME = {0: "MOVE", 5: "FIRE", 6: "SCAN", 7: "END", **{k: f"TURN {v}" for k, v in ACTION_TURN.items()}}


def render(game: Game) -> str:
    """绝对坐标下渲染棋盘(观战用)。"""
    lines = ["   " + " ".join(str(x) for x in range(7))]
    for y in range(7):
        cells = []
        for x in range(7):
            ch = "*" if (x, y) in game.score_zones else "."
            if (x, y) in game.obstacles:
                ch = "#"
            cells.append(ch)
        for color in ("R", "B"):
            p = game.s[color].pos
            if p[1] == y:  # 只盖在哨兵所在的那一行
                cells[p[0]] = color
        lines.append(f" {y} " + " ".join(cells))
    r, b = game.score
    lines.append(
        f"   比分 R {r} : {b} B   回合 {game.turn}   "
        f"R {game.s['R'].pos}{ARROW[game.s['R'].facing]}   "
        f"B {game.s['B'].pos}{ARROW[game.s['B'].facing]}"
    )
    return "\n".join(lines)


REJECT_NAME = {
    "budget": "额度用尽",
    "move": "出界/障碍/被挡",
    "fire_cd": "开火冷却中",
    "scan_cd": "扫描冷却中",
}


def _indent(text: str, pad: str = "      ") -> str:
    return "\n".join(pad + ln for ln in text.splitlines())


class Replay:
    """逐动作回放:把引擎里的每一次行动变成一行。

    双方的动作都流经 `game.apply` —— 我方在 `env.step` 里,对手在
    `env._run_opponent_phase` 里逐个交出 —— 占点结算走 `game.end_phase`。
    在这两个入口上包一层就能拿到完整序列,引擎和环境都不用改。

    `env.reset()` 会新建一个 `Game`(并且可能先让对手走完开场阶段),
    所以这里连 `_reset_state` 一起接住,保证重开一局后回放不断线。

    棋盘只在**状态真的变了**的行动之后打印:失败的行动(被拒、空枪)棋盘
    和上一张一模一样,重复贴是纯噪声。要看每一张棋盘就按回合模式跑。
    """

    def __init__(self, env: SentryEnv, agent_color: str):
        self.env = env
        self.agent = agent_color
        self._phase = None
        self._n = 0
        self._orig_reset = env._reset_state
        env._reset_state = self._reset_state
        self._wrap(env.game)

    # ---- 包装 ----

    def _reset_state(self) -> None:
        self._orig_reset()
        self._wrap(self.env.game)

    def _wrap(self, game: Game) -> None:
        self.game = game
        self._orig_apply = game.apply
        self._orig_end_phase = game.end_phase
        game.apply = self._apply
        game.end_phase = self._end_phase

    # ---- 输出 ----

    def _who(self, color: str) -> str:
        return "我方" if color == self.agent else "敌方"

    def _state(self):
        g = self.game
        return (g.s["R"].pos, g.s["B"].pos, g.s["R"].facing, g.s["B"].facing, g.score)

    def _header(self, color: str) -> None:
        key = (self.game.turn, color)
        if key != self._phase:
            self._phase, self._n = key, 0
            print(f"\n── 回合 {self.game.turn} · {color} 方({self._who(color)})──")

    def _describe(self, color, action, res, ev, p0) -> str:
        g = self.game
        me = g.s[color]
        if action == END:
            body = "结束本阶段"
        elif action == MOVE:
            body = f"移动 {me.facing}"
            if res.success:
                body += f"  {p0} -> {me.pos}"
                if g.in_zone(me.pos):
                    body += "  [进入得分区]"
        elif action in ACTION_TURN:
            body = f"转向 {me.facing}"
        elif action == FIRE:
            if ev.get("kill"):
                body = f"开火 命中!{g.opp(color)} 方阵亡 +2"
            else:
                body = "开火 未命中"
        elif action == SCAN:
            body = "扫描"
        else:
            body = f"动作 {action}"
        if not res.success:
            why = ev.get("rejected")
            body += f"  被拒({REJECT_NAME.get(why, why)})"
        return f"{self._n:>2}. {self._who(color)} {body}"

    def _apply(self, color, action):
        self._header(color)
        g = self.game
        s0 = g.score
        p0 = g.s[color].pos
        before = self._state()
        res, ev = self._orig_apply(color, action)
        self._n += 1
        line = self._describe(color, action, res, ev, p0)
        if g.score != s0:
            line += f"   比分 R {g.score[0]} : {g.score[1]} B"
        print("  " + line)
        if self._state() != before:
            print(_indent(render(g)))
        return res, ev

    def note_agent_end(self) -> None:
        """我方的 END 不流经 `game.apply`(`env.step` 对它直接跳过 apply),
        所以由调用方在 step 之前告诉回放一声,补一行 —— 否则回放里会凭空
        少掉一个"主动收手"的决策,甚至整个阶段都不出现。
        """
        self._header(self.agent)
        self._n += 1
        print(f"  {self._n:>2}. 我方 结束本阶段")

    def _end_phase(self, color) -> None:
        # 每个阶段都要有收尾行:对手可能整段一个动作都不出(official.py 里
        # `if holding: return`),没有这一行这种阶段在回放里会凭空消失。
        self._header(color)
        before = self.game.score
        self._orig_end_phase(color)
        after = self.game.score
        gain = f"{color} 方在得分区 +1" if after != before else f"{color} 方未得分"
        idle = "(本阶段没有任何行动)" if self._n == 0 else ""
        print(f"  ── 阶段结束:{gain}{idle} → 比分 R {after[0]} : {after[1]} B")


def cmd_play(args) -> int:
    set_pack_dir(args.pack_dir or None)
    net, meta, _ = load_checkpoint(args.ckpt)
    rules = None if args.no_rules else VGRules()
    policy = NetPolicy(net, temperature=args.temperature, seed=args.seed, rules=rules)
    print(f"[monet] 规则:{'关(纯网络)' if rules is None else '开(规则 + 网络,复合策略)'}")
    # 名字认不出时**静默退化成随机对手**,不报错 —— 观战时会看到一场轻松的大胜,
    # 还以为是网络变强了。名字写错要在这里自己认出来。
    opp = STATIC_OPPONENTS.get(args.opponent, lambda s: RandomOpponent(seed=s))(args.seed)
    env = SentryEnv(opp, agent_color=args.color, seed=args.seed)
    # 包装必须赶在 reset 之前,否则开局由对手先走的那半个回合会被漏掉
    replay = Replay(env, args.color) if args.verbose else None
    obs, mask, info = env.reset()
    policy.reset()  # 与 env.reset 配对:失明计数是逐局的
    turns_seen = -1
    while True:
        action = policy.act(obs, mask, env.ag_ob)
        if replay is not None and action == END:
            replay.note_agent_end()
        obs, mask, r, term, trunc, info = env.step(action)
        if not args.verbose and env.game.turn != turns_seen:
            turns_seen = env.game.turn
            if args.by_turn:
                print(render(env.game))
                print(f"    -> 我方({args.color}) 行动 {ACTION_NAME.get(action, '?')}\n")
        if term or trunc:
            break
    print(render(env.game))
    w = env.game.winner
    verdict = "平局" if w is None else f"{w} 方胜"
    print(f"\n[monet] 对 {args.opponent}:{verdict}(我方 {args.color},得分率 {env.result():.2f})")
    return 0


# ----------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="monet", description="rl_monet_v1 — 哨兵大战 RL 训练引擎")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="自对弈 PPO 训练")
    # 下面每个 flag 的默认值**一律从 Config 取**,不要在这里写字面量。
    # 两处各写一份 = 训练配置有两个真相,而且 argparse 那个每次都赢:在
    # training/config.py 里改的默认值会被这里的副本静默盖掉,不报错、不提示。
    # 要改默认值就改 monet/training/config.py。
    d = Config()
    t.add_argument("--run-name", default=d.run_name)
    t.add_argument("--out-dir", default=d.out_dir)
    t.add_argument("--steps", type=int, default=d.total_steps)
    t.add_argument("--rollout", type=int, default=d.rollout_steps)
    t.add_argument("--hidden", type=int, default=d.hidden)
    t.add_argument(
        "--arch",
        choices=["mlp", "gru"],
        default=d.arch,
        help="mlp=无记忆(rl_VG_v1.0);gru=编码器+记忆层,走连续片段的截断 BPTT",
    )
    t.add_argument("--gru-hidden", type=int, default=d.gru_hidden,
                   help="记忆层隐状态维度(仅 --arch gru)")
    t.add_argument("--seg-len", type=int, default=d.seg_len,
                   help="截断 BPTT 的段长(决策步数,仅 --arch gru)")
    t.add_argument("--segs-per-mb", type=int, default=d.segs_per_mb,
                   help="每个 minibatch 装几个段(仅 --arch gru)")
    t.add_argument("--lr", type=float, default=d.lr)
    t.add_argument("--seed", type=int, default=d.seed)
    t.add_argument(
        "--league",
        default=",".join(d.league),
        help="逗号分隔;可选 " + "/".join(sorted(STATIC_OPPONENTS)),
    )
    # 评测名单**与联盟分开**是有意的:两份默认名单都是人工维护的口径(评测集默认比
    # 联盟多两个 m3 导出),不该为了做一次实验去改默认值。有了这个开关,"拿某个对手
    # 陪练并且**量出来**"只靠命令行就能完成 —— 否则陪练进去了、评测表里没有那一列,
    # 等于白练。
    t.add_argument(
        "--eval-opponents",
        default=",".join(d.eval_opponents),
        help="逗号分隔;训练期间评测哪些对手(默认同 eval 子命令)",
    )
    t.add_argument(
        "--scan-reveal",
        type=float,
        default=d.scan_reveal,
        help="SCAN 扫出新敌人的奖励(0=关;试 0.1)。只影响训练,会破坏回报=比分差的契约",
    )
    t.add_argument(
        "--snapshot-min-official-winrate",
        type=float,
        default=d.snapshot_min_official_winrate,
        help="官方 AI 平均得分率超过它才加入快照对手;0=不设闸门",
    )
    t.add_argument("--eval-every", type=int, default=d.eval_every)
    t.add_argument("--eval-games", type=int, default=d.eval_games)
    t.add_argument("--resume", default=None)
    t.add_argument(
        "--stop-opponents",
        default=",".join(d.stop_opponents),
        help="早停:评测里对这些对手**全部**到线才收工(逗号分隔;空=不早停)。"
             "全部而非任一 —— 任一达标的时刻往往正是策略最偏科的时候",
    )
    t.add_argument(
        "--stop-winrate",
        type=float,
        default=d.stop_winrate,
        help="早停阈值(0=不早停)。判据用的是 "
             "--stop-confirm-games 那次独立测量,不是 eval_games 那次",
    )
    t.add_argument(
        "--stop-confirm-games",
        type=int,
        default=d.stop_confirm_games,
        help="早停的**决定**局数:另起这么多局专门测 stop-opponents,拿它判。"
             "0=退回用 eval-games 那次的读数(噪声大,不推荐)。"
             "60 局的得分率标准差约 ±0.065,阈值落在噪声带里时真假两个方向都会错",
    )
    t.add_argument(
        "--stop-screen-margin",
        type=float,
        default=d.stop_screen_margin,
        help="便宜筛子:eval-games 那次的读数只要有对手低于 (阈值-它) 就跳过那次"
             "高局数测量。只能筛掉明显没到的,所以给得比噪声宽(默认 0.10)",
    )
    t.add_argument(
        "--guard-official-winrate",
        type=float,
        default=d.guard_official_winrate,
        help="官方保护闸:早停还要求官方档**每一个**都 ≥ 它。挡的是'拿官方换偏科'。"
             "0=关",
    )
    t.add_argument(
        "--stop-metric-slack",
        type=float,
        default=d.stop_metric_slack,
        help="早停还要求当前综合得分率 ≥ 本轮最优-它。挡的是在局部最优尖峰上收工",
    )
    t.add_argument(
        "--league-weight-mode",
        choices=["static", "room"],
        default=d.league_weight_mode,
        help="静态对手配重:static=STATIC_WEIGHTS 定值;"
             "room=按剩余空间(1-胜率)动态配重,谁没打服谁多拿样本",
    )
    t.add_argument(
        "--league-weight-cap",
        type=float,
        default=d.league_weight_cap,
        help="room 模式下单个对手最多占静态额度的比例。不封顶会把额度全压到"
             "唯一还有空间的那家头上,那正是过拟合的形状",
    )
    t.add_argument(
        "--no-rules",
        action="store_true",
        help="训练时不挂规则层(纯网络对照)。注意连'网络不许扫'也一并撤掉,"
        "所以它测的是另一套策略,不能和带规则的 run 逐点比较",
    )
    t.add_argument(
        "--init-pack",
        default=d.init_pack,
        help="起训权重来自哪个参赛包(空=随机初始化);可选 "
        + "/".join(sorted(KNOWN_PACKS)),
    )
    t.add_argument(
        "--pack-dir",
        default="",
        help="外部参赛包目录(内含 rl_weights.h);空=按 MONET_PACK_DIR/默认位置解析",
    )
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="与规则手 / 官方 AI 移植版对战评测")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--games", type=int, default=100)
    e.add_argument(
        "--opponents",
        default=",".join(d.eval_opponents),  # 同上:默认值只认 Config
        help="逗号分隔;可选 " + "/".join(sorted(STATIC_OPPONENTS)),
    )
    e.add_argument("--deterministic", action="store_true")
    e.add_argument("--temperature", type=float, default=1.0)
    e.add_argument("--pack-dir", default="", help="外部参赛包目录(内含 rl_weights.h)")
    e.add_argument(
        "--no-rules",
        action="store_true",
        help="关掉规则,只测纯网络(与开着规则的得分率不可直接比较)",
    )
    e.set_defaults(func=cmd_eval)

    g = sub.add_parser("plot", help="画训练曲线(runs/<名字>/curves.png)")
    g.add_argument("--run", required=True, help="run 名字(runs/ 下的子目录名)")
    g.add_argument("--out-dir", default=d.out_dir)
    g.add_argument("--out", default="", help="输出 PNG 路径;空=runs/<名字>/curves.png")
    g.set_defaults(func=cmd_plot)

    x = sub.add_parser("export", help="导出 C++ 权重头文件")
    x.add_argument("--ckpt", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--source", default="")
    x.add_argument(
        "--max-file-mb",
        type=float,
        default=7.0,
        help="单文件上限(MB),超过则按张量拆分;<=0 表示不拆(评测服务限制 8MB)",
    )
    x.set_defaults(func=cmd_export)

    pl = sub.add_parser("play", help="观战一局(ASCII)")
    pl.add_argument("--ckpt", required=True)
    pl.add_argument("--opponent", default="hunter")
    pl.add_argument("--color", default="R", choices=["R", "B"])
    pl.add_argument("--temperature", type=float, default=1.0)
    pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--pack-dir", default="", help="外部参赛包目录(内含 rl_weights.h)")
    pl.add_argument(
        "--no-rules",
        action="store_true",
        help="关掉规则,只看纯网络怎么下(rl_VG_v1.0 默认开着规则打)",
    )
    pl.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="逐动作回放:每个行动一行(双方都算),棋盘只在状态变化后打印",
    )
    pl.add_argument(
        "--by-turn",
        action="store_true",
        help="逐回合打印棋盘(旧的观战输出);与 -v 同时给时以 -v 为准",
    )
    pl.set_defaults(func=cmd_play)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
