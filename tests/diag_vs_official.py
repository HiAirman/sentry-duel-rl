"""诊断:训练出来的网络到底是怎么输给官方 AI 的。

只报总分看不出该改奖励还是该改联盟,所以把一局拆开:
得分 = 占点分(每回合阶段末 +1)+ 击杀分(每次 +2),两边分别数;
再把"开了几枪 / 中了几枪 / 扫了几次"单独统计 —— 分不清是"不敢开枪"
还是"开枪打不中",给出的结论会完全相反。

统计口径走引擎自己的事件(每个动作的 `res.success` 与 `ev["kill"]`),
不靠分数差分反推,避免 +2 到底是"一次击杀"还是"两次占点"这类歧义。

    python tests/diag_vs_official.py runs/m3/best.npz official_baseline,official_hunter 100
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monet.engine.game import FIRE, MOVE, Game  # noqa: E402
from monet.env import obs as O  # noqa: E402
from monet.env.obs import ObsBuilder  # noqa: E402
from monet.store import load_checkpoint  # noqa: E402
from monet.training.evaluate import NetPolicy  # noqa: E402
from monet.training.selfplay import STATIC_OPPONENTS  # noqa: E402
from tests.duel import play_side  # noqa: E402

TURN_ACTIONS = (1, 2, 3, 4)


def run(ckpt: str, opp_name: str, games: int, seed: int = 0) -> dict:
    net, _, _ = load_checkpoint(ckpt)
    policy = NetPolicy(net, deterministic=False, seed=seed)
    factory = STATIC_OPPONENTS[opp_name]

    # me/op 各自的计数;key 名字短一点,后面排版好写
    s = {k: 0 for k in (
        "win", "draw", "phases",
        "zone", "kill", "death",
        "fire_try", "fire_ok", "scan_try", "scan_ok", "move_ok", "turn_ok",
        "ozone", "okill", "odeath",
        "ofire_try", "ofire_ok", "oscan_try", "oscan_ok",
    )}

    for g in range(games):
        color = "R" if g % 2 == 0 else "B"
        mine = 0 if color == "R" else 1  # 我的分数在 game.score 里的下标
        game = Game()
        other = factory(seed=seed + g)
        ob = ObsBuilder()

        real_apply = game.apply
        # 本阶段内双方各自的击杀数:分数差里含 +2/次,算占点分时要扣掉
        phase_kills = {"me": 0, "op": 0}

        def apply(color_, a, _real=real_apply):
            res, ev = _real(color_, a)
            me = color_ == color
            p = "" if me else "o"
            if a == FIRE:
                s[p + "fire_try"] += 1
                if res.success:
                    s[p + "fire_ok"] += 1
            elif a == O.SCAN:
                s[p + "scan_try"] += 1
                if res.success:
                    s[p + "scan_ok"] += 1
            elif a == MOVE:
                if res.success and me:
                    s["move_ok"] += 1
            elif a in TURN_ACTIONS:
                if res.success and me:
                    s["turn_ok"] += 1
            if ev.get("kill"):
                # 只有开火能击杀 ⇒ 行动方必是击杀方,阵亡的是另一个人。
                s[p + "kill"] += 1
                phase_kills["me" if me else "op"] += 1
                s["odeath" if me else "death"] += 1
            return res, ev

        game.apply = apply

        while not game.done:
            if game.phase == color:
                s["phases"] += 1
                ob.act_start(game.view(color), color)
                before = game.score[mine]
                phase_kills["me"] = 0
                # 一整个行动阶段,与 SentryEnv 的步进一致:8 = max_attempts,
                # 数的是 step 调用次数(被拒的动作不消耗额度但也计入),
                # 其中最多 3 个成功动作
                for _ in range(8):
                    if game.phase_over():
                        break
                    a = policy.act(
                        ob.encode(np.zeros(O.OBS_DIM, dtype=np.float32)),
                        ob.action_mask(np.zeros(O.ACTION_DIM, dtype=np.float32)),
                    )
                    res, _ = game.apply(color, a)
                    ob.on_observation(res.observation, res.consumed, a)
                    if a == O.END:
                        break
                game.end_phase(color)
                s["zone"] += game.score[mine] - before - 2 * phase_kills["me"]
            else:
                before = game.score[1 - mine]
                phase_kills["op"] = 0
                play_side(game, game.phase, other)
                s["ozone"] += game.score[1 - mine] - before - 2 * phase_kills["op"]

        if game.winner == color:
            s["win"] += 1
        elif game.winner is None:
            s["draw"] += 1
    return s


def report(title: str, s: dict, games: int) -> None:
    mine = s["zone"] + 2 * s["kill"]
    theirs = s["ozone"] + 2 * s["okill"]
    print(f"\n=== {title}({games} 局)===")
    print(f"  胜/平      {s['win']}/{s['draw']}   得分率 {s['win']/games:.3f}")
    print(f"  总分       我 {mine:>5}   对方 {theirs:>5}   差 {mine-theirs:+}")
    print(f"  占点分     我 {s['zone']:>5}   对方 {s['ozone']:>5}   差 {s['zone']-s['ozone']:+}")
    print(f"  击杀       我 {s['kill']:>5}   对方 {s['okill']:>5}   差 {s['kill']-s['okill']:+}")
    # 注意:"成功"只表示引擎接受了这个动作(额度被消耗),不表示打中了。
    # 真正的命中看上面的击杀数 —— 开火成功但没人在火力范围内是常事。
    print(
        f"  我方/局    开火 {s['fire_try']/games:.1f}(成功 {s['fire_ok']/games:.1f})  "
        f"SCAN {s['scan_try']/games:.1f}(成功 {s['scan_ok']/games:.1f})  "
        f"移动 {s['move_ok']/games:.1f}  转向 {s['turn_ok']/games:.1f}  "
        f"占点 {s['zone']/games:.1f}  击杀 {s['kill']/games:.2f}"
    )
    print(
        f"  对方/局    开火 {s['ofire_try']/games:.1f}(成功 {s['ofire_ok']/games:.1f})  "
        f"SCAN {s['oscan_try']/games:.1f}(成功 {s['oscan_ok']/games:.1f})  "
        f"占点 {s['ozone']/games:.1f}  击杀 {s['okill']/games:.2f}"
    )
    print(f"  阵亡       我 {s['death']}   对方 {s['odeath']}")


def main() -> int:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/m3/best.npz"
    names = (
        sys.argv[2].split(",")
        if len(sys.argv) > 2
        else ["official_baseline", "official_hunter"]
    )
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    tag = f"{Path(ckpt).parent.name}/{Path(ckpt).name}"
    for n in names:
        report(f"{tag} vs {n}", run(ckpt, n, games), games)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
