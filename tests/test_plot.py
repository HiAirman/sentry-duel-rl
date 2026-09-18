"""训练曲线与逐 update 记录的冒烟测试。

绘图最容易出的岔子不是"画错",而是**在收尾那一步把训练搞崩**:缺列、旧 run 没有
`train_metrics.csv`、无显示环境下后端连不上。所以这里两条主线:

1. 有数据时 PNG 真的生成且非空;
2. 只有评测行(旧 run)时**优雅降级**,不抛异常。

顺带钉住"逐 update 记录真的落盘了且列齐全"——`forced_frac` 那一列是排障用的,
规则静默失效时只有它会说话。
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from monet import plot as P
from monet.training.config import Config
from monet.training.selfplay import SelfPlayTrainer


def _tiny_cfg(td: str, **kw) -> Config:
    base = dict(
        run_name="plottest",
        out_dir=td,
        total_steps=512,
        rollout_steps=256,
        hidden=32,
        epochs=1,
        minibatches=2,
        seed=0,
        league=["random", "camper"],
        eval_opponents=["baseline", "official_baseline"],
        eval_every=2,
        eval_games=6,
        snapshot_every=10 ** 9,
        save_every=10 ** 9,
        progress_every_steps=0,
    )
    base.update(kw)
    return Config(**base)


def test_train_writes_per_update_rows_with_forced_frac():
    """逐 update 的 CSV 必须真的落盘,且带着 `ppo_forced_frac`。

    这一列是规则钩子的脉搏:它恒为 0 就说明规则在训练里静默失效了 ——
    而那正是本仓库最防的"测试全绿但没测"。所以不能只断言文件存在。
    """
    with tempfile.TemporaryDirectory() as td:
        tr = SelfPlayTrainer(_tiny_cfg(td))
        tr.train()
        p = Path(td) / "plottest" / "train_metrics.csv"
        assert p.exists(), "没有 train_metrics.csv"
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        assert len(rows) >= 2, f"只落了 {len(rows)} 行(每个 update 应有一行)"
        need = {
            "update", "step", "ep_return", "ep_win", "episodes", "sps",
            "ppo_policy_loss", "ppo_value_loss", "ppo_entropy", "ppo_forced_frac",
        }
        assert need <= set(rows[0]), f"缺列:{sorted(need - set(rows[0]))}"
        # 步数必须单调递增,否则曲线会自己绕回去
        steps = [int(r["step"]) for r in rows]
        assert steps == sorted(steps) and len(set(steps)) == len(steps)
        assert any(float(r["ppo_forced_frac"]) > 0 for r in rows), (
            "整轮 forced_frac 全是 0 —— 规则没在训练里触发"
        )


def test_fresh_run_truncates_a_previous_train_metrics_csv():
    """从头跑必须丢掉上一轮的逐 update 行,否则曲线会把两次训练连成一条。

    续训(`resume_meta` 非空)则要**追加** —— 那是同一条曲线的下半段。
    """
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "plottest"
        run_dir.mkdir(parents=True)
        csv_path = run_dir / "train_metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["update", "step", "ep_return"])
            w.writeheader()
            w.writerow({"update": 999, "step": 999999, "ep_return": -42.0})

        SelfPlayTrainer(_tiny_cfg(td)).train()
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        assert all(int(r["step"]) < 999999 for r in rows), "上一轮的残留行没被清掉"


def test_plot_smoke_on_a_run_with_both_tables():
    with tempfile.TemporaryDirectory() as td:
        tr = SelfPlayTrainer(_tiny_cfg(td))
        tr.train()
        run_dir = Path(td) / "plottest"
        out = P.plot_run(run_dir)
        assert out == run_dir / "curves.png"
        assert out.exists() and out.stat().st_size > 2000, "PNG 太小,多半是空图"


def test_plot_degrades_gracefully_without_train_metrics():
    """旧 run 只有 `metrics.csv`、没有 `train_metrics.csv` 时的降级路径。

    评测那个面板能画,其余三个面板画一行说明文字 —— 关键是**不抛异常**,
    因为这条路径会在训练收尾被调用,抛了就等于把一次训练跑废。
    """
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "oldrun"
        run_dir.mkdir(parents=True)
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["update", "step", "eval_baseline_winrate", "eval_official_baseline_winrate"],
            )
            w.writeheader()
            for u in (1, 2):
                w.writerow({
                    "update": u, "step": u * 4096,
                    "eval_baseline_winrate": 0.5 + 0.1 * u,
                    "eval_official_baseline_winrate": 0.4 + 0.05 * u,
                })
        out = P.plot_run(run_dir)
        assert out.exists() and out.stat().st_size > 2000


def test_maybe_plot_swallows_errors():
    """收尾钩子对任何异常都只能警告 —— 它挡在 `train()` 返回之前。"""
    with tempfile.TemporaryDirectory() as td:
        P.maybe_plot(Path(td) / "根本不存在的 run")  # 不抛 = 通过


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
