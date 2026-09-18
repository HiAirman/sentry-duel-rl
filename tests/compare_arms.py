"""两条臂的对照图:规则 ON vs 纯网络 OFF。

**为什么单独有一个脚本。** `monet/plot.py` 画的是**一条** run 的四个面板,而下面这些
问题只有把两条臂画在同一根 x 轴上才答得出来:

* 规则到底吃掉了多少棋力?(`rl_v5` 那一格的差)
* 训练能不能把这个差补回来?(差的**趋势**,不是某一次评测的值)
* 补不回来的话,差距是收敛的还是一直那么大?

所以这里复用 `plot.py` 的读取/平滑/字体辅助,只画对照需要的四格。

用法:
    python tests/compare_arms.py "规则ON=runs/vg10" "对照OFF=.probe/vg10_norules" \
        [--out compare_arms.png] [--target 0.80]

注意 `metrics.csv` 里的 `*_winrate` 是**排行榜口径的得分率** `(win + 0.5*draw)/n`
(`monet/training/evaluate.py` 里的 `winrate`),不是纯胜率。两条臂用的都是同一个
口径,所以差值可比。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from monet.plot import (  # noqa: E402
    _pick_cjk_font,
    _panel_title,
    _read_csv,
    _smooth,
    load_gate,
)

# 起跑线,也是后续漂移的对照基线:起训权重(`rl_VG_v0.2`)在 100 局 `rl_v5` 上的
# 实测得分率,由 tests/diag_vg_rules.py 量出。画成水平虚线 —— 不看起点就看不出涨了多少。
V02_REF = {"rules_on": 0.290, "rules_off": 0.510}


def _parse_arm(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"臂的写法是 标签=目录,收到的是 {spec!r}")
    label, _, d = spec.partition("=")
    return label.strip(), Path(d.strip())


def _series(cols: Dict[str, List[float]], key: str) -> np.ndarray:
    return np.asarray(cols.get(key, []), dtype=float)


def _score(arm) -> Optional[float]:
    """臂的最终得分率。取最后一处有限值 —— 早停的那条可能比另一条短。"""
    y = _series(arm["ev"], "eval_rl_v5_winrate")
    ok = np.isfinite(y)
    return float(y[ok][-1]) if ok.any() else None


def _plot_curve(ax, arms, key: str, title: str, target: Optional[float], zh: bool,
                refs: Optional[Dict[str, float]] = None, smooth: int = 1,
                target_label: str = "目标") -> None:
    """一格里画所有臂的同一条曲线。缺列的臂跳过,不报错。

    `target_label` 要如实写:0.80 是**用户要打到的目标**,而 0.50 是**快照闸门**
    (`snapshot_min_official_winrate`)——两个都是横线,但混成同一个词会让人以为
    官方档那格也在追 0.50 的胜率。
    """
    drew = False
    for arm in arms:
        x = _series(arm["ev"], "step")
        y = _series(arm["ev"], key)
        if x.size == 0 or y.size != x.size:
            continue
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            continue
        xs, ys = x[ok], _smooth(y[ok], smooth)
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.6, label=arm["label"], alpha=0.9)
        drew = True
    if target is not None:
        ax.axhline(target, color="crimson", ls="--", lw=1.1, alpha=0.8,
                   label=f"{target_label} {target:.2f}")
    if refs:
        for name, v in refs.items():
            ax.axhline(v, color="gray", ls=":", lw=1.0, alpha=0.7)
            ax.annotate(f"{name} 起点 {v:.3f}", (0.01, v), xycoords=("axes fraction", "data"),
                        fontsize=7, color="gray", va="bottom")
    if not drew:
        ax.text(0.5, 0.5, "无评测记录" if zh else "no eval rows",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, alpha=0.6)
    _panel_title(ax, title, "" if zh else title)
    ax.set_xlabel("step", fontsize=8)
    ax.set_ylabel("得分率", fontsize=8)
    if drew:
        ax.legend(fontsize=7)


def _plot_gap(ax, arms, zh: bool) -> None:
    """规则代价:同一 step 上 规则ON − 对照OFF。负 = 规则拖后腿。

    两条臂的评测点在同一条 step 序列上(同 seed、同 eval_every),所以按 step 直接对齐;
    对不齐的 step 取交集,不插值 —— 插值会凭空造出中间点,把噪声抹成趋势。
    """
    if len(arms) < 2:
        return
    a, b = arms[0], arms[1]
    xa, ya = _series(a["ev"], "step"), _series(a["ev"], "eval_rl_v5_winrate")
    xb, yb = _series(b["ev"], "step"), _series(b["ev"], "eval_rl_v5_winrate")
    common = sorted(set(xa[np.isfinite(xa)]).intersection(xb[np.isfinite(xb)]))
    xs, ds = [], []
    for s in common:
        va = ya[np.where(xa == s)[0][0]]
        vb = yb[np.where(xb == s)[0][0]]
        if np.isfinite(va) and np.isfinite(vb):
            xs.append(s)
            ds.append(va - vb)
    if not xs:
        ax.text(0.5, 0.5, "两条臂没有共同评测点" if zh else "no common evals",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, alpha=0.6)
        _panel_title(ax, f"规则代价({a['label']} − {b['label']})")
        return
    xs_a, ds_a = np.asarray(xs, float), np.asarray(ds, float)
    ax.axhline(0.0, color="black", lw=0.9)
    ax.plot(xs_a, ds_a, marker="o", ms=3.5, lw=1.4, color="darkorange", alpha=0.55,
            label="逐次")
    if len(xs_a) >= 3:
        ax.plot(xs_a, _smooth(ds_a, 3), lw=2.0, color="darkorange", label="滑窗 3")
    ax.fill_between(xs_a, 0, ds_a, where=(ds_a < 0), color="crimson", alpha=0.12)
    _panel_title(ax, f"规则代价({a['label']} − {b['label']})",
                 "负 = 规则拖后腿;看趋势,不看单点" if zh else "")
    ax.set_xlabel("step", fontsize=8)
    ax.set_ylabel("得分率之差", fontsize=8)
    ax.legend(fontsize=7)


def _plot_summary(ax, arms, zh: bool, target: Optional[float],
                  fam: str = "monospace") -> None:
    # `fam` 由调用方给:定宽字体(DejaVu Sans Mono)没有汉字字形,直接用会在
    # 汇总格里画出一排豆腐块 —— 只在没挑到 CJK 字体时才退回定宽。
    ax.axis("off")
    lines = []
    for arm in arms:
        s = _score(arm)
        y = _series(arm["ev"], "eval_rl_v5_winrate")
        ok = np.isfinite(y)
        best = float(y[ok].max()) if ok.any() else float("nan")
        steps = _series(arm["ev"], "step")
        sok = np.isfinite(steps)
        last = int(steps[sok][-1]) if sok.any() else 0
        lines.append(f"{arm['label']}")
        lines.append(f"    末次 {('%.3f' % s) if s is not None else '—'}"
                     f"   最佳 {best:.3f}   step {last:,}"
                     f"   评测 {int(ok.sum())} 次")
    if len(arms) >= 2:
        sa, sb = _score(arms[0]), _score(arms[1])
        if sa is not None and sb is not None:
            lines.append("")
            lines.append(f"末次规则代价 = {sa - sb:+.3f}")
    if target is not None:
        lines.append("")
        hit = [a["label"] for a in arms
               if (_score(a) is not None and _score(a) >= target)]
        lines.append(f"达到 {target:.2f} 的臂:{'、'.join(hit) if hit else '无'}")
    ax.text(0.02, 0.96, "\n".join(lines), transform=ax.transAxes, fontsize=9,
            va="top", family=fam)
    _panel_title(ax, "汇总")


def compare(arms: Sequence[Tuple[str, Path]], out: Path,
            target: Optional[float] = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zh = _pick_cjk_font()
    loaded = []
    for label, d in arms:
        if not d.is_dir():
            raise FileNotFoundError(f"没有这个 run 目录:{d}")
        loaded.append({"label": label, "dir": d, "ev": _read_csv(d / "metrics.csv")})

    gate = load_gate(loaded[0]["dir"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    _plot_curve(axes[0][0], loaded, "eval_rl_v5_winrate", "对 rl_v5 得分率(主判据)",
                target, zh, refs={"规则ON": V02_REF["rules_on"],
                                  "规则OFF": V02_REF["rules_off"]})
    _plot_curve(axes[0][1], loaded, "eval_metric", "官方档平均得分率",
                gate, zh, target_label="快照闸门")
    _plot_gap(axes[1][0], loaded, zh)
    fam = plt.rcParams["font.sans-serif"][0] if zh else "monospace"
    _plot_summary(axes[1][1], loaded, zh, target, fam)

    fig.suptitle("规则 ON vs 对照 OFF — " + " / ".join(a["label"] for a in loaded),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="标签=目录,第一条是规则 ON,第二条是 OFF")
    ap.add_argument("--out", default="compare_arms.png")
    ap.add_argument("--target", type=float, default=0.80)
    a = ap.parse_args()
    arms = [_parse_arm(s) for s in a.arms]
    p = compare(arms, Path(a.out), a.target)
    print(f"[compare] 已写入 {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
