"""训练曲线:把 `runs/<名字>/` 下的两张 CSV 画成一张四格 PNG。

## 两张表各司其职

- `metrics.csv` —— **评测**记录,每个 `eval_every` 一行。对手得分率只在这里。
- `train_metrics.csv` —— **每个 update** 一行:训练局回报/胜率 + PPO 统计。

分开是因为 `metrics.csv` 是对外可比的评测记录(字段名是对外接口,别改),不该往里
掺诊断列;而评测点相对总步数很稀(默认几十个 update 才一次),单靠它画不出训练过程。
画图时两表按 `step` 对齐。

## 缺列不报错

旧 run 可能没有 `train_metrics.csv`,新 run 可能还没跑到一次评测,某些列也会随配置
不同而不存在。绘图是**收尾的锦上添花**,不该让一次训练跑废在最后一行 —— 所以每个
面板先检查列在不在,缺就画一行字说明,不抛异常。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# 必须在 import pyplot **之前**选后端:训练可能在无显示的环境里跑完(甚至被
# 重定向到日志文件),默认后端会去连 X/Quartz,连不上就是一句没头没尾的报错。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 中文标签要用系统中文字体,否则 matplotlib 会把每个字画成方框(还不报错)。
# 按平台常见字体依次试,一个都没有就退回英文标签 —— 宁可换语言,不要满屏豆腐块。
_CJK_CANDIDATES = (
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
    "PingFang SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
)


def _pick_cjk_font() -> bool:
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in have:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def _read_csv(path: Path) -> Dict[str, List[float]]:
    """CSV → 列名到浮点列表。空/非数值单元格按缺失处理(跳过该点)。

    列的并集可能随行变化(`metrics.csv` 是重写整张表,字段集合能变),
    所以按并集收集,缺的补 nan 而不是错位。
    """
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    cols: Dict[str, List[float]] = {k: [] for r in rows for k in r}
    for r in rows:
        for k in cols:
            v = r.get(k)
            try:
                cols[k].append(float(v))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                cols[k].append(float("nan"))
    return cols


def _finite(*arrays: np.ndarray) -> Optional[np.ndarray]:
    """所有输入都有限的掩码 —— 画线与平滑都要先过它,否则 nan 会把整条线抹掉。"""
    ok = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        ok &= np.isfinite(a)
    return ok if ok.any() else None


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    """居中滑动平均,边沿用可用窗口(不补零 —— 补零会把首尾拉向 0,像退步)。"""
    if window <= 1 or len(y) <= 1:
        return y.copy()
    out = np.empty_like(y)
    half = window // 2
    for i in range(len(y)):
        lo, hi = max(0, i - half), min(len(y), i + half + 1)
        out[i] = float(np.mean(y[lo:hi]))
    return out


def _panel_title(ax, title: str, note: str = "") -> None:
    ax.set_title(title + (f"\n{note}" if note else ""), fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)


def _plot_eval(ax, ev: Dict[str, List[float]], gate: Optional[float], zh: bool) -> None:
    """面板 1:各对手评测得分率 + 官方均值 + 快照闸门虚线。"""
    x = np.asarray(ev.get("step", []), dtype=float)
    opp_cols = sorted(k for k in ev if k.startswith("eval_") and k.endswith("_winrate"))
    if x.size == 0 or not opp_cols:
        ax.text(0.5, 0.5, "无评测记录" if zh else "no eval rows",
                ha="center", va="center", fontsize=9, alpha=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        return
    for c in opp_cols:
        name = c[len("eval_"):-len("_winrate")]
        y = np.asarray(ev[c], dtype=float)
        ax.plot(x, y, marker="o", ms=3, lw=1.2, label=name)

    # 官方档均值:按列名前缀认。列名形如 `eval_<对手名>_winrate`,所以剥掉 `eval_`
    # 之后以 `official_` 开头的那些就是官方档 —— 靠命名约定,不必把 selfplay 里的
    # 目标对手表 import 进来(那张表会随训练配置变,画图不该跟着变)。
    off = [c for c in opp_cols if c[len("eval_"):].startswith("official_")]
    if off:
        m = np.nanmean(np.vstack([[float(v) for v in ev[c]] for c in off]), axis=0)
        ax.plot(x, m, color="k", lw=2.0, ls="-", label="official mean" if not zh else "官方均值")
    if gate is not None and gate > 0:
        ax.axhline(gate, color="crimson", ls="--", lw=1.2,
                   label=f"gate {gate:.2f}")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("step", fontsize=8)
    ax.set_ylabel("score rate", fontsize=8)
    _panel_title(ax, "各对手评测得分率" if zh else "eval score rate")
    ax.legend(fontsize=6, ncol=2, loc="lower right")


def _plot_train(ax, tm: Dict[str, List[float]], zh: bool) -> None:
    """面板 2:训练局回报 / 胜率(淡色原始线 + 滑动平均)。"""
    x = np.asarray(tm.get("step", []), dtype=float)
    series = [c for c in ("ep_return", "ep_win") if c in tm]
    if x.size == 0 or not series:
        ax.text(0.5, 0.5, "无逐 update 记录" if zh else "no per-update rows",
                ha="center", va="center", fontsize=9, alpha=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        return
    win = max(5, len(x) // 30)  # 平滑窗:取点数的 1/30,下限 5(点太少时窗要够小才看得出波形)
    for c in series:
        y = np.asarray(tm[c], dtype=float)
        ok = _finite(y)
        if ok is None:
            continue
        ax.plot(x[ok], y[ok], lw=0.6, alpha=0.25, color=f"C{len(ax.lines)}")
        ax.plot(x[ok], _smooth(y[ok], win), lw=1.6, label=f"{c} (ma{win})")
    ax.set_xlabel("step", fontsize=8)
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    _panel_title(ax, "训练局回报 / 胜率" if zh else "train return / win",
                 f"滑窗 {win}" if zh else f"window {win}")
    ax.legend(fontsize=7)


def _plot_losses(ax, tm: Dict[str, List[float]], zh: bool) -> None:
    """面板 3:PPO 损失(策略 / 价值),淡色原始线 + 滑动平均。"""
    x = np.asarray(tm.get("step", []), dtype=float)
    cols = [c for c in ("ppo_policy_loss", "ppo_value_loss") if c in tm]
    if x.size == 0 or not cols:
        ax.text(0.5, 0.5, "无 PPO 统计" if zh else "no PPO stats",
                ha="center", va="center", fontsize=9, alpha=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        return
    for c in cols:
        y = np.asarray(tm[c], dtype=float)
        ok = _finite(y)
        if ok is None:
            continue
        ax.plot(x[ok], y[ok], lw=0.7, alpha=0.3, color=f"C{len(ax.lines)}")
        ax.plot(x[ok], _smooth(y[ok], max(5, len(x) // 30)), lw=1.5, label=c)
    ax.set_xlabel("step", fontsize=8)
    _panel_title(ax, "PPO 损失" if zh else "PPO losses")
    ax.legend(fontsize=7)


def _plot_health(ax, tm: Dict[str, List[float]], zh: bool) -> None:
    """面板 4:PPO 健康度。量纲差得多(熵 ~1、KL ~1e-3),所以用双轴:
    左轴熵,右轴 approx_kl / clip_frac / 规则强制步占比。"""
    x = np.asarray(tm.get("step", []), dtype=float)
    if x.size == 0:
        ax.text(0.5, 0.5, "无 PPO 统计" if zh else "no PPO stats",
                ha="center", va="center", fontsize=9, alpha=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        return
    w = max(5, len(x) // 30)
    right = ax.twinx()
    drew = False
    if "ppo_entropy" in tm:
        y = np.asarray(tm["ppo_entropy"], dtype=float)
        if _finite(y) is not None:
            ax.plot(x, _smooth(y, w), lw=1.5, color="C0", label="entropy")
            drew = True
    for i, c in enumerate(("ppo_approx_kl", "ppo_clip_frac", "ppo_forced_frac")):
        if c not in tm:
            continue
        y = np.asarray(tm[c], dtype=float)
        if _finite(y) is None:
            continue
        right.plot(x, _smooth(y, w), lw=1.2, color=f"C{i + 1}",
                   ls="--" if c == "ppo_approx_kl" else "-", label=c)
        drew = True
    if not drew:
        ax.text(0.5, 0.5, "无 PPO 统计" if zh else "no PPO stats",
                ha="center", va="center", fontsize=9, alpha=0.6)
    else:
        # 强制步对 entropy/KL/clip 的贡献恰好是 0,所以那三列会随 forced_frac
        # 结构性偏低 —— 这一格的意义就是让这件事一眼可见,别把熵下降读成策略崩溃。
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = right.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")
    ax.set_xlabel("step", fontsize=8)
    _panel_title(ax, "PPO 健康度" if zh else "PPO health")
    ax.tick_params(labelsize=8)
    right.tick_params(labelsize=8)


def load_gate(run_dir: Path, default: Optional[float] = None) -> Optional[float]:
    """从 run 的检查点 meta 里读快照闸门值(读不到就用 default)。

    直接 `np.load` 取 meta 那个字符串,不建网络 —— 画图不该顺手解析一份 10 MB 权重。
    """
    for fname in ("best.npz", "final.npz", "last.npz"):
        p = run_dir / fname
        if not p.exists():
            continue
        try:
            meta = json.loads(str(np.load(p, allow_pickle=False)["meta"]))
            cfg = json.loads(meta.get("cfg", "{}"))
            v = cfg.get("snapshot_min_official_winrate")
            if v is not None:
                return float(v)
        except Exception:  # noqa: BLE001 - 读不到闸门值不该挡住画图
            continue
    return default


def plot_run(run_dir, out: Optional[str] = None, gate: Optional[float] = None) -> Path:
    """画 `run_dir` 的曲线,返回 PNG 路径。缺数据的面板画文字说明,不抛异常。"""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"没有这个 run 目录:{run_dir}")

    zh = _pick_cjk_font()
    ev = _read_csv(run_dir / "metrics.csv")
    tm = _read_csv(run_dir / "train_metrics.csv")
    if gate is None:
        gate = load_gate(run_dir)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    _plot_eval(axes[0][0], ev, gate, zh)
    _plot_train(axes[0][1], tm, zh)
    _plot_losses(axes[1][0], tm, zh)
    _plot_health(axes[1][1], tm, zh)

    steps = tm.get("step") or ev.get("step") or []
    n_eval = len(ev.get("step", []))
    n_upd = len(tm.get("step", []))
    fig.suptitle(
        f"{run_dir.name}  —  {n_upd} 个 update / {n_eval} 次评测"
        + (f" / {int(steps[-1])} 步" if steps else "")
        + ("" if zh else f"  ({n_upd} updates, {n_eval} evals)"),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path(out) if out else run_dir / "curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def maybe_plot(run_dir, out: Optional[str] = None) -> None:
    """训练收尾用的包装:**画不出来只警告,绝不把一次训练跑废**。"""
    try:
        p = plot_run(run_dir, out)
        print(f"[rl_monet_v1] 曲线已写入 {p}")
    except Exception as exc:  # noqa: BLE001 - 故意吞掉:锦上添花不该让训练失败
        print(f"[rl_monet_v1] 画曲线失败({type(exc).__name__}: {exc}),训练结果不受影响")
