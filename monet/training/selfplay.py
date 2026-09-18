"""自对弈 PPO 训练主循环。

联盟(league)构成:
  * 固定对手:`STATIC_OPPONENTS` 里注册的那些 —— 手工规则手、官方档对手、参赛包,
    抽样权重见 `STATIC_WEIGHTS`。
  * 自己的历史快照:每隔 snapshot_every 次 update 冻结一份当前权重加进去,
    对手池因此始终跟着策略水涨船高,避免"打固定菜鸡"过拟合。
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..env.official import (
    OfficialAmbusher,
    OfficialBaseline,
    OfficialHunter,
    OfficialPatrol,
    OfficialStalker,
    OfficialWeaver,
)
from ..env.opponents import (
    EndOpponent,
    HeuristicOpponent,
    NeuralOpponent,
    Opponent,
    RandomOpponent,
)
from ..env.rlbest import RLBestOpponent
from ..env.rules_vg import VGRules
from ..env.sentry_env import RewardConfig, SentryEnv
from ..models.factory import arch_of, make_net
from ..models.mlp import MLP
from ..models.ppo import PPOAgent, Rollout
from ..models.ppo_seq import PPOSeqAgent, SeqRollout
from ..pack import LazyPackNet, load_pack_net, resolve_pack_dir
from .config import Config
from .evaluate import evaluate, summary_line


def _official(make):
    """在工厂上打"官方对手"标记。

    标记必须挂在工厂上而不是靠实例化探测:实例化要付构造成本,而 rl_v5 那种
    外部权重包还要碰文件系统(见 pack.LazyPackNet)。在 import 期做这件事会让
    "import 一个训练模块"带上副作用,包缺失时甚至整个引擎都 import 不了。
    """
    make.is_official = True
    return make


STATIC_OPPONENTS = {
    "random": lambda seed: RandomOpponent(seed=seed),
    "end": lambda seed: EndOpponent(),
    # 手工规则手:按规则写的启发式,用作陪练与回归指标
    "baseline": lambda seed: HeuristicOpponent(
        aggression=0.5, use_scan=True, seed=seed, name="baseline"
    ),
    "hunter": lambda seed: HeuristicOpponent(
        aggression=1.0, use_scan=True, seed=seed, name="hunter"
    ),
    "camper": lambda seed: HeuristicOpponent(
        aggression=0.0, use_scan=False, seed=seed, name="camper"
    ),
    # 官方口径对手 1/2:官方 AI 的移植版。**决策逻辑 1:1,但寻路
    # (advance_toward)是我方实现**,详见 env/official.py 顶部的保真度说明。
    # 评测数字不能当打榜预测。
    "official_baseline": _official(lambda seed: OfficialBaseline(seed=seed)),
    "official_hunter": _official(lambda seed: OfficialHunter(seed=seed)),
    # 官方口径对手 3/4:自研规则手(非移植),一个靠 SCAN 抓人杀,一个沿边巡逻
    # 不断 SCAN 见人就杀。四个自研规则手都**不为得分区写任何策略**(不占点、也不
    # 绕开,见 env/official.py 的 hunt_target)。见 env/official.py 末尾。
    "official_stalker": lambda seed: OfficialStalker(seed=seed),
    "official_patrol": lambda seed: OfficialPatrol(seed=seed),
    # 官方口径对手 5/6:自研**记忆型**规则手(见 env/official.py 末尾)。
    # 行为由一个观测里读不到的**内部相位**驱动 —— ambusher 每杀一次就躲三个阶段,
    # weaver 按自己的固定周期出没 —— 所以"该防还是该抢"这件事对无记忆的策略
    # 只能靠巨大的查表逼近。
    #
    # **不进 LEAGUE_DEFAULT / EVAL_DEFAULT**:名单是用户手改的口径。要用它们陪练
    # 就显式 `--league ...,official_ambusher,official_weaver`;要量它们就显式
    # `--eval-opponents`。
    #
    # 工厂**不套** `_official`:名字带 official_ 只是说它们是同一档的规则手,
    # 不代表进了官方档名单(same as official_stalker / official_patrol)。
    # 所以把它们放进评测名单不会触发 `guard_official_winrate` 或快照闸门。
    "official_ambusher": lambda seed: OfficialAmbusher(seed=seed),
    "official_weaver": lambda seed: OfficialWeaver(seed=seed),
    # 参赛包当陪练:直接用已经训练好的权重,不重新训练。
    # 温度 1.0 与部署侧的采样温度一致。包目录见 pack.KNOWN_PACKS。
    "rl_v5": _official(lambda seed: NeuralOpponent(
        LazyPackNet(name="rl_v5"), temperature=1.0, seed=seed, name="rl_v5"
    )),
    # 我们自己的 m3 导出,冻结的若干份(../m3_vN/,与 rl_VG_v0/ 那个会覆盖的活包分开)。
    # 当"别退步"的锚:从 m3 续训时它开局等于自己打自己,价值主要在后期跑偏之后。
    # 每冻一份都要在 pack.KNOWN_PACKS 里注册,否则 LazyPackNet 解析不到目录。
    "m3_v1": lambda seed: NeuralOpponent(
        LazyPackNet(name="m3_v1"), temperature=1.0, seed=seed, name="m3_v1"
    ),
    "m3_v2": lambda seed: NeuralOpponent(
        LazyPackNet(name="m3_v2"), temperature=1.0, seed=seed, name="m3_v2"
    ),
    # rl_VG_v1.0 的**起点**,也是它要超越的基线(在仓库内,见 pack.KNOWN_PACKS)。
    # 纯网络、不带规则 —— 规则只属于 rl_VG_v1.0 自己,当陪练的这个不该有
    # (给了规则它就等于 v1.0 的另一个副本,反而没有区分度)。
    "rl_VG_v0.2": lambda seed: NeuralOpponent(
        LazyPackNet(name="rl_VG_v0.2"), temperature=1.0, seed=seed, name="rl_VG_v0.2"
    ),
    # **已交付的 rl_VG_v1.0 本体**。这是 v2 唯一真正要打赢的对手:vgb4 打
    # rl_v5 / rl_VG_v0.2 / rl_best040 已经是 0.835 / 0.885 / 0.980,而 v1.0 就是
    # vgb4 自己的 final@108 —— 换句话说,"赢过 v1.0"= "赢过当前的自己"。
    #
    # **必须挂 `rules=VGRules()`**:v1.0 是"规则 ∪ 网络"的混合策略,不挂规则
    # 等于在打一个它从没被评测过、也不是它本体的弱化版,赢下来没有任何意义。
    # 也因此它不能进 `_official`(那不是官方档)。
    "rl_VG_v1.0": lambda seed: NeuralOpponent(
        LazyPackNet(name="rl_VG_v1.0"), temperature=1.0, seed=seed,
        name="rl_VG_v1.0", rules=VGRules(),
    ),
    # **已交付的 rl_VG_v2.0 本体**:v1.0 的编码器 + GRU 记忆层(架构标记
    # `kRecurrent=1`)。规则层与 v1.0 逐字相同(v2 换的是记忆,不是规则),所以
    # 同样必须挂 `rules=VGRules()`。
    #
    # `NeuralOpponent` 按 `net.is_recurrent` 自己决定要不要逐局传隐状态,所以这里
    # 不用为"带记忆的对手"多写什么;引擎每局会调 `opponent.reset()`,归零是免费的。
    #
    # **不进 LEAGUE_DEFAULT / EVAL_DEFAULT**:那是用户手改的口径。要用它陪练就
    # 显式 `--league ...,rl_VG_v2.0`;要拿它当评测锚就显式 `--eval-opponents`。
    "rl_VG_v2.0": lambda seed: NeuralOpponent(
        LazyPackNet(name="rl_VG_v2.0"), temperature=1.0, seed=seed,
        name="rl_VG_v2.0", rules=VGRules(),
    ),
    # 外部参赛包 `../RL_best040-source/`:别人的一份 412 维(不是我们的 428)plain MLP,
    # 观测由 `env/obs_rlbest.py` 逐行移植。**它走不了 pack.py**(那里校 obs_dim == 428),
    # 包目录由 `env/rlbest.py` 自己解析(环境变量 $MONET_RLBEST_DIR 可覆盖)。
    #
    # 不是 `_official`:它不是官方档,进不了快照闸门。**也不在 LEAGUE_DEFAULT /
    # EVAL_DEFAULT 里** —— 名单是用户手改的口径,要拿它陪练得显式传 `--league`。
    #
    # 保真度见 `env/obs_rlbest.py` 顶部:引擎头文件(utils.h / sentry_duel.h)不在包里,
    # `can_see` 是**假定**与 `engine.rules` 等价(行为上支持,但无法逐字对拷验证)。
    "rl_best040": lambda seed: RLBestOpponent(seed=seed),
}

# 固定对手的抽样权重:分"低权重档"和"难点档"两档。
#
# 低权重档 = 已经打饱和的对手:对着它们练几乎不产生梯度,但完全撤掉又会丢掉
# "别把它们打回去"的约束,所以留 0.3 的采样额度。
#
# 难点档 = 参赛包(见下面的 STATIC_WEIGHTS.update)。**名字不在 LOW_WEIGHT 里
# 就按 DEFAULT_STATIC_WEIGHT 算**,所以加一个强对手只要注册、不用回来改权重表
# —— 这是有意为之,别把参赛包加进 LOW_WEIGHT。
LOW_WEIGHT = (
    "random",
    "baseline",
    "hunter",
    "camper",
    "official_baseline",
    "official_hunter",
    "official_stalker",
    "official_patrol",
    # 两个记忆型规则手也在这一档 —— 别被"记忆型"三个字误导成难点档。它们的
    # **结构**是记忆型的(相位驱动),但**棋力**不是:`diag_one_opp` 300 局
    # 对 `rl_VG_v1.0` 和 `runs/v2.0` 都是 **300/300**,和 `official_patrol`
    # 的 299/300 同一档。规则手打不过训练出来的网络,这是预期内的;要它们
    # 提供政策梯度,得先有能赢过网络的规则手,那是另一回事。
    # 所以它们在这里的作用和上面六家一样:一条"别把已经赢下来的对手打回去"的
    # 约束,而不是梯度来源。见 README §六。
    "official_ambusher",
    "official_weaver",
)
STATIC_WEIGHTS = {n: 0.3 for n in LOW_WEIGHT}
# **强外部对手再分两档**:rl_v5 给 2.0,另外两个参赛包给 1.0。
#
# 分档的依据是"还剩多少可学的":rl_v5 是这一组里唯一没打服的对手(评测得分率
# ~0.68,还有两成负场),多给它采样才有梯度。rl_VG_v0.2 和 rl_best040 已经接近
# 全胜(~0.98 / 1.00),对着它们练几乎不产生梯度 —— 给 2.0 等于白占一档采样额度,
# 而这份额度正是从 rl_v5 身上挤出来的。1.0 仍高于低权重档(0.3),保住"别把已经
# 赢下来的对手打回去"的约束。
#
# 采样占比(联盟 = 4 个官方 ×0.3 + 上面三个 + 快照 ×1.0):
#   rl_v5 2.0/5.2 = 38% 起步,快照填满 2 个后降到 2.0/7.2 = 28%。
# 快照那一份是**封顶**的(`max_snapshots`),所以快照自对弈最多也就占 28%,不会
# 像不定额时那样一路吃光外部对手的额度、把训练目标从"打赢外部对手"变成"打赢
# 自己的快照"。改这里的权重前先确认 `max_snapshots` 仍是个小数字。
STATIC_WEIGHTS.update({"rl_v5": 2.0, "rl_VG_v0.2": 1.0, "rl_best040": 1.0,
                       "rl_VG_v1.0": 2.0})
# rl_VG_v1.0 拿 2.0 与 rl_v5 同档,不是 1.0:它是 rl_VG_v2.x 唯一真正要打赢的对手
# (vgb4 对另外三家已经是 0.835/0.885/0.980,而 v1.0 就是 vgb4 自己的权重)。
# 给 1.0 的话它只吃到约 14% 的采样,而"打赢当前的自己"是个比打赢 rl_v5 更难的
# 目标,样本不够会先卡在 0.6 附近。
DEFAULT_STATIC_WEIGHT = 1.0
SNAPSHOT_WEIGHT = 1.0  # 快照自对弈

# 官方档对手的注册名(快照闸门只看"打这几家的胜率")。靠工厂上的标记推导,
# 以后多写一个官方档对手不用回来改这张表。
OFFICIAL_OPPONENTS = tuple(
    n for n, make in STATIC_OPPONENTS.items() if getattr(make, "is_official", False)
)


def _read_eval_rows(path: Path) -> List[dict]:
    """把已有的 `metrics.csv` 读回来,当整表重写时的底稿。

    数值能转 float 就转,转不了(空单元格、非数值列)保留原字符串 —— 反正写回去
    是一样的文本。`None` 键是 `DictReader` 对**多出来的列**的兜底,直接丢掉,
    否则写回时会多出一列空名。
    """
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: List[dict] = []
    for r in rows:
        row: dict = {}
        for k, v in r.items():
            if k is None:
                continue
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = v
        out.append(row)
    return out

# 训练联盟与评测集的默认成员。
# 注意:同一份名单在 training/config.py 里还有一份(Config 的字段默认值)——
# 两边不能互相 import(会循环),所以只能各写一份,
# tests/test_train.py::test_default_rosters_do_not_drift 负责焊死漂移。
LEAGUE_DEFAULT = [
    "official_baseline",
    "official_hunter",
    "official_stalker",
    "official_patrol",
    "rl_v5",
    # rl_VG_v1.0 的起点兼基线。自家导出混进**训练**联盟是它的特例:一般自家导出
    # 只进评测集,但 v1.0 要打的就是 v0.2,不给它采样额度就学不到"比 v0.2 强在哪"。
    "rl_VG_v0.2",
]
EVAL_DEFAULT = [
    "official_baseline",
    "official_hunter",
    "official_stalker",
    "official_patrol",
    "rl_v5",
    "m3_v1",
    "m3_v2",
    "rl_VG_v0.2",
]


class League:
    """按权重抽对手;快照对手用冻结的网络副本。

    `max_snapshots` 的默认值只服务于直接构造 `League` 的场景:`SelfPlayTrainer`
    一律显式传 `cfg.max_snapshots`。
    """

    def __init__(self, names: List[str], max_snapshots: int = 6, temperature: float = 1.0, seed: int = 0,
                 weight_mode: str = "static", weight_cap: float = 0.35, weight_ema: float = 0.4,
                 room_eps: float = 0.03):
        # +991 只是把这条随机流和 trainer 自己的 rng 岔开,免得两处抽到同一串数
        self.rng = np.random.default_rng(seed + 991)
        self.temperature = temperature
        self.max_snapshots = max_snapshots
        # 配重(见 Config.league_weight_mode 的注释)。默认 "static" = 旧行为,
        # 直接构造 League 的场景(测试、脚本)不受影响,由 SelfPlayTrainer 显式传。
        self.weight_mode = weight_mode
        self.weight_cap = weight_cap
        self.weight_ema = weight_ema
        self.room_eps = room_eps
        # 每个静态对手的**平滑胜率**,"room" 模式的唯一输入。没有读数的对手不在
        # 表里(当成"还有全部空间"),不预填 0 —— 预填 0 会被读成"已经打服了"。
        self._wr: Dict[str, float] = {}
        self.static: List[Opponent] = []
        self.static_w: List[float] = []  # 与 static 一一对应的抽样权重
        self.snapshots: List[NeuralOpponent] = []
        # 快照名字的计数器,**只增不减**。不能拿 `len(self.snapshots)` 当编号:
        # 它在 `append` 之前求值,而列表又被 `max_snapshots` 裁到定长,于是第
        # `max_snapshots+1` 次冻结起编号就卡死在同一个数上 —— 两个存活的快照
        # 同名,日志里打出 `snap2=13.9% snap2=13.9%`,读起来像联盟里混进了重复项。
        # 名字只用于日志(`_opp_bucket` 明确不按名字认快照),所以这是个显示 bug;
        # 但"名字标识第几代快照"这件事本身得成立,不然下次谁按名字查就踩雷。
        self._snap_seq = 0
        for n in names:
            if n in STATIC_OPPONENTS:
                self.static.append(STATIC_OPPONENTS[n](seed=int(self.rng.integers(1 << 30))))
                self.static_w.append(STATIC_WEIGHTS.get(n, DEFAULT_STATIC_WEIGHT))
            elif n == "snapshot":
                continue  # 快照在训练中动态加入
            else:
                raise ValueError(f"未知对手: {n}")
        # 静态档的**总额度**。"room" 模式只重分这块额度、不改它的大小,所以
        # 静态档与快照档的配比仍由 `STATIC_WEIGHTS` / `SNAPSHOT_WEIGHT` 说了算,
        # `max_snapshots` 对自对弈占比的封顶也就依然成立。
        self._static_total = float(sum(self.static_w))

    def update_weights(self, winrates: Dict[str, float]) -> None:
        """按"剩余空间"重算静态对手的权重(只在 `weight_mode == "room"` 时生效)。

        `winrates` 是最近一次评测的逐对手得分率。**用评测读数而不是训练局的
        实测胜率**:训练局摊到每个对手每轮只有十来局(标准误差 ±0.16),拿它
        配重等于让权重跟着噪声走;评测一次 60 局而且本来就是同一把尺子。

        权重 = 总额度 × 剩余空间 / Σ剩余空间,再对单个对手封顶。剩余空间
        `1 - 胜率` 的含义就是"还有多少可学":打到 1.000 的对手空间为 0,自然
        让出额度;被打回去的对手空间变大,自动多拿样本 —— 保护和清死区是同一个
        动作,不需要两套机制。

        封顶(`weight_cap`)是必须的:联盟里常常只剩一家还有空间,不封顶就会把
        额度几乎全压到它头上,那正是过拟合的形状。
        """
        if self.weight_mode != "room" or not self.static:
            return
        for o in self.static:
            wr = winrates.get(o.name)
            if wr is None:
                continue  # 这次评测没测它 —— 保留上次的读数,别拿"没测"当 0
            prev = self._wr.get(o.name)
            self._wr[o.name] = wr if prev is None else (1.0 - self.weight_ema) * prev + self.weight_ema * wr

        # 没读数的对手当成"还有全部空间":开局谁都还没打服,均分是对的起点。
        room = np.array(
            [max(0.0, 1.0 - self._wr.get(o.name, 0.0)) + self.room_eps for o in self.static],
            dtype=np.float64,
        )
        w = self._static_total * room / room.sum()
        cap = self.weight_cap * self._static_total
        # 把超顶的部分按比例让给没到顶的,直到没人超顶(几轮就收敛)。
        for _ in range(8):
            over = w > cap
            if not over.any():
                break
            excess = float((w[over] - cap).sum())
            w[over] = cap
            under = ~over
            if not under.any():
                break  # 全都到顶了,没地方让 —— 保持现状比胡乱分好
            w[under] += excess * w[under] / w[under].sum()
        self.static_w = [float(x) for x in w]

    def sample(self) -> Opponent:
        pool: List[Opponent] = list(self.static) + list(self.snapshots)
        w = np.array(
            self.static_w + [SNAPSHOT_WEIGHT] * len(self.snapshots), dtype=np.float64
        )
        if w.sum() <= 0:
            return RandomOpponent(seed=int(self.rng.integers(1 << 30)))
        idx = int(self.rng.choice(len(pool), p=w / w.sum()))
        return pool[idx]

    def add_snapshot(self, net: MLP) -> None:
        # 按**同一个架构**克隆:写死 MLP 会把 GRU 的快照塞进去 —— MLP 的
        # load_state_dict 会把 GRU 的键一并收下(它不校验键集合),于是快照网络
        # 里多出一堆永远用不到的键,而 GRU 部分根本没被装载,快照退化成随机 MLP。
        clone = make_net(
            arch=arch_of(net), obs_dim=net.obs_dim, hidden=net.hidden,
            act_dim=net.act_dim, gru_hidden=getattr(net, "gru_hidden", 128),
        )
        clone.load_state_dict(net.state_dict())
        self.snapshots.append(
            NeuralOpponent(
                clone,
                temperature=self.temperature,
                seed=int(self.rng.integers(1 << 30)),
                name=f"snap{self._snap_seq}",
            )
        )
        self._snap_seq += 1
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

    def names(self) -> List[str]:
        return [o.name for o in self.static] + [o.name for o in self.snapshots]

    def distribution(self) -> str:
        """各对手实际吃到的采样占比,一行打印用。"""
        w = np.array(self.static_w + [SNAPSHOT_WEIGHT] * len(self.snapshots), dtype=np.float64)
        if w.sum() <= 0:
            return "(空)"
        p = w / w.sum()
        return " ".join(f"{n}={v:.1%}" for n, v in zip(self.names(), p))


class SelfPlayTrainer:
    def __init__(self, cfg: Config, init_net: Optional[MLP] = None):
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir) / cfg.run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(cfg.seed)

        self.net = init_net or make_net(
            arch=cfg.arch, hidden=cfg.hidden, gru_hidden=cfg.gru_hidden, seed=cfg.seed
        )
        # 架构以**实际的 net** 为准,不看 cfg.arch:起训权重可能是从包/检查点来的,
        # 而 cfg.arch 只决定随机初始化时的样子。两者不一致时(比如拿 GRU 的
        # checkpoint 续训却忘了带 --arch gru)以 net 为准才不会把 RNN 当 MLP 训。
        self.recurrent = bool(getattr(self.net, "is_recurrent", False))
        agent_cls = PPOSeqAgent if self.recurrent else PPOAgent
        self.agent = agent_cls(
            self.net,
            lr=cfg.lr,
            clip=cfg.clip,
            vf_coef=cfg.vf_coef,
            ent_coef=cfg.ent_coef,
            epochs=cfg.epochs,
            minibatches=cfg.minibatches,
            max_grad_norm=cfg.max_grad_norm,
            gamma=cfg.gamma,
            lam=cfg.lam,
            seed=cfg.seed,
            **({"seg_len": cfg.seg_len, "segs_per_mb": cfg.segs_per_mb}
               if self.recurrent else {}),
        )
        # 当前这一步**进入时**的隐状态。每局开头必须归零(见 _pick_opponent)。
        self.h = self.net.new_hidden() if self.recurrent else None
        self.league = League(
            cfg.league,
            max_snapshots=cfg.max_snapshots,
            temperature=cfg.snapshot_temperature,
            seed=cfg.seed,
            weight_mode=cfg.league_weight_mode,
            weight_cap=cfg.league_weight_cap,
            weight_ema=cfg.league_weight_ema,
            room_eps=cfg.league_room_eps,
        )
        self.env = SentryEnv(
            self.league.sample(),
            agent_color=None,
            reward=RewardConfig(scan_reveal=cfg.scan_reveal),
            seed=cfg.seed,
            # 与 env 的默认值、评测侧(evaluate.play_match)保持一致:这个上限
            # 决定哪些局以平局收场,两边不一致时训练日志的 ep_win 和评测得分率
            # 就不是同一个口径。
            max_steps=400,
        )
        # 规则层(rl_VG_v1.0)。策略 = 规则 ∪ 网络:规则命中时把掩码收窄成单动作,
        # 于是那一步的 logp 恰好是 0、策略梯度恰好是 0 —— 规则决定的步对网络**没有**
        # 梯度,网络只学规则没覆盖的残差(走位、占点、接近)。
        #
        # `use_rules=False` 时整条规则路径都不走(连"网络不许扫"的屏蔽也一并撤掉),
        # 这样它才是纯网络对照,而不是"另一个也改过的策略"。
        self.rules = VGRules() if cfg.use_rules else None
        self.obs, self.mask, self.info = self.env.reset(seed=cfg.seed)
        if self.rules is not None:
            self.rules.reset()
        self.ep_return = 0.0
        self.ep_len = 0
        self.best_metric = -1.0
        self.history: List[dict] = []
        self._csv = self.out_dir / "metrics.csv"
        # 逐 update 的训练曲线数据。`metrics.csv` 只在**评测时**落行,`eval_every`
        # 个 update 才一个点,画不出曲线;而回报/胜率/PPO 统计每个 update 都算过了。
        # 这张表补上那部分,`metrics.csv` 一个字段都不动 —— 它是对外可比的评测记录,
        # 不是诊断日志,加列会让新旧行不可比。
        self._train_csv = self.out_dir / "train_metrics.csv"
        self._train_fields: Optional[List[str]] = None
        # 最近一次评测的完整结果(早停要按对手名读,`history` 里的键是拍平的)
        self.last_eval: Dict[str, Dict[str, float]] = {}
        # 上面那份评测是第几个 update 跑出来的,以及早停判据**已经判过**哪个
        # update 的评测。早停判据每个 update 都被调用,但 `last_eval` 只在评测点
        # 刷新,所以同一个评测点只需判一次 —— 判过了没停,下一个 update 读到的
        # 还是同一份数据,再判结果不变(会变就说明早停了,根本走不到那)。
        # 这一条挡掉的浪费很实在:判据里那次高局数测量要跑好几分钟。
        self.last_eval_update: Optional[int] = None
        self._stop_checked: Optional[int] = None
        # 快照闸门状态:最近一次评测里官方对手的平均得分率
        self.last_official_winrate: Optional[float] = None
        self._gate_measurable = any(n in OFFICIAL_OPPONENTS for n in self._eval_names())
        self._gate_warned = False

    # ------------------------------------------------------------------ 工具

    def _pick_opponent(self) -> None:
        self.env.opponent = self.league.sample()
        color = "R" if self.rng.random() < 0.5 else "B"
        self.obs, self.mask, self.info = self.env.reset(
            seed=int(self.rng.integers(1 << 30)), agent_color=color
        )
        if self.rules is not None:
            self.rules.reset()  # 新一局从 turn=0 重来,失明计数必须跟着归零
        if self.recurrent:
            # 新一局 = 清空记忆。漏了这一句,上一局末尾的隐状态会当成"历史"喂进
            # 新局的第一步 —— 模型会真的"记住"上一个对手的打法,而且不报错。
            self.h = self.net.new_hidden()

    def _eval_names(self) -> List[str]:
        return list(getattr(self.cfg, "eval_opponents", None) or EVAL_DEFAULT)

    def _eval_opponents(self) -> Dict[str, Opponent]:
        # 对手种子只由 cfg.seed 决定、不随 update 变(每次评测换的是**对局**种子,
        # 见 _run_eval),这样两次评测之间的抖动只来自策略和采样,不来自对手换手气。
        seed = self.cfg.seed + 12345
        return {n: STATIC_OPPONENTS[n](seed) for n in self._eval_names()}

    # ------------------------------------------------------------ 快照闸门

    def _snapshot_gate_open(self) -> bool:
        """官方 AI 胜率没到线就不加快照 —— 先专心练固定对手。

        闸门看的是**最近一次评测**里官方对手的平均得分率。评测间隔(`eval_every`)
        就是它的取样周期:没评测过 = 关着。`snapshot_min_official_winrate <= 0`
        表示不设闸门。
        """
        need = self.cfg.snapshot_min_official_winrate
        if need <= 0:
            return True
        if not self._gate_measurable:
            # 评测集里没有官方对手 → 这个条件永远测不出来。警告一次后放行,
            # 否则会静默地一个快照都不加。
            if not self._gate_warned:
                self._gate_warned = True
                print(
                    "    !! 快照闸门失效:eval_opponents 里没有官方对手,"
                    "无法测官方胜率,已直接放行"
                )
            return True
        if self.last_official_winrate is None:
            return False
        return self.last_official_winrate > need

    def _stop_reason(self) -> Optional[str]:
        """早停判据:三类条件**同时**成立才收工,少一条就继续训。

        ① `stop_opponents` **全部** ≥ `stop_winrate` —— 用一次专门的
           `stop_confirm_games` 局测量判,不用 `eval_games` 那次;
        ② 官方档**每一个** ≥ `guard_official_winrate`(保护官方 AI);
        ③ 当前综合得分率 ≥ `best_metric - stop_metric_slack`(不许在局部最优尖峰上收工)。

        "全部"而不是"任一"的理由见 `Config.stop_opponents` 的注释。

        **为什么①要另测而不是复核**:见 `Config.stop_confirm_games`。一句话 ——
        复核只挡假阳性,挡不住假阴性,而阈值落在噪声带里时两个方向都会错。

        ②③ 故意排在贵测量之前:它们拦下的正是"已经跑偏了"的情况,那种情况下花
        几分钟去精确测①是纯浪费。它们读的都是 `last_eval`,不要钱。

        读 `last_eval` 而不是 `history`:后者把对手名拍进了列名,取回来要拼字符串,
        拼错了会静默地永不触发。任何一个对手不在评测名单里就整个判据失效(返回
        None),不静默当成"没到线",否则一个拼错的对手名会让早停永远不触发。
        """
        cfg = self.cfg
        opps, need = list(cfg.stop_opponents), cfg.stop_winrate
        if not opps or need <= 0 or not self.last_eval:
            return None
        # 同一个评测点只判一次。判过了没停,下个 update 读的还是同一份 `last_eval`,
        # 再判结果不变 —— 而贵测量要跑好几分钟,不能每个 update 都付一遍。
        if self._stop_checked == self.last_eval_update:
            return None
        self._stop_checked = self.last_eval_update

        # --- 闸②:官方保护。官方档是"没练偏"的锚,不许拿它们换偏科 ---
        if cfg.guard_official_winrate > 0:
            bad = [f"{n} {r['winrate']:.3f}" for n, r in self.last_eval.items()
                   if n in OFFICIAL_OPPONENTS and r["winrate"] < cfg.guard_official_winrate]
            if bad:
                print(f"  · 早停搁置:官方档被打回去了({'、'.join(bad)})—— "
                      f"先补回来(门槛 {cfg.guard_official_winrate:.2f})")
                return None

        # --- 闸③:不许在自己都已经不如自己最好的时候收工 ---
        metric = float(np.mean([r["winrate"] for r in self.last_eval.values()]))
        if self.best_metric > 0 and metric < self.best_metric - cfg.stop_metric_slack:
            print(f"  · 早停搁置:综合得分率 {metric:.4f} 比本轮最优 {self.best_metric:.4f} "
                  f"低出 {cfg.stop_metric_slack:.2f} 以上 —— 这是退步不是达标")
            return None

        if any(o not in self.last_eval for o in opps):
            return None  # 不在评测名单里 —— 条件测不出来

        n = cfg.stop_confirm_games
        if n <= 0:
            # 退回旧行为:拿 `eval_games` 那次的读数直接判。噪声大,只在不舍得
            # 花那几分钟的时候用。
            miss = [f"{o} {self.last_eval[o]['winrate']:.3f}" for o in opps
                    if self.last_eval[o]["winrate"] < need]
            if miss:
                return None
            got = "、".join(f"{o} {self.last_eval[o]['winrate']:.3f}" for o in opps)
            return f"{got} 全部 ≥ {need:.2f}({cfg.eval_games} 局,未另测)"

        # --- 便宜筛子:还有对手明显没到,就别花那几分钟 ---
        # 筛子只能筛掉"明显没到"的,所以 margin 给得比噪声宽(见 Config)。
        if cfg.stop_screen_margin > 0 and any(
            self.last_eval[o]["winrate"] < need - cfg.stop_screen_margin for o in opps
        ):
            return None

        # --- 闸①:决定性测量。换种子 —— 沿用 `seed + update` 等于把同一批局再跑
        # 一遍,那测的还是噪声,只是多花了几分钟 ---
        pool = self._eval_opponents()
        if any(o not in pool for o in opps):
            return None
        res = evaluate(
            self.net,
            {o: pool[o] for o in opps},
            games=n,
            seed=cfg.seed + 90000,
            rules=VGRules() if cfg.use_rules else None,
        )
        miss = [f"{o} {res[o]['winrate']:.3f}" for o in opps if res[o]["winrate"] < need]
        if miss:
            print(f"  · 早停未达成({n} 局):{'、'.join(miss)} 未达 {need:.2f} —— "
                  f"继续训({cfg.eval_games} 局那次是噪声)")
            return None
        got = "、".join(f"{o} {res[o]['winrate']:.3f}" for o in opps)
        return f"{got} 全部 ≥ {need:.2f}({n} 局实测)"

    def _opp_bucket(self) -> str:
        """把当前这局的对手归到一个**固定的**列名上。

        静态对手按名字各占一列;**快照合并成一列** —— 快照是训练中动态生成、
        又被 `max_snapshots` 挤掉的(`add_snapshot` 里 `pop(0)`),逐个建列会让
        表头随训练漂移,续训时新旧表头对不上。

        按**对象身份**认静态对手,不按名字前缀猜:`sample()` 返回的就是
        `self.league.static` 里的那个对象本身,同一性可靠;而"名字以 snap 开头"
        是个约定,哪天改名就静默地全归进快照档了。
        """
        opp = self.env.opponent
        for o in self.league.static:
            if o is opp:
                return o.name
        return "snapshot"

    def _log_train_row(self, update, step, stats, ep_returns, ep_wins, sps,
                       by_opp: Optional[Dict[str, List[float]]] = None) -> None:
        """每个 update 往 `train_metrics.csv` 补一行。字段集合固定(全是数值),
        所以只在第一行写表头;续训时表头沿用文件里已有的那份。

        **逐对手胜率也记在这里**(`ep_wr_<对手>` / `ep_n_<对手>`)。这些局本来
        就要打,记下来不要钱;而 `metrics.csv` 的评测点稀疏得多(`eval_every`
        个 update 才一次),光看那几个点分不清"策略在退化"还是"评测在抖"。

        没抽到的对手记 `nan` 而不是 0.0:`0.0` 会被读成"全输了",真相是"这一轮
        没抽到它"。所以要跟 `ep_n_<对手>` 一起看 —— 那个是局数。
        """
        by_opp = by_opp or {}
        row = {
            "update": update,
            "step": step,
            "ep_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "ep_win": float(np.mean(ep_wins)) if ep_wins else 0.0,
            "episodes": len(ep_returns),
            "sps": sps,
            # 与 metrics.csv 的评测行同名(`ppo_*`),画图代码可以一套列名读两张表
            **{f"ppo_{k}": v for k, v in stats.items()},
        }
        for name in [o.name for o in self.league.static] + ["snapshot"]:
            rs = by_opp.get(name) or []
            row[f"ep_wr_{name}"] = float(np.mean(rs)) if rs else float("nan")
            row[f"ep_n_{name}"] = len(rs)
        if self._train_fields is None:
            self._train_fields = list(row)
            mode, header = ("a" if self._train_csv.exists() else "w"), not self._train_csv.exists()
        else:
            mode, header = "a", False
        with self._train_csv.open(mode, newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self._train_fields, extrasaction="ignore")
            if header:
                w.writeheader()
            w.writerow(row)

    def _gate_status(self) -> str:
        need = self.cfg.snapshot_min_official_winrate
        if need <= 0 or not self._gate_measurable:
            return "不设闸门"
        cur = self.last_official_winrate
        if cur is None:
            return f"闸门关(还没评测过,需 >{need:.2f})"
        return f"闸门{'开' if cur > need else '关'}(官方胜率 {cur:.3f},需 >{need:.2f})"

    # ------------------------------------------------------------------ 训练

    def train(self, resume_meta: Optional[dict] = None) -> MLP:
        cfg = self.cfg
        step = int((resume_meta or {}).get("step", 0))
        update = int((resume_meta or {}).get("update", 0))
        t0 = time.time()
        # `step` 是**绝对**步数(续训时从 checkpoint 接着数),`t0` 却是本进程的起点,
        # 两者相减才是"这段时间真正跑了多少步"。累计速度必须用这个差值:直接拿
        # `step / (now - t0)` 会把续训之前所有轮的步数都算进本进程的速度里,续训一
        # 启动就虚报几十倍,再随运行时间慢慢收敛回真值 —— 看着像"越跑越慢"。
        step0 = step
        print(
            f"[rl_monet_v1] run={cfg.run_name} 输出={self.out_dir}\n"
            f"  网络 {self.net.obs_dim}->{cfg.hidden}x3->{self.net.act_dim}  "
            f"联盟 {self.league.names()}  目标步数 {cfg.total_steps}\n"
            f"  采样占比 {self.league.distribution()}"
        )

        # 新一轮 = 新表。续训(`resume_meta` 非空)则**接着写**,让曲线把两次连起来;
        # 从头跑却留着上一轮的行,会把两次训练画成一条线,看着像"一夜之间退步了"。
        if not resume_meta:
            self._train_csv.unlink(missing_ok=True)
            self._train_fields = None
        else:
            # `metrics.csv` 是**整表重写**(字段集合能随配置变,重写比追加不容易写坏),
            # 所以重写的原料 `self.history` 必须先把旧行读回来 —— 漏了这步不报错,
            # 只是续训后的第一次评测会抹掉之前所有评测行,而追加写的
            # `train_metrics.csv` 安然无恙,两张表的曲线从此对不上。
            self.history = _read_eval_rows(self._csv)

        while step < cfg.total_steps:
            roll = SeqRollout() if self.recurrent else Rollout()
            ep_returns: List[float] = []
            ep_wins: List[float] = []
            by_opp: Dict[str, List[float]] = {}  # 逐对手战绩,与 ep_wins 吃的是同一批局
            last_done = True

            forced_n = 0
            t_win = time.time()
            while len(roll) < cfg.rollout_steps:
                # 规则先判:命中就把掩码收窄成"只有它合法"再采样。这样存进轨迹的
                # logp 是 0.0(而不是网络对另一个动作的概率),ratio 恒为 1,
                # 策略梯度恒为 0 —— 规则不再是一个"事后覆盖",而是策略本身的一部分。
                # 没命中时 `act_mask` 还会把 SCAN 位置零(网络不许自己扫)。
                #
                # 收窄后的掩码必须与动作**成对**存进 Rollout:update 全程用存下来的
                # 掩码,不会回环境重取,两者对不上就会算出错误的概率。
                #
                # `act_mask` 返回的总是新数组,不会动 self.mask —— 它马上被下面的
                # nmask 覆盖,而且 evaluate/cli 也读它,就地改会污染别处。
                if self.rules is not None:
                    mask, k = self.rules.act_mask(self.env.ag_ob, self.mask)
                else:
                    mask, k = self.mask, None  # 纯网络对照:掩码原样交给网络
                if k is not None:
                    forced_n += 1
                # h_in 是**进入这一步时**的隐状态,必须存它而不是 act 之后的新状态:
                # ppo_seq 按段训练时,段起点的 h0 直接取 `h[a]`,存错就整体平移一步。
                h_in = self.h
                if self.recurrent:
                    a, logp, v, self.h = self.agent.act(self.obs, mask, h_in)
                else:
                    a, logp, v = self.agent.act(self.obs, mask)
                nobs, nmask, r, term, trunc, info = self.env.step(a)
                done = term or trunc
                if self.recurrent:
                    roll.add(self.obs, mask, a, logp, v, r, done, h_in)
                else:
                    roll.add(self.obs, mask, a, logp, v, r, done)
                self.ep_return += r
                self.ep_len += 1
                step += 1
                last_done = done
                if done:
                    ep_returns.append(self.ep_return)
                    ep_wins.append(self.env.result())
                    # 必须取**这一局**的对手:下面几行 `_pick_opponent` 一跑,
                    # `env.opponent` 就换成下一局的对手了,顺序反了会张冠李戴。
                    by_opp.setdefault(self._opp_bucket(), []).append(self.env.result())
                    self.ep_return = 0.0
                    self.ep_len = 0
                    self._pick_opponent()
                else:
                    self.obs, self.mask, self.info = nobs, nmask, info

                # 采样进度行。一次 update 要走满 rollout_steps 步,中间不吭声的话
                # 日志看着就像卡死 —— 所以按 progress_every_steps 打点。
                # 速度用**本窗口**算,累计平均会把突发抖动抹平,看不出快慢。
                # 最后一步不打:紧跟其后的 update 行已经报了步数和累计速度。
                pe = cfg.progress_every_steps
                if pe and len(roll) % pe == 0 and len(roll) < cfg.rollout_steps:
                    now = time.time()
                    print(
                        f"      · 采样 {len(roll):>5}/{cfg.rollout_steps}  "
                        f"{pe / max(1e-9, now - t_win):.0f} 步/秒  "
                        f"已完局 {len(ep_returns):>2}  "
                        f"回报 {np.mean(ep_returns) if ep_returns else 0:+.2f}",
                        flush=True,  # 重定向到文件时不做这个,进度行会攒到进程结束才出现
                    )
                    t_win = now

            # 采样在一局**中途**停下时,用当前状态的价值给 GAE 收尾;刚好以终局
            # 收尾则置 0(GAE 里 done 位本来就让这一项乘 0)。注意 done 是
            # term or trunc,所以打满 max_steps 的截断局也算终局、不做 bootstrap。
            if last_done:
                last_value = 0.0
            elif self.recurrent:
                # `self.obs` 在循环里已经前进到**最后一步之后**那个状态(未 done 时
                # 更新过),`self.h` 正是"进入那个状态"的隐状态 —— 与 MLP 分支
                # `value_of(self.obs)` 取的是同一个状态、同一个口径。
                # 返回的新隐状态**丢掉不要**:它已经吃掉了 obs,留着会让下一个
                # rollout 的第一步把同一个观测算两遍。
                last_value, _ = self.agent.value_of(self.obs, self.h)
            else:
                last_value = self.agent.value_of(self.obs)

            progress = step / max(1, cfg.total_steps)
            self.agent.ent_coef = cfg.ent_coef + (cfg.ent_coef_final - cfg.ent_coef) * progress
            stats = self.agent.update(roll, last_value)
            # 规则吃掉了多少比例的决策。这一列是排障用的:规则强制的那几步对
            # entropy / approx_kl / clip_frac 的贡献恰好是 0,所以那三列会随
            # forced_frac 结构性偏低 —— 没有这一列,看到熵掉下去会误判成策略崩溃。
            # 反过来,它要是恒等于 0,就说明规则在训练里静默失效了。
            stats["forced_frac"] = forced_n / max(1, len(roll))
            update += 1
            sps = (step - step0) / max(1e-6, time.time() - t0)
            self._log_train_row(update, step, stats, ep_returns, ep_wins, sps, by_opp)

            if update % cfg.log_every == 0:
                print(
                    f"  upd {update:>4} step {step:>7}  "
                    f"回报 {np.mean(ep_returns) if ep_returns else 0:+.2f}  "
                    f"胜率 {np.mean(ep_wins) if ep_wins else 0:.2f}  "
                    f"π损失 {stats.get('policy_loss', 0):+.4f}  "
                    f"V损失 {stats.get('value_loss', 0):.4f}  "
                    f"熵 {stats.get('entropy', 0):.3f}  "
                    f"KL {stats.get('approx_kl', 0):+.5f}  "
                    f"规则 {stats.get('forced_frac', 0):.1%}  "
                    f"{sps:.0f} 步/秒"
                )

            if update % cfg.save_every == 0:
                from ..store import save_checkpoint

                save_checkpoint(
                    self.out_dir / "last.npz",
                    self.net,
                    self.agent,
                    {"step": step, "update": update, "cfg": cfg.to_json()},
                )

            # 评测放在加快照之前:闸门要读的是**刚算出来**的官方胜率,
            # 否则同一轮里会用上一次评测的旧值(白白晚一个 eval_every)。
            if update % cfg.eval_every == 0 or step >= cfg.total_steps:
                self._run_eval(step, update, stats, ep_returns, ep_wins)

            if update % cfg.snapshot_every == 0:
                if self._snapshot_gate_open():
                    self.league.add_snapshot(self.net)
                    print(
                        f"    -> 加入快照对手,联盟 = {self.league.names()}\n"
                        f"       采样占比 {self.league.distribution()}"
                    )
                else:
                    print(f"    -> 跳过快照({self._gate_status()})")

            # 早停放最后:评测跑过了、快照也加过了、best.npz 该存的也存了,
            # 此时跳出不会丢任何一轮的产物。
            reason = self._stop_reason()
            if reason is not None:
                print(f"  ■ 早停:{reason}(目标已达成,提前收工)")
                break

        from ..store import save_checkpoint

        save_checkpoint(
            self.out_dir / "final.npz",
            self.net,
            self.agent,
            {"step": step, "update": update, "cfg": cfg.to_json()},
        )
        print(f"[rl_monet_v1] 完成:{step} 步,权重在 {self.out_dir/'final.npz'}")
        return self.net

    def _run_eval(self, step, update, stats, ep_returns, ep_wins) -> None:
        cfg = self.cfg
        # 用**新建的** VGRules 而不是 self.rules:评测会把计数器按局重置,而
        # self.rules 正跟着一个没走完的训练局,共享实例会把那个局的失明计数冲掉。
        res = evaluate(
            self.net,
            self._eval_opponents(),
            games=cfg.eval_games,
            seed=cfg.seed + update,
            # 评测必须按**训练时的形态**测:`use_rules=False` 的对照跑就该测纯网络,
            # 否则两条曲线的评测口径不同,比较就失去意义。
            rules=VGRules() if cfg.use_rules else None,
        )
        self.last_eval = res  # 早停按对手名读它,别从拍平的 history 里拼列名
        self.last_eval_update = update  # 早停判据凭它判断"这份读数判过没有"
        print(f"  ── 评测(update {update})──")
        for name, r in res.items():
            print("     " + summary_line(name, r))
        print(f"     快照{self._gate_status()}")
        # 动态配重(league_weight_mode="room"):拿刚测出来的胜率重分静态额度。
        # 放在早停判据**之前** —— 判据里的贵测量不算这一轮,但下一个 update 起
        # 的采样就该用新权重了,不能等到下一轮评测。
        before = self.league.static_w
        self.league.update_weights({n: r["winrate"] for n, r in res.items()})
        if self.league.static_w != before:
            print(f"     配重 {self.league.distribution()}")
        official = [r["winrate"] for n, r in res.items() if n in OFFICIAL_OPPONENTS]
        self.last_official_winrate = float(np.mean(official)) if official else None
        metric = float(np.mean([r["winrate"] for r in res.values()]))

        row = {
            "update": update,
            "step": step,
            "ep_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "train_score_rate": float(np.mean(ep_wins)) if ep_wins else 0.0,
            "eval_metric": metric,
            **{f"eval_{k}_{m}": v for k, r in res.items() for m, v in r.items() if m != "games"},
            **{f"ppo_{k}": v for k, v in stats.items()},
        }
        self.history.append(row)
        # 每次重写整张表:字段集合可能随配置变化,重写比追加更不容易写坏
        fields = list({k: None for r in self.history for k in r})
        with self._csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.history)

        if cfg.keep_best and metric > self.best_metric:
            self.best_metric = metric
            from ..store import save_checkpoint

            save_checkpoint(
                self.out_dir / "best.npz",
                self.net,
                None,
                {"step": step, "update": update, "eval_metric": metric, "cfg": cfg.to_json()},
            )
            print(f"     ★ 新最优(综合得分率 {metric:.4f})→ best.npz")


def load_init_net(cfg: Config) -> Optional[MLP]:
    """按 `cfg.init_pack` 装载起训权重(空 = None,随机初始化)。

    优先用参赛包而不是 `runs/*/best.npz`:best.npz 会随新 run 前移(README §七),
    包是冻结物。这里把解析到的目录打出来,起训权重才是可追溯的。
    """
    if not cfg.init_pack:
        return None
    mlp = load_pack_net(name=cfg.init_pack, hidden=cfg.hidden)
    print(
        f"[rl_monet_v1] 从参赛包 {cfg.init_pack} 起步:"
        f"{resolve_pack_dir(name=cfg.init_pack)}"
    )
    if cfg.arch != "gru":
        return mlp

    from ..models.rnn_policy import RNNPolicy

    if getattr(mlp, "is_recurrent", False):
        # 起点本身就是带记忆的包(gru → gru 的续训),**必须整份装进去**。
        # 走 from_mlp 会把已训好的 GRU 与 P/bP 丢掉、只继承编码器与三个头,
        # 而打印的话术还是"初始前向逐位等于这个包" —— 一路不报错,只是从一个
        # 比预期弱得多的起点开跑。这正是本仓库反复防的那类"静默变弱"。
        if mlp.hidden != cfg.hidden or mlp.gru_hidden != cfg.gru_hidden:
            raise ValueError(
                f"{cfg.init_pack} 的架构与本轮不匹配:"
                f"包 hidden={mlp.hidden} gru={mlp.gru_hidden},"
                f"本轮 hidden={cfg.hidden} gru={cfg.gru_hidden}"
            )
        net = make_net(arch="gru", obs_dim=mlp.obs_dim, hidden=cfg.hidden,
                       act_dim=mlp.act_dim, gru_hidden=cfg.gru_hidden, seed=cfg.seed)
        net.load_state_dict(mlp.state_dict())
        print(
            f"   架构 gru:整份继承 {len(net.p)} 个张量(编码器 + 三个头 + GRU + P/bP)"
            f" —— 初始前向逐位等于 {cfg.init_pack}"
        )
        return net

    # 包里的都是无记忆网络,GRU 从它**继承编码器与三个头**,GRU 和 P/bP 全新。
    # P/bP 零初始化 => 初始化那一刻整个前向逐位等于这个包(见 RNNPolicy 顶部),
    # 所以"最差也只是学成不用记忆,不会一上来就把已经很强的策略弄坏"。
    net = RNNPolicy.from_mlp(mlp, gru_hidden=cfg.gru_hidden, seed=cfg.seed)
    print(
        f"   架构 gru:已继承编码器与三个头({len(mlp.p)} 个张量),"
        f"GRU(隐 {cfg.gru_hidden})与 P/bP 全新且 P/bP 为零 —— "
        f"初始前向逐位等于 {cfg.init_pack}"
    )
    return net


def train(cfg: Config, resume: Optional[str] = None) -> MLP:
    init_net, meta = None, None
    if resume:
        from ..store import load_checkpoint

        init_net, meta, _ = load_checkpoint(resume)
        print(f"[rl_monet_v1] 从 {resume} 续训(step={meta.get('step')})")
    else:
        init_net = load_init_net(cfg)
    trainer = SelfPlayTrainer(cfg, init_net=init_net)
    return trainer.train(resume_meta=meta)
