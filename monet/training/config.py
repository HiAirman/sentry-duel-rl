"""训练配置。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List


@dataclass
class Config:
    # --- 运行 ---
    run_name: str = "monet_v1"
    out_dir: str = "runs"
    seed: int = 0
    total_steps: int = 200_000
    rollout_steps: int = 4096
    # 采样循环里每隔这么多步打一行进度(见 selfplay.py),0 = 关。
    # 想调日志密度别去动 `rollout_steps`:它同时是 PPO 的采样长度,改小会让
    # minibatch 从 512 掉到 128(4096/8)、梯度变噪。`log_every` 按 update 计数、
    # 最小是 1,所以它才是日志密度的下限,不是旋钮。
    progress_every_steps: int = 1024
    log_every: int = 1
    # 外部参赛包目录(rl_v5 的权重)。空 = 走 pack.resolve_pack_dir 的解析链
    # (显式参数 > 进程覆盖 > 环境变量 > 默认位置)。
    pack_dir: str = ""
    # 起训权重来自哪个参赛包(空 = 随机初始化)。用包而不是 .npz 路径,是因为
    # runs/*/best.npz 会随新 run 前移,而包是冻结的(README §七)。
    # 会进 cfg.to_json(),于是每个 checkpoint 都记着"这一轮从哪起步"。
    init_pack: str = ""

    # --- 网络 ---
    hidden: int = 512
    # "mlp" = rl_VG_v1.0 那个无记忆网络;"gru" = 编码器 + 记忆层(rl_VG_v2.x)。
    # 这一项决定**批怎么切** —— gru 走 ppo_seq 的连续片段 + 截断 BPTT,mlp 走
    # ppo.py 的打乱时间步。选错了不报错,只是训练效果差一截。
    arch: str = "mlp"
    gru_hidden: int = 128
    # 截断 BPTT 的段长(决策步数)与每个 minibatch 装几个段。
    # 一局约 54 个决策点,所以 32 意味着多数局只被切成 2 段 —— 记忆要跨的
    # 截断点很少。段越长长程梯度越完整,代价是反向的循环更长、显存/内存更大。
    seg_len: int = 32
    segs_per_mb: int = 8

    # --- PPO ---
    lr: float = 3e-4
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    ent_coef_final: float = 0.003
    epochs: int = 4
    minibatches: int = 8
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    lam: float = 0.95

    # --- 奖励塑形 ---
    # 默认 0 = 奖励严格等于计分事件。>0 时"SCAN 扫出之前看不见的敌人"额外给分,
    # 用来把策略往多用扫描推。会破坏「回报 == 比分差」,只影响训练,不影响评测。
    scan_reveal: float = 0.2

    # --- 联盟/自对弈 ---
    # 含官方档对手:官方 Baseline / Hunter 的移植版(决策逻辑 1:1,寻路是近似的,
    # 见 env/official.py)+ 两个自研规则手 stalker / patrol,以及参赛包 rl_v5 和
    # rl_VG_v0.2(rl_VG_v1.0 的起点兼基线)。
    # 这份名单与 training/selfplay.py 的 LEAGUE_DEFAULT 是两份(不能互相 import,
    # 会循环),test_default_rosters_do_not_drift 负责焊死漂移。
    league: List[str] = field(
        default_factory=lambda: [
            "official_baseline",
            "official_hunter",
            "official_stalker",
            "official_patrol",
            "rl_v5",
            "rl_VG_v0.2",
        ]
    )
    snapshot_every: int = 10      # 每多少次 update 往联盟里加一个自己的冻结快照
    # 联盟里同时留几个快照。快照与外部对手同权重,数量一多就会吃掉大半采样、
    # 把外部对手挤没,所以卡在 2:既保住自对弈对非传递性(猜拳循环)的抑制作用,
    # 又不至于压倒外部对手。两者权重的对比见 selfplay.py 的 STATIC_WEIGHTS。
    max_snapshots: int = 2
    snapshot_temperature: float = 1.0
    # 快照闸门:最近一次评测里**官方 AI 的平均得分率**没超过这个值就不加快照,
    # 免得自对弈过早稀释梯度。0 = 不设闸门。
    snapshot_min_official_winrate: float = 0.50

    # 静态对手的配重方式:
    #   "static" = 用 selfplay.py 的 `STATIC_WEIGHTS` 定值(旧行为);
    #   "room"   = 按**剩余空间**动态配重,谁还没打服谁多拿样本。
    #
    # 动机:定值配重下,已经打到 1.000 的对手会一直吃采样额度却产不出梯度
    # (vgb3 里官方 stalker / patrol 在 48 个 update 里恒为 1.000,五个打服的
    # 对手合计占 44.8% 的采样);而正在**被打回去**的对手权重纹丝不动,没人救它
    # (同期官方 hunter 从 0.929 掉到 0.667)。这两件是同一个信号的两面。
    #
    # 注意它会**盖掉 `STATIC_WEIGHTS` 的逐项值**:动态模式下那些数字只剩下"总额度"
    # 的作用(总采样配比不变,变的是静态档内部怎么分)。想退回定值就设 "static"。
    league_weight_mode: str = "room"
    # 动态配重时,单个对手最多占静态额度的这个比例。
    # **没有它会出事**:vgb3 里只剩 rl_v5 一家有空间,纯按空间分配会把额度几乎
    # 全压到它头上 —— 而那正好是过拟合的形状。封顶把那部分让给次优的对手。
    league_weight_cap: float = 0.35
    # 胜率读数的平滑系数(EMA)。一次评测 60 局,标准误差 ±0.065,直接用会让权重
    # 跟着噪声抖;平滑之后权重跟的是趋势。越小越稳、越迟钝。
    league_weight_ema: float = 0.4
    # "剩余空间"里的保底项:某个对手打到 1.000 时空间为 0,没有这一项它的权重会
    # 归零,从此再不被采样 —— 也就再没机会发现它其实被打回去了(死区变黑洞)。
    league_room_eps: float = 0.03

    # --- 评测/落盘 ---
    eval_opponents: List[str] = field(
        default_factory=lambda: [
            "official_baseline",
            "official_hunter",
            "official_stalker",
            "official_patrol",
            "rl_v5",
            "m3_v1",
            "m3_v2",
            "rl_VG_v0.2",
        ]
    )
    eval_every: int = 40
    eval_games: int = 40
    save_every: int = 40
    keep_best: bool = True
    # 早停:最近一次评测里 `stop_opponents` 里**每一个**的得分率都 ≥ `stop_winrate`
    # 才收工。`total_steps` 是上限,这是下限,谁先到算谁。空列表或
    # `stop_winrate <= 0` = 不早停;只想看一个对手就只填一个名字。
    #
    # **为什么是"全部达标"而不是"任一达标"**:对手强弱不一时,策略会先在对它最
    # 好打的那家达标,而那一刻往往正是它练得最偏、最特化的时候。"任一"等于奖励
    # 偏科,"全部"才是"别练偏"的判据。见 tests/test_train.py 的多对手早停测试。
    #
    # **看噪声再定阈值**:`eval_games` 局的得分率标准差约 `0.5/sqrt(n)`,即默认
    # 40 局是 ±0.08。阈值卡在噪声量级上时,早停会在一次尖峰上触发,不代表策略真
    # 到了那个水平 —— 要么把阈值抬到远高于这个带宽,要么加大 `eval_games`。
    stop_opponents: List[str] = field(default_factory=list)
    stop_winrate: float = 0.0
    # 早停的**决定**局数:判据不读 `eval_games` 那次评测,而是另起一次这么多局的
    # 测量直接判。0 = 退回读 `eval_games` 那次(旧行为)。
    #
    # **为什么必须另测,而不是"低局数先判、达标了再复核"**:复核只挡得住假阳性,
    # 挡不住假阴性。vgb3 那次两个方向同时出错 —— update 84 用 60 局读成 0.617
    # (实为 0.700,低于阈值,该停没停),update 120 读成 0.725(实为 0.647,
    # 不该停却停了)。`eval_games` 局的标准差约 `0.5/sqrt(n)`,60 局是 ±0.065;
    # 阈值只要落在噪声带里,两个方向都会错。200 局(±0.034)把噪声压到判据之下。
    stop_confirm_games: int = 200
    # 便宜筛子:200 局那次测量很贵,不能每个评测点都做。先看 `eval_games` 那次的
    # 读数,只要有**任何一个** stop_opponent 低于 `stop_winrate - 这个值`,就跳过
    # 贵测量直接继续训。
    #
    # 筛子只能筛掉"明显没到"的,不能筛掉"可能到了的",所以 margin 要给足:默认
    # 0.10 ≈ 1.5 倍 60 局的标准差,把假阴性挡在筛子外面。给 0 = 不筛,每个评测点
    # 都做贵测量(最准,也最慢)。
    stop_screen_margin: float = 0.10
    # 官方保护闸:早停还要求**官方档每一个**都 ≥ 这个值。挡的是"拿官方换偏科" ——
    # 策略完全可以在官方掉下去的同时把某一个对手刷上去。`<= 0` = 关。
    #
    # 取 0.60 而不是贴着现状(那些对手平时在 0.92~1.00):这闸是**地板**不是目标,
    # 只在真被打回去时才拦,平时不参与判据。真正的日常守卫是 `stop_metric_slack`。
    guard_official_winrate: float = 0.60
    # 早停还不许在"整体已经不如自己最好那一刻"收工:要求当前综合得分率
    # ≥ `best_metric - 这个值`。它挡的是**局部最优尖峰**。
    #
    # vgb3 就是这个形状:`selfplay.py` 的 best.npz 判据(综合 metric 更大才更新)
    # 早在 update 84 就否决了 update 120 那个点,早停判据却批准了它 —— 两个判据
    # 打架,而早停赢了。这条让早停服从 best.npz 的判断。
    # 容差 0.02 是给"两个点其实一样好"留的余地,不是给退步留的。
    stop_metric_slack: float = 0.02
    # 训练时是否挂着规则层(rl_VG_v1.0 的形态)。False = 纯网络对照:
    # 不只是"规则不触发",而是**完全不走 act_mask**,所以 SCAN 也交还给网络自己选 ——
    # 只把两条规则关掉、却留着"网络不许扫"的屏蔽,那还是另一个策略,当不了对照。
    use_rules: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Config":
        raw = json.loads(s)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Config":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
