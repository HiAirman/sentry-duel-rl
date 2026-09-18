# rl_monet_v1

哨兵大战(sentry_duel V4.4)强化学习训练引擎。**零框架依赖**:规则引擎、观测构造、
网络前向/反向、PPO 全部是纯 Python + NumPy,装上 numpy 就能训。

产出物与既有的 `rl_v5-source/` 部署侧对齐:训练出来的权重可以导出成 `rl_weights.h`,
直接替换进 `rl_ai_v5.cpp` 的推理壳编译成参赛 `.so`。

两条路都通:`export` 把我们的权重**写给**部署侧,`pack` 把部署侧的权重**读回来**
当陪练(见 §七)。头文件格式是同源的,所以两个方向都不需要转换。

```
┌── rl_monet_v1 ─────────────────────────────┐      ┌── 部署侧(已有) ──┐
│ engine/  规则状态机(权威实现)              │      │ rl_ai_v5.cpp      │
│ env/     观测(obs_builder.h 的移植) + 自对弈 │ ───▶ │ rl_weights.h ←────┼─ export
│ models/  NumPy MLP + PPO                   │ ◀─── │ obs_builder.h     │
│ training/训练循环 / 评测 / 导出             │ pack └───────────────────┘
└────────────────────────────────────────────┘
```

---

## 快速开始

```bash
pip install -r requirements.txt

# 训练(默认 20 万步,单进程约 8~15 分钟)
python -m monet.cli train --run-name m1 --steps 200000

# 与规则手对战评测
python -m monet.cli eval --ckpt runs/m1/best.npz --games 200

# 导出 C++ 权重头
python -m monet.cli export --ckpt runs/m1/best.npz --out rl_weights.h

# ASCII 观战一局(默认开规则;加 --no-rules 只看纯网络)
python -m monet.cli play --ckpt runs/m1/best.npz --opponent hunter --verbose

# 从参赛包起步 + 打 rl_v5 到 80% 就早停
python -m monet.cli train --run-name vg10 --steps 200000 \
    --init-pack rl_VG_v0.2 --scan-reveal 0 --eval-every 8 --eval-games 60 \
    --stop-opponents rl_v5 --stop-winrate 0.80

# 训练曲线(训练收尾也会自动出一张 runs/<名字>/curves.png)
python -m monet.cli plot --run vg10
```

### 训练曲线读哪张表

`runs/<名字>/` 下有两张 CSV,**别混**:

| 文件 | 频率 | 内容 |
|---|---|---|
| `metrics.csv` | 每个 `eval_every` 一行 | **评测**记录(各对手得分率),对外可比,历史数字都按它读 |
| `train_metrics.csv` | **每个 update 一行** | 训练局回报/胜率 + 全部 PPO 统计 |

`eval_every=40` 时 20 万步只有 2 个评测点,画不出训练过程 —— 所以有了第二张表。
`plot` 把两张按 `step` 对齐画成四格:各对手得分率(+ 官方均值 + 快照闸门虚线)、
训练回报/胜率、PPO 损失、PPO 健康度。**缺列的面板只画一行说明,不报错**,旧 run
(没有 `train_metrics.csv`)也能画。

> 看 PPO 健康度那一格时注意:`rl_VG_v1.0` 的强制步对 `entropy` / `approx_kl` /
> `clip_frac` 的贡献**恰好是 0**,所以这三列会随 `ppo_forced_frac` 结构性偏低。
> **别把熵下降读成策略崩溃** —— 先看 `forced_frac` 那一列。

`play -v` 是**逐动作回放**:双方每个行动一行(移动带坐标、开火标命中/未命中、
被拒标原因),棋盘只在状态真的变化后打印,每个阶段以一行结算收尾。想回到
"每回合一张棋盘"的旧输出用 `--by-turn`。一局 20 回合大约 900 行,重定向到
文件里看更舒服:

```bash
python -m monet.cli play --ckpt runs/m3/best.npz --opponent official_hunter -v > replay.txt
```

跑测试(全部为自运行脚本,不依赖 pytest):

```bash
python tests/test_engine.py    # 34 条:规则逐条断言
python tests/test_mlp.py       #  5 条:含 float64 数值梯度校验
python tests/test_env.py       # 13 条:观测布局 / 掩码 / 奖励一致性 / 信念差分对拍
python tests/test_train.py     # 23 条:训练→评测→落盘→导出 的集成链路
python tests/test_official.py  # 24 条:官方档对手(几何 / 寻路 / 回合协议 / 记忆相位机)
python tests/test_pack.py      # 26 条:参赛包解析(找不到外部包时响亮跳过 8 条)
python tests/duel.py           # 对手之间循环赛(不是测试,是对拍工具)
```

`test_pack.py` 里靠真包的那几条依赖 `../rl_v5-source/`、`../m3_v1/`、`../m3_v2/` 这三个
**仓库外**的目录。包不在时它不会假装通过,而是明确打印跳过原因和试过的路径;CI 里设
`MONET_REQUIRE_PACK=1` 可以把跳过变成失败(26 条里 8 条会红,退出码 1)。

> Windows 控制台若不是 UTF-8,中文日志可能显示为乱码,`set PYTHONIOENCODING=utf-8`
> 或 `chcp 65001` 即可。

---

## 一、规则引擎(`monet/engine/`)

按 `rules.md` / `api.md`(V4.4)实现,绝对坐标存权威状态,对外通过 `Game.view(color)`
输出**已镜像**的 `Board` 快照 —— 蓝方视角下自己永远在 `(0,0)` 朝 `E`,与选手 `act()`
收到的视图一致。

覆盖的规则点(每条都有对应测试):

| 规则 | 实现位置 |
|---|---|
| 7×7、出生点、障碍、中心十字得分区 | `rules.py` |
| 每回合每方最多 3 次**成功**行动,失败不消耗(§7.1) | `game.py:apply` |
| 开局/复活后仍在出生点的首个成功 `TURN` 免费,离点即失效 | `game.py` `free_turn` |
| 视野 T 形 + **逐格独立的障碍遮挡**(§5.3) | `rules.visible_cells` |
| 火力 3×3 锥形 + **同通道**障碍阻挡 | `rules.fire_hits` |
| FIRE 后 CD=2 → 下一回合不可用、再下回合恢复 | `game.py` |
| SCAN 后 CD=3、临时视野仅本回合有效 | `game.py` `scan_reveal` |
| 击杀 +2、对方回出生点且**保留 CD** | `game.py:respawn` |
| 占点分**每方行动结束时分别结算**,互不排斥 | `game.py:end_phase` |
| 20 回合 → 平分进加时,最多 5 回合,仍平则平局 | `game.py:_check_terminal` |
| 超时判负(对方 +1,已结算行动不回滚) | `Game.force_timeout` |

镜像的正确性单独测过:蓝方在**视图坐标系**下发 `MOVE`/`TURN N` 会正确落到绝对坐标
的 `(5,6)` / 朝向 `S`。

## 二、观测与动作(`monet/env/obs.py`)

`ObsBuilder` 是 `rl_v5-source/obs_builder.h` 的逐行移植:8 个 7×7 平面 + 36 个标量 =
**428 维**,动作 **8 个**(`0=move, 1..4=NESW, 5=fire, 6=scan, 7=end`),与
`rl_weights.h` 里的 `kObsDim/kActDim` 一致(`tests/test_env.py::test_obs_dim_matches_deploy_header`
把这条钉死)。

包含原实现里的信念追踪:敌方位移信念按 BFS≤3 扩张、视野内证伪剔除、SCAN 与击杀改写信念、
路径指纹(反路径坍缩)等。

> ⚠️ 这是**人工移植**,不是共享同一份源码。两侧若要长期并行,建议把观测逻辑收敛到
> 单一实现(例如让 C++ 侧读取同一份规格),否则容易漂移。导出权重与 C++ 前向的**结构**
> 一致性是严格对齐的(`tests/test_mlp.py::test_forward_matches_reference_formula`)。

`ObsBuilder.action_mask` 只用公开信息(不含隐形敌人占格),被引擎拒绝的动作不消耗额度,
所以掩码漏掉的情况会被环境自动重试兜住。

> ⚠️ **`ObsBuilder` 必须一局一个,不能每个行动阶段新建。** `act_start` 依赖跨阶段状态
> (`my_score`/`opp_score`/`prev_act_end_pos`/`belief`/`steps_since_move`/`my_zone_turns_ago`
> …),每阶段 new 一个等于把这些全清零,喂给网络的是它训练时从没见过的分布。
> 后果不是报错而是**静默变傻**:同一份 `m3` 权重,一局一个 builder 打 `official_hunter`
> 是 `0.167`,每阶段新建则是 **0/120 全败**,连 SCAN 频率都会翻 5 倍。
> 写临时评测脚本时照 `SentryEnv` 和 `tests/diag_vs_official.py` 来。

## 三、自对弈训练(`monet/training/`)

**PPO**(掩码离散动作)+ **联盟式自对弈**:

* 对手池 = **默认 7 个固定对手**(官方四家 + 三个参赛包,`LEAGUE_DEFAULT`) +
  每隔 `snapshot_every` 次更新冻结一份当前权重的**历史快照**(滚动窗口 6 个,
  **有闸门**,见下)。`STATIC_OPPONENTS` 里注册了 14 个名字(`random` / `end` /
  三档手工 / 官方两家移植 / 四个自研规则手 / 三个包),其余的要靠 `--league` 显式
  点名。抽样权重分两档:

  | 名字 | 来源 | 抽样权重 | 说明 |
  |---|---|---|---|
  | `random` / `camper` | 手工 | 0.3 | 均匀采样 / 只占点不扫描 |
  | `baseline` / `hunter` | 手工 | 0.3 | 按规则写的启发式,回归指标 |
  | `official_baseline` / `official_hunter` | **官方 AI 移植** | 0.3 | 见 §六,决策逻辑 1:1,寻路是近似的 |
  | `official_stalker` / `official_patrol` | 自研规则手 | 0.3 | 见 §六,靠 SCAN 抓人杀 |
  | `official_ambusher` / `official_weaver` | 自研**记忆型**规则手 | 0.3 | 见 §六,行为由观测里读不到的相位驱动。**默认不在联盟/评测名单里**,要用得显式 `--league` / `--eval-opponents` 点名 |
  | `rl_v5` | **外部参赛包** | 1.0 | 见 §七,读 `../rl_v5-source/rl_weights.h`,不重训 |
  | `m3_v1` | **自研导出(冻结)** | 1.0 | 见 §七,冻结在 `m3` 的 step 753664 |
  | `m3_v2` | **自研导出(冻结)** | 1.0 | 见 §七,冻结在 `m3` 的 step 901120 |
  | `snap0..5` | 自对弈快照 | 1.0 | 滚动窗口 6 个 |

  分档的理由是**梯度**:手工四家早就被打满(`1.000`),官方四家里三家也到了
  `1.000`(`official_hunter` 是例外,`0.66~0.96`,见 §五末尾那段),对着它们练
  几乎不产生梯度;但完全撤掉又会丢掉"别把它们打回去"的约束,所以留 `0.3`。
  真正的难点是三个参赛包,得分率只有 `0.5~0.6`。
  两种状态下的实际占比(训练时会打印,不用自己算):

  | | 三个包 | 官方四家 | 自对弈 |
  |---|---|---|---|
  | 闸门关着(还没快照) | **71.4%**(各 23.8%) | 28.6%(各 7.1%) | — |
  | 闸门开、6 个快照 | 29.4%(各 9.8%) | 11.8%(各 2.9%) | 58.8% |

  权重在 `selfplay.py` 顶部的 `STATIC_WEIGHTS` / `SNAPSHOT_WEIGHT`:**不在
  `STATIC_WEIGHTS` 里的名字自动吃 `DEFAULT_STATIC_WEIGHT` = 1.0**,所以加一个
  强对手只要注册,不用回来改权重表(`tests/test_train.py::test_default_rosters_do_not_drift`
  焊住这条)。低权重档那张表叫 `LOW_WEIGHT`(以前叫 `HANDWRITTEN`,后来官方四家也
  进了这一档,名字就不对了)。

  **上面那张表描述的是 `--league-weight-mode static`。默认已经是 `room`**(见下),
  那种模式下 `STATIC_WEIGHTS` 的**逐项值被盖掉、只剩总额度有用**,表里的百分比
  不再成立。

#### 动态配重(默认):谁还没打服谁多拿

`--league-weight-mode room` 不再用定值,而是每次评测后按**剩余空间** `1 - 得分率`
重分静态档的额度:

- 打到 `1.000` 的对手空间为 0、让出额度 —— 它们本来就不产生梯度,只是白占采样;
- **被打回去**的对手空间变大、自动多拿样本。清死区和保护是同一个动作。

单个对手有 `--league-weight-cap`(默认 35%)封顶。没有它,联盟里常常只剩一家还有
空间,额度会几乎全压到它头上 —— 而那正是过拟合的形状。

代入 `runs/vgb3/best.npz` 的 200 局实测(vgb3 的真实形状:官方 stalker/patrol 恒在
`1.000`、官方 hunter 从 `0.929` 掉到 `0.667`):

| 对手 | 定值占比 | room 占比 | 得分率 |
|---|---|---|---|
| official_hunter | 4.2% | **22.2%** | 0.740 |
| official_stalker | 4.2% | 2.3% | 1.000 |
| official_patrol | 4.2% | 2.3% | 1.000 |
| rl_best040 | 13.9% | 3.8% | 0.980 |
| rl_v5 | 27.8% | 25.2% | 0.700 |

额度**总额守恒**,所以静态档与快照档的配比不变,`max_snapshots` 对自对弈的封顶依然
成立。胜率读数走 EMA 平滑(`Config.league_weight_ema`):一次评测 60 局的标准误差是
±0.065,不平滑的话权重会跟着噪声抖。想退回定值:`--league-weight-mode static`。

#### 早停:三条闸同时成立才收工

`--stop-opponents` + `--stop-winrate` 只是第一条。

1. **达标** —— 名单里每一个对手都到线。判据用**另起的一次** `--stop-confirm-games`
   局(默认 200)测量,不是 `--eval-games` 那次的读数。
2. **官方保护闸** —— 官方档每一个都 ≥ `--guard-official-winrate`(默认 0.60)。
   挡的是"拿官方换偏科":策略完全可以一边把 rl_v5 刷上去、一边把官方 hunter 打下去。
3. **不许在退步的点收工** —— 当前综合得分率 ≥ 本轮最优 − `--stop-metric-slack`
   (默认 0.02)。

第 1 条为什么必须**另测**、而不是"低局数先判、达标了再复核":复核只挡得住假阳性,
挡不住假阴性。`eval_games` 局的标准差约 `0.5/sqrt(n)`(60 局 ±0.065),阈值落在噪声
带里时**两个方向都会错**。`runs/vgb3` 正是两个方向同时错的实例 —— update 84 用 60 局
读成 `0.617`(实为 `0.700`,**该停没停**),update 120 读成 `0.725`(实为 `0.647`,
**不该停却停了**)。整轮跑的命运由噪声决定。

第 3 条挡的是同一个实例的另一面:`best.npz` 的判据(综合得分率更大才更新)早在
update 84 就否决了 update 120,而早停判据批准了它 —— **两个判据打架,早停赢了**。
这条让早停服从 `best.npz` 的判断。

那次高局数测量只在"便宜筛子"放行时才跑(有对手低于 `阈值 − --stop-screen-margin`
就跳过),而且**同一个评测点只判一次**,所以不会拖慢每个 update。

### 快照闸门:打不过官方 AI 就先别自对弈

快照**不是到点就加**。只有当最近一次评测里**官方 AI 的平均得分率 > `0.40`** 时才加,
否则跳过一个快照周期:

```
     ★ 新最优(综合得分率 0.0000)→ best.npz
     快照闸门关(官方胜率 0.135,需 >0.40)
    -> 跳过快照(闸门关(官方胜率 0.135,需 >0.40))
```

理由:`m1`~`m3` 从没跟官方 AI 练过,一开始就把大半采样额度喂给自对弈,梯度全花在
"打自己"上。闸门关着的阶段联盟只有 7 个固定对手(官方四家各占 7.1% + 三个包各 23.8%),先把官方这四家啃下来
再引入自对弈。命令行 `--snapshot-min-official-winrate 0` 可以关掉闸门恢复旧行为。

* 判定用**最近一次评测**的结果,所以闸门的取样周期就是 `eval_every`;
* 评分口径是 `eval_opponents` 里所有官方对手的平均得分率(靠**工厂上的 `is_official`
  标记**识别,以后多写一个官方档对手不用改代码)。标记挂在工厂上而不是靠实例化探测:
  注册表里有 `rl_v5`(要读仓库外的包),实例化探测会让"import 一个训练模块"附带
  文件系统副作用,包缺失时连 `export` 都跑不了;
* ⚠️ **名单是 `official_baseline` / `official_hunter` / `rl_v5` 三家**(钉在
  `tests/test_train.py::test_official_opponents_are_registered_and_flagged`),
  不是"官方四家"、也不是"官方四家 + rl_v5"。`rl_v5` 是**算进去的** —— 闸门问的是
  "能不能打赢手上最难的对手",所以把外部强敌也计进均值是有意的。代价是闸门语义
  变成了"两家移植 + 一份外部包"的混合均值。
  四个**自研**规则手(stalker / patrol / ambusher / weaver)则**都不在**这个名单里:
  它们的工厂没套 `_official`,名字里的 `official_` 只表示"同一档的规则手"。
  要动这个集合,先回 `test_train.py` 对口径 —— 它会改 `guard_official_winrate`
  和快照闸门的语义,现有 run 的读数跟着变;
* ⚠️ 以后往这个名单里加更强的对手,均值会**更难达标**(分母多了强敌),沿用 `0.40`
  前请重新标定;
* ⚠️ `eval_games` 默认只有 40,官方胜率的标准误约 5.7%,**卡在 0.40 附近时闸门会来回跳**。
  要它稳定就用 `--eval-games 200`(标准误约 2.5%);
* ⚠️ 如果 `eval_opponents` 里没有官方对手,这个条件根本测不出来 —— 此时会打一行警告并
  **放行**,而不是静默地一个快照都不加。

* 每局随机执红/执蓝。
* 奖励与计分事件严格对应:击杀 `+2`、阵亡 `-2`、占点 `+1`,终局胜 `+1` / 平 `0` / 负 `-1`。
  `tests/test_env.py::test_episode_reward_matches_score_difference` 校验
  `回报 == 我方得分 - 2×阵亡 + 终局奖励`。
  唯一的例外是**默认关闭**的 `--scan-reveal`(见下)。
* 熵系数从 `ent_coef` 线性退火到 `ent_coef_final`。

### `--scan-reveal`:把策略往"多用扫描"推

策略几乎不扫描(实测 `1.2` 次/局,而两个官方 AI 是 `5.7~7.0` 次/局)。要让扫描多起来,
唯一干净的办法是给它一个价:SCAN 成功、且**确实扫出了之前看不见的敌人**时额外给一份奖励。

```bash
python -m monet.cli train --run-name m4 --steps 400000 --scan-reveal 0.1
```

* 默认 `0` = 关闭,奖励严格等于计分事件(`回报 == 比分差` 的契约不受影响);
* 只奖"扫出来了",空扫不给,敌人本来就看得见时的扫描也不给;
* 只作用于**训练** —— 评测走默认奖励配置,数字不受影响;
* 它确实会打破「回报 == 比分差」这条不变量,所以是显式 opt-in。

⚠️ **别用"强制扫描"代替它。** 实测把策略改成"每个阶段只要有 SCAN 就先用掉":
打 `official_baseline` 从 `0.517` 掉到 `0.000`,打 `official_hunter` 从 `0.100` 掉到 `0.050`。
原因是扫描要吃掉 3 个动作里的 1 个,而策略本来就靠**移动**获取视野 —— `m3/best` 开火
`4.2` 次/局、其中约 `92%` 是有效击杀(`3.87`),它出手时基本不落空。所以视野不是瓶颈,
动作经济才是。塑形的作用是让策略自己学会"什么时候值得扫",不是替它决定。

> 60k 步的对照实验(`--scan-reveal 0.1` vs 默认)确认这个旋钮**确实推得动**:
> SCAN 从 `3.0` 次/局涨到 `6.9` 次/局。但 60k 步的策略本身还是退化的(几乎不移动),
> 所以这只是"梯度方向对"的验证,不代表胜率会涨 —— 要看得跑完 40 万步。

关键超参见 `monet/training/config.py`,命令行可覆盖常用项。产物落在 `runs/<name>/`:

```
runs/m1/
  final.npz    最后一版权重(+Adam 状态)
  best.npz     评测综合得分率最高的一版
  last.npz     周期性存档(可 --resume)
  metrics.csv  每次评测的完整指标
```

## 四、部署(`monet/training/export_cpp.py`)

`export` 生成的 `rl_weights.h` 与原文件符号完全一致(`kW0..kW3`、`kB0..kB3`、
`kLN1..3Gamma/Beta`,外加推理不用的 `kW4/kB4` 价值头),把它放进 AI 源码包替换同名文件
重新编译即可 —— `rl_ai_v5.cpp` 的 `forward()` 一行都不用改。

网络结构 `428 → 512 → 512(+残差) → 512(+残差) → 8/1`,LayerNorm + GELU(精确 erf),
与 C++ 侧逐层同构。

### 带记忆的架构(v2,`arch=gru`)

导出器**按架构选符号表**(`factory.tensor_table`),所以同一个 `export` 命令两种网络都
能导。GRU 包在前 16 个 MLP 符号之后多接 14 个,总共 30 个张量:

```
428 → 512 → 512(+残差) → 512(+残差) → a3        ← 与 v1 逐层同构的编码器
        h  = GRU_128(a3, h_prev)                ← 逐决策点推进,每局从零开始
        feat = a3 ⊕ (kMemProj·h + kMemBias)     ← 残差回注(P/bP 零初始化)
        8/1 = kW3/kB3 · feat
```

部署侧多两件事,都写在 `rl_VG_v2.0/rl_VG.cpp` 里:

1. **隐状态必须是全局的**(`g_h[kGruHidden]`)—— 引擎每个阶段只调一次 `act()`,记忆
   要跨这些调用活下来才叫记忆;`board.turn == 0` 时和 `g_ob`/`g_rules` 一起归零。
2. **sigmoid 必须在 ±60 处截断**,与 Python 侧 `gru._SIG_CLIP` 一致 —— 不截断在
   float32 下 `exp(-x)` 会在 x < −88 溢出成 `inf`。

头部另有 `kRecurrent` / `kGruHidden` 两行**显式**架构标记。不靠"有没有 `kGruHidden`"
去猜是刻意的:隐式约定少写一行就静默退化成纯 MLP,而表现只是"发出去的包比测出来的弱"。

> **本机没有 C++ 工具链**(g++/clang++/cl/MSVC 全无),所以部署侧代码**没有被编译过**。
> 补偿手段是 `tests/test_gru_deploy.py`:把 `rl_VG.cpp` 的前向逐行转写成 Python 做
> 逐位对拍(单步 9.1e-15、24 步 1.2e-14、真实包 16 步 6.2e-15),外加两条**词法**检查
> 盖住转写结构上表达不了的两个坑(先存 `h_prev` 再写 `h`、sigmoid 截断值)。
> 这些能抓规格漂移,抓不到语法/类型错误 —— **上线前必须真编译一次**。

**⚠️ 单文件 8MB 上限**:评测服务对多文件源码包限制单文件 ≤8MB(api.md §9),而 512 宽的
权重头光文本就有 **10.3MB**,直接提交会被拒。所以 `export` 默认按 7MB 自动拆分:

```
rl_weights.h          0.00 MB   只放 banner + #include 各分片
rl_weights_part1.h    6.64 MB   前 8 个张量
rl_weights_part2.h    3.68 MB   后 8 个张量
```

拆分是**无损**的(不降精度),推理代码除 include 外不动。`--max-file-mb 0` 可关掉拆分
(单文件提交或本地自用)。改完用 `python tests/check_pack_headers.py <包目录>` 自检
include 链/括号/符号,不需要 C++ 编译器。

> 该上限对**单文件源码提交**不适用(api.md 只对 §9 的多文件包规定)。
> 另外 api.md §9 提到「主源码优先取 `my_ai.cpp`」—— 若选开源展示,主推源码文件名
> 建议叫 `my_ai.cpp`,否则展示的源码不一定是这个。

---

## 五、实测(CPU,单进程)

> **`rl_VG_v2.3`(GRU 记忆层)的战绩与探索过程见 [`EXPLORATION_v2.md`](EXPLORATION_v2.md)** ——
> 完整的探索报告(架构、验证手段、踩过的坑、以及一条负面发现:记忆对胜率的贡献
> 落在测量噪声里,提升主要来自"在含 `rl_VG_v1.0` 的联盟里继续训练")。
> 当前交付读数:4 个目标对手 ×**400 局** × 两个独立种子,**8 格全过 0.70**,
> 最紧的一格是 `rl_v5` 的 0.785(seed 90000:0.785 / 0.960 / 0.973 / 0.970;
> seed 424242:0.835 / 0.963 / 0.963 / 0.945)。它相对 `rl_VG_v2.0` 涨在哪里、
> 以及**为什么这不能算推翻报告 §6** 见 §7。
> 中途版 `runs/v2.1`(多训 2.4 倍步数、记忆反而缩小)与 `runs/v2.3/best.npz`
> (最低格与 `final` 打平、均值更低)**未打包**。

三次运行:`m1` = 6 万步(约 4 分钟),`m2` = 40 万步(约 22 分钟),`m3` = 40 万步
(约 25 分钟,官方 AI 的 BFS 更贵)。各权重对固定对手打 200 局:

| 对手 | m2 得分率 | m2 净胜分 | m3/best | m3/final | 备注 |
|---|---|---|---|---|---|
| random | 1.000 | +15.62 | — | — | 手工,不在默认评测集里 |
| baseline | 1.000 | +28.73 | 1.000 | 1.000 | 手工 |
| hunter | 1.000 | +28.77 | 1.000 | 1.000 | 手工 |
| camper | 1.000 | +31.52 | 0.995 | 1.000 | 手工 |
| **official_baseline** | **0.210** | **−3.41** | **0.470** | **0.540** | 官方移植 |
| **official_hunter** | **0.220** | **−4.92** | **0.220** | **0.225** | 官方移植 |
| 综合(5 家均值) | 0.686 | | 0.737 | 0.753 | |

> ⚠️ **评测表的数字只在名单相同时可比。** `evaluate()` 对整份名单只用**一个**
> `NetPolicy`(`evaluate.py:93`),而 `play_match` 只重设环境的种子、**不给策略重设**
> ——所以策略采样的随机流是**跨对手连续**的。同一个检查点、同样 40 局,单独跑
> `--opponents rl_v5,m3_v1` 与跑全名单,`rl_v5` 会从 0.550 变成 0.625。以后加了
> 对手或改了顺序,旧数字对不上是正常的,别当成回归。

### 这两组数字要分开读

四个手工规则手从 `m2` 起就是 **200 胜 0 平 0 负全打满** —— 早就饱和了,再堆步数对
这批规则手没有区分度。而面对官方 AI 的移植版,`m2` 只有 **21%/22%** 的得分率。

`m3` 把 `official_baseline` 从 `0.210` 拉到 `0.470~0.540`,但 **`official_hunter` 一步没动**
(`0.220 → 0.220/0.225`)。拆开看输在哪(`tests/diag_vs_official.py`,`m3/best` 各 60 局):

| | vs official_baseline | vs official_hunter |
|---|---|---|
| 总分差 | −1.63 /局 | −5.30 /局 |
| **占点分差** | **−2.03 /局** | **−3.43 /局** |
| 击杀差 | +0.20 /局 | −0.93 /局 |
| 对方占点 / 我方占点 | 11.3 / 9.3 | 11.4 / 7.9 |
| 对方 SCAN / 我方 SCAN | 7.0 / 1.2 | 5.7 / 1.1 |

**输的是占点,不是对枪。** 对 `official_baseline` 我方击杀还小赢,总分却输 —— 两个官方
AI 每局稳定在十字上薅 `11.3~11.4` 个占点分(约 57% 的阶段),我方只有 `7.9~9.3`。
机制很直白:`baseline_ai.cpp` 的分支就是「`in_score_zone` → 直接 `return`」,它一旦钉在
十字上就每阶段白拿 1 分,想让它停下来只能杀掉它。

> ⚠️ **`m1`/`m2`/`m3` 的训练联盟里根本没有官方 AI。** 三个检查点里存的 `league` 都是
> `['random','baseline','hunter','camper']` —— 官方 AI 只出现在**评测集**里(`eval_opponents`
> 含它们),所以每轮评测都在给这个策略"打一个它从没练过的对手"的分。
> `m3` 对 `official_baseline` 的 `0.470`、以及对 `official_hunter` 的 `0.220` 都是
> **零样本迁移**,不是练出来的;这也解释了为什么 `official_hunter` 从 `m2` 到 `m3`
> 一动不动 —— 两轮都没练过它,自然都在同一个水平上。
>
> 根因是 `cli.py` 的 `--league` 默认值一直停在旧的 4 家手工对手(Config 里的默认值早就
> 改成 6 家了,被命令行默认值盖掉)。**已修**:现在默认取 `LEAGUE_DEFAULT`。
> 下面这张拆解表说明的是"**怎么输的**"(占点争不过),而不是"为什么没练出来"。
>
> ⚠️ 上表还是在**旧采样权重**(手工对手 `1.0`)下训的。权重改成 `0.1` 之后官方 AI 的
> 采样占比从 `13.3%` 涨到 `17.5%`,而且是真的进联盟了 —— 要对比得重新训一轮。

> ⚠️ `eval_games=40` 太小:单列标准误约 8%,`best.npz` 基本是按噪声挑的(见上表 ——
> `m3/final` 的 200 局重测反而比 `best` 好)。要挑权重就把 `--eval-games` 提到 200。

### ⚠️ 规则层(`rl_VG_v1.0`)的代价:实测是负的,别当成"加了规则就更强"

`rl_VG_v1.0` 的设想是"击杀 / SCAN 这两件事是纯几何、可精确判定的,从学习问题里摘出去,
网络只学残差"。**在没带规则训过的网络上,这个设想是错的**,而且错得很大。

`tests/diag_vg_rules.py` 在 `rl_VG_v0.2` 权重上对 `rl_v5` 各打 100 局:

| 配置 | 得分率 | SCAN/局 | 强制占比 |
|---|---|---|---|
| **纯网络(基线)** | **0.510** | – | – |
| 只开击杀 | 0.440 | 0 | 3.9% |
| 只开扫描,阈值 2(原规格) | **0.110** | 5.84 | 11.4% |
| 只开扫描,阈值 4 | 0.230 | 2.72 | 5.2% |
| 只开扫描,阈值 8 | 0.310 | 0.83 | 1.6% |
| 只开扫描,阈值 12 | 0.470 | 0.21 | 0.4% |
| 击杀 + 扫描(阈值 6) | 0.240 | 1.51 | 6.7% |

**代价随扫描频率单调上升**,而且**不是动作开销**:阈值 4 时扫描只吃掉 1.8% 的动作,
却每扫一次掉 **~1.4 分**(我方得分 `13.50 → 11.27`,而局均步数几乎不变)。

机制:SCAN 把 `opp_visible` 翻成 `True`,于是网络切进它训练过的"看得见敌人"分支 ——
**那个分支会离开得分区去追**。而每阶段站在得分区里 `END` 就是 **+1 分**,所以一次扫描
几乎正好抵掉一分。README §五 上面那句"我方输在占点、SCAN 比对方少"是真的,但
**结论不能反过来推成"那就强制它多扫"**:对方是"扫得多 **且** 守得住点",v0.2 只学会了
后者,把前者硬塞进去就会把后者一起挤掉。

> **但这只是"没带规则训过的网络"的代价,不是稳态代价。** 规则真正不可回收的部分只有
> 那 1.8% 的动作开销 —— 剩下的都是"网络没见过这个观测"。带规则训练之后网络应当学会
> "看得见敌人时照样守点",这部分能收回来多少,就是 `runs/vg10` 对照
> `runs/vg10_norules`(纯网络对照,`--no-rules`)要回答的问题。
> **判据是先量后训**:除非带规则的曲线追平对照,否则不要用带规则的权重交付。

### 753664 → 901120:换来了 `rl_v5`,代价是 `official_hunter`

接上参赛包之后联盟换成"官方四家 + 三个包",`m3` 又往下训到 step 901120。用冻结的
`m3_v1`(753664)当锚**逐个对手单独测**(单对手名单,避免跨对手的随机流干扰,各 100 局):

| 对手 | `m3_v1`(753664) | `best.npz`(901120) | 判读 |
|---|---|---|---|
| `official_hunter` | 0.870 / 0.960 / 0.930 | **0.660 / 0.780 / 0.700** | ⚠️ **退了 ~21 个百分点**(三个种子一致) |
| `official_baseline` | 0.980 | 0.980 | 没动 |
| `rl_v5` | 0.55~0.625 | **0.775** | 涨了 |

（`official_hunter` 那行是 seed=1000/2000/3000,100 局,单对手的随机误差约 ±4.7%,
21 个百分点的差不是噪声。净胜分同步从 `+6.4` 掉到 `+2.5`。）

```bash
python -c "import sys;sys.path.insert(0,'.');from monet.store import load_checkpoint;from monet.training.selfplay import STATIC_OPPONENTS;from monet.training.evaluate import evaluate;import monet.pack as P;best,m,_=load_checkpoint('runs/m3/best.npz');[print(l, round(evaluate(n,{'official_hunter':STATIC_OPPONENTS['official_hunter'](1000)},games=100,seed=1000)['official_hunter']['winrate'],3)) for l,n in {'m3_v1':P.load_pack_net(name='m3_v1'),'best':best}.items()]"
```

**为什么没被 `best.npz` 的挑选机制挡住:** 综合指标是**名单均值**,`rl_v5` 涨的那部分
盖过了 `official_hunter` 跌的部分,于是"总分更好"的权重被存成 `best`。这正是锚的用处 ——
均值会掩盖单对手的退步,锚那一行不会。要修就往联盟里保留/加大 `official_hunter` 的
采样额度,而不是只看综合分。

**结论:之前那句「打满规则手不等于能打榜」是对的,而且差距很大。** 手工启发式
(`baseline`/`hunter`/`camper`)是同一个人按同一份规则写的,策略空间小、套路固定,
被吃透很快;官方 Baseline 的占点路线和中距离火力覆盖是另一套东西,`m2` 没见过。
用 `tests/duel.py` 做对手循环赛(40 局/对,行方视角)可以看清层次:

| | baseline | hunter | camper | official_baseline | official_hunter |
|---|---|---|---|---|---|
| **baseline** | — | 0.60 | 0.55 | 0.03 | 0.40 |
| **hunter** | 0.47 | — | 0.38 | 0.03 | 0.40 |
| **camper** | 0.50 | 0.70 | — | 0.00 | 0.30 |
| **official_baseline** | 0.95 | 0.95 | 1.00 | — | 0.62 |
| **official_hunter** | 0.60 | 0.60 | 0.65 | 0.40 | — |

官方移植版对三个手工规则手是碾压(0.95~1.00);两个官方移植版之间 `official_baseline`
占优(0.62)。所以 `m2` 现在的真实水平大致是「稳吃手工规则手、打不过官方逻辑」。

### m1 的训练过程(每 2 万步一评,30 局)

| 步数 | 训练得分率 | vs baseline | vs hunter | vs camper |
|---|---|---|---|---|
| 20k | 0.120 | 0.00 | 0.00 | 0.00 |
| 41k | 0.355 | 0.13 | 0.20 | 0.37 |
| 61k | 0.678 | 0.60 | 0.80 | 0.83 |

> ⚠️ 上表是 `m1` 在**旧联盟**(只有 4 个手工规则手)下训出来的,只用于确认
> 「引擎通、策略在变强」。`league` 默认值现在是 7 个固定对手(官方四家 + 三个参赛包),
> **要复现/超越 `m2`,得用新联盟重新训一轮** —— `m2` 是在旧联盟下训的,
> 它没见过官方对手。

导出的 `rl_weights.h`(10.3 MB,16 个数组)维度与部署侧完全一致
(`kW0[219136]`=512×428、`kW3[4096]`=8×512、`kW4[512]`、`kObsDim=428`、`kActDim=8`),
替换进源码包重编译即可,`rl_ai_v5.cpp` 一行都不用改。

## 六、已知边界

* **速度**:纯 Python 规则引擎约 2500 步/秒,端到端训练约 250~300 步/秒(CPU)。
  上量训练建议把 `engine/` 换成 C++/pybind 实现,观测与训练侧不用动。
* **官方 AI 是移植版,不是官方二进制**(见下节):`official_baseline`/`official_hunter`
  的决策逻辑是 1:1 照搬的,但寻路是我方实现。**评测数字不能当打榜预测。**
* **对手策略不共享**训练策略,`snapshot` 只增不减(滚动窗口 6 个),没有做 PSRO / 优先级采样。
* **默认联盟依赖仓库外目录**:`rl_v5` 要读 `../rl_v5-source/rl_weights.h`、`m3_v1` 要读
  `../m3_v1/rl_weights.h`,两个都在 `rl_monet_v1/` 之外。包缺失时训练会在**启动阶段**
  大声失败,而不是静默少一个对手 —— 悄悄降级会让训练数据在不知情的情况下变样。
  ⚠️ 检查在构造 `League` 时做(即 `SelfPlayTrainer.__init__`)。`--resume` **不会**
  恢复检查点里存的那份 `cfg`(只取 `step`/`update`,`cli.py:52-56`),所以续训用的是
  **当前**的名单:从 `m3` 续训会自动带上 `m3_v1`(以及包里冻结的那份自己),
  而包挪了地方就得 `--pack-dir` 指回去。临时绕开就 `--league` 手动列出
  不含这两个包的名单(见 §七)。

### 官方 AI 移植(`monet/env/official.py`)

`official_baseline` / `official_hunter` 从 `baseline_ai.cpp` / `hunter_ai.cpp` 移植而来。

**决策逻辑(优先级、分支顺序、动作计数)是 1:1 照搬的**,包括那些看着奇怪的细节
(比如 baseline 在必杀分支里 turn→fire→scan 完全不检查额度)。但这两个 AI 依赖
`utils.h` / `navigation.h`,而这两个文件不在手边,所以:

| 函数 | 状态 |
|---|---|
| `same_pos` `blocked` `in_score_zone` `best_turn_to_face` `in_fire_range` | 语义明确,按规则重写 |
| `can_step_fire` | 照 `hunter_ai.cpp` 的同名函数重写(含自带的障碍射线判定) |
| `advance_toward` | ⚠️ **寻路核心是我方实现**(BFS 最短路 + 同长度优先不转身),非官方原版 |

官方的 `advance_toward` 还带可选参数(hunter 传了敌人位置与朝向,疑似"边压边找"),
本实现忽略。**官方版在 (1,1) 障碍附近的路线可能与这里不同** —— hunter 的路径点
`(2,1)`/`(1,2)` 正好贴着那个障碍,而"同长度优先不转身"这条就是为了在那里不来回摆
(`test_advance_toward_prefers_not_turning` 钉住了它;没有这条时会退化成
`(2,1)↔(3,1)` 无限往返、整局 0:0 拖进加时)。

**结论:这是「官方决策逻辑 + 我方寻路」,不是官方 AI 本身。** 当陪练、当难度基准都
没问题;拿它评测出来的数字**不能当打榜预测**。要准确评测请走官方 API(api.md §10,
榜单本来就含官方 Baseline / Hunter)。

### 自研官方档:`official_stalker` / `official_patrol` / `official_ambusher` / `official_weaver`

这四个**不是移植,是自己写的规则手**,所以上面那条保真度警告对它们不适用 ——
它们没有"原版"可言,口径就是我们自己定的。

**`is_official = True` 写在类上,但那不决定任何事。** 真正划档的是
`selfplay.py` 里注册工厂上的标记:`OFFICIAL_OPPONENTS` 是拿
`getattr(工厂, "is_official", False)` 筛的,而工厂是那些 lambda。这四个的工厂
**没套 `_official`**,所以它们**不在**官方档名单里,放进评测名单也**不会**触发
`guard_official_winrate`(官方保护闸)或快照闸门。成员名单钉在
`tests/test_train.py::test_official_opponents_are_registered_and_flagged`。

| | 打法 | 记忆型? |
|---|---|---|
| `official_stalker` | 靠 SCAN + 视野找人击杀,直扑敌人;没有情报就朝敌方出生角搜 | 否 |
| `official_patrol` | 沿棋盘边界环(24 格)巡逻,SCAN 一冷却就用(实测 7.1 次/局,是该规则的上限);敌人进入"单回合可击杀"范围就直接打死 | 否 |
| `official_ambusher` | 每击杀一次,就在**三个自己的行动阶段内**绕到离自己开火位置最远的角躲起来(提前到就提前结束),躲完再出来找人杀 | **是** |
| `official_weaver` | 固定周期出没:主动找 4 个阶段 → 消失 3 个阶段 → 往复,每次消失轮换一个角 | **是** |

#### 得分区:不写策略,正反两面都不写

**这四个 AI 对得分区一无所知。** 它们不占点、也不绕着走 —— 推进目标永远是敌人
本身(`hunt_target`),路怎么短怎么走,穿过中心十字就穿过。

这一条最早写错了:"不占点"被实现成了"绕着得分区走"(`kill_in_one_turn(...,
forbid=zones)` 管走位、`advance_toward(..., avoid_extra=zones)` 把得分区从最短路
里排除)。那是**反向的占点策略**,同样是特意为得分区写策略,而且会从行为里漏出来:
对手只要看它绕开中心就能推断打法。对两个记忆型对手尤其致命 —— 它们的相位本该是
观测里**唯一读不出来的东西**,一旦能被"它又绕中心了"这种走位反推出来,记忆价值
就没了。现在 `forbid` / `zone_safe_target` / `_outside_zone_cells` 都已删除,
`test_zone_avoidance_helpers_stay_deleted` 守着它们别长回来。

副作用要记住:它们**会**顺路拿占点分。所以 `OfficialAmbusher` 判定"刚杀过人"用的是
**比分差 ≥ 2**,靠的是结算频率(`Game.end_phase` 里占点一次只 +1,击杀 +2),而不是
"它们不占点"—— 口径变了这条判据依然成立。

实测(`m3/best`,40 局):`official_stalker` 得分率 **0.700**、`official_patrol` **0.525**
(净胜分 −0.15 —— m3 只能跟它打平)。

> 拿它们当难度基准要看准它们弱在哪:它们**不规划**占点,所以对局里不会出现"守中心
> 十字"这种压力,占点分只在路过时顺手拿 —— 提供的是**击杀压力**,不是比分压力。
> **别把这条读成"它们不占点"**:`test_self_authored_opponents_do_cross_the_score_zone`
> 钉的正是反面,stalker / ambusher / weaver 都会走进得分区。唯一的例外是
> `official_patrol`,它的路线是边界环,恰好不与任何得分区相交 —— 属于路线选的结果,
> 不是绕开。所以 m3 打它们的得分率反而比打移植版低不了多少(对 `official_baseline`
> 是 1.000)。

#### 两个记忆型对手为什么"吃记忆"

它们的行动由**内部相位**驱动,而这个相位同时满足两条:

1. **观测里读不到**。428 维观测只有即时量(位置、朝向、CD、比分、回合数),没有任何
   一个字段在说"对手现在处于什么模式"。
2. **由事件推进,而不是由棋盘上可读的量推进**。凡是"位置 / 回合数的函数"的行为,
   无记忆的网络查一张足够大的表就能逼近 —— 那些量本来就在观测里。而"某件事发生之后
   又过了几个阶段",单帧里根本不存在,只能靠**差分一个可观测量**得到。

差分正是循环结构擅长、MLP 不擅长的事 —— MLP 只能把它压成一张巨大的
(比分 × 回合 × 位置) 查表,换个起点权重就失效。

`official_ambusher` 的藏身点取**自己开火位置**的最远角,有两个理由都跟记忆有关:

* **不能是常量**。固定躲同一个角会被直接记成事实,对手退化成普通靶子。
* **不能只靠当前帧**。它由"三个阶段前那次击杀发生在哪"决定,而那个位置在需要它的
  时候早已不在画面里。

> 训练侧真正可学的是那个**窗口**:"它刚杀完人,接下来两三个阶段既不扫描也不追人,
> 这段时间去占点最划算"。无记忆的策略拿不到这个窗口的开合时刻。

实现上有一点值得记:官方 `act()` 一次调用内部就走完整回合(最多 3 个动作,且控制流
依赖每个动作返回的观测)。所以 `Opponent.turn()` 写成**生成器** —— 每个 `yield` 出去的
动作由环境调 `game.apply` 执行、结果 `send` 回来(`SentryEnv._run_opponent_phase`)。
这样对手造成的击杀仍然流经环境,我方的 `-2` 阵亡奖励和 `deaths` 计数不会漏。
(`test_official_opponent_kills_still_reach_the_reward` 钉住了这条:对手若绕过环境
自己驱动引擎,该测试会挂。)
* **超时规则**在训练里没有建模(不模拟 1 秒墙);`Game.force_timeout` 只在评测/规则测试里用。
  另外 §7.3 的「崩溃判负」在 API 里没有对应入口,只有一个 +1 惩罚钩子,表达不了判负。

### 与官方引擎待确认的三处(需要 `utils.h` / 官方引擎才能定论)

1. **`force_timeout` 会不会给超时方结算占点分**。本实现走字面读法「超时结束该方阶段 ⇒
   阶段结束就结算」,所以超时方站着得分区时双方各得 1 分(实测 1:1)。若官方把「放弃」
   理解为什么都不给,这里就不一致。**对刷分无影响**(两边都赚),但值得对齐。
2. **开火当次 `observation.fire_cd` 报什么**。§4.3 说「开火回合为 0」,本实现报 `2`
   (内部值,回合末才 -1)。所有判定都在阶段开头读 `fire_cd == 0`,所以行为无差异,
   但读中间态的 AI 会看到不同数字。
3. **`can_see(pos, f, pos)` 的约定**。本实现返回 `True`(同格可见),
   `visible_cells` 不含自己所在格。这是**故意的**:击杀者站在对方出生点上时,复活会让两人叠在
   同一格(`tests/test_engine.py::test_killer_can_share_spawn_with_victim`),此时
   `directly_visible` 必须为真,否则会出现隐形的敌人。代价是
   `ObsBuilder._subtract_visible_cells` 会多清掉一格信念(实测占 0.046% 的调用),
   而 C++ 侧走 `utils.h::can_see`,该文件不在本仓库,无法逐位比对。

> `obs.py` 是 `obs_builder.h` 的**逐位移植**,两边必须保持一致;上面第 3 条这类分歧
> 只能改引擎侧或接受,不能去「修」`obs.py`。已用独立转写的 C++ 版本做过差分对拍:
> 默认地图 + 非对称地图共 80 局(含强制击杀/阵亡/SCAN/传送),约 3.3 万次向量+掩码比对、
> 7118 次内部状态比对,**0 差异** —— 上面第 3 条是唯一残留项。

> **「不能修」的准确含义是不能改语义,不是不能改实现。** `_dilate_belief` 与
> `_subtract_visible_cells` 做过一次纯性能的等价重写(信念当 49 位掩码、邻居关系
> 按障碍集合预计算、可见集一次算好再查表,原来每个格子都要重算一遍
> `can_see` → `visible_cells` → `set(obstacles)`)。两者的行为契约由
> `tests/test_env.py::test_belief_rewrite_is_bitwise_equivalent_to_the_slow_version`
> 焊死:重写前的逐格原版逐行抄在该测试里当规格,在真实对局上**每次调用**两版都跑、
> 逐格比对。`test_can_see_equals_visible_cells_membership` 单独钉住那次重写依赖的
> 恒等式(`visible_cells` 从不返回障碍格,所以 `can_see(t)` 等价于
> `t == my_pos or t in visible_cells(...)`),上面第 3 条那个多清一格的怪癖因此原样保留。

### 引擎侧已修的契约 bug

`Game.legal_actions` 曾与 `Game.apply` 不一致:漏掉同朝向 `TURN`(§3.4 明确它是成功行动),
且不检查行动额度、会返回 `apply` 必然拒绝的行动。现已对齐并由
`test_legal_actions_contract_matches_apply` 压测钉死。训练路径不依赖该函数。

## 七、参赛包当陪练(`monet/pack.py`)

把参赛包里的 `rl_weights.h` 直接接进联盟和评测集,不重新训练。注册表在
`pack.KNOWN_PACKS`:**包目录里直接放 `rl_weights.h`**(不要多套一层,
`resolve_pack_dir` 只认 `<包目录>/rl_weights.h`)。外部包与 `rl_monet_v1/` 并列,
自研包在仓库内(`PACK_ROOTS` 两个根按序找):

| 注册名 | 目录 | 架构 | 来源 | 算官方? |
|---|---|---|---|---|
| `rl_v5` | `../rl_v5-source/` | MLP | **外部**强 AI,别人训的 | 否 |
| `m3_v1` | `../m3_v1/` | MLP | 我们自己 `m3` 的导出,**冻结在 step 753664** | 否 |
| `m3_v2` | `../m3_v2/` | MLP | 同上,**冻结在 step 901120**(更新的那一份) | 否 |
| `m1_v1` | `m1_v1/` | MLP | 仓库内;`runs/m1/best.npz` 的导出(6.1 万步,**训练度最低**)。**只当起训权重** | 否 |
| `rl_VG_v0.2` | `rl_VG_v0.2/` | MLP | 仓库内;`rl_VG_v1.0` 的起点兼基线 | 否 |
| `rl_VG_v1.0` | `rl_VG_v1.0/` | MLP | 仓库内;规则+网络,已交付的冻结包 | 否 |
| `rl_VG_v2.0` | `rl_VG_v2.0/` | **GRU** | 仓库内;v1.0 的编码器 + 记忆层,已交付的冻结包 | 否 |
| `rl_VG_v2.3` | `rl_VG_v2.3/` | **GRU** | 仓库内;`v2.0` 续训 400k 步的冻结导出,已交付的冻结包 | 否 |

> `m1_v1` 是这几份里**唯一不是对手**的包:注册进 `KNOWN_PACKS` 只是为了让
> `--init-pack m1_v1` 能解析到目录,`selfplay.py` 的对手表里**没有**它的条目
> (加进去会改变联盟配重与评测口径)。目录里只有 `rl_weights.h` 与一份说明,
> 没有 `obs_builder.h` / `rl_VG.cpp` —— 它从来没提交过,别把它打进 `.so` 交上去。

`rl_VG_v2.0` / `rl_VG_v2.3` 是**循环**包:`rl_weights.h` 里 `kRecurrent = 1`、`kGruHidden = 128`,
比 MLP 包多 14 个符号(`kGru*` 12 个 + `kMemProj`/`kMemBias`)。解析侧按 `kRecurrent`
选符号表(`pack._load_weights`),所以同一个 `load_pack_net` 两种架构都能读。老包没有
这两行 —— 缺省当纯 MLP,那正是它们的身份。

能直接接是因为**头文件格式同源**:`rl_v5-source/rl_weights.h` 的头部注释就是
「由 `export_cpp.py` 自动生成」,16 个符号与 MLP 参数一一对应,`obs_builder.h`
与我们移植的那份 md5 相同。同一套约定,所以解析出来就能前向。

```bash
# 默认位置就是上面那三处,一般不用传
python -m monet.cli eval --ckpt runs/m3/best.npz --games 40
python -m monet.cli play --ckpt runs/m3/best.npz --opponent rl_v5 -v
python -m monet.cli play --ckpt runs/m3/final.npz --opponent m3_v1 -v
python -m monet.cli play --ckpt runs/m3/final.npz --opponent m3_v2 -v

# 外部包在别处(--pack-dir 只作用于 rl_v5,不会挪动 m3_v1 / m3_v2)
python -m monet.cli train --pack-dir /path/to/pack ...
set MONET_PACK_DIR=D:/some/pack          # 或环境变量
```

路径解析优先级(`name == PRIMARY_PACK` 即 `rl_v5` 时):`--pack-dir` >
`set_pack_dir()` > `$MONET_PACK_DIR` > 默认位置(锚在 `pack.py` 的文件位置,
整体挪仓库仍然成立)。**其它注册名按固定位置解析** —— 那几个口子是给"外部包
挪了地方"准备的,不该顺带把我们自己冻结的包换掉(`test_pack_dir_override_only_moves_the_primary_pack`
钉住这条)。

### 为什么值得接

`rl_v5` 是联盟里**唯一不依赖我们自己实现**的对手:`official_*` 全是规则手或移植版,
强弱都掺着我们自己的实现假设。`rl_v5` 是原样的外部 AI,是唯一干净的参照物。

`m3/best`(step 753664)打它 **0.550~0.625**(40 局,全名单;采样与 `--deterministic`
各一次),是名单里最低的一档 —— 同期打两个移植版是 0.975 / 0.950。这正是"接进来了"
的接受标准:如果它也刷到 0.9x,说明权重装错了(或被装成了废物),该回查判据。

```bash
python -m monet.cli eval --ckpt runs/m3/best.npz --games 40 --opponents rl_v5,m3_v1 --deterministic
#   rl_v5  0.575   m3_v1  0.500
```

`m3_v1` / `m3_v2` 的用途不一样:**它们是"别退步"的锚**,分别是 `m3` 在 step 753664 与
901120 的冻结。而 `runs/m3/best.npz` **还会继续往下走** —— 锚与它本来就会越差越远,
这是设计,不是漂移。价值在于:策略跑偏之后,打它们的得分率会掉下去,比看综合指标
更早报警。每冻一个新锚就往 `../m3_vN/` 新开一个目录,不能跟着 `rl_VG_v0/`
(每次导出都覆盖的活包)走。

"锚"的基线是导出那一刻量到的:`m3_v1` = **0.500**(净胜分 +0.80,采样口径 0.550),
而同期 `rl_v5` 是 0.550~0.625 —— 前者是**镜像对局**(当时 `best.npz` 也是 step 753664),
对称性没破,说明 `../m3_v1/` 里装的确实是那一版 `best.npz`(装错的话它会明显偏一边:
要么被打穿,要么把 `best` 打穿)。以后从 `m3` 续训,锚那一行应该从 0.5 上下起步,
涨上去才算真进步。

`m3_v2` 是同一个做法冻的 step 901120,导出时与当时的 `best.npz` **逐位相同**
(最大权重差 `0.000000`)。冻每一份锚都照这个流程走:导出 → 新目录 → 注册进
`pack.KNOWN_PACKS` → 加进 `STATIC_OPPONENTS` 与两边名单。

```bash
python -m monet.cli eval --ckpt runs/m3/best.npz --games 40 --opponents rl_v5,m3_v1,m3_v2 --deterministic
#   step 753664 时量到:rl_v5 0.575   m3_v1 0.500
```

> ⚠️ 别再拿"`m3_vN` == `runs/m3/best.npz`"当断言(或当测试):`best.npz` 每刷新一次
> 最高分就往前走一步,而锚不动。要确认差多远,手动比一下就行(换个名字即可):
> ```bash
> python -c "import sys;sys.path.insert(0,'.');import numpy as np;from monet.store import load_checkpoint;import monet.pack as P;n,_ ,_=load_checkpoint('runs/m3/best.npz');p=P.load_pack_net(name='m3_v2');print(max(float(np.abs(n.p[k]-p.p[k]).max()) for k in n.p))"
> ```
> 0.0 表示还停在同一版;一旦不是 0 就说明 `best.npz` 已经往前走了(正常)。

### 怎么知道权重装对了

装错**不会抛异常**,只会让 `rl_v5` 变成弱智,而弱智对手混在联盟里几乎看不出来。
`tests/test_pack.py` 用五层判据挡这件事,分层是实测标定过的:

| 判据 | 手段 | 实测判别力 |
|---|---|---|
| A 结构 | 16 个张量齐全、形状/规格/有限性 | 挡不住形状合法的错位 |
| B 前向 | 照 `rl_ai_v5.cpp` 逐字转写一遍(扁平寻址、不 reshape) | 正确 ≈6e-07;列主序误读 ≈7.0 |
| C 行为 | **打 `camper`** 40 局确定性 | 正确 1.000;`W1`↔`W2` 互换 0.500 |
| D 往返 | 导出→解析逐位比对(强制走分片路径) | 覆盖分片 `#include` 与 `%.9g` 解析 |
| E 命名 | 按命名约定独立推导符号表 | 挡"导出/解析共用的那张表本身写错" |
| F 接线 | **默认名单逐个真的构造一遍** | 挡"加了包却没注册 / 目录多套一层"这类压根起不来的错 |

> 判据 F 是补上一个真实事故:把 `m3_v2` 加进 `STATIC_OPPONENTS` 和两边名单,却忘了
> 注册进 `KNOWN_PACKS`,`eval` / `train` 的默认名单会**在启动时**抛 `PackError`。
> 当时 `test_pack.py` 25 条全绿 —— 光比对"两张名单一致"是不够的,得真的构造一次。
> 注意判据 A~E 全都是**权重装错**的判据,而 F 管的是**压根没接上**:两类错法互不覆盖。

> ⚠️ **判据 C 必须打 `camper`,不能打 `random`** —— 这条与直觉相反。`random` 自己
> 几乎不得分,所以一个错位到近乎随机的网络照样能赢它 **1.000**;`camper` 缩在得分区
> 里不出来,逼出"能否主动结束对局",才分得开(1.000 vs 0.500)。拿 `random` 当阈值
> 等于没判。

### ⚠️ 别把导出写进参赛包

`export --out` 指向参赛包目录会**覆盖掉参考 AI 的权重**,而 `lru_cache` 在同一进程里
仍返回旧对象,当场可能看不出来。`rl_v5-source/` 是外部产物,覆盖了没法从我们的
检查点重新生成;`m3_v1/` 名义上能从 `runs/m3/best.npz` 重做,但那已经不是同一份了
(见上:`best.npz` 早往前走了),**冻结一旦破功,"别退步"这个信号就悄悄失效了**。`export_cpp` 现在会**挨个挡下 `all_pack_dirs()` 里
的每一个包**(子目录也挡),只放行 `rl_VG_v0/` 那种待提交的活包。真要从 `runs/m3/`
再导出一份冻结包,换个新名字新建目录(如 `m3_v2/`),别覆盖 `m3_v1/`。

## 八、目录

```
monet/
  engine/    rules.py(几何常量与纯函数) types.py(与 sentry_duel.h 对齐) game.py(状态机)
  env/       obs.py(428 维观测) opponents.py(陪练) official.py(官方档对手:移植 + 自研)
             sentry_env.py(动作级 MDP)
  models/    mlp.py(前向/反向) ppo.py(GAE + Adam + 掩码裁剪目标)
  training/  config.py selfplay.py evaluate.py export_cpp.py
  pack.py    参赛包解析(读 rl_weights.h 当陪练;注册表 KNOWN_PACKS 见 §七)
  store.py   检查点读写      cli.py  命令行入口
tests/       test_engine.py test_mlp.py test_env.py test_train.py test_official.py
             test_pack.py duel.py(对手循环赛工具)
```
