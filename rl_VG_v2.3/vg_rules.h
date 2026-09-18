// vg_rules.h - 规则层:C++ 移植(本包与 rl_VG_v1.0 的规则**逐字相同** ——
// 规则是策略里与网络无关的那一半,v2 换的是记忆架构,不是规则)
//
// **本文件是 monet/env/rules_vg.py 的逐行移植,不是"另一个实现"。**
// 训练和评测走 Python 那份,交付的 .so 走这份。两边一旦漂移,表现是"测得强、
// 发出去弱",而且不会报错、不会崩 —— 只会静默地掉分。改一边就必须改另一边。
//
// tests/test_rules_vg.py 有差分测试盯着这份移植,但它只能抓**规格**漂移(常量、
// 循环序、平局裁决),抓不到本文件里的编译/编码错误:写这个包的机器上没有 C++
// 工具链。这个局限是已知的,不要因为"测试全绿"就认为它编译得过。
//
// 策略是三层优先级:
//   1. 本回合剩余额度内能击杀 → 击杀(killPlan)
//   2. 否则,敌方连续失明 MISS_THRESHOLD 个阶段且 SCAN 可用 → SCAN
//   3. 否则交给网络
//
// 两条规则都只用**公开信息**:自己的位置/朝向/CD,以及 opp_visible 为真时的
// intel_pos。引擎在 opp_visible 为真时把敌人实时位置原样写进 intel,而敌人只在
// 我方阶段之外行动(分阶段状态机),所以一次搜索之内敌方格是静止的 —— 这是**证明**,
// 不是预测。

#pragma once

#include <cmath>
#include <vector>

#include "sentry_duel.h"  // Pos

namespace vg {

// ---------------------------------------------------------------- 常量

inline constexpr int kBoardSize = 7;
inline constexpr int kCells = kBoardSize * kBoardSize;
inline constexpr int kFireRange = 3;
inline constexpr int kActionsPerTurn = 3;
// 一个阶段最多 kActionsPerTurn 个行动,搜索深度不会超过它。
inline constexpr int kMaxSearchDepth = kActionsPerTurn;

// 失明几个**完整阶段**才扫。语义:阈值 4 ⇒ 第 5 个失明阶段才扫(4 个阶段都"走完了"
// 才计数)。想让它在第 2 个失明阶段就扫,改成 1 即可 —— 必须与 rules_vg.py 的
// MISS_THRESHOLD 同步。
//
// 取值偏大是刻意的:扫得越勤越掉分。机制是 SCAN 把 opp_visible 翻成 true,网络的
// "看得见敌人"分支会离开得分区去追,而一个阶段站在得分区里 END 就是 +1 分。
inline constexpr int kMissThreshold = 4;

// 网络**不能**自己选 SCAN —— SCAN 完全归规则 2 管:该扫的时候规则强制扫,不该扫的
// 时候一次都不许浪费(SCAN 花掉一个行动额度,而不扫的代价只是情报旧一点)。
//
// 这是"关掉"而不是"改掉",所以必须在**采样之前**把它从掩码里抹掉:事后覆盖动作会
// 让存下来的 logp 变成网络对另一个动作的概率,PPO 的 ratio 就废了。部署侧没有 PPO,
// 但保持同一个口径才能让"离线测出来的分数"等于"发出去的分数"。
// 必须与 rules_vg.py 的 NETWORK_SCAN 同步。
inline constexpr bool kNetworkScan = false;

// ---- 动作 id(与 obs_builder.h 的 action_mask、rl_VG.cpp 的 do_action 一致)----
inline constexpr int kMove = 0;
inline constexpr int kTurnN = 1;
inline constexpr int kTurnE = 2;
inline constexpr int kTurnS = 3;
inline constexpr int kTurnW = 4;
inline constexpr int kFire = 5;
inline constexpr int kScan = 6;
inline constexpr int kEnd = 7;

// 方向顺序 N/E/S/W —— **平局裁决就靠它**,不能改成别的顺序。
// 与 rules_vg.py 的 R.DIRS 同序。
inline constexpr int kNDirs = 4;
inline constexpr char kDirs[kNDirs] = {'N', 'E', 'S', 'W'};

// 观测是**视角坐标**(蓝方被镜像过,实测红蓝开局都是 my_pos=(0,0)、my_facing='E'),
// 所以"我的出生点/初始朝向"恒为 (0,0)/'E',不需要分辨红蓝。
inline constexpr char kStartFacing = 'E';

inline int dirIndex(char d) {
    switch (d) {
        case 'N': return 0;
        case 'E': return 1;
        case 'S': return 2;
        case 'W': return 3;
        default:  return 0;
    }
}

inline int turnAction(char d) {
    switch (d) {
        case 'N': return kTurnN;
        case 'E': return kTurnE;
        case 'S': return kTurnS;
        case 'W': return kTurnW;
        default:  return -1;
    }
}

inline void deltaOf(char f, int& dx, int& dy) {
    switch (f) {
        case 'N': dx = 0;  dy = -1; break;
        case 'E': dx = 1;  dy = 0;  break;
        case 'S': dx = 0;  dy = 1;  break;
        case 'W': dx = -1; dy = 0;  break;
        default:  dx = 0;  dy = 0;  break;
    }
}

// 垂直方向:用于 3x3 火力范围的横向展开。与 rules.py 的 PERP 同表。
inline void perpOf(char f, int& px, int& py) {
    switch (f) {
        case 'N':
        case 'S': px = 1; py = 0; break;
        default:  px = 0; py = 1; break;  // E / W
    }
}

inline bool inBounds(int x, int y) {
    return x >= 0 && x < kBoardSize && y >= 0 && y < kBoardSize;
}

inline bool samePos(Pos a, Pos b) { return a.x == b.x && a.y == b.y; }

inline int manhattan(Pos a, Pos b) {
    return std::abs(a.x - b.x) + std::abs(a.y - b.y);
}

inline bool isObstacle(const std::vector<Pos>& obstacles, int x, int y) {
    for (std::size_t i = 0; i < obstacles.size(); ++i) {
        if (obstacles[i].x == x && obstacles[i].y == y) return true;
    }
    return false;
}

// ---------------------------------------------------------------- 规则输入

// 一个决策点规则需要的全部信息。**刻意做成扁平 struct** 而不是直接吃 ObsBuilder:
// 这样差分测试能脱离引擎构造任意状态(见 tests/test_rules_vg.py 的镜像测试),
// 而不用去伪造一个 Board。
struct RuleState {
    Pos my_pos{0, 0};
    char my_facing = 'E';
    int fire_cd = 0;
    int scan_cd = 0;
    int actions_used = 0;
    int turn = 0;
    bool free_turn = false;
    bool opp_visible = false;
    Pos intel_pos{-1, -1};
    std::vector<Pos> obstacles;
};

// ---------------------------------------------------------------- 几何

// 3x3 火力范围内的命中判定(含障碍阻挡)。与 rules.py::fire_hits 逐行对应。
//
// 火力范围 = 朝向前方距离 1..kFireRange 的 3x3 区域。障碍会阻挡**同一火力通道**
// (同一行/列平行射线)上障碍之后的格子 —— 所以内层循环遇到障碍要 break 而不是
// continue,这一点弄错会让规则报出打不中的"击杀方案"。
inline bool fireHits(Pos shooter, char facing, Pos target, const std::vector<Pos>& obstacles) {
    int fx = 0, fy = 0;
    deltaOf(facing, fx, fy);
    int px = 0, py = 0;
    perpOf(facing, px, py);
    for (int lat = -1; lat <= 1; ++lat) {
        const int sx = shooter.x + lat * px;
        const int sy = shooter.y + lat * py;
        for (int step = 1; step <= kFireRange; ++step) {
            const int cx = sx + step * fx;
            const int cy = sy + step * fy;
            if (!inBounds(cx, cy)) break;
            if (isObstacle(obstacles, cx, cy)) break;
            if (cx == target.x && cy == target.y) return true;
        }
    }
    return false;
}

// 此刻还能不能"免费转身"。与 rules_vg.py::free_turn_available 逐行对应。
//
// 引擎只在"复活/开局后的那一次 act 内"给免费 TURN,而观测里的 free_turn 位漏掉了
// **开局第一个阶段**(obs.py 的 act_start 把它设成 i_died,那时还没人死过)。
// 这个观测缺口不能去修 —— 改它等于重新定义 428 维观测、废掉所有已训权重 ——
// 只能在规则里绕过去:turn==0 + 一个行动都没消耗 + 朝向还是初始的 E,三者同时成立
// 就等价于"开局阶段的第一次决策"。
//
// (免费转身**不消耗**额度,所以转身之后 actions_used 仍是 0,必须靠朝向把自己和
//  "还没动过"区分开。)
inline bool freeTurnAvailable(const RuleState& s) {
    if (s.my_pos.x != 0 || s.my_pos.y != 0) return false;
    if (s.free_turn) return true;
    return s.turn == 0 && s.actions_used == 0 && s.my_facing == kStartFacing;
}

// ---------------------------------------------------------------- 规则 1:击杀搜索

struct KillNode {
    Pos pos;
    char facing;
    int first;  // 这条路径的**第一个**动作;-1 = 还没花过行动额度(即 first is None)
};

// 本回合内能击杀就返回最短方案的第一个动作,否则 -1。
//
// 只回答"能不能",不返回整条路径:调用方照做第一步,下一个决策点重新调用即可。
// 每步都从当前观测重推,所以中途 SCAN 揭示、走位被挡之类的意外都能立刻修正,也
// 不需要跨行动维护计划状态。
//
// **全程在视角坐标里算。** 180 度旋转对 fireHits 是把障碍集映到自身、把前方映到
// 前方的等距变换,所以在视角坐标里判定命中等价于引擎在绝对坐标里判定。
inline int killPlan(const RuleState& s, int budget = -1) {
    if (s.fire_cd != 0) return -1;
    // 只凭乐观信念开枪不算"能击杀"(opp_visible 为真 ⟹ intel_pos 是实时位置)。
    if (!s.opp_visible) return -1;
    const int tx = s.intel_pos.x, ty = s.intel_pos.y;
    if (tx < 0 || ty < 0) return -1;

    if (budget < 0) budget = kActionsPerTurn - s.actions_used;
    if (budget <= 0) return -1;
    // **只截断循环次数,不截断 budget 本身。** 下面那个可达性剪枝用的是**原始**
    // budget —— Python 侧写的是 `for _ in range(min(budget, MAX_SEARCH_DEPTH))`,
    // 剪枝那行的 budget 没有被 min() 碰过。默认路径上 budget = 3 - actions_used
    // 本来就不会超过 MAX_SEARCH_DEPTH,所以两边碰不到差别;但只要有人显式传一个
    // 更大的 budget,提前截断就会让 C++ 比 Python 少搜一层。
    const int max_depth = (budget < kMaxSearchDepth) ? budget : kMaxSearchDepth;

    const Pos start = s.my_pos;
    const Pos target = s.intel_pos;

    // 开火前最多走 budget-1 步,而一发子弹最远打到曼哈顿距离 kFireRange+1
    // (前向 3 格、横向最多偏 1 格)。够不着就直接退出 —— 绝大多数决策点在这里
    // 出局,省掉整棵 BFS。
    if (manhattan(start, target) > (budget - 1) + kFireRange + 1) return -1;

    // 根节点。顺序就是平局裁决:C++ 与 Python 必须用同一套顺序,否则差分测试会红。
    //   先"不转身";再按 N/E/S/W 展开免费转身(它们代价 0,first 就是那次转身)。
    std::vector<KillNode> frontier;
    {
        KillNode root;
        root.pos = start;
        root.facing = s.my_facing;
        root.first = -1;
        frontier.push_back(root);
    }
    if (freeTurnAvailable(s)) {
        for (int i = 0; i < kNDirs; ++i) {
            const char d = kDirs[i];
            if (d == s.my_facing) continue;
            KillNode n;
            n.pos = start;
            n.facing = d;
            n.first = turnAction(d);
            frontier.push_back(n);
        }
    }

    // seen 记 (格, 朝向)。棋盘 49 格 × 4 朝向,定长数组就够,不必上哈希表 ——
    // 而且定长数组的遍历顺序是确定的,不会因为哈希实现不同而换掉平局裁决。
    bool seen[kCells * kNDirs];
    for (int i = 0; i < kCells * kNDirs; ++i) seen[i] = false;
    for (std::size_t i = 0; i < frontier.size(); ++i) {
        seen[(frontier[i].pos.y * kBoardSize + frontier[i].pos.x) * kNDirs +
             dirIndex(frontier[i].facing)] = true;
    }

    // 按"开火前花掉的行动数"递增扫描,第一次命中就是最短方案;同一层内按 frontier
    // 顺序取,即平局裁决。
    for (int depth = 0; depth < max_depth; ++depth) {
        for (std::size_t i = 0; i < frontier.size(); ++i) {
            if (fireHits(frontier[i].pos, frontier[i].facing, target, s.obstacles)) {
                return frontier[i].first < 0 ? kFire : frontier[i].first;
            }
        }

        std::vector<KillNode> nxt;
        for (std::size_t i = 0; i < frontier.size(); ++i) {
            const Pos pos = frontier[i].pos;
            const char facing = frontier[i].facing;
            const int first = frontier[i].first;

            // 子节点顺序:TURN 先(N/E/S/W),再 MOVE。与 Python 侧一致。
            for (int d = 0; d < kNDirs; ++d) {
                const char nd = kDirs[d];
                if (nd == facing) continue;  // 掩码不允许转向当前朝向,转了也是白花一步
                const int key = (pos.y * kBoardSize + pos.x) * kNDirs + dirIndex(nd);
                if (seen[key]) continue;
                seen[key] = true;
                KillNode n;
                n.pos = pos;
                n.facing = nd;
                n.first = (first < 0) ? turnAction(nd) : first;
                nxt.push_back(n);
            }

            int dx = 0, dy = 0;
            deltaOf(facing, dx, dy);
            Pos step;
            step.x = pos.x + dx;
            step.y = pos.y + dy;
            // **先判出界再算下标。** Python 那边 `key in seen` 对任意 tuple 都安全,
            // 移植过来就不是了:出界的 step 会算出负的 mkey,拿它索引 seen[] 是越界读,
            // 表现为随机行为或崩溃。四个条件都无副作用,换序与 Python 等价。
            //
            // 与 action_mask 同口径:出界 / 障碍 / 敌人所在格都不能进。
            // (引擎会拒绝走进敌方格,而这里比掩码更严:只要有 opp_visible 就不进,
            //  而掩码只在"直接看见"时才挡 —— 严格的那一侧才不会让规则给出被引擎
            //  拒绝的动作,被拒的动作不消耗额度,反复给出就会卡死。)
            if (!inBounds(step.x, step.y)) continue;
            if (samePos(step, target)) continue;
            if (isObstacle(s.obstacles, step.x, step.y)) continue;
            const int mkey = (step.y * kBoardSize + step.x) * kNDirs + dirIndex(facing);
            if (seen[mkey]) continue;
            seen[mkey] = true;
            KillNode n;
            n.pos = step;
            n.facing = facing;
            n.first = (first < 0) ? kMove : first;
            nxt.push_back(n);
        }

        if (nxt.empty()) break;
        frontier.swap(nxt);
    }
    return -1;
}

// ---------------------------------------------------------------- 规则 2 + 合流

// 规则 2 的阶段计数器 + 两条规则的决策点钩子。
//
// 每个决策点调用**恰好一次** forced();它内部借机推进阶段计数,幂等。
class VgRules {
public:
    VgRules() { reset(); }

    void reset() {
        turn_ = -1;
        seen_ = false;
        miss_ = 0;
    }

    // 诊断计数。交付侧不用,留着是为了让"规则到底有没有生效"在日志里看得见 ——
    // 静默失效是本项目反复踩的坑。
    int forcedCount() const { return forced_count_; }
    int killCount() const { return kill_count_; }
    int scanCount() const { return scan_count_; }
    int blindPhases() const { return miss_; }

    // 返回规则强制执行的合法动作 id;两条规则都没命中则 -1。
    // mask 是环境给的**原始**掩码 —— 规则要看到它才能判断 SCAN 可不可用。
    int forced(const RuleState& s, const float* mask) {
        note(s);

        const int k = killPlan(s);
        // mask[k] > 0 是安全网:被引擎拒绝的动作不消耗额度,规则若反复给出同一个
        // 被拒的动作就会卡死。
        if (k >= 0 && mask[k] > 0.0f) {
            ++forced_count_;
            ++kill_count_;
            return k;
        }

        if (scanWorthIt(s) && mask[kScan] > 0.0f) {
            ++forced_count_;
            ++scan_count_;
            return kScan;
        }
        return -1;
    }

    // 一个决策点的完整规则处理:把 mask_in 整成"交给网络采样"的掩码写进 mask_out,
    // 并返回强制动作 id(-1 = 没强制)。
    //
    // 训练与部署两条路径都只走这一个入口,免得各写一遍而分叉(分叉的表现是
    // "训练时很强、部署后变弱",而且不会报错)。
    int actMask(const RuleState& s, const float* mask_in, float* mask_out, int n_act) {
        const int k = forced(s, mask_in);
        for (int i = 0; i < n_act; ++i) mask_out[i] = mask_in[i];
        if (k >= 0) {
            // 收窄成"只有 k 合法"。收窄之后再采样,存进轨迹的 logp 恰好是 0.0、
            // ratio 恰好是 1.0、策略梯度恰好是 0。
            for (int i = 0; i < n_act; ++i) mask_out[i] = 0.0f;
            mask_out[k] = 1.0f;
        } else if (!kNetworkScan) {
            mask_out[kScan] = 0.0f;
        }
        return k;
    }

private:
    void note(const RuleState& s) {
        const int turn = s.turn;
        if (turn_ < 0 || turn < turn_) {
            // 新的一局从 turn=0 重来。少了这一句,上一局残留的 turn 会把 0 当成
            // "又一个阶段结束",把连败跨局累加下去。
            turn_ = turn;
            seen_ = false;
            miss_ = 0;
        } else if (turn != turn_) {
            // 上一个阶段走完了:整个过程都没见到敌人就算一次失明。
            miss_ = seen_ ? 0 : miss_ + 1;
            turn_ = turn;
            seen_ = false;
        }
        if (s.opp_visible) {
            // 直接视野或 SCAN 揭示都算"看到",看到就清零。
            seen_ = true;
            miss_ = 0;
        }
    }

    bool scanWorthIt(const RuleState& s) const {
        if (miss_ < kMissThreshold) return false;
        if (s.scan_cd != 0) return false;
        return true;
    }

    int turn_ = -1;
    bool seen_ = false;
    int miss_ = 0;
    int forced_count_ = 0;
    int kill_count_ = 0;
    int scan_count_ = 0;
};

}  // namespace vg
