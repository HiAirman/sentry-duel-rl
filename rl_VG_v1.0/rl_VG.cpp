// rl_VG.cpp - RL + 规则 混合策略部署 AI(哨兵大战参赛 .so)
//
// 权重来源:rl_monet_v1 自对弈训练(vg10 → vgb3 → vgb4,规则全程开启)
//           → monet/training/export_cpp.py → rl_weights.h(同目录)
//
// 结构:
//   - 观测:复用同目录 obs_builder.h(与 rl_monet_v1/monet/env/obs.py 同一规格)
//   - 规则:同目录 vg_rules.h(monet/env/rules_vg.py 的逐行移植)。每个决策点先问
//           规则,命中就把掩码收窄成只有那一个动作合法,没命中则关掉 SCAN 再交给
//           网络 —— 收窄发生在**采样之前**,所以"离线测出来的分数"和"发出去的
//           分数"是同一个策略。规则细节见 vg_rules.h 顶部。
//   - 推理:手写 MLP 前向(428→512→512→512→8,带 LayerNorm + 残差 + GELU),
//           权重在同目录 rl_weights.h
//   - 决策:masked softmax 温度采样(T=1.0,与训练时采样温度一致),行动被引擎拒绝时
//           屏蔽该动作重试
//   - 安全壳:任何异常/异常状态回退到简单启发式,保证不崩溃、不超时
//
// 推理预算:每局 60 次 act × 4 层 512 matmul,实测 < 1s
//
// 构建:提交后由评测服务在项目根自动编译(无 Makefile 时编译根层全部 .cpp),
//       sentry_duel.h / utils.h 由引擎提供。本地等价命令:
//   g++ -std=c++17 -O2 -fPIC -shared -I$(ENGINE_INCLUDE) rl_VG.cpp -o rl_VG.so

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

#include "sentry_duel.h"
#include "utils.h"
#include "obs_builder.h"
#include "vg_rules.h"
// 生成文件:kW0..kW3 + kB0..kB3 + kLN1Gamma..kLN3Beta,外加价值头 kW4/kB4
// (推理不用价值头,导出器保留它是为了让同一套 export 能复用到别处)。
#include "rl_weights.h"

namespace {

rl::ObsBuilder g_ob;
vg::VgRules g_rules;
bool g_game_started = false;

// 部署温度:T=1.0 = 标准 softmax 采样;T→0 = argmax;T<1 更确定;T>1 更随机
static constexpr float kDeployTemperature = 1.0f;

// 4 层 512 网络的常量
static constexpr int kN = kObsDim;     // 428
static constexpr int kH = kHidden;     // 512
static constexpr int kA = kActDim;     // 8

// thread-local RNG,避免共享状态
static thread_local std::mt19937 g_rng{std::random_device{}()};

// ---- GELU 激活(精确版) ----
inline float gelu(float x) {
    // 0.5 * x * (1 + erf(x / sqrt(2)))
    return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865475f));
}

// ---- LayerNorm:对单个向量(无 batch)计算 mean/var,应用 gamma/beta ----
// 输入/输出 buffer 都指向同一处(原地)
void layer_norm_inplace(float* x, const float* gamma, const float* beta, int n) {
    float mean = 0.0f;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    float var = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = x[i] - mean;
        var += d * d;
    }
    var /= n;
    const float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    for (int i = 0; i < n; ++i) {
        x[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// ---- MatMul:y = W * x + b ----
// y[o] = sum_i W[o, i] * x[i] + b[o]
//   W 形状 (out, in) 行主序
void linear(const float* W, const float* b, const float* x, float* y,
            int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; ++o) {
        const float* w_row = W + static_cast<size_t>(o) * in_dim;
        float s = b[o];
        for (int i = 0; i < in_dim; ++i) {
            s += w_row[i] * x[i];
        }
        y[o] = s;
    }
}

// ---- 手写 MLP 前向(428 → 512 → 512(+残差) → 512(+残差) → 8) ----
// Layer 1: x -> Linear(428,512) -> LayerNorm -> GELU -> h1
// Layer 2: h1 -> Linear(512,512) -> +h1 -> LayerNorm -> GELU -> h2
// Layer 3: h2 -> Linear(512,512) -> +h2 -> LayerNorm -> GELU -> h3
// Head:    h3 -> Linear(512,8) -> logits
void forward(const float* x, float* logits) {
    static thread_local float h1[kH], h2[kH], h3[kH];

    // Layer 1: 428 → 512
    linear(kW0, kB0, x, h1, kN, kH);
    layer_norm_inplace(h1, kLN1Gamma, kLN1Beta, kH);
    for (int i = 0; i < kH; ++i) h1[i] = gelu(h1[i]);

    // Layer 2: 512 → 512 (残差)
    linear(kW1, kB1, h1, h2, kH, kH);
    for (int i = 0; i < kH; ++i) h2[i] += h1[i];  // residual
    layer_norm_inplace(h2, kLN2Gamma, kLN2Beta, kH);
    for (int i = 0; i < kH; ++i) h2[i] = gelu(h2[i]);

    // Layer 3: 512 → 512 (残差)
    linear(kW2, kB2, h2, h3, kH, kH);
    for (int i = 0; i < kH; ++i) h3[i] += h2[i];  // residual
    layer_norm_inplace(h3, kLN3Gamma, kLN3Beta, kH);
    for (int i = 0; i < kH; ++i) h3[i] = gelu(h3[i]);

    // Head: 512 → 8
    linear(kW3, kB3, h3, logits, kH, kA);
}

// ---- masked 温度采样:T=1.0(与训练一致),带 fallback 到 argmax ----
int sample_with_temperature(const float* logits, const float* mask, float temperature) {
    if (temperature <= 1e-3f) {  // argmax fallback
        int best = -1;
        float best_v = -1e30f;
        for (int a = 0; a < kA; ++a) {
            if (mask[a] <= 0.0f) continue;
            if (logits[a] > best_v) { best_v = logits[a]; best = a; }
        }
        return best;
    }
    float mx = -1e30f;
    for (int a = 0; a < kA; ++a)
        if (mask[a] > 0.0f) mx = std::max(mx, logits[a]);
    float sum = 0.0f;
    float prob[kA];
    for (int a = 0; a < kA; ++a) {
        prob[a] = (mask[a] > 0.0f) ? std::exp((logits[a] - mx) / temperature) : 0.0f;
        sum += prob[a];
    }
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(g_rng);
    int chosen = kA - 1;
    for (int a = 0; a < kA; ++a) {
        if (prob[a] <= 0.0f) continue;
        r -= prob[a];
        if (r <= 0.0f) { chosen = a; break; }
        chosen = a;
    }
    return chosen;
}

// ---- 从观测构造规则层要的状态 ----
// ObsBuilder 里的 my_pos_/my_facing_ 来自**视角坐标**的 Board(引擎给蓝方的视图
// 已做 180 度镜像,与 monet/engine/game.py::view 同口径),所以出生点恒为 (0,0)、
// 初始朝向恒为 'E' —— vg::kStartFacing 就是靠这一点成立的。
vg::RuleState make_rule_state() {
    vg::RuleState s;
    s.my_pos = g_ob.my_pos();
    s.my_facing = g_ob.my_facing();
    s.fire_cd = g_ob.fire_cd();
    s.scan_cd = g_ob.scan_cd();
    s.actions_used = g_ob.actions_used();
    s.turn = g_ob.turn();
    s.free_turn = g_ob.free_turn();
    s.opp_visible = g_ob.opp_visible();
    s.intel_pos = g_ob.intel_pos();
    s.obstacles = g_ob.obstacles();
    return s;
}

// ---- 执行一个动作 ----
bool do_action(int a, ActionObservation& ob, bool& consumed) {
    switch (a) {
        case 0: { ActionResult r = move();     ob = r.observation; consumed = r.consumed; return r.success; }
        case 1: { ActionResult r = turn('N');  ob = r.observation; consumed = r.consumed; return r.success; }
        case 2: { ActionResult r = turn('E');  ob = r.observation; consumed = r.consumed; return r.success; }
        case 3: { ActionResult r = turn('S');  ob = r.observation; consumed = r.consumed; return r.success; }
        case 4: { ActionResult r = turn('W');  ob = r.observation; consumed = r.consumed; return r.success; }
        case 5: { ActionResult r = fire();     ob = r.observation; consumed = r.consumed; return r.success; }
        case 6: { ScanResult  r = scan();      ob = r.observation; consumed = r.consumed; return r.success; }
        default: return false;  // 7 = end
    }
}

// ---- 启发式回退(任何异常时使用) ----
void fallback_act(const Board& board, char my_color) {
    const Sentry& me = (my_color == 'R') ? board.red : board.blue;
    const Sentry& opp = (my_color == 'R') ? board.blue : board.red;
    if (opp.visible && me.fire_cd == 0) { fire(); return; }
    if (me.scan_cd == 0) { scan(); return; }
    if (can_move_forward(me, opp.last_known_pos, board.obstacles, board.size)) move();
}

void rl_act(const Board& board, char my_color) {
    if (board.turn == 0 || !g_game_started) {
        g_ob.reset();
        // 规则 2 的失明计数必须逐局清零,否则上一局的连败会跨局累加下去,
        // 新一局开局就误判成"已经失明很久"。与 NetPolicy.reset 同一纪律。
        g_rules.reset();
        g_game_started = true;
    }
    g_ob.act_start(board, my_color);

    float obs[rl::kObsDim];
    float mask[kA];
    float net_mask[kA];
    float banned[kA] = {0};
    int used = 0;
    while (used < vg::kActionsPerTurn) {
        g_ob.encode(obs);
        g_ob.action_mask(mask);
        // 被引擎拒过的动作从掩码里抹掉 —— 被拒的动作不消耗额度,不抹就会原地打转。
        for (int i = 0; i < kA; ++i) {
            if (banned[i] > 0.0f) mask[i] = 0.0f;
        }

        // 规则层:命中就把掩码收窄成只有它合法,没命中则关掉 SCAN 再把掩码交给网络。
        // net_mask 才是采样用的那个 —— 与 NetPolicy.act 走同一个口径。
        const vg::RuleState st = make_rule_state();
        const int forced = g_rules.actMask(st, mask, net_mask, kA);
        (void)forced;  // 规则命中与否已经从 net_mask 上体现,这里只留个名字便于调试

        float logits[kA];
        forward(obs, logits);

        int best = sample_with_temperature(logits, net_mask, kDeployTemperature);
        if (best < 0 || net_mask[best] <= 0.0f || best == 7) break;  // 不可执行或 end

        ActionObservation ob{};
        bool consumed = false;
        const bool ok = do_action(best, ob, consumed);
        g_ob.on_observation(ob, consumed, best);
        if (!ok) {
            // 该动作被引擎拒绝(不消耗额度),换次优。规则若还命中同一个动作,
            // 下一轮它会被 banned 挡掉,所以不会死循环 —— 这也是 vg_rules.h 里
            // `mask[k] > 0` 那道安全网的部署侧对应物。
            banned[best] = 1.0f;
            continue;
        }
        if (consumed) ++used;
    }
}

}  // namespace

extern "C" void act(const Board& board, char my_color) {
    try {
        rl_act(board, my_color);
    } catch (...) {
        try { fallback_act(board, my_color); } catch (...) {}
    }
}
