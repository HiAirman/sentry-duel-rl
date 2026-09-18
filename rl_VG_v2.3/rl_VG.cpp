// rl_VG.cpp - RL(带记忆) + 规则 混合策略部署 AI(哨兵大战参赛 .so)
//
// 权重来源:rl_monet_v1 自对弈训练(从 rl_VG_v1.0 的权重起步,架构换成 GRU)
//           → monet/training/export_cpp.py → rl_weights.h(同目录)
//
// 与 rl_VG_v1.0 的唯一区别是**多了一段记忆**:决策不再只看当前观测,而是把整局
// 历史压在一个 GRU 隐状态里。编码器(428→512→512→512)和三个头一字未改 —— 记忆
// 是以**零初始化残差**回注的:
//
//     a3   = Encoder(obs)                      与 v1.0 逐层同构
//     h    = GRU_128(a3, h_prev)               本文件新增
//     feat = a3 + kMemProj·h + kMemBias        kMemProj/kMemBias 训练起点是全零
//     π    = kW3·feat + kB3
//
// kMemProj/kMemBias 从零开始,所以这一步在训练**起点**上逐位等价于 v1.0;记忆层最差
// 也只能学成"不用它"。同步的 Python 实现在 monet/models/rnn_policy.py。
//
// 结构:
//   - 观测:复用同目录 obs_builder.h(与 rl_monet_v1/monet/env/obs.py 同一规格)
//   - 规则:同目录 vg_rules.h(monet/env/rules_vg.py 的逐行移植)。每个决策点先问
//           规则,命中就把掩码收窄成只有那一个动作合法,没命中则关掉 SCAN 再交给
//           网络 —— 收窄发生在**采样之前**,所以"离线测出来的分数"和"发出去的
//           分数"是同一个策略。规则细节见 vg_rules.h 顶部。
//   - 推理:手写 MLP 编码器 + 手写 GRU 单步(见 gru_step)
//   - 决策:masked softmax 温度采样(T=1.0,与训练时采样温度一致),行动被引擎拒绝时
//           屏蔽该动作重试
//   - 安全壳:任何异常/异常状态回退到简单启发式,保证不崩溃、不超时
//
// 推理预算:每局 60 次决策 × (4 层 512 matmul + 6 个 128 matmul),实测 < 1s
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
// 生成文件:kW0..kW3 + kB0..kB3 + kLN1Gamma..kLN3Beta(+ 价值头 kW4/kB4,
// 推理不用,导出器保留它是为了让同一套 export 能复用到别处)+ GRU 的 12 个
// kGru* + 残差回注的 kMemProj/kMemBias + 架构标记 kRecurrent/kGruHidden。
#include "rl_weights.h"

// 拿到一份纯 MLP 的权重头(比如手滑 copy 了 v1.0 的)时**编译期就炸**,而不是
// 静默地跳过整个记忆段 —— 后者的表现是"发出去的包比本地测的弱",而评测分数
// 低几个点几乎不会被归因到一行 include。
static_assert(kRecurrent == 1, "这个部署壳需要带 GRU 的权重头(kRecurrent=1);"
                              "纯 MLP 的包请用 rl_VG_v1.0 那份 rl_VG.cpp");

namespace {

rl::ObsBuilder g_ob;
vg::VgRules g_rules;
bool g_game_started = false;

// 部署温度:T=1.0 = 标准 softmax 采样;T→0 = argmax;T<1 更确定;T>1 更随机
static constexpr float kDeployTemperature = 1.0f;

// 网络维度。kObsDim/kHidden/kActDim/kGruHidden 都来自权重头。
static constexpr int kN = kObsDim;       // 428
static constexpr int kH = kHidden;       // 512
static constexpr int kA = kActDim;       // 8
static constexpr int kG = kGruHidden;    // GRU 隐状态宽度

// ---- 跨调用保持的 GRU 隐状态 ----
// **必须是全局的**:引擎每个阶段只调一次 act(),记忆要跨这些调用活下来才叫记忆。
// 每局开场(board.turn == 0)清零,见 rl_act。
// 不用 thread_local:评测框架可能在同一线程上交替跑两局,而隐状态属于"这一局",
// 不属于"这个线程";真正的隔离靠 turn==0 的重置。
float g_h[kG];

// thread-local RNG,避免共享状态
static thread_local std::mt19937 g_rng{std::random_device{}()};

// ---- GELU 激活(精确版) ----
inline float gelu(float x) {
    // 0.5 * x * (1 + erf(x / sqrt(2)))
    return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865475f));
}

// ---- sigmoid ----
// 夹到 ±60 是为了不让 std::exp 溢出:gru.py 的 _SIG_CLIP 也是 60,两边同口径。
// **这个夹子会改数值**(不是纯防溢出的装饰):float32 下 exp(60)≈1.1e26 还活着,
// 但 x < -88 时 exp(-x) 就 inf 了,结果从"接近 0"变成 nan。夹了之后两端都饱和到
// 0/1,与 numpy 侧逐位一致。
inline float sigmoid(float x) {
    if (x < -60.0f) x = -60.0f;
    if (x > 60.0f) x = 60.0f;
    return 1.0f / (1.0f + std::exp(-x));
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
//   W 形状 (out, in) 行主序;b 传 nullptr 表示这一层没有偏置(kMemProj 就是)。
void linear(const float* W, const float* b, const float* x, float* y,
            int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; ++o) {
        const float* w_row = W + static_cast<size_t>(o) * in_dim;
        float s = (b != nullptr) ? b[o] : 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            s += w_row[i] * x[i];
        }
        y[o] = s;
    }
}

// ---- GRU 单步:h = GRU(x, h),**原地更新 h** ----
// 与 monet/models/gru.py::GRU.forward 的单步同构,门序 r/z/n(PyTorch 约定):
//
//   r = sigmoid(W_ir·x + b_ir + W_hr·h + b_hr)
//   z = sigmoid(W_iz·x + b_iz + W_hz·h + b_hz)
//   n = tanh   (W_in·x + b_in + r ⊙ (W_hn·h + b_hn))
//   h'= (1-z)⊙n + z⊙h
//
// 三处容易写错、且写错了不会报错只会变弱的地方:
//   1. 六个投影必须全部读**更新前的** h。下面先把 h 拷进 hprev 再写 h,所以三个
//      `linear(..., h, ...)` 与 r⊙(W_hn·h) 用的都是旧值。
//   2. n 的门控是 r ⊙ (W_hn·h + b_hn),**不是** r ⊙ (W_hn·(r⊙h)) —— 后者是另一
//      种(错的)GRU 变体,形状一样、跑得起来、梯度也有,只有数值对不上。
//   3. h' 的插值是 (1-z)⊙n + z⊙h(新信息权重是 1-z)。
void gru_step(const float* x, float* h) {
    static thread_local float xr[kG], xz[kG], xn[kG];
    static thread_local float hr[kG], hz[kG], hn[kG];
    static thread_local float hprev[kG], n[kG];

    // 输入侧三个投影(与 h 无关)
    linear(kGruWIr, kGruBIr, x, xr, kH, kG);
    linear(kGruWIz, kGruBIz, x, xz, kH, kG);
    linear(kGruWIn, kGruBIn, x, xn, kH, kG);
    // 隐状态侧三个投影,全部用**旧** h
    linear(kGruWHr, kGruBHr, h, hr, kG, kG);
    linear(kGruWHz, kGruBHz, h, hz, kG, kG);
    linear(kGruWHn, kGruBHn, h, hn, kG, kG);
    for (int i = 0; i < kG; ++i) hprev[i] = h[i];

    for (int i = 0; i < kG; ++i) {
        const float r = sigmoid(xr[i] + hr[i]);
        const float z = sigmoid(xz[i] + hz[i]);
        n[i] = std::tanh(xn[i] + r * hn[i]);
        h[i] = (1.0f - z) * n[i] + z * hprev[i];
    }
}

// ---- 编码器:428 → 512 → 512(+残差) → 512(+残差) → a3 ----
// 与 rl_VG_v1.0 的同名函数逐行一致 —— 这一段没有记忆,不属于本次改动。
void encode(const float* x, float* a3) {
    static thread_local float h1[kH], h2[kH];

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
    linear(kW2, kB2, h2, a3, kH, kH);
    for (int i = 0; i < kH; ++i) a3[i] += h2[i];  // residual
    layer_norm_inplace(a3, kLN3Gamma, kLN3Beta, kH);
    for (int i = 0; i < kH; ++i) a3[i] = gelu(a3[i]);
}

// ---- 完整前向:编码 → 推进记忆 → 残差回注 → 策略头 ----
// h **原地推进**(同时是输入和输出)。调用方传的是跨调用保持的 g_h。
void forward(const float* x, float* h, float* logits) {
    static thread_local float a3[kH], feat[kH];

    encode(x, a3);
    gru_step(a3, h);

    // feat = a3 + kMemProj·h + kMemBias。kMemProj 没有偏置项,所以 b 传 nullptr,
    // bP 单独加 —— 两者在训练侧也是分开的两个张量,合并没有好处。
    linear(kMemProj, nullptr, h, feat, kG, kH);
    for (int i = 0; i < kH; ++i) feat[i] += a3[i] + kMemBias[i];

    linear(kW3, kB3, feat, logits, kH, kA);
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
        // 记忆同样逐局清零 —— 隐状态属于"这一局",不属于"这个进程"。忘了这一句,
        // 新一局会继承上一局的残局印象(比如"刚才一直在被追"),而训练侧每局都从
        // 零隐状态开始,两边就不是同一个策略了。
        for (int i = 0; i < kG; ++i) g_h[i] = 0.0f;
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
        // 注意 forward 会**原地推进 g_h**。重试路径(下面的 continue)会再推进一步,
        // 这正是训练侧的行为:引擎拒一次,训练循环就多走一次 act()。
        forward(obs, g_h, logits);

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
