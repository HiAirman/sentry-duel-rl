// obs_builder.h - 观测构建 + 敌方信念追踪(rl/SPEC.md §2/§5)
//
// 关键设计:本头文件只使用选手可得的公开信息(act() 收到的 Board 视图、
// 行动返回的 ActionObservation、utils.h 工具函数),训练环境与部署 .so 共用
// 同一份代码,从构造上保证训练/部署观测一致,杜绝信息泄漏。
//
// 用法(与选手 act() 的生命周期一致):
//   ObsBuilder ob;             // 每局一个实例(部署侧为静态单例,turn==0 时 reset)
//   ob.reset();                // 开局
//   ob.act_start(view, 'R');   // 每次 act() 开始
//   ob.on_observation(res.observation, res.consumed);  // 每次行动返回后
//   ob.encode(buf);            // buf 为 428 个 float(rl::OBS_DIM)

#pragma once

#include <array>
#include <cstring>
#include <vector>

#include "sentry_duel.h"
#include "utils.h"

namespace rl {

inline constexpr int kBoardSize = 7;
inline constexpr int kCells = kBoardSize * kBoardSize;
inline constexpr int kPlanes = 8;
inline constexpr int kScalars = 36;
inline constexpr int kObsDim = kPlanes * kCells + kScalars;  // 428

class ObsBuilder {
public:
    void reset() {
        belief_.fill(0.0f);
        belief_[cell(6, 6)] = 1.0f;  // 对方出生在镜像视角的 (6,6)
        intel_pos_ = {-1, -1};
        intel_facing_ = '?';
        intel_turn_ = -100;
        my_pos_ = {0, 0};
        my_facing_ = 'E';
        my_score_ = opp_score_ = 0;
        fire_cd_ = scan_cd_ = 0;
        turn_ = 0;
        actions_used_ = 0;
        is_blue_ = false;
        opp_visible_ = opp_directly_visible_ = false;
        free_turn_ = false;
        first_act_ = true;
        prev_act_end_pos_ = {0, 0};
        seen_now_ = {-1, -1};
        // 新增:死亡/重复操作/得分区占用/可见性追踪
        recent_deaths_ = 0;
        total_deaths_ = 0;
        steps_since_move_ = 0;
        kills_ = 0;
        my_zone_turns_ago_ = 99;
        opp_zone_turns_ago_ = 99;
        my_turn_zone_streak_ = 0;
        last_was_fire_ = false;
        // 新增:路径指纹追踪
        start_pos_ = my_pos_;
        path_anchor_x_ = my_pos_.x;
        path_anchor_y_ = my_pos_.y;
        recent_displacement_ = 0;
        recent_unique_cells_ = 0;
        consecutive_same_pos_turns_ = 0;
        death_count_short_window_ = 0;
        last_turn_pos_ = my_pos_;
    }

    // act() 开始:view 为引擎传入的(已镜像)Board 快照。
    void act_start(const Board& view, char my_color) {
        const Sentry& me = (my_color == 'R') ? view.red : view.blue;
        const Sentry& opp = (my_color == 'R') ? view.blue : view.red;
        is_blue_ = (my_color == 'B');
        turn_ = view.turn;
        obstacles_ = view.obstacles;
        zones_ = view.score_zones;

        // --- 击杀/死亡侦查(分数差是公开信息)---
        // 对方 +2 的唯一来源是击杀我(占点只有 +1),比位置判断更可靠
        const int my_delta = me.score - my_score_;
        const int opp_delta = opp.score - opp_score_;
        const bool i_died = opp_delta >= 2 ||
                            (same(me.last_known_pos, Pos{0, 0}) &&
                             !same(me.last_known_pos, prev_act_end_pos_) && !first_act_);
        if (my_delta >= 2) {
            // 我击杀了对方:对方回其出生点(公开规则),情报立刻更新
            belief_.fill(0.0f);
            belief_[cell(6, 6)] = 1.0f;
            intel_pos_ = {6, 6};
            intel_facing_ = 'W';
            intel_turn_ = turn_;
        }
        // 免费 TURN 仅在复活后第一个 act 内有效;本 act 未被杀则收回
        free_turn_ = i_died;

        my_score_ = me.score;
        opp_score_ = opp.score;

        // --- 新增:死亡计数 ---
        if (i_died) {
            total_deaths_++;
            recent_deaths_++;  // 跨多 turn 衰减:每个 turn_start 时减一
        }
        if (my_delta >= 2) {
            kills_++;
        }
        // --- 新增:本回合首动作时的位置变化检测(steps_since_move) ---
        // 如果本 act 起点与上一 act 末点不同,说明移动过(否则至少一步没动)
        if (!first_act_) {
            if (!same(me.last_known_pos, prev_act_end_pos_)) {
                steps_since_move_ = 0;  // 复活/被传送视为一次位移
            } else {
                steps_since_move_++;
            }
        }
        // --- 新增:得分区占用追踪(双方) ---
        const bool me_in_zone = in_zone(me.last_known_pos, zones_);
        const bool opp_in_zone = in_zone(opp.last_known_pos, zones_);
        if (me_in_zone) {
            my_turn_zone_streak_++;
            my_zone_turns_ago_ = 0;
        } else {
            my_zone_turns_ago_++;
        }
        if (opp_in_zone) {
            opp_zone_turns_ago_ = 0;
        } else {
            opp_zone_turns_ago_++;
        }
        // 衰减 recent_deaths_(每 turn 减 1,下限 0)
        if (recent_deaths_ > 0 && turn_ > 0) recent_deaths_--;
        // ---- 路径指纹更新 ----
        // 起点位移向量
        start_pos_ = me.last_known_pos;
        // 本回合起点与上一回合起点距离(累加近似"总位移")
        const int dx_start = std::abs(me.last_known_pos.x - last_turn_pos_.x);
        const int dy_start = std::abs(me.last_known_pos.y - last_turn_pos_.y);
        if (dx_start + dy_start > 0) {
            recent_displacement_ += 1;
            // 简化:每 3 个 act 记一个 unique cell
            if ((actions_used_ == 0) && (turn_ % 2 == 0)) recent_unique_cells_++;
            // 起点变化 → 重置"同一格停留"计数
            consecutive_same_pos_turns_ = 0;
        } else if (!first_act_) {
            consecutive_same_pos_turns_++;
        }
        last_turn_pos_ = me.last_known_pos;
        // 短窗口死亡计数(最近 3 turn)
        death_count_short_window_ = std::min(recent_deaths_, 3);
        // path_anchor(每 4 turn 重新定位为当前位置,作为路径"锚点")
        if (turn_ % 4 == 0) {
            path_anchor_x_ = me.last_known_pos.x;
            path_anchor_y_ = me.last_known_pos.y;
        }
        my_pos_ = me.last_known_pos;
        my_facing_ = me.last_known_facing;
        fire_cd_ = me.fire_cd;
        scan_cd_ = me.scan_cd;
        actions_used_ = 0;

        // --- 信念时间推进:敌方自上一来我方 act 起行动过一个阶段 ---
        if (!first_act_) dilate_belief();

        // --- 视野证伪:当前 T 形视野内的格子若有人我必看到 ---
        opp_visible_ = opp.visible;
        opp_directly_visible_ = opp.visible && can_see_me(opp.last_known_pos);
        if (opp.visible && opp.last_known_pos.x >= 0) {
            belief_.fill(0.0f);
            belief_[cell(opp.last_known_pos.x, opp.last_known_pos.y)] = 1.0f;
            intel_pos_ = opp.last_known_pos;
            intel_facing_ = opp.last_known_facing;
            intel_turn_ = turn_;
        }
        if (!opp_directly_visible_) subtract_visible_cells();

        seen_now_ = opp_directly_visible_ ? opp.last_known_pos : Pos{-1, -1};
        first_act_ = false;
    }

    // 每次行动函数返回后调用,更新观测(行动立即结算)。
    // action_id: 0=move 1..4=NESW 5=fire 6=scan 7=end(用于 last_was_fire_)
    void on_observation(const ActionObservation& o, bool consumed, int action_id = -1) {
        last_action_id_ = action_id;
        // 朝向变了但额度未消耗 = 复活免费 TURN 被用掉
        if (free_turn_ && !consumed && o.my_facing != my_facing_) free_turn_ = false;
        my_pos_ = o.my_pos;
        my_facing_ = o.my_facing;
        fire_cd_ = o.fire_cd;
        scan_cd_ = o.scan_cd;
        if (consumed) {
            ++actions_used_;
            last_was_fire_ = (last_action_id_ == 5);
        }
        opp_visible_ = o.opp_visible;
        opp_directly_visible_ = o.opp_directly_visible;
        if (o.opp_visible && o.opp_last_known_pos.x >= 0) {
            belief_.fill(0.0f);
            belief_[cell(o.opp_last_known_pos.x, o.opp_last_known_pos.y)] = 1.0f;
            intel_pos_ = o.opp_last_known_pos;
            intel_facing_ = o.opp_last_known_facing;
            intel_turn_ = turn_;
        }
        if (!opp_directly_visible_) subtract_visible_cells();
        seen_now_ = opp_directly_visible_ ? o.opp_last_known_pos : Pos{-1, -1};
        prev_act_end_pos_ = my_pos_;
    }

    // 当前观测编码到 out[kObsDim]。
    void encode(float* out) const {
        std::memset(out, 0, sizeof(float) * kObsDim);
        // 平面 0:障碍;1:得分区
        for (const Pos& o : obstacles_) out[0 * kCells + cell(o.x, o.y)] = 1.0f;
        for (const Pos& z : zones_) out[1 * kCells + cell(z.x, z.y)] = 1.0f;
        // 平面 2:我方位置
        out[2 * kCells + cell(my_pos_.x, my_pos_.y)] = 1.0f;
        // 平面 3:敌方信念
        std::memcpy(out + 3 * kCells, belief_.data(), sizeof(float) * kCells);
        // 平面 4:当前直接看到
        if (seen_now_.x >= 0) out[4 * kCells + cell(seen_now_.x, seen_now_.y)] = 1.0f;
        // 平面 5:最后已知情报
        if (intel_pos_.x >= 0) out[5 * kCells + cell(intel_pos_.x, intel_pos_.y)] = 1.0f;
        // 平面 6/7:出生点常量
        out[6 * kCells + cell(0, 0)] = 1.0f;
        out[7 * kCells + cell(6, 6)] = 1.0f;
        // 标量
        float* s = out + kPlanes * kCells;
        switch (my_facing_) {
            case 'N': s[0] = 1.0f; break;
            case 'E': s[1] = 1.0f; break;
            case 'S': s[2] = 1.0f; break;
            case 'W': s[3] = 1.0f; break;
            default: break;
        }
        switch (intel_facing_) {
            case 'N': s[4] = 1.0f; break;
            case 'E': s[5] = 1.0f; break;
            case 'S': s[6] = 1.0f; break;
            case 'W': s[7] = 1.0f; break;
            default: s[8] = 1.0f; break;  // 未知
        }
        s[9] = fire_cd_ / 3.0f;
        s[10] = scan_cd_ / 3.0f;
        s[11] = my_score_ / 20.0f;
        s[12] = opp_score_ / 20.0f;
        s[13] = turn_ / 24.0f;
        s[14] = actions_used_ / 3.0f;
        s[15] = is_blue_ ? 1.0f : 0.0f;
        s[16] = opp_visible_ ? 1.0f : 0.0f;
        s[17] = opp_directly_visible_ ? 1.0f : 0.0f;
        s[18] = intel_pos_.x >= 0 ? (turn_ - intel_turn_) / 24.0f : 1.0f;
        s[19] = free_turn_ ? 1.0f : 0.0f;
        // ---- 新增:死亡/重复操作/得分区/可见性追踪 ----
        s[20] = std::min(recent_deaths_, 4) / 4.0f;       // 最近 4 turn 内被击杀次数
        s[21] = std::min(total_deaths_, 10) / 10.0f;       // 本局累计被击杀次数
        s[22] = std::min(steps_since_move_, 10) / 10.0f;  // 连续未移动步数
        s[23] = std::min(kills_, 5) / 5.0f;               // 本局累计击杀数
        s[24] = std::min(my_zone_turns_ago_, 10) / 10.0f; // 距上次在得分区回合数
        s[25] = std::min(opp_zone_turns_ago_, 10) / 10.0f;
        s[26] = std::min(my_turn_zone_streak_, 6) / 6.0f;  // 我方连续站得分区回合数
        s[27] = last_was_fire_ ? 1.0f : 0.0f;
        // ---- 新增:路径指纹(反路径坍缩信号)----
        // s[28..31] = 我方近 4 turn 起点位置(归一化到 [0,1])
        s[28] = static_cast<float>(path_anchor_x_) / 6.0f;
        s[29] = static_cast<float>(path_anchor_y_) / 6.0f;
        // s[30..31] = 本局起点-当前位置位移向量
        s[30] = (my_pos_.x - start_pos_.x) / 6.0f;
        s[31] = (my_pos_.y - start_pos_.y) / 6.0f;
        // s[32..33] = 近 4 turn 累计位移量(0..6)
        s[32] = std::min(recent_displacement_, 6) / 6.0f;
        s[33] = std::min(recent_unique_cells_, 7) / 7.0f;
        // s[34..35] = 重复路径信号(连续两次回到同位置 / 同一格停留回合数)
        s[34] = std::min(consecutive_same_pos_turns_, 4) / 4.0f;
        s[35] = std::min(death_count_short_window_, 3) / 3.0f;
    }

    // 公开掩码(只用公开信息;隐形敌人占据格不掩,交给引擎拒绝,失败不消耗)
    void action_mask(float* mask) const {
        for (int i = 0; i < 8; ++i) mask[i] = 0.0f;
        if (actions_used_ >= 3) {
            mask[7] = 1.0f;  // 只能 end
            return;
        }
        // move
        int dx = 0, dy = 0;
        delta(my_facing_, dx, dy);
        const int nx = my_pos_.x + dx, ny = my_pos_.y + dy;
        if (nx >= 0 && nx < kBoardSize && ny >= 0 && ny < kBoardSize &&
            !is_obstacle(nx, ny) && !(opp_directly_visible_ && nx == intel_pos_.x && ny == intel_pos_.y))
            mask[0] = 1.0f;
        // turn 1..4 = N/E/S/W
        static const char kDirs[4] = {'N', 'E', 'S', 'W'};
        for (int i = 0; i < 4; ++i)
            if (my_facing_ != kDirs[i]) mask[1 + i] = 1.0f;
        // fire / scan
        if (fire_cd_ == 0) mask[5] = 1.0f;
        if (scan_cd_ == 0) mask[6] = 1.0f;
        // end
        mask[7] = 1.0f;
    }

    // 供环境读取的内部状态
    Pos my_pos() const { return my_pos_; }
    char my_facing() const { return my_facing_; }
    int actions_used() const { return actions_used_; }
    int fire_cd() const { return fire_cd_; }
    int scan_cd() const { return scan_cd_; }
    bool free_turn() const { return free_turn_; }
    int total_deaths() const { return total_deaths_; }
    int recent_deaths() const { return recent_deaths_; }
    int steps_since_move() const { return steps_since_move_; }
    void clear_free_turn() { free_turn_ = false; }

private:
    static int cell(int x, int y) { return y * kBoardSize + x; }
    static bool same(Pos a, Pos b) { return a.x == b.x && a.y == b.y; }
    static void delta(char f, int& dx, int& dy) {
        switch (f) {
            case 'N': dx = 0; dy = -1; break;
            case 'E': dx = 1; dy = 0; break;
            case 'S': dx = 0; dy = 1; break;
            case 'W': dx = -1; dy = 0; break;
            default: dx = 0; dy = 0; break;
        }
    }
    bool in_zone(Pos p, const std::vector<Pos>& zones) const {
        for (const Pos& z : zones) if (z.x == p.x && z.y == p.y) return true;
        return false;
    }
    bool is_obstacle(int x, int y) const {
        for (const Pos& o : obstacles_)
            if (o.x == x && o.y == y) return true;
        return false;
    }
    bool can_see_me(Pos target) const {
        Sentry me{};
        me.last_known_pos = my_pos_;
        me.last_known_facing = my_facing_;
        return can_see(me, target, obstacles_);
    }

    // 敌方一个行动阶段最多移动 3 格:信念按 BFS≤3(绕障碍、不占我方格)扩张
    void dilate_belief() {
        std::array<float, kCells> next = belief_;
        for (int step = 0; step < 3; ++step) {
            std::array<float, kCells> cur = next;
            for (int y = 0; y < kBoardSize; ++y) {
                for (int x = 0; x < kBoardSize; ++x) {
                    if (cur[cell(x, y)] <= 0.0f) continue;
                    static const int kDx[4] = {0, 1, 0, -1};
                    static const int kDy[4] = {-1, 0, 1, 0};
                    for (int d = 0; d < 4; ++d) {
                        const int nx = x + kDx[d], ny = y + kDy[d];
                        if (nx < 0 || nx >= kBoardSize || ny < 0 || ny >= kBoardSize) continue;
                        if (is_obstacle(nx, ny)) continue;
                        if (nx == my_pos_.x && ny == my_pos_.y) continue;
                        next[cell(nx, ny)] = 1.0f;
                    }
                }
            }
        }
        belief_ = next;
    }

    // 当前视野内确认无人的格子从信念剔除
    void subtract_visible_cells() {
        for (int y = 0; y < kBoardSize; ++y)
            for (int x = 0; x < kBoardSize; ++x)
                if (belief_[cell(x, y)] > 0.0f && can_see_me({x, y}))
                    belief_[cell(x, y)] = 0.0f;
    }

    std::array<float, kCells> belief_{};
    Pos intel_pos_{-1, -1};
    char intel_facing_ = '?';
    int intel_turn_ = -100;
    Pos my_pos_{0, 0};
    char my_facing_ = 'E';
    int my_score_ = 0, opp_score_ = 0;
    int fire_cd_ = 0, scan_cd_ = 0;
    int turn_ = 0;
    int actions_used_ = 0;
    bool is_blue_ = false;
    bool opp_visible_ = false;
    bool opp_directly_visible_ = false;
    bool free_turn_ = false;
    bool first_act_ = true;
    Pos prev_act_end_pos_{0, 0};
    Pos seen_now_{-1, -1};
    std::vector<Pos> obstacles_;
    std::vector<Pos> zones_;
    // ---- 新增追踪状态 ----
    int recent_deaths_ = 0;          // 最近 4 turn 内被击杀次数(用于"死亡惩罚"信号)
    int total_deaths_ = 0;           // 本局累计被击杀次数
    int steps_since_move_ = 0;       // 连续未移动的步数(>2 触发蹲坑信号)
    int kills_ = 0;                  // 本局累计击杀数
    int my_zone_turns_ago_ = 99;     // 距上次在得分区回合数
    int opp_zone_turns_ago_ = 99;
    int my_turn_zone_streak_ = 0;    // 我方连续占点回合数(用于"蹲坑得分"识别)
    bool last_was_fire_ = false;
    int last_action_id_ = -1;
    // ---- 路径指纹追踪状态 ----
    Pos start_pos_{0, 0};            // 本局初始位置
    Pos last_turn_pos_{0, 0};        // 上一回合起始位置(用于位移检测)
    int path_anchor_x_ = 0;          // 路径锚点 x(每 4 turn 更新)
    int path_anchor_y_ = 0;
    int recent_displacement_ = 0;    // 累计位移(0..6)
    int recent_unique_cells_ = 0;    // 经过的不同格子数(0..7)
    int consecutive_same_pos_turns_ = 0;  // 连续未移动回合数
    int death_count_short_window_ = 0;     // 最近 3 turn 死亡计数
};

}  // namespace rl
