"""PPO 的循环网络版本:按**连续片段**做截断 BPTT。

与 `ppo.py` 的唯一实质差异在 `update()` 的批构造。那边把 rollout 里的**单个时间步**
打乱成 minibatch —— 对无记忆的 MLP 完全正确,对循环网络则是**错的**:同一 minibatch
里混着来自不同时刻、不同局的步,隐状态传不下去,记忆层永远学不到跨步依赖,而且
不会报错、不会崩,只是"加了 RNN 但没变强"。

这里的做法:
1. 按 `done` 把 rollout 切成**局**;
2. 每局再切成定长**段**(`seg_len`);
3. minibatch 由**若干整段**组成;
4. 每段的初始隐状态用采样时存下来的那个(= 段起点**之前**那一步的隐状态),它天然
   detached —— 这就是截断点所在。

**为什么初始隐状态要从轨迹里存,而不是"从零开始重跑一遍"**:重跑要用当前参数、
而且要把这一局之前的所有步都跑一遍,代价随局长线性涨;存下来的是**行为策略**下的
隐状态,这正是截断 BPTT 的标准做法(R2D2 的 stored state),偏差有界且实现简单。

段尾不足一整段的用零补齐,并用 `valid` 掩码把补出来的步**从损失里剔除**。补齐放在
**段尾**是安全的:GRU 是因果的,后面的零补不到前面去。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .ppo import _NEG, PPOAgent, Rollout, compute_gae, log_softmax, masked_logits


class SeqRollout:
    """带隐状态的轨迹。

    单独一个类而不是给 `Rollout` 加字段:`ppo.py` 的 `arrays()` 是按位置解包的,
    往中间插一个字段会让 MLP 那条路径静默错位(全是 float 数组,错了不报错)。
    """

    def __init__(self):
        self.obs: list = []
        self.mask: list = []
        self.action: list = []
        self.logp: list = []
        self.value: list = []
        self.reward: list = []
        self.done: list = []
        self.h: list = []

    def add(self, obs, mask, action, logp, value, reward, done, h):
        """`h` 是**进入这一步时**的隐状态(= 上一步算出来的那个)。

        存"进入时"而不是"离开时",段起点的 h0 才能直接取 `h[a]`,不用回头找上一步;
        更重要的是语义清楚:段起点用到的信息**严格早于**段内任何一步。
        """
        self.obs.append(obs)
        self.mask.append(mask)
        self.action.append(action)
        self.logp.append(logp)
        self.value.append(value)
        self.reward.append(reward)
        self.done.append(done)
        self.h.append(h)

    def __len__(self):
        return len(self.reward)

    def arrays(self):
        return (
            np.asarray(self.obs, dtype=np.float32),
            np.asarray(self.mask, dtype=np.float32),
            np.asarray(self.action, dtype=np.int64),
            np.asarray(self.logp, dtype=np.float32),
            np.asarray(self.value, dtype=np.float32),
            np.asarray(self.reward, dtype=np.float32),
            np.asarray(self.done, dtype=np.float32),
            np.asarray(self.h, dtype=np.float32),
        )


def episode_segments(dones: np.ndarray, seg_len: int) -> List[Tuple[int, int]]:
    """按 `done` 切局、再切段,返回 [start, end) 列表(左闭右开)。

    **段绝不允许跨局。** 跨局的段会把上一局末尾的隐状态带进新的一局,而新一局
    开局 h 必须归零 —— 那正是 `_pick_opponent` 里 reset 的语义。
    """
    segs: List[Tuple[int, int]] = []
    T = len(dones)
    start = 0
    for t in range(T):
        if dones[t] > 0 or t == T - 1:
            end = t + 1
            a = start
            while a < end:
                segs.append((a, min(a + seg_len, end)))
                a += seg_len
            start = end
    return segs


class PPOSeqAgent(PPOAgent):
    """继承 `PPOAgent` 复用 Adam/裁剪(`_clip_and_step` 只依赖 `net.p`/`net.g`,
    对 RNNPolicy 的合并参数视图同样适用),只重写采样与批构造。"""

    def __init__(self, net, seg_len: int = 32, segs_per_mb: int = 8, **kw):
        super().__init__(net, **kw)
        self.seg_len = seg_len
        self.segs_per_mb = segs_per_mb

    # ------------------------------------------------------------------ 采样

    def act(self, obs: np.ndarray, mask: np.ndarray, h: np.ndarray,
            deterministic: bool = False):
        """返回 (action, logp, value, h_new)。"""
        lg, v, h_new = self.net.step(obs, h)
        ml = masked_logits(lg, mask)
        lp = log_softmax(ml)
        if deterministic:
            a = int(np.argmax(np.where(mask > 0, lg, _NEG)))
        else:
            p = np.exp(lp)
            a = int(self.rng.choice(len(p), p=p / p.sum()))
        return a, float(lp[a]), float(v), h_new

    def value_of(self, obs: np.ndarray, h: np.ndarray) -> Tuple[float, np.ndarray]:
        _, v, h_new = self.net.step(obs, h)
        return float(v), h_new

    # ------------------------------------------------------------------ 更新

    def update(self, roll: SeqRollout, last_value: float = 0.0) -> Dict[str, float]:
        obs, mask, act, old_logp, values, rewards, dones, hs = roll.arrays()
        T = len(rewards)
        if T == 0:
            return {}
        adv, returns = compute_gae(rewards, dones, values, last_value, self.gamma, self.lam)
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

        segs = episode_segments(dones, self.seg_len)
        if not segs:
            return {}
        S, L, A = len(segs), self.seg_len, self.net.act_dim

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0}
        n_upd = 0
        order = np.arange(S)

        for _ in range(self.epochs):
            self.rng.shuffle(order)
            for s0 in range(0, S, self.segs_per_mb):
                mb = order[s0:s0 + self.segs_per_mb]
                X, M, Vld, H0 = self._pack(obs, mask, hs, segs, mb, L)
                Aa = self._pack_vec(act, segs, mb, L, np.int64)
                OLP, R, ADV = (self._pack_vec(old_logp, segs, mb, L, np.float32),
                               self._pack_vec(returns, segs, mb, L, np.float32),
                               self._pack_vec(adv_norm, segs, mb, L, np.float32))

                logits, vpred = self.net.forward_seq(X, H0, cache=True)
                ml = masked_logits(logits, M)
                lp_all = log_softmax(ml)                       # (S,L,A)
                lp_a = np.take_along_axis(lp_all, Aa[..., None], axis=-1)[..., 0]
                p = np.exp(lp_all)

                ratio = np.exp(lp_a - OLP)
                surr1 = ratio * ADV
                surr2 = np.clip(ratio, 1.0 - self.clip, 1.0 + self.clip) * ADV
                use_surr1 = surr1 <= surr2

                # 归一化用**有效步数**,不是补齐后的槽位数:否则补出来的零会把
                # 损失和梯度整体缩小,而缩小比例随每批补齐多少而变 —— 学得慢,
                # 且慢得不规律。
                Nv = max(1.0, float(Vld.sum()))
                policy_loss = -float((np.minimum(surr1, surr2) * Vld).sum() / Nv)

                # 行熵(H_i 而不是批量均值),理由见 ppo.py 那段注释。
                ent_row = -(p * lp_all).sum(axis=-1)           # (S,L)
                ent = float((ent_row * Vld).sum() / Nv)
                value_loss = 0.5 * float((((vpred - R) ** 2) * Vld).sum() / Nv)
                approx_kl = float(((OLP - lp_a) * Vld).sum() / Nv)
                clip_frac = float(((np.abs(ratio - 1.0) > self.clip) * Vld).sum() / Nv)

                w = Vld / Nv
                # 不把 dlogits/dvalue 强行压成 float32:dtype 跟着参数走。
                # 实际训练里参数就是 float32,压不压都一样;但 tests/test_ppo_seq.py
                # 的有限差分要在 float64 下跑,压成 float32 会让差分噪声盖住真误差。
                g_logp = np.where(use_surr1, -ADV, 0.0).astype(p.dtype) * ratio * w
                eye = np.eye(A, dtype=p.dtype)
                dlogits = g_logp[..., None] * (eye[Aa] - p)
                dlogits += (self.ent_coef * w)[..., None] * p * (lp_all + ent_row[..., None])
                dlogits *= M  # 被屏蔽的动作梯度归零
                dvalue = self.vf_coef * (vpred - R) * w

                self.net.zero_grad()
                self.net.backward_seq(dlogits, dvalue)
                gn = self._clip_and_step()

                stats["policy_loss"] += policy_loss
                stats["value_loss"] += value_loss
                stats["entropy"] += ent
                stats["approx_kl"] += approx_kl
                stats["clip_frac"] += clip_frac
                stats["grad_norm"] += gn
                n_upd += 1

        if n_upd:
            for k in stats:
                stats[k] /= n_upd
        stats["samples"] = float(T)
        stats["segments"] = float(S)
        return stats

    # -------------------------------------------------------------- 批构造

    def _pack(self, obs, mask, hs, segs, mb, L):
        """把 |mb| 个段铺成 (S, L, ...) 的定长数组,段尾补零。"""
        S = len(mb)
        X = np.zeros((S, L, obs.shape[-1]), np.float32)
        M = np.zeros((S, L, mask.shape[-1]), np.float32)
        Vld = np.zeros((S, L), np.float32)
        H0 = np.zeros((S, hs.shape[-1]), np.float32)
        for i, si in enumerate(mb):
            a, b = segs[si]
            n = b - a
            X[i, :n] = obs[a:b]
            M[i, :n] = mask[a:b]
            Vld[i, :n] = 1.0
            H0[i] = hs[a]        # 段起点**进入时**的隐状态,天然 detached
        return X, M, Vld, H0

    def _pack_vec(self, arr, segs, mb, L, dtype):
        out = np.zeros((len(mb), L), dtype)
        for i, si in enumerate(mb):
            a, b = segs[si]
            out[i, :b - a] = arr[a:b]
        return out
