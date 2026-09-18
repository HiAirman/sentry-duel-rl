"""PPO(掩码离散动作)—— 纯 NumPy 实现,含 GAE 与 Adam。

与常见 PPO 实现的差异点:
* 策略头带 action mask(引擎不允许的动作在采样与损失里都被屏蔽);
* 价值头独立于策略头(Wv/bv 只用于训练)。头文件里照导(kW4/kB4),
  只是 rl_ai_v5.cpp 那类推理壳不读它 —— 这样导出的 rl_weights.h 与推理壳完全对齐。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .mlp import MLP

# 被屏蔽的动作在 logits 上替换成这个值。用有限值而不是 -inf:exp(-1e9) 会下溢到
# 0(正是想要的),但 -inf 会让 log_softmax 里的减法出 nan。
_NEG = -1e9


def masked_logits(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """被屏蔽的动作替换成哨兵值。

    **必须保持入参的 dtype,不能写死 float32。** `ppo_seq` 的有限差分检查在 float64
    下跑 —— float32 的 1e-7 舍入噪声会把可用的差分步长压到 1e-3 以上,分辨不出
    "梯度错了 1%" 这类错。哨兵也要取同 dtype 的标量,否则 numpy 会把整块提升到
    float64,同样毁掉那个检查。
    """
    logits = np.asarray(logits)
    return np.where(mask > 0, logits, np.asarray(_NEG, dtype=logits.dtype))


def log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def probs(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(masked_logits(logits, mask)))


@dataclass
class Rollout:
    """一条轨迹(可跨多局拼接;局与局之间用 done=1 隔开)。"""

    obs: list = field(default_factory=list)
    mask: list = field(default_factory=list)
    action: list = field(default_factory=list)
    logp: list = field(default_factory=list)
    value: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    done: list = field(default_factory=list)

    def add(self, obs, mask, action, logp, value, reward, done):
        self.obs.append(obs)
        self.mask.append(mask)
        self.action.append(action)
        self.logp.append(logp)
        self.value.append(value)
        self.reward.append(reward)
        self.done.append(done)

    def __len__(self):
        return len(self.reward)

    def arrays(self):
        # 返回顺序与 update() 里的解包顺序一一对应,改这里必须同步改那里
        # (两者都是同 dtype 的数组,顺序错了不会报错,只会静默训坏)。
        return (
            np.asarray(self.obs, dtype=np.float32),
            np.asarray(self.mask, dtype=np.float32),
            np.asarray(self.action, dtype=np.int64),
            np.asarray(self.logp, dtype=np.float32),
            np.asarray(self.value, dtype=np.float32),
            np.asarray(self.reward, dtype=np.float32),
            np.asarray(self.done, dtype=np.float32),
        )


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """标准 GAE-λ。dones[t]=1 表示 t 之后是新的一局(不 bootstrap)。"""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in range(T - 1, -1, -1):
        nonterminal = 1.0 - dones[t]
        next_v = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    returns = adv + values
    return adv, returns


class PPOAgent:
    """策略 + 优化器。"""

    def __init__(
        self,
        net: MLP,
        lr: float = 3e-4,
        clip: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        epochs: int = 4,
        minibatches: int = 8,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        lam: float = 0.95,
        seed: int = 0,
    ):
        self.net = net
        self.lr, self.clip = lr, clip
        self.vf_coef, self.ent_coef = vf_coef, ent_coef
        self.epochs, self.minibatches = epochs, minibatches
        self.max_grad_norm = max_grad_norm
        self.gamma, self.lam = gamma, lam
        self.rng = np.random.default_rng(seed)
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in net.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in net.p.items()}

    # ------------------------------------------------------------------ 采样

    def act(self, obs: np.ndarray, mask: np.ndarray, deterministic: bool = False):
        """返回 (action, logp, value)。单次前向同时拿到策略与价值。"""
        X = obs.reshape(1, -1).astype(np.float32)
        lg, v = self.net.forward(X, cache=False)
        lg = lg[0]
        ml = masked_logits(lg, mask)
        lp = log_softmax(ml)
        if deterministic:
            a = int(np.argmax(np.where(mask > 0, lg, _NEG)))
        else:
            p = np.exp(lp)
            a = int(self.rng.choice(len(p), p=p / p.sum()))
        return a, float(lp[a]), float(v[0])

    def value_of(self, obs: np.ndarray) -> float:
        _, v = self.net.forward(obs.reshape(1, -1).astype(np.float32), cache=False)
        return float(v[0])

    # ------------------------------------------------------------------ 更新

    def update(self, roll: Rollout, last_value: float = 0.0) -> Dict[str, float]:
        obs, mask, act, old_logp, values, rewards, dones = roll.arrays()
        T = len(rewards)
        if T == 0:
            return {}
        adv, returns = compute_gae(rewards, dones, values, last_value, self.gamma, self.lam)
        adv_std = adv.std()
        # 优势标准化。eps 是为 T=1 准备的:此时 std=0,不加 eps 会得到 nan/inf。
        adv_norm = (adv - adv.mean()) / (adv_std + 1e-8)

        batch = T
        mb_size = max(1, batch // self.minibatches)
        idx_all = np.arange(T)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0}
        n_upd = 0

        for _ in range(self.epochs):
            self.rng.shuffle(idx_all)
            for start in range(0, batch, mb_size):
                idx = idx_all[start : start + mb_size]
                if len(idx) == 0:
                    continue
                X = obs[idx]
                M = mask[idx]
                A = act[idx]
                OLP = old_logp[idx]
                R = returns[idx]
                ADV = adv_norm[idx]
                B = len(idx)

                logits, vpred = self.net.forward(X, cache=True)
                ml = masked_logits(logits, M)
                lp_all = log_softmax(ml)
                lp_a = lp_all[np.arange(B), A]
                p = np.exp(lp_all)

                ratio = np.exp(lp_a - OLP)
                surr1 = ratio * ADV
                surr2 = np.clip(ratio, 1.0 - self.clip, 1.0 + self.clip) * ADV
                use_surr1 = surr1 <= surr2
                policy_loss = -float(np.mean(np.minimum(surr1, surr2)))

                # 熵必须**逐行**算:软熵对 logits 的梯度是
                #   dL/dz_ij = ent_coef * p_ij * (logp_ij + H_i),
                # 其中 H_i 是第 i 行**自己**的熵。若用批量均值 H_batch 代替,每行会多出
                # 一个 (H_batch - H_i) 的伪项;在掩码被收窄成单动作的行上(规则强制的步:
                # p 是 one-hot、lp_all[k] 恰好为 0)它等于 1.0*(0 + H_batch) != 0,而
                # `dlogits *= M` 挡不住(M[k] 仍是 1),于是 Adam 会把那个动作的 logit
                # 一路压低 —— 不报错,只是策略越来越不肯选它。行熵在这里恰好是 0。
                ent_row = -(p * lp_all).sum(axis=1)
                # 日志报的是各行熵的均值,逐行化不改变它的数值,所以这条曲线与更早的
                # 读数仍可直接比较。**同一份 ent_row 既进日志又进梯度**(见下面 dlogits
                # 那行),它符号写反时曲线和训练会一起歪,而且哪里都不报错:只是"熵有
                # 没有塌"这一列失去意义 —— 而判断策略崩没崩主要就看它。
                ent = float(np.mean(ent_row))
                # 0.5 是为了让 d(0.5*(vpred-R)^2)/dvpred 干净地等于 (vpred-R),所以
                # 下面 dvalue 里不再出现 0.5 —— 两处要一起看。
                value_loss = 0.5 * float(np.mean((vpred - R) ** 2))
                approx_kl = float(np.mean(OLP - lp_a))
                clip_frac = float(np.mean(np.abs(ratio - 1.0) > self.clip))

                # --- 策略梯度 ---
                g_logp = np.where(use_surr1, -ADV, 0.0).astype(np.float32) * ratio / B
                dlogits = g_logp[:, None] * (
                    np.eye(self.net.act_dim, dtype=np.float32)[A] - p
                )
                # --- 熵正则(dL/dz = ent_coef * p * (logp + H_i),H_i 见上面的 ent_row)---
                dlogits += (self.ent_coef / B) * p * (lp_all + ent_row[:, None])
                dlogits *= M  # 被屏蔽的动作梯度归零

                # 值函数梯度:0.5 已与 value_loss 里的 0.5 相抵,/B 来自批量均值。
                dvalue = (self.vf_coef * (vpred - R) / B).astype(np.float32)

                self.net.zero_grad()
                self.net.backward(dlogits.astype(np.float32), dvalue)

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
        return stats

    def _clip_and_step(self) -> float:
        """Adam 更新一步,然后清空梯度(`update()` 里梯度在此处才消费完)。

        返回的是**裁剪前**的全局范数 —— 日志里的 grad_norm 要的是它,不是裁剪后的。

        **更新必须原地写(`p -= ...`)。** 对 RNNPolicy 而言 `net.p[k]` 与
        `enc.p[k]` / `gru.p[k]` 是**同一个 ndarray**;写成 `p = p - ...` 只会重绑
        循环变量,`net.p` 和编码器读到的仍是旧数组,前向于是用没更新过的权重 ——
        表现为"训练完全不动",且不报错。
        """
        g = self.net.g
        total = np.sqrt(sum(float((v.astype(np.float64) ** 2).sum()) for v in g.values()))
        scale = 1.0
        if self.max_grad_norm > 0 and total > self.max_grad_norm:
            scale = self.max_grad_norm / (total + 1e-8)
        self.t += 1
        b1, b2, eps = self.beta1, self.beta2, self.eps
        for k, p in self.net.p.items():
            grad = g[k] * scale
            m = self.m[k]
            v = self.v[k]
            m *= b1
            m += (1.0 - b1) * grad
            v *= b2
            v += (1.0 - b2) * (grad * grad)
            mhat = m / (1.0 - b1 ** self.t)
            vhat = v / (1.0 - b2 ** self.t)
            p -= (self.lr * mhat / (np.sqrt(vhat) + eps)).astype(np.float32)
        self.net.zero_grad()
        return float(total)
