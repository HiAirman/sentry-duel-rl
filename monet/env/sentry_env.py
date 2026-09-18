"""自我对弈环境:动作级 MDP,粒度与部署侧 rl_ai_v5.cpp 的 act() 循环一致。

一次 step = 我方的一次行动函数调用(不是一整个回合)。这样:
  * 训练时的决策点与 .so 里的推理点完全相同;
  * 掩码重试(引擎拒绝 → 不消耗额度 → 换动作)天然可用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..engine.game import Game
from ..engine.types import Board
from . import obs as O
from .obs import ObsBuilder


@dataclass
class RewardConfig:
    """事件奖励默认与 rules.md 的计分完全一致,另加终局胜负奖励。

    `scan_reveal` 是唯一一项**不对应任何计分事件**的塑形项,默认 0(关闭)。
    打开后,"SCAN 成功且这一扫真的把之前看不见的敌人扫出来了"额外给一份奖励,
    用来把策略往"多用扫描"推。注意它会破坏「回报 == 比分差」这条不变量 ——
    所以默认关闭,`tests/test_env.py` 里钉的是默认值下的契约。
    """

    kill: float = 2.0
    death: float = -2.0
    zone: float = 1.0
    win: float = 1.0
    lose: float = -1.0
    draw: float = 0.0
    step: float = 0.0
    scan_reveal: float = 0.0


class SentryEnv:
    def __init__(
        self,
        opponent,
        agent_color: Optional[str] = None,
        reward: Optional[RewardConfig] = None,
        max_attempts: int = 8,
        max_steps: int = 400,
        seed: int = 0,
        game_kwargs: Optional[dict] = None,
    ):
        self.opponent = opponent
        self.fixed_color = agent_color
        self.reward_cfg = reward or RewardConfig()
        self.max_attempts = max_attempts
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.game_kwargs = game_kwargs or {}

        self._obs_buf = np.zeros(O.OBS_DIM, dtype=np.float32)
        self._mask_buf = np.zeros(O.ACTION_DIM, dtype=np.float32)
        self._reset_state()

    # ------------------------------------------------------------------ 内部

    def _reset_state(self) -> None:
        self.game = Game(**self.game_kwargs)
        self.ag_ob = ObsBuilder()
        self.opp_ob = ObsBuilder()
        self.agent_color = "R"
        self.attempts = 0
        self.steps = 0
        self.done = False
        self.deaths = 0  # 我方被击杀次数(用于校验奖励与比分一致)

    def _emit(self) -> Tuple[np.ndarray, np.ndarray]:
        obs = self.ag_ob.encode(self._obs_buf)
        mask = self.ag_ob.action_mask(self._mask_buf)
        return obs.copy(), mask.copy()

    def _enter_agent_phase(self) -> None:
        self.ag_ob.act_start(self.game.view(self.agent_color), self.agent_color)
        self.attempts = 0

    def _run_opponent_phase(self) -> float:
        """跑完对手的一整个行动阶段,返回我方应得的奖励(主要是被击杀)。

        对手用**生成器**逐个交出动作(`Opponent.turn`),每个动作都由本环境调
        `game.apply` 执行。整回合型对手(官方 Baseline / Hunter)因此能在自己的
        控制流里拿到每个动作的观测,而击杀事件仍然流经这里 —— 我方的阵亡惩罚和
        `deaths` 计数不会漏。若让对手自己驱动引擎,这两项就丢了。
        """
        color = self.game.phase
        assert color != self.agent_color
        ob = self.opp_ob
        ob.act_start(self.game.view(color), color)
        reward = 0.0
        gen = self.opponent.turn(lambda: self.game.view(color), color, ob)
        last = None
        try:
            # 上限放宽到 3×:整合回合的对手可能因引擎拒绝而重试同一个动作。
            for _ in range(3 * self.max_attempts):
                if self.game.phase_over():
                    break
                try:
                    a = gen.send(last)
                except StopIteration:
                    break
                res, ev = self.game.apply(color, a)
                ob.on_observation(res.observation, res.consumed, a)
                if ev.get("kill") and ev.get("death_color") == self.agent_color:
                    reward += self.reward_cfg.death
                    self.deaths += 1
                last = res
                if a == O.END:
                    break
        finally:
            gen.close()
        self.game.end_phase(color)
        return reward

    def _finish_agent_phase(self) -> float:
        """结束我方行动阶段:结算占点分 → 让对手行动 → 回到我方。"""
        reward = 0.0
        if self.game.in_zone(self.game.s[self.agent_color].pos):
            reward += self.reward_cfg.zone
        self.game.end_phase(self.agent_color)
        if not self.game.done:
            reward += self._run_opponent_phase()
        return reward

    def _terminal_reward(self) -> float:
        w = self.game.winner
        if w is None:
            return self.reward_cfg.draw
        return self.reward_cfg.win if w == self.agent_color else self.reward_cfg.lose

    # ------------------------------------------------------------------- API

    def reset(self, seed: Optional[int] = None, agent_color: Optional[str] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        color = agent_color or self.fixed_color
        if color is None:
            color = "R" if self.rng.random() < 0.5 else "B"
        self.agent_color = color
        self.opponent.reset()

        if self.game.phase != color:  # 我方执蓝:对手(红)先手
            self._run_opponent_phase()
        self._enter_agent_phase()
        return (*self._emit(), self._info())

    def step(self, action: int):
        """执行**一个**我方动作,返回 `(obs, mask, reward, terminated, truncated, info)`。

        一个 step ≠ 一个回合:额度用满(3 个动作)、主动 END、或引擎判阶段结束才换阶段,
        而换阶段会顺带把对手那一整个阶段跑完并结算占点分(见 `_finish_agent_phase`)。
        所以同一个 `reward` 里可能同时装着"本动作的事件分"和"本阶段的占点/被击杀分" ——
        读训练曲线时别把它当成单步即时奖励。

        `terminated` 与 `truncated` 分开报:前者是分出胜负、带终局奖励,后者是撞到
        `max_steps` 被截断、不带。**但调用方目前并不区分** —— `selfplay.py` 一律
        `done = term or trunc`(截断局不做 bootstrap,见那里的注释),`evaluate.py`
        同理。分成两个字段是为了留下区分的余地,不是因为现在有两条路径。
        """
        if self.done:
            raise RuntimeError("episode already done; call reset()")
        cfg = self.reward_cfg
        reward = cfg.step
        self.steps += 1

        if action != O.END:
            was_visible = self.ag_ob.opp_visible
            res, ev = self.game.apply(self.agent_color, action)
            self.ag_ob.on_observation(res.observation, res.consumed, action)
            if ev.get("kill"):
                reward += cfg.kill if ev.get("death_color") != self.agent_color else cfg.death
            if (
                cfg.scan_reveal
                and action == O.SCAN
                and res.success
                and res.observation.opp_visible
                and not was_visible
            ):
                # 只奖"扫出来了",不奖空扫,也不奖敌人本来就看得见时的扫描
                reward += cfg.scan_reveal
            # 动作被引擎拒绝(撞墙、越界……):**不加任何惩罚**。拒绝不消耗额度,
            # 合法性已经由掩码约束住了,再补一份惩罚只会让策略"怕"这个动作,
            # 教不会它"别做"—— 掩码之外它本来就不会被采样到。
        self.attempts += 1

        # attempts 数的是 step 调用次数,被引擎拒绝的动作也计入(它们不消耗额度),
        # 所以 max_attempts 是防死循环的上限,不是"本阶段还剩几步"。
        phase_done = (
            action == O.END
            or self.game.phase_over()
            or self.attempts >= self.max_attempts
        )
        truncated = False
        if phase_done:
            reward += self._finish_agent_phase()

        terminated = self.game.done
        if terminated:
            reward += self._terminal_reward()
            self.done = True
        elif self.steps >= self.max_steps:
            truncated = True
            self.done = True
        elif phase_done:
            self._enter_agent_phase()

        obs, mask = self._emit()
        return obs, mask, float(reward), terminated, truncated, self._info()

    def _info(self) -> dict:
        r, b = self.game.score
        return {
            "turn": self.game.turn,
            "score": (r, b),
            "agent_color": self.agent_color,
            "my_score": r if self.agent_color == "R" else b,
            "opp_score": b if self.agent_color == "R" else r,
            "winner": self.game.winner,
            "actions_used": self.game.actions_used,
            "deaths": self.deaths,
        }

    # -------------------------------------------------------------- 便捷封装

    @property
    def obs_dim(self) -> int:
        return O.OBS_DIM

    @property
    def action_dim(self) -> int:
        return O.ACTION_DIM

    def result(self) -> float:
        """终局结果:1.0 胜 / 0.5 平 / 0.0 负(用于评测)。"""
        w = self.game.winner
        if w is None:
            return 0.5
        return 1.0 if w == self.agent_color else 0.0
