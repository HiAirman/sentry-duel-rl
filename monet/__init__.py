"""rl_monet_v1 —— 哨兵大战(sentry_duel V4.4)强化学习训练引擎。

模块划分
--------
engine/    规则状态机(纯 Python,权威实现,可独立测试)
env/       观测(obs_builder.h 的 Python 移植)+ 自我对弈环境 + 陪练对手
models/    纯 NumPy MLP 与 PPO
training/  自对弈训练循环、评测、导出 C++ 权重头
"""

__version__ = "1.0.0"
__all__ = ["engine", "env", "models", "training"]
