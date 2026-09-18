"""按 `arch` 建网络 —— 检查点、训练器、导出三处共用同一个入口。

单独抽出来是因为**架构名散落在三处**:训练器按它选 PPO 变体、store 按它重建
网络、export 按它选导出格式。三处各写一份 `if arch == "gru"` 迟早漏一处,而漏
的表现是"存下来是 GRU、读回来是 MLP",权重键对不上,`load_state_dict` 宽容地
跳过所有 GRU 键 —— 于是加载"成功",模型却退化成随机初始化的 MLP。
"""

from __future__ import annotations

from .mlp import MLP

ARCHS = ("mlp", "gru")


def make_net(arch: str = "mlp", obs_dim: int = 428, hidden: int = 512,
             act_dim: int = 8, gru_hidden: int = 128, seed: int = 0):
    if arch == "mlp":
        return MLP(obs_dim=obs_dim, hidden=hidden, act_dim=act_dim, seed=seed)
    if arch == "gru":
        from .rnn_policy import RNNPolicy

        return RNNPolicy(obs_dim=obs_dim, hidden=hidden, act_dim=act_dim,
                         gru_hidden=gru_hidden, seed=seed)
    raise ValueError(f"未知架构 {arch!r},可选:{'、'.join(ARCHS)}")


def arch_of(net) -> str:
    return "gru" if getattr(net, "is_recurrent", False) else "mlp"


def tensor_table(net):
    """这个网络的 `(C++ 符号, 参数名)` 导出表。

    和 `make_net` 同一个理由放这里:导出器要按架构选符号表、解析器要按架构选同一
    张表,**两边各写一份 `if` 就会分叉**,而分叉的表现是符号名对不上 —— 解析端
    报"找不到定义"还算好的,真错位了就是网络照常前向、只是变弱。
    """
    if getattr(net, "is_recurrent", False):
        from .rnn_policy import TENSORS

        return TENSORS
    from .mlp import TENSORS

    return TENSORS
