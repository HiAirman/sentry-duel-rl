"""检查点读写(numpy .npz)。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .models.factory import arch_of, make_net
from .models.mlp import MLP
from .models.ppo import PPOAgent


def save_checkpoint(path, net: MLP, agent: Optional[PPOAgent] = None, meta: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 键前缀区分三块数据:`p/` = 网络参数,`m/`、`v/` = Adam 的一/二阶矩。
    # load_checkpoint 按同样的前缀切回去,两边必须一致 —— 名字对不上不会报错,
    # 只会少装参数。
    blobs = {f"p/{k}": v for k, v in net.p.items()}
    if agent is not None:
        blobs.update({f"m/{k}": v for k, v in agent.m.items()})
        blobs.update({f"v/{k}": v for k, v in agent.v.items()})
    payload = dict(meta or {})
    payload.update({"obs_dim": net.obs_dim, "hidden": net.hidden, "act_dim": net.act_dim})
    # 架构必须写进检查点:读的时候要按它重建网络。少了这一项,GRU 的检查点会被
    # 当成 MLP 读回来 —— `load_state_dict` 宽容地跳过所有 GRU 键,加载"成功",
    # 模型却是随机初始化的 MLP。
    payload["arch"] = arch_of(net)
    if getattr(net, "is_recurrent", False):
        payload["gru_hidden"] = int(net.gru_hidden)
    if agent is not None:
        payload["adam_t"] = agent.t
    # meta 序列化成 JSON 字符串、而不是直接存 dict:load_checkpoint 用
    # `allow_pickle=False` 打开,存成对象数组的 dict 会让加载直接失败。
    blobs["meta"] = np.array(json.dumps(payload, ensure_ascii=False))
    np.savez_compressed(path, **blobs)


def load_checkpoint(path, with_optimizer: bool = False) -> Tuple[MLP, Optional[dict], Optional[PPOAgent]]:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    # 形状从 meta 里读,所以检查点是自描述的;网络参数的**集合**则直接来自文件里的 p/ 键
    # (不是从 TENSORS 重建),缺键不会报错,只会让网络少几层参数。
    net = make_net(
        arch=meta.get("arch", "mlp"),
        obs_dim=int(meta["obs_dim"]),
        hidden=int(meta["hidden"]),
        act_dim=int(meta["act_dim"]),
        gru_hidden=int(meta.get("gru_hidden", 128)),
    )
    sd = {k[len("p/") :]: data[k] for k in data.files if k.startswith("p/")}
    if getattr(net, "is_recurrent", False):
        # 循环网络**不能**直接 `net.p = sd`:它的 p 是 enc/gru/自身三处的合并视图,
        # 整体替换会把共享打断 —— Adam 更新新字典、前向读旧数组,表现为训练完全
        # 不动。走 load_state_dict 让它自己重新合并。
        net.load_state_dict(sd)
    else:
        net.p = sd
    agent = None
    if with_optimizer:
        # 只有显式要求才还原 Adam 状态(m/v + 步数)。这里建的 PPOAgent 只是个容器:
        # 检查点不存超参,lr 等走的是默认值,调用方必须自己用 Config 指定,
        # 否则续训会静默换成默认超参。
        from .models.ppo import PPOAgent

        agent = PPOAgent(net, lr=3e-4)
        agent.m = {k[len("m/") :]: data[k] for k in data.files if k.startswith("m/")}
        agent.v = {k[len("v/") :]: data[k] for k in data.files if k.startswith("v/")}
        agent.t = int(meta.get("adam_t", 0))
    return net, meta, agent
