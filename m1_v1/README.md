# `m1_v1` —— 起训权重,不是参赛包

**这个目录里只有 `rl_weights.h`(由 `export` 从检查点重新生成,未入版本库)。**

- 来源:`runs/m1/best.npz`,**step 61,440** —— 全仓库训练度最低的一份冻结导出。
- 用途:`python -m monet.cli train --arch gru --init-pack m1_v1 ...`,即**从一份很弱的
  编码器起步**,给 GRU 记忆层(`P`/`bP`)留出生长空间。见 `EXPLORATION_v2.md` §5 与 §7.4。
- **它不是一个可提交的参赛包**:没有 `obs_builder.h`,也没有 `rl_VG.cpp`。目录里没有
  这两个文件是**故意的** —— 它从没提交过,也不该被交上去。
- 解析路径:MLP 包 → `load_init_net` 走 `RNNPolicy.from_mlp`,**只继承编码器与三个头
  (16 个张量)**,GRU 与 `P`/`bP` 全新且 `P`/`bP` 为零,所以初始前向逐位等于本包。
- 注册位置:`monet/pack.py::KNOWN_PACKS`。**不在** `selfplay.py` 的对手表里,也**不在**
  任何联盟/评测名单里 —— 加进去会改变联盟配重与评测口径。

重新生成:

```bash
python -m monet.cli export --ckpt runs/m1/best.npz --out m1_v1/rl_weights.h
```

注意 `export` 的护栏会拒绝写入**已注册**的包目录,所以上面这条命令现在会报错 ——
要重新生成得先临时从 `KNOWN_PACKS` 里注释掉这一行,生成完再放开。
