# V46_COMPACT — 共享嵌入 + 训练协议升级

## 动机
v43 达到 AUC 0.80116（assist2017 fold0/seed42），后续 UMK 变体（v45 两方向）均无增益。
v46 目标：**在不损失创新性的前提下压缩参数量、升级训练协议**，同时提升效率与泛化。

## 两处改动（全部向后兼容，默认关闭）

### 1. 共享嵌入（weight tying）— `use_shared_embeddings`（config 开关，默认 0）
- 现状：`concept_emb`(3109×256=796K) 与 `hist_concept_emb`(796K)、
  `item_emb`(97×256=24.8K) 与 `hist_item_emb`(24.8K) 是**四个独立表**，共 ~1.64M 参数（32%）。
- 改动：`use_shared_embeddings=1` 时 `hist_*_emb = *_emb`（PyTorch 权重绑定）。
  target/history 双流复用同一词表 —— 同一概念空间在不同时间位置用同一向量。
- 依据（跨域借鉴）：Transformer 权重共享（Press & Wolf 2017）、ALBERT 参数共享；
  与 A2G 的"双流输入"创新不冲突 —— 双流结构与边界拆分、近因加权等机制全部保留。
- 收益：参数 5,107,701 → 4,286,709（**−16.1%**），named_modules 97 → 95。

### 2. 训练协议升级 — `config["training"]` 新字段（缺省 = v43 旧行为）
| 字段 | v43 旧值 | v46 新值 | 依据（跨域借鉴） |
|---|---|---|---|
| `optimizer` | adam（缺省） | `adamw` | AdamW（Loshchilov & Hutter 2019）解耦权重衰减 |
| `weight_decay` | 0.0 | 1e-4 | 无正则 → v43 早停 @32 过拟合信号 |
| `shuffle` | false（FixedOrder 顺序取批） | true（seed×1000003+epoch 确定性打乱） | 深度学习通用批量训练实践 |
| `warmup_epochs` | 0（恒定 lr） | 3（3 轮线性 warmup → 余弦衰减到 10%） | warmup+cosine（NLP/CV 标准调度） |

- `FixedOrderBatchSampler.__init__` 加 `shuffle=False` 参数；`signature`/`state_dict` 记录
  shuffle 标志，训练可复现（同 seed+epoch 打乱序列一致）。
- `experiment.py`：`loader(..., shuffle=config)`；AdamW 可选；`warmup_epochs>0` 时创建
  LambdaLR（warmup 后余弦衰减），每 epoch 评估后 `scheduler.step()`。

## 兼容性
- 根默认（无新 config 键）与 v43 逐位兼容：95/95 测试通过（含 server 逐位对照、
  消融 gate 对照、梯度对照、RNG 流对照）。
- v46 为**新架构（共享表）+ 新协议**，需完整重训，不继承 v43 checkpoint。
- UMK（v45 r2）保持 `use_umk_*=0` 默认关闭 —— attention 位置调制线已证伪，不纳入 v46。

## 状态
- [x] 代码：`_initialization.py` 开关、`data.py` sampler、`experiment.py` 协议
- [x] 测试：95/95 unittest 通过
- [x] config：`configs/assist2017_v46_compact.json`（`use_shared_embeddings:1` + 新协议）
- [ ] 冻结 candidates/v46_compact + 冻结版 smoke + full 训练
