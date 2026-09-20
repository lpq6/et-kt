# Assist2017 实验协议

## 固定边界

- 数据：服务器训练折词表版 Assist2017，不重新划分学生。
- 训练 folds 1,2,3,4；验证 fold0；seed42。
- 训练 3582 切片 / 1093 学生 / 606414 个计分交互。
- 验证 905 切片 / 274 学生 / 153067 个计分交互。
- 长度 200，width256，FFN512，4 attention layers / 8 heads。
- Adam，lr=0.0001，weight_decay=0，训练 batch64，评估 batch128。
- 最大 200 epoch，patience20，固定 CSV 顺序，无跨切片状态。
- 最大验证 AUC，依次以更低 NLL、Brier、ECE15 打破平局，保留更早 epoch。
- 初始化只允许随机参数与训练折统计先验，无预训练模型或优化器状态复用。
- 不读取测试集、不执行窗口测试、不自动扩展数据集或 seed。

## 显式 Batch256 候选

`assist2017_fold0_seed42_batch64_v1` 是原参考协议，无协议字段的旧配置仍解释为它。
`assist2017_fold0_seed42_batch256_direct_v1` 是另行命名的直接 batch256 候选，
只改训练 batch，保持评估 batch128、Adam lr=0.0001、最大 200 epoch、
patience20、seed、fold 和数据绑定不变。没有同步放大学习率。

3582 个训练切片，batch64 每轮 56 次更新，batch256 每轮 14 次更新；
最后一批分别为 62 和 254 条。两个协议的优化过程不同，结果必须分别记录。
新运行保存协议标识、`gradient_accumulation_steps=1`、`microbatching=false`，
并在每轮记录 `optimizer_steps`。独立审计拒绝与声明不符的批量和更新次数。

本机三个版本直接 batch256 均在前向阶段 OOM；并未启动 batch256 正式训练。
batch128 只做了合成一步的容量测试，没有新增其正式训练协议。
梯度累积和混合精度均未实现或启用，不能把它们记成此次的 FP32 直接 batch256。

## 三种结论不可混淆

1. **可运行/等价**：代码、状态字典、预测、梯度、因果性和重载测试通过。
2. **点门槛通过**：完整独立训练完成，检查点重评一致，验证 AUC 严格 > 0.8174。
3. **论文级完成**：独立终态审计、同协议基线、固定五个数据集与模块重训均达标。

上述第三项绝不能由第一项或一次局部 smoke 推导。

服务器协议中的“5/5”明确指选定的五个数据集，不是五个随机 seed。
总目标是在原来相同的八个候选数据集中，至少同一组五个同时满足基线和模块门槛。

## 可比性限制

服务器 `trainfold_data_summary.json` 标明：
`legacy_scores_not_comparable_after_remap=true`，
`baseline_retraining_required=true`。

因此 0.8174 是当前固定描述性门槛，不自动等于“已超越同协议全部基线”。
跨操作系统、PyTorch/CUDA 版本不能预设逐位相同；本地测试的逐位等价是在
同一运行环境内，对比服务器源码与整理源码。

## 消融

已实现九个 v33 模块开关，v43 另有输入记忆/多模态残差开关。
它们必须各自从匹配的初始状态重新训练，不能拿 Full 权重关闭模块后评分
冒充独立重训消融。当前 CLI 不自动执行完整消融或跨数据集实验。

版本源码分别在 `versions/` 的独立子目录中保存，
不是同一目录内切换配置后覆盖代码。已冻结版本的修改应另建版本目录；
默认 `run.py` 会拒绝哈希变化和跨版本架构/重计算配置。

正向贡献需要 Full AUC 严格大于对应独立重训消融，按学生聚类的不确定性分析
和多重比较也需单独核查。零初始化等价、模块有梯度都不是正向贡献证据。

## 本地历史对齐候选

`v44_aligned_history_local` 单独把基础统计窗口的历史对角线从 -1 改为 0。
输入历史本来已右移，因此它只增加最近已观察的 t-1 事件，不增加当前标签。
参数、初始化和上述训练协议不变；详见 `V44_ALIGNED_HISTORY.md`。
旧 v33/v43 代码行为不变，正在执行的 v43 实验不切换候选。

新对照 `no_aligned_history_statistics` 恢复旧窗口，必须独立重训；
若采用该架构，模块对照总数为 12，八数据集的预声明比较集合为 96。
完整消融训练仍需另行冻结执行协议，不能用短测试或关闭训练后权重代替。

## 产物

每次完整运行输出：
`run.json`、`initial_state.pt`、`epochs.jsonl`、`selected_model.pt`、
`validation_predictions.npz`、`result.json`。

异常只写 `incomplete.json`，不当作科学失败，也不自动重试或覆盖旧运行。
预测文件包含 label、probability、learner_uid、csv_row_index、position，
用于后续独立重算与学生级统计分析。
