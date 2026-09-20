# V43 迭代与消融核对

记录日期：2026-09-18。正在训练的 `versions/v43_lowmem` 不改写。
`versions/v43_lowmem_r2` 与 `versions/v43_lowmem_r3` 已另存为完整独立副本，
补齐独立消融入口和诊断；`r3` 还修复了 CPU 测试报告的跳过项序列化。
Full 模型、参数、初始化、
训练数据、batch64、优化器与停止规则不变，不作为新的 Full 性能候选重训。

## 服务器材料核对

新读取的材料放在根目录 `provenance/review_20260918/`，不覆盖旧来源包。
`v43_protocol.json`、`v43_source_review.json`、`v43_design_audit.md`、
`v43_reference_snapshot.json` 与服务器 v43 原 manifest 对应条目逐字节核查。
远程只读目录盘点截至本次读取未见更晚的 `independent_a2g*` 候选，
也未见 v43 的正式训练 artifacts 目录。这不是全服务器遍历或结果存在性证明。

实际代码与文字的差异必须披露：

- v43 `design_audit.md` 仍主要描述 v39 四步输入记忆，并未完整描述新增 modal 残差。
- v43 协议主结构记载新增 7168 参数，但继承的容量限制文字还写 3072。
  实际为输入记忆 3072，加上两组 `[8,256]` 路由参数 4096，总计 7168。
- `no_input_memory` 的历史文字指同时去掉输入记忆与 modal，而代码有两个独立开关。
  组合对照不能单独证明这两个模块各自有效。
- 服务器 `paired_learner_audit.py` 的默认 `VARIANTS` 仍只有 Full 和三个早期对照，
  CLI 默认 family-size 为 15；与 v43 协议的十个对照 / family80 不一致。
  该脚本的加权学生重采样实现可供核查，但其默认入口不是完整 v43 消融证据。

这些是说明与覆盖问题，不是在本轮训练中临时改变模型的理由。

## R2 独立干预

| CLI variant | 直接关闭 | 保留或连带影响 |
| --- | --- | --- |
| `no_evidence` | priors、统计证据、直接证据及支持量缩放 | 保留原始历史 q/c/r |
| `no_ssm` | SSM 主路径和读出 | 同时失去输入记忆与 modal 的输出贡献 |
| `no_attention` | attention 与其 FFN | 同时失去内嵌 FFN gate |
| `no_factorized_input` | 语义/证据分流残差 | 保留原 token 投影 |
| `no_item_attempt_stage` | 题目尝试次数读出 | 其余模块保留 |
| `no_history_pace` | 历史时间调制 | Newton 与图保留 |
| `no_prequential_newton` | 在线 Newton 修正 | 图与时间调制保留 |
| `no_concept_graph` | 概念图递推 | FFN gate 和两种 SSM 记忆保留 |
| `no_ffn_gate` | FFN 隐层输入门 | attention/FFN 与两种 SSM 记忆保留 |
| `no_input_memory` | 四步投影输入记忆 | modal 残差仍启用 |
| `no_multimode_residual` | 多时间尺度 modal 残差 | 四步输入记忆仍启用 |

Full 与各对照从同 seed 的相同状态张量重新生成，优化器全新。
CPU 测试会显式激活两类参数，确认关闭目标模块后预测改变且目标梯度消失，
另一模块保留梯度；这些只是干预语义检查，不是 AUC 正向作用证明。

新的两项独立控制使完整集合为 11 个对照；若正式采用，八个候选数据集的
预先声明多重比较集合为 88，而不是沿用历史 family80。尚未冻结正式执行包，
CLI 仍拒绝 `train --variant` 非 Full，未启动任意消融或额外数据集训练。

## 下一阶段需要的证据

1. 当前 Assist2017 Full 按原停止规则完成，进程正常退出。
2. 完整 153067 个验证交互经 CSV 绑定、指标独立重算及检查点重评一致。
3. AUC 严格 > 0.8174 后才能另行冻结跨数据集及独立消融执行协议。
4. 同一组至少五个数据集同时超过各自固定基线数值门槛，且全部独立模块对照有效。
5. 学生级配对 bootstrap 与多重比较显式报告；单 seed 不证明跨 seed 稳健性。

基线表的五折均值、论文数值与本任务单折验证结果不可冒充匹配实验。
既有词表重映射也影响可比性；未重训基线的比较只能标注为描述性门槛。
不能把“缺基线”当零，不能以一次 Assist2017 通过宣布五数据集目标完成。
