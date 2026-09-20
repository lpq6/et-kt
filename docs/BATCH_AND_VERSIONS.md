# Batch 与独立版本记录

日期：2026-09-18。设备：RTX 3060 Laptop 6GB；PyTorch 2.2.0 / CUDA 11.8。
测试保持 width256、length200、完整模型、FP32 参数和激活、float64 BCE。
没有梯度累积、microbatching、混合精度或模型剪枝。

## 本机 Batch 实测

| 版本 | 直接 batch | 结果 | 峰值 allocated |
| --- | --- | --- | --- |
| v33 | 256 | 前向 OOM | 5322996736 字节 |
| v43 | 256 | 前向 OOM | 5323025920 字节 |
| v43_lowmem | 256 | 前向 OOM | 5700026368 字节 |
| v43_lowmem | 128 | 前向、反向、Adam 一步完成，梯度有限 | 5261673472 字节 |
| v43_lowmem | 64 | 先前完整合成一步通过 | 2670045184 字节 |

OOM 行记录的是失败前的峰值，不是该 batch 的完整显存需求。
batch128 单步成功、余量较小，不能推出长训稳定；batch64 仍是默认配置。
这里的结论只针对本机当前实现与环境，不推断 172 服务器的容量。

原始记录：

- `runs/profile_v33_batch256_full_budget_20260918.json`
- `runs/profile_v43_batch256_full_budget_20260918.json`
- `runs/profile_v43_lowmem_batch256_full_budget_20260918.json`
- `runs/profile_v43_lowmem_batch128_full_budget_20260918.json`
- `runs/profile_v43_lowmem_batch64.json`

`configs/assist2017_v43_lowmem_batch256.json` 保留为明确命名的候选，
可在另一环境先执行 `run.py profile --batch-size 256`。
其是否提升 AUC 必须实际独立训练验证，不能仅由 batch 大小推断。

直接 batch256 并非模型在原则上禁止，而是本机当前完整 FP32 实现实测放不下。
Assist2017 的训练集有 3582 个切片；batch64 每轮更新 56 次，直接 batch256
每轮仅更新 14 次。因此更换 batch 还会改变训练过程，不能在正在运行的参考
实验中直接替换。batch256 配置已有独立协议名，未启动正式性能实验。
没有把梯度累积、混合精度或裁剪模型冒充为直接 batch256。

## 完整版本

| 路径 | 默认主模型 | 重计算 |
| --- | --- | --- |
| `versions/v33` | A2G v33 | 关闭 |
| `versions/v43` | A2GModal v43 | 关闭 |
| `versions/v43_lowmem` | A2GModal v43 | 概念图每 16 步 |
| `versions/v43_lowmem_r2` | A2GModal v43 | 同上，独立消融与审计工具 |
| `versions/v43_lowmem_r3` | A2GModal v43 | 同上，修复测试报告序列化 |
| `versions/v44_aligned_history_local` | A2GAlignedHistory 本地候选 | 同上，基础统计包含最近已观察事件 |

每个目录都是实际文件副本，不是链接或只保留差异补丁。
包含 `src/a2g` 全部代码、所有回归配置、测试、独立训练入口、审计工具、
中文模型阅读文档、服务器来源依据和依赖版本记录。代码通过相对包导入闭合，
不依赖相邻版本或上层 `src`。外部依赖仅为 Python 环境和显式传入的冻结数据。

为保留回归测试，包内含全部候选模块和对照配置；不代表每版启用了全部分支。
`VERSION.json` 指定本版默认配置、架构、重计算方式，并逐文件记录 SHA256。
`run.py` 会拒绝把 v43 配置交给 v33，或把 lowmem 配置当普通 v43 启动。
因此现有 batch256 候选应从 `versions/v43_lowmem/run.py` 启动。
v44 是另一架构，不能使用该 v43 配置；未定义 v44 的正式 batch256 协议。

本机示例，在任何目录执行：

```powershell
& 'E:\StudyWork\sota_kt_lab\a2g\.venv\Scripts\python.exe' `
  'E:\StudyWork\sota_kt_lab\a2g\versions\v43_lowmem\run.py' verify

& 'E:\StudyWork\sota_kt_lab\a2g\.venv\Scripts\python.exe' `
  'E:\StudyWork\sota_kt_lab\a2g\versions\v43_lowmem\run.py' smoke `
  --data-dir 'E:\StudyWork\sota_kt_lab\a2g\data\assist2017' `
  --output runs\new_smoke --device cpu
```

相对参数路径以版本目录为基准。运行目录必须不存在。
不要依靠 Windows 的 `.py` 文件关联启动；使用明确的 Python 解释器。

后续迭代保留旧目录，在新目录修改和验证，重新冻结为新版本。
根目录工具 `tools/freeze_versions.py` 拒绝覆盖已有目标，不能用于掩盖旧版修改。
`v43_lowmem_r2` 和 `v43_lowmem_r3` 已另存为完整物理副本，包含新增控制、诊断、
终审和复评工具；`r3` 还修复了 CPU 测试报告序列化。它们的 Full 模型、默认配置
和重计算方式与 `v43_lowmem` 相同，不重复启动 Full 性能实验。
v44 是另存的 216 文件完整候选，不是覆盖旧文件；其新统计行为与 v43 不等价。
2026-09-18 复核六个版本的全部文件哈希通过，根目录 CPU 测试 75 项中
73 项通过、2 项 CUDA 测试跳过。报告：
`runs/tests_v44_and_dataset_boundaries_20260918.json`。

## 完整训练结果

原 v43_lowmem 正式训练于 2026-09-18 05:22:07 +0800 正常退出，
第 32 轮最佳 AUC 为 0.8011643751561428，第 52 轮按 patience20 早停。
全部 153067 个验证交互通过独立 CSV 审计和检查点重载复评。
该版本未超过严格 0.8174 门槛；不是基础设施失败，也不作自动重试。
完整结果另行保存于 `runs/v43_full_outcome_20260918_0524.json`，
不回写冻结代码或原始训练结果。v44 的执行状态以独立登记和运行记录为准。
