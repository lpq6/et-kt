# A2G 本地完整模型

本项目位于 `E:\StudyWork\sota_kt_lab\a2g`，与原有 EduStateKT、KeenKT 等代码隔离。
源码来自 `lpq@172.25.114.0:/home/lpq/a2g_mambakt`，不是其他模型改名。

## V45 开发候选

开发版已增加三个默认关闭的统一记忆核开关，完整公式、兼容性边界与命令见
[V45 UMK](docs/V45_UMK.md)。配置为 `configs/assist2017_v45_umk_{off,ssm,attn,rwce,all}.json`，
使用根目录 `run.py smoke`。这是 v43 上的独立候选，不叠加 v44 历史窗口修改；
`versions/` 下所有冻结副本不变。短测试不代表 AUC 改善。

## 先看结论

- `src/a2g/model.py`：整理后的完整 v33，重复的继承式前向合并为一处。
- `src/a2g/candidate.py`：服务器 v43 的有限输入记忆与多时间尺度残差候选。
- `versions/v33`、`versions/v43`、`versions/v43_lowmem`、`versions/v43_lowmem_r2`、
  `versions/v43_lowmem_r3`：
  各自完整、独立保存的版本，不引用根目录的可变源码。
- `provenance/server_172/`：服务器原文件，不参与正常训练导入。
- 已验证 v33 的初始化、激活模块后的预测、消融输出和梯度与原版一致。
- v43 是待验证候选，不能因为版本更新、单元测试通过就声称 AUC 提升。
- `versions/v44_aligned_history_local`：本地历史窗口对齐候选，保持相同参数与
  初始化，补入最近已观察事件；已通过因果性、旧开关恢复和真实数据短测试，
  尚无完整重训 AUC。其完整代码独立保存，旧版本未回写。
- **尚不能宣称 Assist2017 AUC > 0.8174、五数据集达标或正向消融成立。**

阅读路线见 [完整模型阅读指南](docs/MODEL_GUIDE.md)，实验边界见
[实验协议](docs/EXPERIMENT_PROTOCOL.md)，证据与限制见
[当前核查记录](docs/AUDIT_STATUS.md)。

## 运行

**学习和复现固定版本请优先使用 `versions/`，根目录 `src/` 保留作后续开发。**
每个版本含模型、全部模块、数据读取、训练/评估代码、测试和来源材料。
`v43_lowmem_r2` 和 `v43_lowmem_r3` 是后续独立副本，增加了输入记忆与多时间
尺度残差的分开消融入口、验证诊断和终审绑定工具；`r3` 还修复了 CPU 测试
报告的跳过项 JSON 序列化。它们不改写旧版本，也不宣称性能已提升。
详见 [版本索引](versions/README.md)。

在本机从任何目录使用明确的解释器启动：

```powershell
& 'E:\StudyWork\sota_kt_lab\a2g\.venv\Scripts\python.exe' `
  'E:\StudyWork\sota_kt_lab\a2g\versions\v43_lowmem\run.py' verify
```

`run.py` 强制选择本版本源码并校验 `VERSION.json`。相对路径以版本目录为基准；
数据用 `--data-dir` 显式传入，不重复复制。已有版本不覆盖，后续迭代另存新目录。

在本目录打开 PowerShell：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m a2g.experiment audit-data --output runs\data_audit.json
.\.venv\Scripts\python.exe -m a2g.experiment smoke --config configs\assist2017_v43.json --output runs\v43_smoke --device cuda
.\.venv\Scripts\python.exe -m a2g.experiment train --config configs\assist2017_v43.json --output runs\v43_full_fold0_seed42 --device cuda
```

本机 6GB GPU 的正式运行建议使用已验证的激活重计算配置及文件日志启动器：

```powershell
.\.venv\Scripts\python.exe tools\run_logged.py train --config configs\assist2017_v43_lowmem.json --output runs\v43_lowmem_full --control runs\v43_lowmem_process
```

该配置不改参数量、batch64、序列长度或优化器；概念图每 16 步重新计算激活。
生产 batch64 的峰值 allocated 显存由 4827106304 降至 2670045184 字节。
CPU/CUDA 梯度对照和真实数据短训练模型、预测哈希一致；不是 AUC 改进声明。
`--control` 目录记录标准输出、错误和进程退出码，不覆盖已有目录或自动重试。

直接 batch256 已单独保存为
`configs/assist2017_v43_lowmem_batch256.json`，不冒充 batch64 参考协议。
本机 v33、v43、v43_lowmem 均在 width256 / length200 / FP32 的前向阶段 OOM。
低显存版 batch128 完成一个合成训练步，峰值 allocated 约 4.90 GiB，
但余量小，未验证长训稳定性；默认仍为 batch64。详见
[batch 与版本记录](docs/BATCH_AND_VERSIONS.md)。

运行目录必须不存在，防止覆盖实验。`smoke` 只验证流水线，不能报告为正式 AUC。
`train` 从随机参数和训练折先验初始化，不使用服务器训练检查点。

本地 `.venv` 通过 `--system-site-packages` 只读复用
`D:\Anaconda\envs\machine_learning` 的 PyTorch 2.2.0 / CUDA 11.8、
NumPy 1.26.4；并非完全独立复制的依赖环境。没有修改父环境。
在另一台机器上可创建独立环境，再安装 `pyproject.toml` 中的依赖。

## 目录

```text
src/a2g/
  model.py             完整 v33 主前向
  candidate.py         v43 候选包装
  _initialization.py   旧初始化顺序与基础序列/统计
  modules/             SSM、证据、图、Newton、尝试次数、时间、FFN 等
  data.py              严格训练/验证数据集
  uid.py               学生隔离与切片元数据
  randomness.py        对齐 dropout 随机子流
  experiment.py        训练、早停、重载和结果导出
  metrics.py           AUC/NLL/Brier/ECE 与严格阈值
configs/               冻结数据绑定与模型配置
tests/                 等价性、因果性、数据、重载测试
tools/                 来源迁移、传输、验证、显存测量
provenance/            原始代码和来源映射
data/assist2017/       原协议 train_valid CSV 与训练折词表
runs/                  不覆盖的实验结果
versions/              v33、v43、v43_lowmem、v43_lowmem_r2、v43_lowmem_r3 的独立代码、
                        配置和入口
```

正常使用不需要运行 `tools/consolidate_sources.py`。
该工具仅记录此次机械迁移过程，默认生成到新的 `work/extracted_reference`，
拒绝覆盖维护中的 `src/a2g`。
