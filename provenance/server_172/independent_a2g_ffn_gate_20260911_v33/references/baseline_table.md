---
type: result-summary
created: 2026-07-10
updated: 2026-08-31
tags: [KT, EduKTM, baseline, 5-fold, std, result-table, reproducibility]
source: baseline_run_20260716_final + baseline_expansion_8datasets_20260718_v3
---

## 复现工作台导航（2026-08-05）

这份主笔记保留结果与审计的总入口；真正可复用、可核验的细节分散在下面五个模块里。

| 模块 | 入口文件 | 当前定位 |
|---|---|---|
| pyKT 原生 38 模型 / YAML | [PYKT_38_MODEL_YAML_INDEX_20260731.md](./PYKT_38_MODEL_YAML_INDEX_20260731.md) | 只列已固定 registry、default YAML、公开 tuning 记录；不声称 paper/global optimum |
| 2025–2026 新开源模型 / adapter | [RECENT_2025_2026_MODEL_ADAPTER_MATRIX_20260731.md](./RECENT_2025_2026_MODEL_ADAPTER_MATRIX_20260731.md) | 只保留已核实源码、适配状态、参数边界与可运行边界 |
| 数据集清单 | [DATASETS_20260727.md](./DATASETS_20260727.md) | 只保留准入数据集；显式排除 KeenKT、EdNet、裸 ASSIST2009 与去重/未折叠版本 |
| 预处理命令 | [PREPROCESSING_20260727.md](./PREPROCESSING_20260727.md) | 负责下载、清洗、时序防泄露、转换为 pyKT 输入格式 |
| 训练命令 | [TRAINING_COMMANDS_228_20260729.md](./TRAINING_COMMANDS_228_20260729.md) | 仅保留 228-only 可直接复现的命令与前置检查 |

当前可靠结论只按证据写法保留为：pyKT 固定 registry `38/38`、default YAML `38/38`；公开 tuning 记录存在，但独立核实的最优 YAML 仍为 `0`；8 个准入数据集均保留原始官方下载/权威页与原始描述；严格预处理与 228 端静态可构造性已经审计，但这不自动等价于全部模型都可跑通。

### 目标覆盖快照（不是完成声明）

| 目标项 | 证据状态 | 主要证据 |
|---|---|---|
| pyKT 自带全部基线模型与 default YAML | 已覆盖 | `PYKT_38_MODEL_YAML_INDEX_20260731.md` |
| pyKT 公布的 tuning 记录边界 | 已覆盖但不称最优 | `PYKT_38_MODEL_YAML_INDEX_20260731.md` + 相关审计 |
| 近两年新开源 KT 模型 / 适配状态 / 开源地址 / 参数边界 | 已覆盖 | `RECENT_2025_2026_MODEL_ADAPTER_MATRIX_20260731.md` |
| 数据集官方下载链接与原始描述 | 已覆盖 | `DATASETS_20260727.md` |
| 可运行预处理：下载、清洗、时序防泄露、pyKT 输入格式 | 已覆盖为命令与审计，不等于全部运行完成 | `PREPROCESSING_20260727.md` + `STRICT_PREPROCESSING_EVIDENCE_RECONCILIATION_20260805.json` |
| 228-only 训练/复现命令 | 已覆盖为命令入口 | `TRAINING_COMMANDS_228_20260729.md` |
| KeenKT / EdNet 排除 | 已覆盖并显式写入 | `DATASETS_20260727.md` + delivery audit |
| 全部代码无需修改即可直接复现实验 | 未完成，仍需以逐项运行结果证明 | `VALIDATION_TUNING_GATE_SUMMARY_20260805.json` + `COMPLETE_BASELINE_DELIVERY_AUDIT_20260805_LOCAL.json` |

### 关键脚本层（直接入口，不是结果声明）

| 任务 | 主要脚本 |
|---|---|
| 数据下载 / 原始校验 | `scripts/download_datasets.py`；`scripts/audit_raw_source_descriptions.py` |
| 单数据集预处理 / 注册 | `scripts/reproduce_dataset.py`；`scripts/register_processed_dataset.py` |
| 转移后复核 | `scripts/reverify_transferred_dataset.py` |
| DenoiseKT 辅助矩阵 | `scripts/build_denoisekt_matrix.py` |
| 228 复现封装 | `scripts/run_kt_repro_228.sh`；`scripts/reproduce_all_approved_datasets_20260729.py` |
| profile / 五折执行 | `scripts/run_profile.py`；`scripts/run_five_folds.py` |

### 入口资产位置复核（2026-08-05）

本地 PaperGraph 镜像中可直接找到 5 个模块笔记与 4 个审计 JSON；`run_kt_repro_228.sh` 和 `reproduce_all_approved_datasets_20260729.py` 也在本目录。其余 public package 脚本不是散落在 PaperGraph 根目录，而是在 delivery root 的 `outputs/pykt_reproducible_20260727_public_v3/scripts/` 下存在。这个差异只说明“镜像布局不同”，不等于代码缺失。

| 脚本 | PaperGraph 根目录 | delivery public v3 |
|---|---|---|
| `download_datasets.py` | 根目录未镜像 | 存在 |
| `audit_raw_source_descriptions.py` | 根目录未镜像 | 存在 |
| `reproduce_dataset.py` | 根目录未镜像 | 存在 |
| `register_processed_dataset.py` | 根目录未镜像 | 存在 |
| `reverify_transferred_dataset.py` | 根目录未镜像 | 存在 |
| `build_denoisekt_matrix.py` | 根目录未镜像 | 存在 |
| `run_profile.py` | 根目录未镜像 | 存在 |
| `run_five_folds.py` | 根目录未镜像 | 存在 |
| `run_kt_repro_228.sh` | 存在 | wrapper/overlay 路径另见训练命令 |
| `reproduce_all_approved_datasets_20260729.py` | 存在 | overlay 路径另见训练命令 |

### public v3 脚本静态编译复核（2026-08-05）

对 delivery root 的 `outputs/pykt_reproducible_20260727_public_v3/scripts/` 做了只读 `python -m py_compile`，未下载数据、未预处理、未训练、未评估。结果：8/8 通过。

| 脚本 | py_compile |
|---|---|
| `download_datasets.py` | pass |
| `audit_raw_source_descriptions.py` | pass |
| `reproduce_dataset.py` | pass |
| `register_processed_dataset.py` | pass |
| `reverify_transferred_dataset.py` | pass |
| `build_denoisekt_matrix.py` | pass |
| `run_profile.py` | pass |
| `run_five_folds.py` | pass |

该复核只证明 public v3 入口脚本语法可编译；“无需修改可直接复现实验”仍必须由 228 环境、真实原始数据、严格 split、命令运行和结果 artifact 逐项证明。

### 228 wrapper / helper 静态复核（2026-08-05）

本次只做本地静态检查。第一次直接 `bash -n <Windows path>` 因 WSL 路径翻译失败而不可判定；随后改用 `Get-Content -Raw | bash -n -` 对脚本文本检查，4/4 通过。另对 6 个 228 Python helper 做 `python -m py_compile`，6/6 通过。未连接 GPU、未启动远程队列、未下载、未预处理、未训练、未评估。

| 入口 | 静态检查 | 结果 |
|---|---|---|
| `run_kt_repro_228.sh` | `bash -n -` | pass |
| `run_continuous_baseline_gpu_queue_228_20260803.sh` | `bash -n -` | pass |
| `run_complete_strict_preprocessing_228_20260803.sh` | `bash -n -` | pass |
| `run_strict_preprocessing_backlog_guarded_228.sh` | `bash -n -` | pass |
| `check_training_environment_228.py` | `py_compile` | pass |
| `reproduce_all_approved_datasets_20260729.py` | `py_compile` | pass |
| `run_validation_candidate_228.py` | `py_compile` | pass |
| `run_profile_fold_train_vocab_228.py` | `py_compile` | pass |
| `run_pykt_38x8_strict_entry_dryrun_228_20260803.py` | `py_compile` | pass |
| `run_strict_pykt_dkt_dryrun_matrix_228_20260803.py` | `py_compile` | pass |

这个静态门禁比“文件存在”更强，但仍不能替代 228 实机依赖、原始数据 hash、严格 fold、训练命令和结果 artifact 的端到端证明。

### 关键 JSON 审计 artifact 解析复核（2026-08-05）

对主笔记当前引用的 8 个关键 JSON 做本地只读 `ConvertFrom-Json` 解析和 `status/failures` 摘要检查，8/8 可解析。该检查用于防止“笔记引用了损坏 JSON”或“状态口径写错”，不启动任何实验。

| JSON artifact | parse | status | failures |
|---|---|---|---:|
| `COMPLETE_BASELINE_DELIVERY_AUDIT_20260805_LOCAL.json` | pass | `pass` | 0 |
| `STRICT_PREPROCESSING_EVIDENCE_RECONCILIATION_20260805.json` | pass | `reconciled_terminal_pass` | 0 |
| `LOCAL_STATIC_EXECUTABILITY_AUDIT_20260805.json` | pass | `pass_with_layout_caveat` | 0 |
| `VALIDATION_TUNING_GATE_SUMMARY_20260805.json` | pass | `partial_gate_only` | 0 |
| `STRICT_PREPROCESSING_TERMINAL_VERIFY_20260803.json` | pass | `pass` | 0 |
| `STRICT_PREPROCESSING_TERMINAL_INVENTORY_20260803.json` | pass | `pass` | no explicit `failures` field |
| `STRICT_PYKT_DKT_8DATASET_40FOLD_DRYRUN_AUDIT_20260803_V2.json` | pass | `pass` | 0 |
| `PYKT_38MODEL_8DATASET_STRICT_ENTRY_MATRIX_20260803.json` | pass | `pass` | 0 |

注意：`VALIDATION_TUNING_GATE_SUMMARY_20260805.json` 的 `partial_gate_only` 是有意保守状态，说明 validation tuning 仍未真正执行；它不能作为“最优 YAML 已得到”的证据。

### 数据集 registry 本地结构复核（2026-08-05）

对 `DATASETS_20260727.md` 的 approved dataset table 做只读结构检查：8/8 数据集行存在，每一行的 authority / exact download 字段都含有 HTTP(S) 链接；`keenkt`、`KeenKT`、`ednet`、`EdNet` 在该 registry 文件中均为 0 次匹配。该复核只证明本地 registry 结构与排除项正确，不代表外部网页当前仍可访问。

| 检查项 | 结果 |
|---|---|
| approved dataset rows | 8 |
| rows with authority/download HTTP(S) link | 8/8 |
| KeenKT string matches in registry | 0 |
| EdNet string matches in registry | 0 |

准入数据集仍为：`assist2009_corrected_collapsed`、`assist2012`、`junyi2015`、`assist2015`、`nips_task34`、`assist2017`、`slepemapy`、`statics2011`。裸 `assist2009`、EdNet aliases 和 KeenKT 不进入复现实验包。

### pyKT 38-model YAML index 本地结构复核（2026-08-05）

对 `PYKT_38_MODEL_YAML_INDEX_20260731.md` 做只读结构检查：模型行 `38/38`，default YAML 字段 `38/38`，public tuning YAML 计数合计 `33`。文件中明确写有 independently verified optimal YAML=`0`；出现 `KeenKT` 和 `EdNet` 的位置是硬排除声明，不是模型/数据集准入行。

| 检查项 | 结果 |
|---|---|
| model rows | 38 |
| rows with `configs/models/defaults/*.yaml` | 38/38 |
| public tuning YAML count | 33 |
| independently verified optimal YAML | 0 |
| KeenKT / EdNet as admitted row | 0 |

参数口径继续保持：`public_tuning_available_no_independent_optimality_claim` 和 `upstream_default_only_no_optimality_claim` 都不能写成论文最优或全局最优；当前只能作为 default/released tuning 边界进入复现实验包。

### 2025–2026 model / adapter matrix 本地结构复核（2026-08-05）

对 `RECENT_2025_2026_MODEL_ADAPTER_MATRIX_20260731.md` 做只读结构检查：披露模型行 `16`，每行都有 source/adapter 字段；参数/profile 计数合计 `21`，`optimal=0`。文件中的覆盖摘要与矩阵行一致：verified 2025–2026 paper records=`14`，source-only/unresolved=`2`。

| class | rows |
|---|---:|
| `runnable_admitted` | 5 |
| `runnable_unadmitted` | 3 |
| `conditional` | 3 |
| `adapter_blocked` | 2 |
| `source_only` | 2 |
| `paper_only_protocol_rejected` | 1 |

继续保留 fail-closed 边界：MCSKT 是 paper-only / protocol-rejected；MCKT 是 source-verified 但 adapter-blocked；SAINT++ 与 PromptKT 仍为 specialized source-only，不能当作标准 runner 已接入。

### 预处理模块阶段覆盖复核（2026-08-05）

对 `PREPROCESSING_20260727.md` 做只读结构检查，确认它覆盖目标要求的预处理链路：下载、原始描述审计、单数据集预处理、注册、转移复核、DenoiseKT 辅助矩阵、泄露审计记录和 pyKT 配置安装。该检查只证明文档/命令入口覆盖，不启动下载或预处理。

| 目标阶段 / artifact | 本地匹配 |
|---|---:|
| `download_datasets.py` | 2 |
| `audit_raw_source_descriptions.py` | 1 |
| `reproduce_dataset.py` | 4 |
| `register_processed_dataset.py` | 3 |
| `reverify_transferred_dataset.py` | 1 |
| `build_denoisekt_matrix.py` | 1 |
| `leakage_audit` | 1 |
| `repro_run.json` | 7 |
| `train_valid_sequences.csv` | 1 |
| `test_sequences.csv` | 1 |
| `data_config.json` | 11 |

因此，预处理模块已具备“可按命令入口复核”的结构覆盖；是否在 228 上端到端重跑成功仍要看真实原始数据、hash、fold 输出和 terminal verification artifact。

### 228 训练命令模块阶段覆盖复核（2026-08-05）

对 `TRAINING_COMMANDS_228_20260729.md` 做只读结构检查，确认它覆盖 228-only 复现实验的命令层：环境 preflight、dependency audit、下载/预处理、strict vocab/fold dry-run、真实 strict fold run 门禁、validation-only tuning plan/collect/select、单 fold profile、38×8 matrix audit、five-fold execution 和 evidence contract。该检查只证明命令模块覆盖，不启动远程任务。

| 命令阶段 / marker | 本地匹配 |
|---|---:|
| `check_training_environment_228.py` | 2 |
| `dependency-audit` | 1 |
| `reproduce_all_approved_datasets_20260729.py` | 2 |
| `strict-vocab-dry-run` | 1 |
| `strict-fold-dry-run` | 1 |
| `strict-fold-run` | 1 |
| `validation-plan` | 1 |
| `validation-collect` | 1 |
| `validation-select` | 1 |
| `run_profile.py` | 1 |
| `model-matrix-audit` | 1 |
| `run_five_folds.py` | 1 |
| `Evidence contract` | 1 |
| `KT_ALLOW_REAL_RUN=1` explicit real-run gate | 4 |
| EdNet / KEENKT / bare ASSIST2009 rejection markers | present |

因此，训练命令模块已经具备“可按 228-only wrapper 追踪”的结构覆盖；但所有 `KT_ALLOW_REAL_RUN=1` 路径仍需实际授权窗口和结果 artifact 才能证明端到端可复现。

### 模块入口与关键审计 SHA256 快照（2026-08-05）

为防止后续笔记/审计文件漂移，对 5 个模块入口和 7 个关键审计 JSON 做本地只读 SHA256 快照。该快照只锁定当前文件内容，不代表外部 URL、远程 228 状态或 GPU 结果已经完成。

| 文件 | SHA256 |
|---|---|
| `PYKT_38_MODEL_YAML_INDEX_20260731.md` | `5ae7ce31ed7723c60c4f7b4f12db2e51c28778076d332ac6f0451d8e81e52bcd` |
| `RECENT_2025_2026_MODEL_ADAPTER_MATRIX_20260731.md` | `591311607e935b75a26aba427dd0f05761a1ff198d9edcda920d9f0e114114b1` |
| `DATASETS_20260727.md` | `34105bdde565034161d399f1b60b39d72024da20ed06308848eebfd73a6a5191` |
| `PREPROCESSING_20260727.md` | `0608b4ddf1a2b51b4cdb07304fd1f9f0a809ef474fe79b435b9760e8fe8be24c` |
| `TRAINING_COMMANDS_228_20260729.md` | `9ede970060f05896aae0e67f087cd9daa562292f7d8009321d6b7e3a04ad46a7` |
| `COMPLETE_BASELINE_DELIVERY_AUDIT_20260805_LOCAL.json` | `8e38d185340764d87711f1a4848ba4694f57428b82c45a58220e04b8b3eca18d` |
| `STRICT_PREPROCESSING_EVIDENCE_RECONCILIATION_20260805.json` | `c798b5476660b418f0996cc24b15b2170b9b2f61fd3a320d3027ff078af2aff5` |
| `LOCAL_STATIC_EXECUTABILITY_AUDIT_20260805.json` | `14416e1c654ca78ca77d72a96d7ea2d923477ddde7d9d8221d6cc7599247d390` |
| `VALIDATION_TUNING_GATE_SUMMARY_20260805.json` | `e11f8ac272ec084110b6a1673d340745a0b4e8f98f42a486dc98167ba34e44f3` |
| `STRICT_PREPROCESSING_TERMINAL_VERIFY_20260803.json` | `3cef156bf5351f920754e3c4561623bd28c81d0cb0ef4075735f8925d9a155db` |
| `STRICT_PYKT_DKT_8DATASET_40FOLD_DRYRUN_AUDIT_20260803_V2.json` | `728b921ae66dc1fc194f9a7e711d7216c228e9dca5a1ffdcbd50245e06e77c45` |
| `PYKT_38MODEL_8DATASET_STRICT_ENTRY_MATRIX_20260803.json` | `bd52bf78c7e09e17d266e1a2d0015c8c6bbc466f2567b7d2532d27ddc9609239` |

<!-- CODEX-KT-GOAL-CURRENT-20260803:START -->
## KT 完整复现目标当前证据（2026-08-03）

- pyKT 固定 registry：`38/38` 模型、`38/38` default YAML；`32` 个模型有 `33` 份 pyKT released tuning YAML。
- 参数语义必须保持准确：released tuning row、作者默认值、论文搜索空间都不是独立核实的 paper/global optimum；当前 independently verified optimal YAML=`0`。无法从论文/源码证明的“最优值”不得杜撰，改用 validation-only equal-budget 选择后只能称 declared-pool validation-best。
- 2025–2026/专用源码矩阵：`16` 条披露、`14` 篇已核实近年论文、`21` 份参数/搜索空间 artifact；MCSKT 仍 paper-only 且动态 k 的 test-selected 口径被严格拒绝，MCKT/HCGKT 等保持 fail-closed。
- 数据：8 个准入库均保留官方下载/权威页、原始 artifact 字节数、SHA256、原始成员/列描述与使用边界；EdNet 全部 alias、KeenKT、裸 ASSIST2009 与 dedup/uncollapsed ASSIST2009 硬排除。
- 当前严格预处理：fresh source manifest `8/8`；compact fold-train-only learner-disjoint 视图 `40/40`，共复核 `160` 个标准 pyKT train/validation/one-step 文件。question/concept vocab 只用其他四个训练 folds 拟合，validation/test 未见 ID 统一映射到 OOV；training_started=false、evaluation_started=false、test_feedback_used=false。
- 228 终态归档已在 `complete.marker` 后重新盘点：payload `318` 个文件、`27390792006` bytes；非自引用 checksum 共 `319` 条，独立逐项重算全部匹配，并抽样解析 `16` 份 JSON 与 `16` 份 CSV，失败 `0`。
- 固定 pyKT DKT 入口已对真实 strict config 做 `8` 数据集 × `5` folds dry-run：`40/40` pass、`0` fail；每 fold 绑定独立 runtime/data_config，未打开 test CSV、未训练、未评估。这只证明严格数据入口可消费，不代表 38 个模型均已通过模型级门禁。
- 全 pyKT strict 入口矩阵已进一步验证：`236/304` model-dataset cells 的 folds0–4 命令通过，合计 `1180/1180`；`68` cells 保持 HR/DB/SB/RB 阻断。该状态只叫 strict command dry-run，不叫 GPU smoke、训练完成或 effectiveness 结果。
- 近年模型 DenoiseKT 的 fold-aware auxiliary 已补齐：6 个具备 question IDs 的数据集 × 5 folds=`30/30` pass；active-validation 与 final-test metadata 均排除、responses 不读取且合成 response-invariance 测试通过。Assist2015/Statics2011 因 `num_q=0` 保持不适用。本条是 2026-08-03 的阶段快照；训练及冻结 final-test 后续已完成，并由文末 2026-08-29 独立复算收据取代“仍未启动”的旧状态。
- 空间安全边界：40 个视图只物化训练/验证与 one-step final-test 必需文件；Window 文件未冒充完成，必须在正式 Window evaluation 授权后用对应 fold 的 `train_only_vocab.json` 流式 remap。当前 Window-ready=`0/40`。
- 可运行边界：严格数据已不再是 native 模型的统一前置缺口，但模型级源码/依赖/schema 门禁仍有效；GKT/LPKT/DIMKT/RKT/HCGKT、DAT-AKT、FA-KT 等不得因预处理完成而被误写成 runnable。

当前证据：`PYKT_38_MODEL_YAML_INDEX_20260731.*`、`RECENT_2025_2026_MODEL_ADAPTER_MATRIX_20260731.*`、`DATASETS_20260727.md`、`PREPROCESSING_20260727.md`、`STRICT_PREPROCESSING_COMPLETE_8DATASET_40FOLD_AUDIT_20260803.json`、`STRICT_PREPROCESSING_TERMINAL_*_20260803.*`、`STRICT_PYKT_DKT_8DATASET_40FOLD_DRYRUN_AUDIT_20260803_V2.json`、`PYKT_38MODEL_8DATASET_STRICT_ENTRY_MATRIX_20260803.*`、`DENOISEKT_6DATASET_30FOLD_TRAIN_ONLY_AUXILIARY_*_20260803.json`。本区块不声称所有基线已训练或所有缺失主表数值已补齐。
<!-- CODEX-KT-GOAL-CURRENT-20260803:END -->
<!-- CODEX-KT-CATALOG-20260729:START -->
## 2026-07-29 复现资产总账校正（只读审计）

- 权威机器总账：KT完整可复现实验资产总账_20260729.json；可读索引：KT完整可复现实验资产总账_20260729.md。
- 固定 pyKT：38 个模型、38 份 default YAML；另有 post-pin FlucKT、ACE-KT、ASIKT 两个外部 overlay。
- 2025–2026 注册口径：12 条；SAINT++/PromptKT 仅 source-only。MCSKT 论文/DOI 已核实，但作者仓未核实且动态 `k` 使用测试集选择；其 `†` 点估计仅作 paper-only 参考，不进入 60 组严格复现矩阵。MCKT 仍为 source-verified、adapter-blocked。
- 该 2026-07-29 盘点中的 `2/8` 已由 2026-08-03 终态证据取代：当前 fresh source manifests 为 `8/8`；详见 `CODEX-KT-GOAL-CURRENT-20260803`。
- 结果矩阵：60 个完整 5/5 组合；common-five 40/40；校准已物化 13 组，不能把缺失 ECE 当作已测。
- 参数边界：公开 tuning row 没有指标或选择规则；独立核实的 global/paper-optimal YAML 为 0，不得写成“所有模型最优参数”。
- 新增复现 overlay：228 observed package lock、hash-locked simple-extra dry-run、fail-closed Mamba/FA-KT 门禁、38 模型精确 profile index 和通用 228-only 命令；它不修改冻结 public v3 包。
- 38×8 直接运行矩阵：304 格；静态可构造 236，fail-closed 68；当前严格真实可启动 0。DAT-AKT default 前向缺陷、FA-KT 依赖和五个协议硬拒绝均已单独标记。
- 预处理边界：用户级 train/valid/test 无重叠且时序字段受审计；stock pyKT 的 question/concept ID 映射在用户拆分前建立，因此当前是 new-learner、transductive-item 口径，不是 strict train-only vocabulary。
- 该 2026-07-29 的“真实 8 数据集尚未物化”已过期：2026-08-03 已完成 `40/40` fold-train-only compact 视图；Window 仍按 fold 延后流式 remap。
- 新增等预算 validation-only tuning 链：候选计划、15-trial 合成收集、test-feedback 拒绝、`validation_tuned` 选择器和 228 candidate dry-run 均通过；真实 tuning 为 0 次。
- ASIKT 旧 pending/readiness 文件已由 overlay supersession 证据覆盖；当前只承认 NIPS Task 3&4 的严格 5/5 author-source causal-mask adapter 结果。
- MCSKT canonical PDF/DOI、论文参数和 paper-only 指标已核实；作者仓未核实，动态 k 使用测试集反馈，因此 adapter、严格结果和 paper-optimal profile 均未准入。
- 执行门禁：KT 只允许 228；127/172 禁止；228 无时间截止。本次同步只读，未启动下载、预处理、训练或评估。
- 消融仍为中间证据：no-item-dropout 已物化 58/75，效应方向最后严格对齐为 50/75；其他核心模块尚未运行。
<!-- CODEX-KT-CATALOG-20260729:END -->































# 5折结果均值标准差汇总 20260710

<!-- CODEX-A2G-PUBLICATION-BASELINE-TARGET-20260803:START -->
## A2G-MambaKT 投稿级完整基线待补总表（2026-08-03）

本节先冻结投稿所需的完整对比范围，再补实验。它综合 MCSKT 的 Table 3/4、Fig. 3 与 ASIKT 的 Table 2、Fig. 7，但不直接混排两篇论文在不同硬件、batch、数据处理和实现下的数值。

状态定义：`5/5`=已有可准入五折 final-test 结果；`V-FROZEN`=validation-only 五折已冻结但 final test 未启动；`FAIL`=已实际执行但无可用 checkpoint/result；`P0`=仅 fold0 pilot，不入五折主表；`TODO`=代码/协议可继续补跑；`SRC`=只有源码路径，尚无标准 runner/profile；`ADAPT`=需先完成适配和门禁；`PAPER†`=仅论文报告值；`BLOCK`=当前源码、依赖或泄露协议阻断。

### 投稿核心与强基线：八数据集覆盖矩阵

| Model | Family / role | Assist09-c | Assist12 | Assist15 | Assist17 | NIPS34 | Junyi15 | Slepemapy | Statics11 | 当前动作 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BKT | classical | TODO | TODO | TODO | TODO | 5/5 | TODO | TODO | TODO | 至少保留一个 classical 行；其余数据集按需扩展 |
| DKT | recurrent | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | effectiveness 已覆盖；效率已有四字段实测 |
| DKT+ | recurrent regularized | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 作为补充基线，不必重复训练 |
| DKT-Forget | recurrent + forgetting | BLOCK | TODO | BLOCK | TODO | TODO | TODO | TODO | TODO | 仅时间戳数据可用；必须使用 train-only 固定 gap-domain adapter |
| DKVMN | memory network | TODO | V-FROZEN | TODO | TODO | 5/5 | TODO | TODO | TODO | Assist12 validation 5/5 已冻结；final test 未启动 |
| DeepIRT | memory / IRT | TODO | TODO | TODO | TODO | 5/5 | TODO | TODO | TODO | NIPS34 已有；其他库属于补充优先级 |
| SAKT | attention | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | effectiveness 已覆盖；效率已有四字段实测 |
| SAINT | Transformer encoder-decoder | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | DTransformer fail-fast 后未启动；不得标成 SAINT++ |
| SAINT++ | response-aware Transformer | SRC | SRC | SRC | SRC | SRC | SRC | SRC | SRC | pyKT 专用源码存在，但未进入标准 registry/profile；先适配再跑 |
| AKT | monotonic attention / Rasch | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | effectiveness 已覆盖；效率已有四字段实测 |
| SimpleKT | strong simple attention | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | effectiveness 已覆盖；效率已有四字段实测 |
| SparseKT | sparse attention | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | MCSKT 直接效率基线；需冻结 soft/top-k 口径 |
| DTransformer | contrastive Transformer | TODO | FAIL | TODO | TODO | 5/5 | TODO | TODO | TODO | Assist12 fold0 Triton/gcc 失败，无 checkpoint |
| UKT | 2025 uncertainty-aware | 5/5 | TODO | 5/5 | 5/5 | 5/5 | 5/5 | TODO | 5/5 | DTransformer fail-fast 后未启动；Assist12/Slepemapy 仍缺 |
| ACE-KT | 2026 stage-wise | 5/5 | TODO | 5/5 | 5/5 | 5/5 | TODO | TODO | 5/5 | 优先补 Assist12、Junyi15、Slepemapy |
| ASIKT | 2025 mechanism-matched SSM | TODO | TODO | TODO | P0 | 5/5 | P0 | P0 | TODO | NIPS34 已准入；三库 pilot 不得冒充五折 |
| CSKT | recent strong attention | TODO | TODO | TODO | TODO | 5/5 | TODO | TODO | TODO | 作为 NIPS34 强基线保留，扩展为次优先级 |
| FlucKT | recent strong attention | TODO | TODO | TODO | TODO | 5/5 | TODO | TODO | TODO | 作为 NIPS34 强基线保留，扩展为次优先级 |
| Mamba4KT | canonical Mamba comparator | ADAPT | ADAPT | ADAPT | ADAPT | ADAPT | ADAPT | ADAPT | ADAPT | 必须补成 Mamba/SSM 效率对照，先完成 hash-bound adapter |
| MCSKT | 2026 paper comparator | BLOCK | PAPER†/BLOCK | BLOCK | PAPER†/BLOCK | PAPER†/BLOCK | BLOCK | BLOCK | BLOCK | 无已核实作者仓；动态 k 使用 test accuracy，严格结果禁止准入 |

NIPS34 已有的 QIKT、FolibiKT、DeepIRT 等 5/5 行继续保留在下方 60 组审计明细中。它们是有价值的补充强基线，但不替代上表中 DKVMN、SAINT/SAINT++、DTransformer、UKT、ACE-KT、ASIKT 和 Mamba4KT 的缺口。

<!-- CODEX-PUBLICATION-MAIN-TABLES-20260803:START -->
### 投稿 effectiveness 主表（核心模型；五数据集 one-step test）

这是正文候选主表。已跑出的同协议五折 final-test 数值从不可变 artifact 生成；尚未跑出或尚未准入的模型明确保留 `—`。该表不会用 validation、fold0 pilot、smoke 或 paper-only 数值补格。

| Dataset | Metric | DKT | DKT+ | DKVMN | SAKT | SAINT++ | AKT | SimpleKT | DTransformer | UKT | ACE-KT | ASIKT | Mamba4KT | MCSKT† | A2G-MambaKT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Junyi2015 | AUC | 0.754553 +/- 0.000097 | 0.753959 +/- 0.000110 | 0.751666 +/- 0.000426 | 0.749412 +/- 0.000755 | — | 0.803541 +/- 0.000284 | 0.803113 +/- 0.000211 | — | **0.804888 +/- 0.000119** | — | — | — | — | <u>0.804826 +/- 0.000111</u> |
| Junyi2015 | ACC | 0.846524 +/- 0.000034 | 0.846219 +/- 0.000023 | 0.846014 +/- 0.000052 | 0.845151 +/- 0.000173 | — | 0.854726 +/- 0.000083 | 0.854816 +/- 0.000048 | — | **0.855215 +/- 0.000073** | — | — | — | — | <u>0.854974 +/- 0.000070</u> |
| ASSIST2015 | AUC | 0.726861 +/- 0.000479 | <u>0.728353 +/- 0.000729</u> | 0.721298 +/- 0.000465 | 0.703955 +/- 0.000590 | — | 0.725735 +/- 0.000952 | 0.723972 +/- 0.000425 | — | 0.726737 +/- 0.000882 | 0.725765 +/- 0.000276 | — | — | — | **0.729012 +/- 0.000420** |
| ASSIST2015 | ACC | 0.750645 +/- 0.000322 | 0.751046 +/- 0.000339 | 0.750609 +/- 0.000412 | 0.746135 +/- 0.000316 | — | <u>0.751355 +/- 0.000508</u> | 0.750584 +/- 0.000280 | — | 0.750840 +/- 0.000663 | 0.750363 +/- 0.000186 | — | — | — | **0.751662 +/- 0.000156** |
| NIPS Task 3&4 | AUC | 0.770967 +/- 0.000394 | 0.771500 +/- 0.000211 | 0.768590 +/- 0.000869 | 0.747161 +/- 0.001053 | — | 0.804626 +/- 0.001219 | 0.802279 +/- 0.000347 | 0.801547 +/- 0.000604 | 0.805728 +/- 0.000510 | 0.803960 +/- 0.000329 | 0.798763 +/- 0.001047 | — | 0.827300† | 0.805565 +/- 0.000243 |
| NIPS Task 3&4 | ACC | 0.704490 +/- 0.000381 | 0.704851 +/- 0.000455 | 0.702165 +/- 0.000767 | 0.682991 +/- 0.000938 | — | 0.732949 +/- 0.000612 | 0.731513 +/- 0.000412 | 0.730464 +/- 0.000663 | 0.733253 +/- 0.000697 | 0.732686 +/- 0.000242 | 0.728213 +/- 0.001147 | — | 0.763500† | 0.731878 +/- 0.000644 |
| ASSIST2017 | AUC | 0.716026 +/- 0.000940 | 0.717607 +/- 0.000674 | 0.710194 +/- 0.000595 | 0.653189 +/- 0.000930 | — | 0.766839 +/- 0.000793 | 0.753958 +/- 0.001005 | — | 0.762522 +/- 0.004354 | <u>0.782030 +/- 0.000544</u> | — | — | 0.817400† | **0.784384 +/- 0.000836** |
| ASSIST2017 | ACC | 0.685402 +/- 0.001047 | 0.687453 +/- 0.000458 | 0.683436 +/- 0.000245 | 0.664698 +/- 0.000799 | — | 0.714914 +/- 0.000469 | 0.707607 +/- 0.000773 | — | 0.712861 +/- 0.003303 | <u>0.725020 +/- 0.001395</u> | — | — | 0.760400† | **0.727199 +/- 0.000465** |
| Slepemapy | AUC | 0.783850 +/- 0.000288 | 0.786570 +/- 0.000477 | 0.785218 +/- 0.000425 | 0.771241 +/- 0.000470 | — | **0.800163 +/- 0.000465** | 0.794192 +/- 0.000419 | — | — | — | — | — | — | <u>0.799000 +/- 0.000209</u> |
| Slepemapy | ACC | 0.797203 +/- 0.000081 | 0.798261 +/- 0.000238 | 0.797235 +/- 0.000204 | 0.791721 +/- 0.000312 | — | <u>0.803394 +/- 0.000844</u> | 0.801021 +/- 0.000172 | — | — | — | — | — | — | **0.803450 +/- 0.000260** |

<!-- CODEX-FIVE-PRIORITY-MAIN-TABLE-20260908:START -->
### 五库优先投稿主表（现有合法五折结果直用，2026-09-08）

本节把用户指定的五个优先数据集集中到一个可直接用于论文排版的视图：ASSIST2009 corrected/collapsed、ASSIST2015、ASSIST2017、NIPS Task 3&4、Slepemapy。表中数值均复用本文件已有的同口径五折 final-test artifact；没有该级别证据的单元明确保留 `—`。`†` 表示 MCSKT 论文值（paper-only），不参与本项目排名；validation、fold0 pilot、smoke 和未完成 tuning 均不回填。

| Dataset | Metric | BKT | DKT | DKT+ | DKVMN | DKT-Forget | DeepIRT | SAKT | SAINT | SAINT++ | AKT | SimpleKT | DTransformer | UKT | ACE-KT | ASIKT | CSKT | FlucKT | FoLiBiKT | QIKT | DenoiseKT | MCKT | Mamba4KT | MCSKT† | A2G-MambaKT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ASSIST2009 corrected/collapsed | AUC | — | 0.825756 +/- 0.001019 | 0.825614 +/- 0.000588 | 0.816374 +/- 0.001572 | — | — | 0.795565 +/- 0.001660 | — | — | 0.840313 +/- 0.001020 | 0.840088 +/- 0.001059 | — | **0.848004 +/- 0.001405** | <u>0.845685 +/- 0.000872</u> | — | — | — | — | — | 0.786573 +/- 0.001450 | — | — | — | — |
| ASSIST2009 corrected/collapsed | ACC | — | 0.768648 +/- 0.001041 | 0.769230 +/- 0.000781 | 0.761925 +/- 0.001480 | — | — | 0.748478 +/- 0.002482 | — | — | 0.777064 +/- 0.001267 | 0.776476 +/- 0.001133 | — | **0.781122 +/- 0.000806** | <u>0.779969 +/- 0.001030</u> | — | — | — | — | — | 0.741494 +/- 0.001019 | — | — | — | — |
| ASSIST2015 | AUC | — | 0.726861 +/- 0.000479 | <u>0.728353 +/- 0.000729</u> | 0.721298 +/- 0.000465 | — | — | 0.703955 +/- 0.000590 | — | — | 0.725735 +/- 0.000952 | 0.723972 +/- 0.000425 | — | 0.726737 +/- 0.000882 | 0.725765 +/- 0.000276 | — | — | — | — | — | NA (no qid) | — | — | — | **0.729012 +/- 0.000420** |
| ASSIST2015 | ACC | — | 0.750645 +/- 0.000322 | 0.751046 +/- 0.000339 | 0.750609 +/- 0.000412 | — | — | 0.746135 +/- 0.000316 | — | — | <u>0.751355 +/- 0.000508</u> | 0.750584 +/- 0.000280 | — | 0.750840 +/- 0.000663 | 0.750363 +/- 0.000186 | — | — | — | — | — | NA (no qid) | — | — | — | **0.751662 +/- 0.000156** |
| ASSIST2017 | AUC | — | 0.716026 +/- 0.000940 | 0.717607 +/- 0.000674 | 0.710194 +/- 0.000595 | — | — | 0.653189 +/- 0.000930 | — | — | 0.766839 +/- 0.000793 | 0.753958 +/- 0.001005 | — | 0.762522 +/- 0.004354 | <u>0.782030 +/- 0.000544</u> | — | — | — | — | — | 0.768272 +/- 0.000750 | — | 0.817400† | **0.784384 +/- 0.000836** |
| ASSIST2017 | ACC | — | 0.685402 +/- 0.001047 | 0.687453 +/- 0.000458 | 0.683436 +/- 0.000245 | — | — | 0.664698 +/- 0.000799 | — | — | 0.714914 +/- 0.000469 | 0.707607 +/- 0.000773 | — | 0.712861 +/- 0.003303 | <u>0.725020 +/- 0.001395</u> | — | — | — | — | — | 0.718766 +/- 0.000664 | — | 0.760400† | **0.727199 +/- 0.000465** |
| NIPS Task 3&4 | AUC | 0.687601 +/- 0.000174 | 0.770967 +/- 0.000394 | 0.771500 +/- 0.000211 | 0.769581 +/- 0.000318 | — | 0.768468 +/- 0.001061 | 0.747161 +/- 0.001053 | — | — | 0.804626 +/- 0.001219 | 0.802279 +/- 0.000347 | 0.801547 +/- 0.000604 | <u>0.805728 +/- 0.000510</u> | 0.803960 +/- 0.000329 | 0.798763 +/- 0.001047 | **0.807984 +/- 0.000624** | 0.806469 +/- 0.000713 | 0.805674 +/- 0.000305 | 0.802748 +/- 0.000442 | 0.801381 +/- 0.000507 | — | — | 0.827300† | 0.805565 +/- 0.000243 |
| NIPS Task 3&4 | ACC | 0.644598 +/- 0.000610 | 0.704490 +/- 0.000381 | 0.704851 +/- 0.000455 | 0.702736 +/- 0.000380 | — | 0.701929 +/- 0.000764 | 0.682991 +/- 0.000938 | — | — | <u>0.732949 +/- 0.000612</u> | 0.731513 +/- 0.000412 | 0.730464 +/- 0.000663 | **0.733253 +/- 0.000697** | 0.732686 +/- 0.000242 | 0.728213 +/- 0.001147 | 0.735711 +/- 0.000518 | 0.734940 +/- 0.000862 | 0.733578 +/- 0.000759 | 0.731415 +/- 0.000478 | 0.730151 +/- 0.000661 | — | — | 0.763500† | 0.731878 +/- 0.000644 |
| Slepemapy | AUC | — | 0.783850 +/- 0.000288 | 0.786570 +/- 0.000477 | 0.785218 +/- 0.000425 | — | — | 0.771241 +/- 0.000470 | — | — | **0.800163 +/- 0.000465** | 0.794192 +/- 0.000419 | — | — | — | — | — | — | — | — | 0.792618 +/- 0.000592 | — | — | — | <u>0.799000 +/- 0.000209</u> |
| Slepemapy | ACC | — | 0.797203 +/- 0.000081 | 0.798261 +/- 0.000238 | 0.797235 +/- 0.000204 | — | — | 0.791721 +/- 0.000312 | — | — | <u>0.803394 +/- 0.000844</u> | 0.801021 +/- 0.000172 | — | — | — | — | — | — | — | — | 0.801298 +/- 0.000418 | — | — | — | **0.803450 +/- 0.000260** |

说明：`—` 是缺少同口径五折 final-test artifact，不是零分；`NA (no qid)` 是 DenoiseKT 协议不适用。该集中表与前文历史/核心表的数值逐项一致，仅增加了 ASSIST2009 与扩展模型列，便于直接排版和后续 append-only 回填。

### 五库 model efficiency 字段总表（先冻结字段，缺值不伪造）

参考 MCSKT/ASIKT 的效率报告结构，统一保留 Model、Dataset、TT、IT、Throughput、Parameters、FLOPs、GPU usage/peak memory、硬件与证据状态。当前已有的效率实测只在 Assist2012 strict 根和 NIPS34 forward-only 补充表，不能移植到这五库；因此五库缺失值先留 `—`，GPU 空闲时由 228 的串行 watcher 按同一协议补测。

| Dataset | Model | TT (s/epoch) | IT (ms/batch) | Throughput (tokens/s) | Parameters (M) | FLOPs (G/step) | GPU usage (peak MiB / util %) | Evidence status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| ASSIST2009 corrected/collapsed | DKT / DKT+ / DKVMN / SAKT / SAINT++ / AKT / SimpleKT / DTransformer / UKT / ACE-KT / ASIKT / Mamba4KT / A2G-MambaKT | — | — | — | — | — | — | NOT_MEASURED_228 |
| ASSIST2015 | DKT / DKT+ / DKVMN / SAKT / SAINT++ / AKT / SimpleKT / DTransformer / UKT / ACE-KT / ASIKT / Mamba4KT / A2G-MambaKT | — | — | — | — | — | — | NOT_MEASURED_228 |
| ASSIST2017 | DKT / DKT+ / DKVMN / SAKT / SAINT++ / AKT / SimpleKT / DTransformer / UKT / ACE-KT / ASIKT / Mamba4KT / A2G-MambaKT | — | — | — | — | — | — | NOT_MEASURED_228 |
| NIPS Task 3&4 | DKT / DKT+ / DKVMN / SAKT / SAINT++ / AKT / SimpleKT / DTransformer / UKT / ACE-KT / ASIKT / Mamba4KT / A2G-MambaKT | — | — | — | — | — | — | NOT_MEASURED_228 (NIPS34 forward-only values remain in separate table) |
| Slepemapy | DKT / DKT+ / DKVMN / SAKT / SAINT++ / AKT / SimpleKT / DTransformer / UKT / ACE-KT / ASIKT / Mamba4KT / A2G-MambaKT | — | — | — | — | — | — | NOT_MEASURED_228 |

效率字段的空值是当前真实缺口，不构成投稿完成声明；不得从其它数据集、论文异硬件或 validation 运行时推导。完成同一硬件/精度/batch/序列长度/计时边界的独立测量后，才可逐单元替换 `—`。
<!-- CODEX-FIVE-PRIORITY-MAIN-TABLE-20260908:END -->
### 投稿 effectiveness 补充表（经典、扩展强基线与 paper-only 参照）

| Dataset | Metric | BKT | DKT-Forget | DeepIRT | SAINT | SparseKT | CSKT | FlucKT | FoLiBiKT | QIKT | DenoiseKT | MCKT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ASSIST2009 corrected/collapsed | AUC | — | — | — | — | — | — | — | — | — | 0.786573 +/- 0.001450 | — |
| ASSIST2009 corrected/collapsed | ACC | — | — | — | — | — | — | — | — | — | 0.741494 +/- 0.001019 | — |
| ASSIST2012 | AUC | — | — | — | — | — | — | — | — | — | 0.772206 +/- 0.000437 | — |
| ASSIST2012 | ACC | — | — | — | — | — | — | — | — | — | 0.752842 +/- 0.000311 | — |
| Junyi2015 | AUC | — | — | — | — | — | — | — | — | — | 0.800210 +/- 0.000208 | — |
| Junyi2015 | ACC | — | — | — | — | — | — | — | — | — | 0.852887 +/- 0.000321 | — |
| ASSIST2015 | AUC | — | — | — | — | — | — | — | — | — | NA (no qid) | — |
| ASSIST2015 | ACC | — | — | — | — | — | — | — | — | — | NA (no qid) | — |
| NIPS Task 3&4 | AUC | 0.687601 +/- 0.000174 | — | 0.768468 +/- 0.001061 | — | — | **0.807984 +/- 0.000624** | <u>0.806469 +/- 0.000713</u> | 0.805674 +/- 0.000305 | 0.802748 +/- 0.000442 | 0.801381 +/- 0.000507 | — |
| NIPS Task 3&4 | ACC | 0.644598 +/- 0.000610 | — | 0.701929 +/- 0.000764 | — | — | **0.735711 +/- 0.000518** | <u>0.734940 +/- 0.000862</u> | 0.733578 +/- 0.000759 | 0.731415 +/- 0.000478 | 0.730151 +/- 0.000661 | — |
| ASSIST2017 | AUC | — | — | — | — | — | — | — | — | — | 0.768272 +/- 0.000750 | — |
| ASSIST2017 | ACC | — | — | — | — | — | — | — | — | — | 0.718766 +/- 0.000664 | — |
| Slepemapy | AUC | — | — | — | — | — | — | — | — | — | 0.792618 +/- 0.000592 | — |
| Slepemapy | ACC | — | — | — | — | — | — | — | — | — | 0.801298 +/- 0.000418 | — |
| Statics2011 | AUC | — | — | — | — | — | — | — | — | — | NA (no qid) | — |
| Statics2011 | ACC | — | — | — | — | — | — | — | — | — | NA (no qid) | — |

核心表与补充表合并后按同一 `dataset × metric` 排名：**粗体**=同项目可审计结果最佳，<u>下划线</u>=次佳。`—` 表示当前没有同口径、可审计的五折 final-test 结果。`NA (no qid)` 表示 DenoiseKT 在该数据集没有 question ID，模型协议不适用，不是待补跑的空值。`†` 为 MCSKT 论文单点值，不是本项目复现结果，不参与项目内排名。A2G 数值从已审计五数据集 artifact 读取；ASIKT 仅 NIPS34 严格五折，其他三库 fold0 pilot 不进入本表。DKVMN Assist12 目前只有冻结的 validation 指标，也必须保持 `—`。

#### DenoiseKT 五折概率质量（one-step final-test）

| Dataset | NLL | Brier | ECE-15 |
|---|---:|---:|---:|
| ASSIST2009 corrected/collapsed | 0.525084 +/- 0.001682 | 0.174978 +/- 0.000645 | 0.018728 +/- 0.004011 |
| ASSIST2012 | 0.509621 +/- 0.000537 | 0.168410 +/- 0.000181 | 0.016107 +/- 0.003243 |
| Junyi2015 | 0.362041 +/- 0.000203 | 0.110705 +/- 0.000102 | 0.008751 +/- 0.002597 |
| NIPS Task 3&4 | 0.533123 +/- 0.002070 | 0.180076 +/- 0.000716 | 0.021296 +/- 0.010774 |
| ASSIST2017 | 0.546031 +/- 0.000848 | 0.184294 +/- 0.000275 | 0.009125 +/- 0.003676 |
| Slepemapy | 0.426305 +/- 0.000734 | 0.137549 +/- 0.000206 | 0.009454 +/- 0.001817 |

以上为五折 sample std；冻结前选择未使用 final-test，`test_feedback_used_for_selection=false`。DenoiseKT Window 尚未材料化或评估，不能从 one-step 数值推断 Window 结果。

### 审计分表 A：经典与通用强基线

| Dataset | Metric | BKT | DKT | DKT+ | DKT-Forget | DKVMN | DeepIRT | SAKT | SAINT | AKT | SimpleKT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Junyi2015 | AUC | — | 0.754553 +/- 0.000097 | 0.753959 +/- 0.000110 | — | 0.751666 +/- 0.000426 | — | 0.749412 +/- 0.000755 | — | 0.803541 +/- 0.000284 | 0.803113 +/- 0.000211 |
| Junyi2015 | ACC | — | 0.846524 +/- 0.000034 | 0.846219 +/- 0.000023 | — | 0.846014 +/- 0.000052 | — | 0.845151 +/- 0.000173 | — | 0.854726 +/- 0.000083 | 0.854816 +/- 0.000048 |
| ASSIST2015 | AUC | — | 0.726861 +/- 0.000479 | <u>0.728353 +/- 0.000729</u> | — | 0.721298 +/- 0.000465 | — | 0.703955 +/- 0.000590 | — | 0.725735 +/- 0.000952 | 0.723972 +/- 0.000425 |
| ASSIST2015 | ACC | — | 0.750645 +/- 0.000322 | 0.751046 +/- 0.000339 | — | 0.750609 +/- 0.000412 | — | 0.746135 +/- 0.000316 | — | <u>0.751355 +/- 0.000508</u> | 0.750584 +/- 0.000280 |
| NIPS Task 3&4 | AUC | 0.687601 +/- 0.000174 | 0.770967 +/- 0.000394 | 0.771500 +/- 0.000211 | — | 0.768590 +/- 0.000869 | 0.768468 +/- 0.001061 | 0.747161 +/- 0.001053 | — | 0.804626 +/- 0.001219 | 0.802279 +/- 0.000347 |
| NIPS Task 3&4 | ACC | 0.644598 +/- 0.000610 | 0.704490 +/- 0.000381 | 0.704851 +/- 0.000455 | — | 0.702165 +/- 0.000767 | 0.701929 +/- 0.000764 | 0.682991 +/- 0.000938 | — | 0.732949 +/- 0.000612 | 0.731513 +/- 0.000412 |
| ASSIST2017 | AUC | — | 0.716026 +/- 0.000940 | 0.717607 +/- 0.000674 | — | 0.710194 +/- 0.000595 | — | 0.653189 +/- 0.000930 | — | 0.766839 +/- 0.000793 | 0.753958 +/- 0.001005 |
| ASSIST2017 | ACC | — | 0.685402 +/- 0.001047 | 0.687453 +/- 0.000458 | — | 0.683436 +/- 0.000245 | — | 0.664698 +/- 0.000799 | — | 0.714914 +/- 0.000469 | 0.707607 +/- 0.000773 |
| Slepemapy | AUC | — | 0.783850 +/- 0.000288 | 0.786570 +/- 0.000477 | — | 0.785218 +/- 0.000425 | — | 0.771241 +/- 0.000470 | — | **0.800163 +/- 0.000465** | 0.794192 +/- 0.000419 |
| Slepemapy | ACC | — | 0.797203 +/- 0.000081 | 0.798261 +/- 0.000238 | — | 0.797235 +/- 0.000204 | — | 0.791721 +/- 0.000312 | — | <u>0.803394 +/- 0.000844</u> | 0.801021 +/- 0.000172 |

### 审计分表 B：现代强基线、机制匹配与论文对照

| Dataset | Metric | SAINT++ | SparseKT | DTransformer | UKT | ACE-KT | ASIKT | CSKT | FlucKT | FoLiBiKT | QIKT | Mamba4KT | MCSKT† | A2G-MambaKT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Junyi2015 | AUC | — | — | — | **0.804888 +/- 0.000119** | — | — | — | — | — | — | — | — | <u>0.804826 +/- 0.000111</u> |
| Junyi2015 | ACC | — | — | — | **0.855215 +/- 0.000073** | — | — | — | — | — | — | — | — | <u>0.854974 +/- 0.000070</u> |
| ASSIST2015 | AUC | — | — | — | 0.726737 +/- 0.000882 | 0.725765 +/- 0.000276 | — | — | — | — | — | — | — | **0.729012 +/- 0.000420** |
| ASSIST2015 | ACC | — | — | — | 0.750840 +/- 0.000663 | 0.750363 +/- 0.000186 | — | — | — | — | — | — | — | **0.751662 +/- 0.000156** |
| NIPS Task 3&4 | AUC | — | — | 0.801547 +/- 0.000604 | 0.805728 +/- 0.000510 | 0.803960 +/- 0.000329 | 0.798763 +/- 0.001047 | **0.807984 +/- 0.000624** | <u>0.806469 +/- 0.000713</u> | 0.805674 +/- 0.000305 | 0.802748 +/- 0.000442 | — | 0.827300† | 0.805565 +/- 0.000243 |
| NIPS Task 3&4 | ACC | — | — | 0.730464 +/- 0.000663 | 0.733253 +/- 0.000697 | 0.732686 +/- 0.000242 | 0.728213 +/- 0.001147 | **0.735711 +/- 0.000518** | <u>0.734940 +/- 0.000862</u> | 0.733578 +/- 0.000759 | 0.731415 +/- 0.000478 | — | 0.763500† | 0.731878 +/- 0.000644 |
| ASSIST2017 | AUC | — | — | — | 0.762522 +/- 0.004354 | <u>0.782030 +/- 0.000544</u> | — | — | — | — | — | — | 0.817400† | **0.784384 +/- 0.000836** |
| ASSIST2017 | ACC | — | — | — | 0.712861 +/- 0.003303 | <u>0.725020 +/- 0.001395</u> | — | — | — | — | — | — | 0.760400† | **0.727199 +/- 0.000465** |
| Slepemapy | AUC | — | — | — | — | — | — | — | — | — | — | — | — | <u>0.799000 +/- 0.000209</u> |
| Slepemapy | ACC | — | — | — | — | — | — | — | — | — | — | — | — | **0.803450 +/- 0.000260** |

输入证据：严格基线矩阵 SHA256 `d8d9dd80f4535a815059578255df7d44fe849eabe42ed978d21244b93f0f2479`；A2G 五数据集 metrics SHA256 `b2c9a37d87a6782a1fe6f3624e41aaca3075103c2cfa5fa4f5c9e7056cd70f24`；Assist12 效率审计 SHA256 `5e7e8744b2755a1ec32a1fe0a760edf2683085e376ab507ce22ddab34dfb9d6b`；A2G/ASIKT 同 batch 效率 SHA256 `0686249a2f02f8a39a60b1824bd9217b9ee4c20c0dae2017311fd158209cf70d`。表格由 artifact 自动生成，禁止手填结果。

### Overall results of model efficiency（正文候选主表）

共同硬件与张量协议：`Assist2012 / fold0 / seed42 / batch64 / seq200 / FP32 / 同一 RTX 4090 D`。数据根分两层：DKT/SAKT/AKT/SimpleKT 为历史 Assist2012 根，DKVMN/UKT/SAINT 为 strict fold-train-only-vocab 根。TT 是每 epoch 训练时间；IT 是固定完整 test loader 的推理时间；GPU usage 是训练 epoch 的峰值 allocated/reserved。两层可在同表披露，但 TT/IT 不得跨数据根排序或计算加速比。

| Model | TT (s/epoch) | IT (s/full test) | Throughput (int/s) | Total params (M) | Trainable params (M) | GPU peak alloc/reserved (GiB) | FLOPs (GFLOPs) | 证据状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BKT | — | — | — | — | — | — | N/A | 非神经优化器基线；效率口径另定 |
| DKT | 3.216 +/- 0.048 | 1.265 +/- 0.028 | 423,790 | 0.480865 | 0.480865 | 0.395 / 3.408 | NA | 228 历史 Assist2012 根实测；FLOPs 待该根 operator-complete |
| DKT+ | — | — | — | 0.480865 | 0.480865 | — | NA | CPU exact-load/rebind pass；TT/IT/GPU/FLOPs 待测 |
| DKT-Forget | — | — | — | — | — | — | NA | train-only gap-domain adapter 后测 |
| DKVMN | 10.521 +/- 0.105 | 1.878 +/- 0.037 | 285,512 +/- 5,414 | 0.339801 | 0.339801 | 2.673 / 4.916 | 5.269 | 228 strict 根实测；analytic FLOPs convention v1 |
| DeepIRT | — | — | — | — | — | — | NA | 待同协议测量 |
| SAKT | 3.113 +/- 0.295 | 1.482 +/- 0.426 | 385,117 | 0.650753 | 0.650753 | 0.614 / 3.416 | NA | 228 历史 Assist2012 根实测；FLOPs 待该根 operator-complete |
| SAINT | 20.079 +/- 0.080 | 2.826 +/- 0.031 | 189,744 +/- 2,058 | 57.344257 | 57.344257 | 4.398 / 4.764 | 141.237 | 228 strict 根实测；analytic FLOPs convention v1 |
| SAINT++ | — | — | — | — | — | — | NA | 待标准 runner/profile 后测 |
| AKT | 43.008 +/- 0.036 | 6.588 +/- 0.028 | 81,377 | 6.255536 | 6.255536 | 8.710 / 8.898 | NA | 228 历史 Assist2012 根实测；FLOPs 待该根 operator-complete |
| SimpleKT | 5.157 +/- 0.750 | 1.516 +/- 0.367 | 368,127 | 14.767105 | 14.715905 | 1.023 / 3.523 | NA | 228 历史 Assist2012 根实测；FLOPs 待该根 operator-complete |
| LPKT | — | — | — | — | — | — | NA | train-only 资源已审计；v35/v36 已排队，数值待执行 |
| DIMKT | — | — | — | — | — | — | NA | train-only difficulty 已审计；v35/v36 已排队，数值待执行 |
| SparseKT | — | — | — | — | — | — | NA | adapter smoke 不能填效率数值 |
| DTransformer | — | — | — | — | — | — | NA | Assist12 runtime 失败，暂空 |
| PKT | — | — | — | — | — | — | NA | ASIKT 论文对照；本项目 adapter/result 未准入 |
| MIKT | — | — | — | — | — | — | NA | ASIKT 论文对照；本项目 adapter/result 未准入 |
| UKT | 27.376 +/- 0.040 | 4.325 +/- 0.028 | 123,944 +/- 782 | 17.657633 | 17.555233 | 4.892 / 5.166 | 140.163 | 228 strict 根实测；analytic FLOPs convention v1 |
| ACE-KT | — | — | — | — | — | — | NA | CPU smoke 不能填效率数值 |
| ASIKT | — | — | — | — | — | — | NA | NIPS34 forward-only 值另列，不混入 Assist12 主表 |
| CSKT | — | — | — | — | — | — | NA | 待同协议测量 |
| FlucKT | — | — | — | — | — | — | NA | CPU smoke 不能填效率数值 |
| FoLiBiKT | — | — | — | — | — | — | NA | 待同协议测量 |
| QIKT | — | — | — | — | — | — | NA | 待同协议测量 |
| DenoiseKT | — | — | — | — | — | — | NA | 6 数据集五折 effectiveness 已完成；Assist12 同协议 TT/IT/GPU/FLOPs 仍待测 |
| MCKT | — | — | — | — | — | — | NA | adapter/protocol 阻断，暂空 |
| Mamba4KT | — | — | — | — | — | — | NA | hash-bound adapter 缺失 |
| MCSKT | — | — | — | — | — | — | NA | paper-only；无可准入 228 实现 |
| A2G-MambaKT | — | — | — | — | — | — | NA | 待 A2G 提供冻结的 Assist12 同协议 artifact |

效率表只有数据根、硬件、batch、精度、序列长度和计时边界全部一致的值才能横向排序。当前 strict 三行内部可比较，历史四行内部可比较；两层之间仅披露，不排序。smoke、pilot、其他数据集或其他硬件的数值不填入主表。analytic FLOPs 只与同一 convention v1 结果比较，不能直接与 MCSKT paper-only FLOPs 比较。

### ASIKT 风格多数据集效率图待填主表

这里先冻结 Fig. 7 类图所需的完整字段；正式正文主图至少报告五个主数据集，所有单元必须在相同 GPU、batch、精度、序列长度和计时边界下测量。

| Dataset | Metric | DKT | DKVMN | SAKT | AKT | SimpleKT | DTransformer | UKT | ACE-KT | ASIKT | Mamba4KT | A2G-MambaKT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Junyi2015 | TT (s/epoch) | — | — | — | — | — | — | — | — | — | — | — |
| Junyi2015 | Peak GPU alloc (GiB) | — | — | — | — | — | — | — | — | — | — | — |
| ASSIST2015 | TT (s/epoch) | — | — | — | — | — | — | — | — | — | — | — |
| ASSIST2015 | Peak GPU alloc (GiB) | — | — | — | — | — | — | — | — | — | — | — |
| NIPS Task 3&4 | TT (s/epoch) | — | — | — | — | — | — | — | — | — | — | — |
| NIPS Task 3&4 | Peak GPU alloc (GiB) | — | — | — | — | — | — | — | — | — | — | — |
| ASSIST2017 | TT (s/epoch) | — | — | — | — | — | — | — | — | — | — | — |
| ASSIST2017 | Peak GPU alloc (GiB) | — | — | — | — | — | — | — | — | — | — | — |
| Slepemapy | TT (s/epoch) | — | — | — | — | — | — | — | — | — | — | — |
| Slepemapy | Peak GPU alloc (GiB) | — | — | — | — | — | — | — | — | — | — | — |

### A2G / ASIKT 同 batch forward-only 补充效率表（NIPS34）

该表使用一个持久真实 NIPS34 test batch（batch=48，7483 valid tokens）、FP32、10 warmups、50 次同步重复，只支持模型 forward microbenchmark；它不是上述 Assist12 TT/IT 主表，也不支持训练时间结论。

| Model | p50 latency (ms) | p95 latency (ms) | Throughput p50 (token/s) | Params (M) | Peak alloc/reserved (GiB) |
|---|---:|---:|---:|---:|---:|
| A2G-MambaKT | 23.125 | 23.279 | 323,588 | 3.028492 | 0.294 / 1.996 |
| ASIKT | 25.351 | 25.400 | 295,173 | 5.353844 | 1.844 / 3.027 |
<!-- CODEX-PUBLICATION-MAIN-TABLES-20260803:END -->

### MCSKT Table 4 论文报告值（只作协议设计参考）

MCSKT 原文协议为 `Assist12 / batch64 / seq200 / NVIDIA RTX 3060 Ultra`。这些数值带 `†`，不得与上面的 RTX 4090 D 实测值直接比较、排序或计算加速比。原文把 Parameter 描述为 MB/“M”尺度，本项目保留其原始数值，不转换成 trainable parameter count。

| Model | TT† | IT† | Parameter† | FLOPs† | GPU usage† |
|---|---:|---:|---:|---:|---:|
| DKT | 17.53 | 2.25 | 0.37 | 4.82 | 0.37 |
| DKT-Forget | 19.20 | 2.43 | 0.44 | 5.77 | 0.76 |
| LPKT | 287.84 | 38.07 | 0.07 | 42.31 | 4.90 |
| DIMKT | 206.33 | 26.25 | 0.23 | 3.76 | 0.36 |
| AKT | 169.18 | 18.18 | 4.88 | 72.41 | 9.30 |
| SparseKT | 187.35 | 23.14 | 1.85 | 39.46 | 7.96 |
| DTransformer | 784.26 | 15.66 | 1.12 | 64.00 | 16.38 |
| Mamba4KT | 39.52 | 4.20 | 2.90 | 28.23 | 2.00 |
| MCSKT | 77.86 | 4.52 | 5.69 | 33.22 | 1.90 |

ASIKT Figure 7 使用 `RTX 3090 / batch24 / seq200`，展示 DKT、DTransformer、SAKT、AKT、PKT、MIKT、ASIKT 在四数据集上的每 epoch 训练时间与显存，但图中没有可可靠抄录的精确表格值。因此只采用其图形设计与测量维度，不从像素估算数值。

### 投稿效率图片清单

| Figure | 参考 | 横轴 / 分组 | 纵轴 | 待补状态 |
|---|---|---|---|---|
| Efficiency-performance scatter | MCSKT Fig. 3 | TT 或 IT | AUC / ACC | 用 228 同协议实测生成；不混入 paper-only 值 |
| Training time + memory comparison | ASIKT Fig. 7 | model x dataset | TT 与 peak GPU memory | Assist12 主图先完成；多数据集作为补充 |
| Sequence-length scaling | MCSKT long-sequence experiment | seq=200/512/1024/2048 | TT、IT、throughput、peak memory | A2G、Mamba4KT、ASIKT、AKT/DTransformer 至少四类 comparator |
| Complexity/resource bars | MCSKT Table 4 | model | params、FLOPs、GPU peak | FLOPs 仅在 operator-complete 后绘制 |
| Calibration reliability | 本项目补强 | confidence bin | accuracy / confidence | 从已保存概率生成，不为 ECE 重训 |

### 当前不可直接纳入的论文对照

| Model | 来源角色 | 当前结论 |
|---|---|---|
| LPKT | MCSKT Table 3/4 | stock pyKT 使用 test-inclusive time vocab/Q-matrix，当前严格阻断 |
| DIMKT | MCSKT Table 3/4 | active validation fold 响应参与 difficulty 构造，当前严格阻断 |
| PKT / MIKT | ASIKT Table 2 | 仅作为论文对照语境；没有当前 hash-bound pyKT adapter/result，不能填入本项目实测格 |
| TCKT / HD-KT | MCSKT Table 3 | 仅论文报告语境；未通过本项目源码、依赖、字段映射与泄露门禁 |
| GKT / RKT / HCGKT | pyKT 扩展候选 | 分别因 test-inclusive graph、relation provenance 缺失、运行时缺陷而 fail-closed |

队列终态：DKVMN Assist12 validation-only 5/5 已完成并冻结；DTransformer fold0 因 Triton/gcc 编译失败后队列 fail-fast，UKT/SAINT 未启动。GPU 已释放并交接 A2G；本基线任务未重启，final test 仍未授权。
<!-- CODEX-PUBLICATION-BASELINE-SMOKE-20260803:START -->
### 投稿基线 smoke 门禁（2026-08-03 实际执行）

`smoke` 只验证构造、forward、backward 与 optimizer wiring，不产生 AUC/ACC，不替代 validation/final test。固定 pyKT commit 的 13 个原生模型标准入口已在 228 逐模型隔离执行 CPU-only optimizer-step smoke：**12/13 通过**；SparseKT 的标准入口失败证据原样保留，之后其独立 hash-bound 4-to-3 兼容适配器已通过。SAINT++ 专用源码、ACE-KT 归档作者 overlay、FlucKT 官方 post-pin runtime 也已分别完成 CPU optimizer-step smoke。另有 DKT、SAKT、AKT、SimpleKT 四个模型已在 228 的 RTX 4090 D 上用真实 Assist12 batch64/seq200 完成 forward + backward + optimizer step 和严格 checkpoint load；ASIKT 仅有既存的同 GPU、真实 NIPS34 batch forward-only 证据，不得写成训练 smoke。

| Model | 本轮投稿 smoke 状态 | 可声称边界 |
|---|---|---|
| BKT | N/A | 非神经优化器基线；保留已准入的 NIPS34 effectiveness 结果 |
| DKT | GPU 真实 batch PASS | forward/backward/optimizer step；未启动正式训练，未用 test 选择 |
| DKT+ | CPU 合成 batch PASS | 构造与 optimizer wiring 通过；不是 effectiveness 结果 |
| DKT-Forget | CPU 合成 batch PASS | 模型 wiring 通过；真实数据仍须 train-only gap-domain adapter |
| DKVMN | CPU 合成 batch PASS | wiring 通过；Assist12 validation-only 五折队列另行运行 |
| DeepIRT | CPU 合成 batch PASS | wiring 通过；不是 effectiveness 结果 |
| SAKT | GPU 真实 batch PASS | forward/backward/optimizer step；未启动正式训练，未用 test 选择 |
| SAINT | CPU 合成 batch PASS | SAINT wiring 通过；禁止冒充 SAINT++ |
| SAINT++ | 专用源码 CPU PASS | 作者入口默认参数的 forward/backward/optimizer step 通过；标准 registry 仍未接入 |
| AKT | GPU 真实 batch PASS | forward/backward/optimizer step；未启动正式训练，未用 test 选择 |
| SimpleKT | GPU 真实 batch PASS | forward/backward/optimizer step；未启动正式训练，未用 test 选择 |
| SparseKT | 固定适配器 CPU PASS | 标准入口仍因 4→3 解包失败；独立 hash-bound adapter 已完成 forward/backward/optimizer step，未修改 runtime |
| DTransformer | CPU 合成 batch PASS | 构造与 optimizer wiring 通过；不是 effectiveness 结果 |
| UKT | CPU 合成 batch PASS | 构造与 optimizer wiring 通过；不是 effectiveness 结果 |
| ACE-KT | 归档 overlay CPU PASS | hash-bound 作者 overlay + 作者默认参数的 optimizer-step wiring 通过；已有五组 effectiveness 继续保留 |
| ASIKT | GPU 真实 batch forward-only PASS | 仅支持同 batch forward latency/显存/参数；TT 与训练 smoke 仍为 NA |
| CSKT | CPU 合成 batch PASS | 构造与 optimizer wiring 通过；不是 effectiveness 结果 |
| FlucKT | post-pin runtime CPU PASS（shim） | 源码硬编码 `.cuda()`；显式 compatibility shim 下 wiring 通过，不声称原生 CPU 支持 |
| Mamba4KT | BLOCK | canonical Mamba comparator 的 hash-bound adapter 未完成 |
| MCSKT | BLOCK | 未核实作者仓且动态 k 存在 test-accuracy 反馈风险；只保留 paper-only 参考 |

机器可读 v2 总账：`outputs/PUBLICATION_BASELINE_SMOKE_INVENTORY_20260803_V2.json`，SHA256 `de5546d53a9efe7ae311558f13dab22af1a78b70a84640b5ae2729d515907edb`。原始 13-model 标准入口尝试：`outputs/PUBLICATION_NATIVE_BASELINE_CPU_SMOKE_AUDIT_20260803_ATTEMPT1.json`，SHA256 `063ffa59f09e3cbfd3d433f948fa7727a865396da2c443b9f608ae2353cf438f`。v1 inventory 保留为适配前快照；当前总状态仍是 `partial_pass_with_explicit_blocks`，因为 Mamba4KT 与 MCSKT 尚未通过执行门禁。
<!-- CODEX-PUBLICATION-BASELINE-SMOKE-20260803:END -->

<!-- CODEX-ASSIST2012-VALIDATION-QUEUE-20260803:START -->
### Assist2012 validation-only 队列终态（2026-08-03）

队列没有完成原计划 20/20，而是在 DTransformer fold0 fail-fast 后释放 GPU。DKVMN 的唯一预声明候选 `cand_45cc1e54cd7c` 已完成 folds0-4：validation AUC 为 `0.730646 +/- 0.003731`，validation ACC 为 `0.735355 +/- 0.002275`；逐折 AUC 为 `0.724723 / 0.732371 / 0.731826 / 0.734572 / 0.729735`。五折 `testauc/testacc/window_testauc/window_testacc` 均为 `-1`，profile 已冻结，final test 未启动、未授权，因此这些 validation 数值不得进入 effectiveness 主表。

DTransformer fold0 因 Triton `cuda_utils` 本地 gcc 编译 `CalledProcessError` 失败，没有 checkpoint、没有可用结果；DTransformer folds1-4、UKT folds0-4、SAINT folds0-4 均未启动。GPU 已异常 fail-fast 释放，并已向 A2G 发送 handoff；本基线任务未重启队列。

证据：`outputs/ASSIST2012_VALIDATION_QUEUE_PARTIAL_AUDIT_20260803_V2.json` SHA256 `c9326526b81a19dc350acdd3f842016a9e5807bcf93af791c381b562118a4a1e`；`outputs/DKVMN_ASSIST2012_VALIDATION_PROFILE_FREEZE_20260803.json` SHA256 `ba3a00881690666f2535871be24ed33999c8b529b8a844cfc8dcab41352298c1`。
<!-- CODEX-ASSIST2012-VALIDATION-QUEUE-20260803:END -->

<!-- CODEX-A2G-PUBLICATION-BASELINE-TARGET-20260803:END -->

## 2026-07-17 安全口径结论

本页已更新为当前可引用的统一 pyKT 参考 baseline 口径：只展示 `mean +/- sample std`，不展示方差。

当前论文对比基线表只纳入 8 个来源明确、5 折结果完整、原始预测可复算、实际 loader 切分无用户泄露且可复现边界清楚的数据集：

- `assist2009_corrected_collapsed`
- `assist2012`
- `junyi2015`
- `assist2015`
- `nips_task34`
- `assist2017`
- `slepemapy`
- `statics2011`

当前已完成历史/固定协议 artifact 审计的共同 baseline 为：

- `dkt`
- `dkt+`
- `akt`
- `simplekt`
- `sakt`

近期模型只纳入已单独通过 5/5 门禁的组合：`ukt`、`acekt`、`cskt`、`fluckt`；ASIKT 作为机制匹配的 author-source adapter 另列入 NIPS34 专项行。模型名出现在运行队列中不等于结果可用。

`keenkt` 已从本页移除：它不能作为当前论文主基线引用，原因是存在未来信息/泄露风险，且不属于本轮固定协议共同 baseline。

`ednet`/`ednet_official` 已从本页移除：当前切分与样本口径的泄露风险尚未排除。现有结果不能作为当前论文主基线引用。

`assist2009_raw`/`assist2009_dedup_uncollapsed` 不进入当前主表。`assist2009_corrected_collapsed` 现已有 DKT、DKT+、AKT、SimpleKT、SAKT、UKT、ACE-KT 七条五折合格行：其中 DKT、DKT+、SimpleKT、SAKT、UKT、ACE-KT 是固定官方 pyKT/固定 overlay 的 `v3` 严格运行，AKT 是 prediction、checkpoint/config 和同哈希 loader 三门禁通过的历史固定配置结果，二者的 evidence 类型不得混写。

## 统计口径

以下结果均为 5 个 validation-fold 模型和 one-step test，格式为：

`AUC mean +/- sample std`，`ACC mean +/- sample std`

这里的 `std` 是 5 个 fold 之间的样本标准差，不是方差。论文正文主表只报告 mean 和标准差。

历史 2026-07-16/更早固定协议行使用 seed 42；2026-07-18 严格 `v3` 行按预先冻结的 YAML 逐折解析 seed 和参数，不能把全部新行统一写成 seed 42。每折实际 seed、命令和 resolved profile 均在 `launch_manifest.json`。

准确切分语义：先固定留出一个用户不重叠的 test set，再在剩余用户中建立 5 个 validation folds；5 个模型都在同一个 held-out test 上评估。因此正文应写 `five validation-fold runs with a fixed held-out test`，不要写成每个样本轮流作为测试集的传统 5-fold cross-validation。

## 评估实现核验（2026-07-17）

现有表中 2026-07-16 及历史复用结果没有直接执行 upstream `examples/wandb_predict.py`，实际命令链为 `evaluate_one_step_baseline.py -> pykt_one_step_eval_entry.py -> pykt.models.evaluate()`。入口从 pyKT 导入 `load_model` 与 `evaluate`，为 one-step test 单独构造 pyKT `KTDataset` loader，然后执行 `evaluate(model, test_loader, ...)`。因此普通 `dkt`、`dkt+`、`akt`、`simplekt`、`sakt` 的模型预测与聚合 AUC/ACC 使用的是 pyKT 核心评估函数，不是另写一套 sklearn 评估取代 pyKT。

但旧 172 runtime 是 `HEAD=98a790756068e07ccfe3b99db335a856723cff24` 的修改工作树，不是未修改的官方 pyKT。运行快照记录的完整 tracked diff SHA256 为 `5efba02976d97ff681c6234984ee7eff99d8583da890f7d237ecac1cd7bb33b4`；2026-07-17 只读复核时 HEAD 与该完整 diff 哈希均完全一致。源码和可无损解码的完整 diff 已保存到 `outputs/remote_172_records/baseline_run_20260716_final/runtime/legacy_eval_scripts/`。代码差异复核表明：对这五个普通模型，评估修改只涉及新增模型 dispatch、debug 输出和单类别序列行 AUC 显式回退为 `-1`，没有改变整体 one-step AUC/ACC 公式。因此旧表应称 **pyKT-based fixed-profile results**，不能写成“未修改官方 pyKT 脚本的严格复现”。

2026-07-17 已完成统一独立二次核验：五个数据集、五个模型、五个 fold 共 **125/125** 份 prediction 均逐行读取标签与概率，使用 sklearn 重算全局 AUC/ACC，并与保留的 pyKT metric 在绝对误差 `1e-12` 内一致。ASSIST2012 与 Junyi 的旧 prediction 在单类别序列的非权威“逐序列 AUC”字段中含小写裸 `nan`；审计器只将该独立字段规范化为 `null` 以解析记录，未修改标签或概率，且在报告中分别登记 13,000 与 203,375 行。

同日完成两项补充门禁：其一，五个数据集实际被 checkpoint `data_config` 解析到的 10 份 `train_valid_sequences.csv`/`test_sequences.csv` 均与预处理 manifest 的字节数和 SHA256 一致，训练/验证 fold 为 `0–4`、固定 test fold 为 `-1`，五个数据集用户交集均为 0，有时间戳的数据未发现逆序；其二，125/125 份 checkpoint、`config.json`、metric/run JSON 与训练/评估日志均已哈希，25 个 dataset-model 组合在五个 fold 中各自只有一套固定 train/model 参数。由此本页结果可作为论文的 **pyKT-based fixed-profile/reference comparison baselines**；但它仍不是“全部由未修改官方 pyKT 固定 commit 重新训练”的结果。

今后的严格结果统一使用官方 pyKT commit `f766468f1d3083f737e5fde22a17a08de0d52552`：`run_five_folds.py` 先调用 upstream `wandb_predict.py`，随后从同一 `*_test_predictions.txt` 用 sklearn 重算 AUC/ACC，并要求与 pyKT 报告值在绝对误差 `1e-12` 内一致；二次核验是审计，不替代 pyKT 评估。

| 数据集 | pyKT 处理证据 | 旧表评估证据 | 当前可复现结论 |
|---|---|---|---|
| `assist2015` | 官方原件；已用固定官方 `f766468...` 再处理并得到与 172 完全相同的输出；实际 loader 用户交集为 0 | pyKT `evaluate()`；25/25 prediction 全行重算一致；25/25 checkpoint/config 完整 | **可作为统一参考基线**；五者中端到端处理证据最完整 |
| `nips_task34` | 官方 `data.zip`；pyKT 标准处理 argv、代码、输入/输出哈希齐全；实际 loader 用户交集为 0 | pyKT `evaluate()`；25/25 prediction 与 checkpoint/config 门禁通过 | **可作为统一参考基线**；来源、处理、结果均闭环 |
| `assist2012` | 官方 ZIP 与工作 CSV 成员字节一致；旧标准处理命令、代码和输出哈希齐全；实际 loader 用户交集为 0 | 历史 pyKT `evaluate()`；25/25 prediction 与 checkpoint/config 门禁通过 | **可作为审计后的历史统一参考基线**；不是本轮固定官方 commit 端到端重训 |
| `junyi2015` | DataShop 权威页 + 字节匹配 USTC 镜像；旧标准处理命令、代码和输出哈希齐全；实际 loader 用户交集为 0 | 历史 pyKT `evaluate()`；25/25 prediction 与 checkpoint/config 门禁通过 | **可作为审计后的历史统一参考基线**；必须披露镜像及非商业/引用边界 |
| `assist2017` | 发布方 legacy Drive 原件字节闭环；旧 pyKT 处理代码/输出哈希齐全；实际 loader 用户交集为 0 | 隔离 legacy pyKT `evaluate()`；整数值 decimal-time parser repair；25/25 prediction 与 checkpoint/config 门禁通过 | **可作为兼容 runtime 下的统一参考基线**；必须披露 repair，不能称未修改官方 runtime |

对应五份预处理 manifest 已本地保存于 `outputs/remote_172_records/baseline_run_20260716_final/runtime/legacy_preprocess_manifests/`。

<!-- CODEX-A2G-PRIMARY5-WIDE-TABLE:START -->
## 五个主数据集横向主表（正文候选；自动同步）

该位置保留用于兼容旧笔记链接；表格字节来自文首投稿主表生成 artifact。`—` 表示当前尚无可准入的同协议五折 final-test 数值。

这是正文候选主表。已跑出的同协议五折 final-test 数值从不可变 artifact 生成；尚未跑出或尚未准入的模型明确保留 `—`。该表不会用 validation、fold0 pilot、smoke 或 paper-only 数值补格。

| Dataset | Metric | DKT | DKT+ | DKVMN | SAKT | SAINT++ | AKT | SimpleKT | DTransformer | UKT | ACE-KT | ASIKT | Mamba4KT | MCSKT† | A2G-MambaKT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Junyi2015 | AUC | 0.754553 +/- 0.000097 | 0.753959 +/- 0.000110 | — | 0.749412 +/- 0.000755 | — | 0.803541 +/- 0.000284 | 0.803113 +/- 0.000211 | — | **0.804888 +/- 0.000119** | — | — | — | — | <u>0.804826 +/- 0.000111</u> |
| Junyi2015 | ACC | 0.846524 +/- 0.000034 | 0.846219 +/- 0.000023 | — | 0.845151 +/- 0.000173 | — | 0.854726 +/- 0.000083 | 0.854816 +/- 0.000048 | — | **0.855215 +/- 0.000073** | — | — | — | — | <u>0.854974 +/- 0.000070</u> |
| ASSIST2015 | AUC | 0.726861 +/- 0.000479 | <u>0.728353 +/- 0.000729</u> | — | 0.703955 +/- 0.000590 | — | 0.725735 +/- 0.000952 | 0.723972 +/- 0.000425 | — | 0.726737 +/- 0.000882 | 0.725765 +/- 0.000276 | — | — | — | **0.729012 +/- 0.000420** |
| ASSIST2015 | ACC | 0.750645 +/- 0.000322 | 0.751046 +/- 0.000339 | — | 0.746135 +/- 0.000316 | — | <u>0.751355 +/- 0.000508</u> | 0.750584 +/- 0.000280 | — | 0.750840 +/- 0.000663 | 0.750363 +/- 0.000186 | — | — | — | **0.751662 +/- 0.000156** |
| NIPS Task 3&4 | AUC | 0.770967 +/- 0.000394 | 0.771500 +/- 0.000211 | 0.769581 +/- 0.000318 | 0.747161 +/- 0.001053 | — | 0.804626 +/- 0.001219 | 0.802279 +/- 0.000347 | 0.801547 +/- 0.000604 | 0.805728 +/- 0.000510 | 0.803960 +/- 0.000329 | 0.798763 +/- 0.001047 | — | 0.827300† | 0.805565 +/- 0.000243 |
| NIPS Task 3&4 | ACC | 0.704490 +/- 0.000381 | 0.704851 +/- 0.000455 | 0.702736 +/- 0.000380 | 0.682991 +/- 0.000938 | — | 0.732949 +/- 0.000612 | 0.731513 +/- 0.000412 | 0.730464 +/- 0.000663 | 0.733253 +/- 0.000697 | 0.732686 +/- 0.000242 | 0.728213 +/- 0.001147 | — | 0.763500† | 0.731878 +/- 0.000644 |
| ASSIST2017 | AUC | 0.716026 +/- 0.000940 | 0.717607 +/- 0.000674 | — | 0.653189 +/- 0.000930 | — | 0.766839 +/- 0.000793 | 0.753958 +/- 0.001005 | — | 0.762522 +/- 0.004354 | <u>0.782030 +/- 0.000544</u> | — | — | 0.817400† | **0.784384 +/- 0.000836** |
| ASSIST2017 | ACC | 0.685402 +/- 0.001047 | 0.687453 +/- 0.000458 | — | 0.664698 +/- 0.000799 | — | 0.714914 +/- 0.000469 | 0.707607 +/- 0.000773 | — | 0.712861 +/- 0.003303 | <u>0.725020 +/- 0.001395</u> | — | — | 0.760400† | **0.727199 +/- 0.000465** |
| Slepemapy | AUC | 0.783850 +/- 0.000288 | 0.786570 +/- 0.000477 | — | 0.771241 +/- 0.000470 | — | **0.800163 +/- 0.000465** | 0.794192 +/- 0.000419 | — | — | — | — | — | — | <u>0.799000 +/- 0.000209</u> |
| Slepemapy | ACC | 0.797203 +/- 0.000081 | 0.798261 +/- 0.000238 | — | 0.791721 +/- 0.000312 | — | <u>0.803394 +/- 0.000844</u> | 0.801021 +/- 0.000172 | — | — | — | — | — | — | **0.803450 +/- 0.000260** |
<!-- CODEX-A2G-PRIMARY5-WIDE-TABLE:END -->

<!-- CODEX-KT-COMPLETE-MATRIX-20260728:START -->
## 完整基线审计明细表（60 个 5/5 组合）

本表直接由最终机器可读矩阵生成，覆盖 60 个完整 dataset-model 组合；同名 CSV 已同步为相同 60 行。A2G-MambaKT 是目标方法，不计入这 60 条 baseline 行。

| Dataset | Model | Folds | AUC mean +/- std | ACC mean +/- std | Window AUC mean +/- std | Window ACC mean +/- std | NLL | Brier | ECE-15 | Evidence tier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| assist2009_corrected_collapsed | acekt | 5 | 0.845685 +/- 0.000872 | 0.779969 +/- 0.001030 | 0.846436 +/- 0.001241 | 0.779968 +/- 0.001308 | NA | NA | NA | v16_strict_5of5 |
| assist2009_corrected_collapsed | akt | 5 | 0.840313 +/- 0.001020 | 0.777064 +/- 0.001267 | 0.841987 +/- 0.001063 | 0.778485 +/- 0.001224 | NA | NA | NA | v16_strict_5of5 |
| assist2009_corrected_collapsed | dkt | 5 | 0.825756 +/- 0.001019 | 0.768648 +/- 0.001041 | 0.826537 +/- 0.000987 | 0.769120 +/- 0.000758 | NA | NA | NA | v16_strict_5of5 |
| assist2009_corrected_collapsed | dkt+ | 5 | 0.825614 +/- 0.000588 | 0.769230 +/- 0.000781 | 0.826391 +/- 0.000529 | 0.769578 +/- 0.000665 | NA | NA | NA | v16_strict_5of5 |
| assist2009_corrected_collapsed | sakt | 5 | 0.795565 +/- 0.001660 | 0.748478 +/- 0.002482 | 0.780229 +/- 0.015777 | 0.739884 +/- 0.009758 | NA | NA | NA | v16_strict_5of5 |
| assist2009_corrected_collapsed | simplekt | 5 | 0.840088 +/- 0.001059 | 0.776476 +/- 0.001133 | 0.839497 +/- 0.001236 | 0.776125 +/- 0.000954 | NA | NA | NA | v16_strict_5of5 |
| assist2009_corrected_collapsed | ukt | 5 | 0.848004 +/- 0.001405 | 0.781122 +/- 0.000806 | 0.849608 +/- 0.001323 | 0.782453 +/- 0.000962 | NA | NA | NA | v16_strict_5of5 |
| assist2012 | akt | 5 | 0.779282 +/- 0.000552 | 0.757258 +/- 0.000853 | 0.780863 +/- 0.000547 | 0.758163 +/- 0.000822 | NA | NA | NA | v16_strict_5of5 |
| assist2012 | dkt | 5 | 0.730352 +/- 0.000212 | 0.734183 +/- 0.000205 | 0.731599 +/- 0.000228 | 0.734822 +/- 0.000188 | NA | NA | NA | v16_strict_5of5 |
| assist2012 | dkt+ | 5 | 0.731710 +/- 0.000420 | 0.734358 +/- 0.000365 | 0.733006 +/- 0.000414 | 0.734999 +/- 0.000390 | NA | NA | NA | v16_strict_5of5 |
| assist2012 | sakt | 5 | 0.707173 +/- 0.001031 | 0.724079 +/- 0.000560 | 0.706969 +/- 0.001142 | 0.724056 +/- 0.000465 | NA | NA | NA | v16_strict_5of5 |
| assist2012 | simplekt | 5 | 0.773878 +/- 0.000295 | 0.750971 +/- 0.001224 | 0.774774 +/- 0.000307 | 0.751877 +/- 0.001032 | NA | NA | NA | v16_strict_5of5 |
| assist2015 | acekt | 5 | 0.725765 +/- 0.000276 | 0.750363 +/- 0.000186 | 0.725949 +/- 0.000294 | 0.750499 +/- 0.000143 | NA | NA | NA | v16_strict_5of5 |
| assist2015 | akt | 5 | 0.725735 +/- 0.000952 | 0.751355 +/- 0.000508 | 0.725897 +/- 0.000930 | 0.751367 +/- 0.000438 | NA | NA | NA | v16_strict_5of5 |
| assist2015 | dkt | 5 | 0.726861 +/- 0.000479 | 0.750645 +/- 0.000322 | 0.726927 +/- 0.000474 | 0.750569 +/- 0.000312 | NA | NA | NA | v16_strict_5of5 |
| assist2015 | dkt+ | 5 | 0.728353 +/- 0.000729 | 0.751046 +/- 0.000339 | 0.728392 +/- 0.000718 | 0.750966 +/- 0.000316 | NA | NA | NA | v16_strict_5of5 |
| assist2015 | sakt | 5 | 0.703955 +/- 0.000590 | 0.746135 +/- 0.000316 | 0.703862 +/- 0.000566 | 0.746196 +/- 0.000254 | NA | NA | NA | v16_strict_5of5 |
| assist2015 | simplekt | 5 | 0.723972 +/- 0.000425 | 0.750584 +/- 0.000280 | 0.724029 +/- 0.000440 | 0.750545 +/- 0.000269 | NA | NA | NA | v16_strict_5of5 |
| assist2015 | ukt | 5 | 0.726737 +/- 0.000882 | 0.750840 +/- 0.000663 | 0.726847 +/- 0.000893 | 0.750860 +/- 0.000636 | NA | NA | NA | v16_strict_5of5 |
| assist2017 | acekt | 5 | 0.782030 +/- 0.000544 | 0.725020 +/- 0.001395 | 0.784691 +/- 0.001455 | 0.726362 +/- 0.001812 | NA | NA | NA | v16_strict_5of5 |
| assist2017 | akt | 5 | 0.766839 +/- 0.000793 | 0.714914 +/- 0.000469 | 0.771074 +/- 0.000883 | 0.717751 +/- 0.000952 | NA | NA | NA | v16_strict_5of5 |
| assist2017 | dkt | 5 | 0.716026 +/- 0.000940 | 0.685402 +/- 0.001047 | 0.719415 +/- 0.001174 | 0.687395 +/- 0.001070 | NA | NA | NA | v16_strict_5of5 |
| assist2017 | dkt+ | 5 | 0.717607 +/- 0.000674 | 0.687453 +/- 0.000458 | 0.721051 +/- 0.000640 | 0.689222 +/- 0.000450 | NA | NA | NA | v16_strict_5of5 |
| assist2017 | sakt | 5 | 0.653189 +/- 0.000930 | 0.664698 +/- 0.000799 | 0.655104 +/- 0.000705 | 0.664991 +/- 0.000584 | NA | NA | NA | v16_strict_5of5 |
| assist2017 | simplekt | 5 | 0.753958 +/- 0.001005 | 0.707607 +/- 0.000773 | 0.754247 +/- 0.000490 | 0.708384 +/- 0.000656 | NA | NA | NA | v16_strict_5of5 |
| assist2017 | ukt | 5 | 0.762522 +/- 0.004354 | 0.712861 +/- 0.003303 | 0.766571 +/- 0.004163 | 0.714900 +/- 0.003121 | NA | NA | NA | v16_strict_5of5 |
| junyi2015 | akt | 5 | 0.803541 +/- 0.000284 | 0.854726 +/- 0.000083 | 0.806148 +/- 0.000321 | 0.855529 +/- 0.000072 | NA | NA | NA | v16_strict_5of5 |
| junyi2015 | dkt | 5 | 0.754553 +/- 0.000097 | 0.846524 +/- 0.000034 | 0.756816 +/- 0.000111 | 0.846840 +/- 0.000021 | NA | NA | NA | v16_strict_5of5 |
| junyi2015 | dkt+ | 5 | 0.753959 +/- 0.000110 | 0.846219 +/- 0.000023 | 0.756096 +/- 0.000139 | 0.846456 +/- 0.000022 | NA | NA | NA | v16_strict_5of5 |
| junyi2015 | sakt | 5 | 0.749412 +/- 0.000755 | 0.845151 +/- 0.000173 | 0.751607 +/- 0.000734 | 0.845419 +/- 0.000191 | NA | NA | NA | v16_strict_5of5 |
| junyi2015 | simplekt | 5 | 0.803113 +/- 0.000211 | 0.854816 +/- 0.000048 | 0.805707 +/- 0.000199 | 0.855496 +/- 0.000035 | NA | NA | NA | v16_strict_5of5 |
| junyi2015 | ukt | 5 | 0.804888 +/- 0.000119 | 0.855215 +/- 0.000073 | 0.807673 +/- 0.000163 | 0.856046 +/- 0.000070 | NA | NA | NA | v16_strict_5of5 |
| nips_task34 | acekt | 5 | 0.803960 +/- 0.000329 | 0.732686 +/- 0.000242 | 0.805519 +/- 0.000356 | 0.734386 +/- 0.000281 | NA | NA | NA | mixed_evaluator_semantic_audited_strict_5of5 |
| nips_task34 | akt | 5 | 0.804626 +/- 0.001219 | 0.732949 +/- 0.000612 | 0.806989 +/- 0.001128 | 0.734957 +/- 0.000644 | NA | NA | NA | v16_strict_5of5 |
| nips_task34 | asikt_author | 5 | 0.798763 +/- 0.001047 | 0.728213 +/- 0.001147 | 0.800982 +/- 0.001012 | 0.729957 +/- 0.001061 | NA | NA | NA | author_source_causal_mask_adapter_strict_5of5 |
| nips_task34 | bkt | 5 | 0.687601 +/- 0.000174 | 0.644598 +/- 0.000610 | 0.691723 +/- 0.000204 | 0.648938 +/- 0.000625 | NA | NA | NA | mature_library_prediction_audited_strict_5of5 |
| nips_task34 | cskt | 5 | 0.807984 +/- 0.000624 | 0.735711 +/- 0.000518 | 0.810503 +/- 0.000607 | 0.737761 +/- 0.000537 | NA | NA | NA | v16_strict_5of5 |
| nips_task34 | deep_irt | 5 | 0.768468 +/- 0.001061 | 0.701929 +/- 0.000764 | 0.771062 +/- 0.001043 | 0.704206 +/- 0.000675 | NA | NA | NA | uniform_evaluator_external_split_audited_strict_5of5 |
| nips_task34 | dkt | 5 | 0.770967 +/- 0.000394 | 0.704490 +/- 0.000381 | 0.773068 +/- 0.000378 | 0.706148 +/- 0.000291 | NA | NA | NA | v16_strict_5of5 |
| nips_task34 | dkt+ | 5 | 0.771500 +/- 0.000211 | 0.704851 +/- 0.000455 | 0.773569 +/- 0.000229 | 0.706738 +/- 0.000352 | NA | NA | NA | v16_strict_5of5 |
| nips_task34 | dkvmn | 5 | 0.769581 +/- 0.000318 | 0.702736 +/- 0.000380 | 0.772071 +/- 0.000345 | 0.705010 +/- 0.000370 | NA | NA | NA | uniform_evaluator_external_split_audited_strict_5of5 |
| nips_task34 | dtransformer | 5 | 0.801547 +/- 0.000604 | 0.730464 +/- 0.000663 | 0.803524 +/- 0.000707 | 0.732090 +/- 0.000603 | NA | NA | NA | strict_uniform_evaluator_prediction_audited_5of5 |
| nips_task34 | fluckt | 5 | 0.806469 +/- 0.000713 | 0.734940 +/- 0.000862 | 0.808800 +/- 0.000666 | 0.736958 +/- 0.000978 | NA | NA | NA | v16_strict_5of5 |
| nips_task34 | folibikt | 5 | 0.805674 +/- 0.000305 | 0.733578 +/- 0.000759 | 0.807888 +/- 0.000301 | 0.735590 +/- 0.000895 | NA | NA | NA | strict_uniform_evaluator_prediction_audited_5of5 |
| nips_task34 | qikt | 5 | 0.802748 +/- 0.000442 | 0.731415 +/- 0.000478 | 0.804391 +/- 0.000418 | 0.732957 +/- 0.000381 | NA | NA | NA | strict_uniform_evaluator_prediction_audited_5of5 |
| nips_task34 | sakt | 5 | 0.747161 +/- 0.001053 | 0.682991 +/- 0.000938 | 0.744529 +/- 0.001010 | 0.680983 +/- 0.001395 | NA | NA | NA | v16_strict_5of5 |
| nips_task34 | simplekt | 5 | 0.802279 +/- 0.000347 | 0.731513 +/- 0.000412 | 0.804230 +/- 0.000257 | 0.733328 +/- 0.000486 | NA | NA | NA | v16_strict_5of5 |
| nips_task34 | ukt | 5 | 0.805728 +/- 0.000510 | 0.733253 +/- 0.000697 | 0.807907 +/- 0.000529 | 0.735266 +/- 0.000697 | NA | NA | NA | mixed_evaluator_semantic_audited_strict_5of5 |
| slepemapy | akt | 5 | 0.800163 +/- 0.000465 | 0.803394 +/- 0.000844 | 0.804273 +/- 0.000425 | 0.804622 +/- 0.000789 | NA | NA | NA | strict_uniform_evaluator_prediction_audited_5of5 |
| slepemapy | dkt | 5 | 0.783850 +/- 0.000288 | 0.797203 +/- 0.000081 | 0.786415 +/- 0.000291 | 0.797660 +/- 0.000132 | NA | NA | NA | v16_strict_5of5 |
| slepemapy | dkt+ | 5 | 0.786570 +/- 0.000477 | 0.798261 +/- 0.000238 | 0.789184 +/- 0.000478 | 0.798737 +/- 0.000240 | NA | NA | NA | strict_uniform_evaluator_prediction_audited_5of5 |
| slepemapy | sakt | 5 | 0.771241 +/- 0.000470 | 0.791721 +/- 0.000312 | 0.771982 +/- 0.000405 | 0.791467 +/- 0.000193 | NA | NA | NA | strict_uniform_evaluator_prediction_audited_5of5 |
| slepemapy | simplekt | 5 | 0.794192 +/- 0.000419 | 0.801021 +/- 0.000172 | 0.797179 +/- 0.000459 | 0.801387 +/- 0.000174 | NA | NA | NA | strict_uniform_evaluator_prediction_audited_5of5 |
| statics2011 | acekt | 5 | 0.809976 +/- 0.002144 | 0.789348 +/- 0.001157 | 0.810983 +/- 0.001266 | 0.788128 +/- 0.001151 | NA | NA | NA | v16_strict_5of5 |
| statics2011 | akt | 5 | 0.827680 +/- 0.001827 | 0.799565 +/- 0.000801 | 0.830634 +/- 0.001569 | 0.801771 +/- 0.001135 | NA | NA | NA | v16_strict_5of5 |
| statics2011 | dkt | 5 | 0.820605 +/- 0.001642 | 0.796952 +/- 0.000490 | 0.821992 +/- 0.001682 | 0.797290 +/- 0.000304 | NA | NA | NA | v16_strict_5of5 |
| statics2011 | dkt+ | 5 | 0.825382 +/- 0.000578 | 0.796861 +/- 0.000967 | 0.827628 +/- 0.000828 | 0.797460 +/- 0.000826 | NA | NA | NA | v16_strict_5of5 |
| statics2011 | sakt | 5 | 0.791061 +/- 0.003281 | 0.785362 +/- 0.000825 | 0.796306 +/- 0.001778 | 0.788293 +/- 0.001640 | NA | NA | NA | v16_strict_5of5 |
| statics2011 | simplekt | 5 | 0.818371 +/- 0.000511 | 0.793985 +/- 0.002310 | 0.819544 +/- 0.001016 | 0.795547 +/- 0.003486 | NA | NA | NA | v16_strict_5of5 |
| statics2011 | ukt | 5 | 0.822622 +/- 0.001679 | 0.796341 +/- 0.002362 | 0.823934 +/- 0.001224 | 0.796999 +/- 0.001699 | NA | NA | NA | v16_strict_5of5 |

- 所有 `std` 均为五个 validation-fold 模型之间的样本标准差。
- `NA` 表示该校准指标尚未物化到最终矩阵，不等于需要重训；已有概率文件时可直接复算。
- EdNet aliases 与 KEENKT 继续硬排除；partial、OOM、smoke 和仅注册结果均不入表。
<!-- CODEX-KT-COMPLETE-MATRIX-20260728:END -->

2026-07-19 00:40 的统一只读审计从每折 manifest 重新核对出 `15` 个严格 `v3` 完整组合，全部满足 5/5 train returncode、checkpoint/config/prediction 文件记录、固定 runtime/profile hash、用户零重叠及 prediction 重算一致。ASSIST2009 corrected/collapsed 的 AKT 另由三份独立审计证明 5/5 通过。以上 16 行才是本轮新增到主表的行；本次审计 JSON SHA256 为 `feb9a4e43b89e023a1ac77991649e809befd33686d6be731640c3a57bd28aea0`。

2026-07-19 05:06 的 ASSIST2017 ACE-KT retry2 独立审计新增 1 行：五折 AUC `0.782030 +/- 0.000544`、ACC `0.725020 +/- 0.001395`。UKT retry2 尚未完成 5/5，未写入本表；不纳入任何不完整或 OOM 结果。

## 可引用边界

这张表现在可以作为当前论文的统一 pyKT 对比 baseline 主表。历史行的统一措辞必须是 `pyKT-based fixed-profile/reference baselines`；标为 strict `v3` 的行才可写固定官方 pyKT commit 下的严格运行。两类都不能写成 `paper-optimal`、`tuned-best` 或“复现各模型原论文最优参数”。

ASSIST2012 与 Junyi2015 是历史 172 结果的严格审计复用，不是本轮重训；ASSIST2015、NIPS Task 3&4、ASSIST2017 是 2026-07-16 固定协议结果。

ASSIST2017 使用隔离兼容 runtime 修复合法整数值的 decimal-form `usetimes` token，正文需要说明原 shared-runtime 失败历史与最终 repair driver 成功边界。

五套 `configs/legacy_common5_fixed_profiles/*.yaml` 已从 125 个 checkpoint 的解析配置生成，覆盖 25 个 dataset-model 组合并通过 25/25 命令 dry-run；它们用于在固定官方 commit 下启动未来的严格参数 replay，不能把未来 replay 与当前 legacy 数字混称为同一次运行。

## 证据路径

- 最终证据包：`<USER_HOME>\Documents\Codex\2026-07-14\228-172-3\outputs\remote_172_records\baseline_run_20260716_final`
- ASSIST2012/Junyi 审计：`outputs/remote_172_records/baseline_run_20260716_final/legacy_assist2012_junyi_audit_20260716/legacy_assist2012_junyi_common5_audit_20260716.json`
- 新跑结果 JSON：`outputs/remote_172_records/baseline_run_20260716_final/paper_metrics/one_step_<dataset>_<model>_fold<fold>_seed42.json`
- `v3` 13 个严格完整组合审计：`outputs/baseline_expansion_172_20260718_v3/audits/completed_groups_audit_20260718.json`，SHA256 `4a1264808ce50d1fa2a876723fad9f75ecc4fb891c616afb200cfb09c2aca9e0`
- ASSIST2009 corrected/collapsed + AKT 三门禁：`outputs/baseline_expansion_172_20260718_v3/audits/legacy_assist2009_corrected_akt/`
- 数据来源与许可总账：`<USER_HOME>\Documents\ObsidianVault\KT-papers\13-EduKTM-Baselines\数据集资产_原始位置与来源链接_20260706.md`
- 下载、预处理、全部基线训练与评估的唯一命令总账：`<USER_HOME>\Documents\ObsidianVault\KT-papers\13-EduKTM-Baselines\统一复现命令总账_数据预处理与全部基线_20260718.md`
- 2026-07-17 固定 pyKT 完整复现包：`<USER_HOME>\Documents\Codex\2026-07-14\228-172-3\work\pykt_reproducible_20260717`
- 125/125 prediction 全行复算：`work/pykt_reproducible_20260717/evidence/legacy_common5_prediction_recompute_20260717.json`
- 5/5 实际 loader 切分/时序审计：`work/pykt_reproducible_20260717/evidence/legacy_common5_loader_split_audit_20260717.json`
- 125/125 checkpoint/config/日志哈希审计：`work/pykt_reproducible_20260717/evidence/legacy_common5_checkpoint_audit_20260717.json`
- 五套历史固定参数 replay YAML 与直接命令：`work/pykt_reproducible_20260717/configs/legacy_common5_fixed_profiles/`、`work/pykt_reproducible_20260717/docs/LEGACY_COMMON5_PAPER_BASELINES.md`
- 38 模型、近年模型、数据与直接命令总账：`<USER_HOME>\Documents\ObsidianVault\KT-papers\13-EduKTM-Baselines\pyKT完整基线与2025-2026开源模型复现总账_20260717.md`

## 2025–2026 扩展状态

`ukt`、`cskt`、`robustkt`、`mockt` 已进入扩展审计清单；`denoisekt` 只在有 question ID 的兼容数据集（优先 `nips_task34`）运行，不再错误安排到 concept-only ASSIST2015。固定源码已有 38 个模型、38 份 default YAML、33 份 pyKT 公共调参结果 YAML；公共 CSV 没有指标/选择准则，因此这些配置不称“最优”。HCGKT 因 upstream 训练分支缺陷、FA-KT 因 172 缺 `mamba-ssm`，当前不进主矩阵。MoC-KT 项目页已核实为 <https://pykt.org/mockt>。

2026-07-17 代码级安全复核发现：固定 pyKT 的 `init_dataset4train` 会为 `dkt_forget`、`mtkt`、`fa_kt` 读取 held-out test 元数据来扩展时间间隔词表。新严格 runner 已阻断该行为并固定测试无关 gap domain 为 `128/128/16`；未经过该门禁的新结果不得混入本表。另确认 GKT 用 train+test 生成转移图、LPKT 用 train+test 生成时间词表/Q-matrix、DIMKT 的全 folds 难度含当前 validation 响应、RKT 缺可追溯 relation 生成器，因此四者当前硬拒绝。真实训练还必须在启动前复核 `repro_run.json` 和实际 loader 文件哈希。最终静态审计同时覆盖：33 份公共 YAML=`30` 个命令通过+`3` 个安全拒绝；38 份默认 YAML=`33` 个命令通过+`5` 个安全拒绝；五套历史固定参数 replay YAML=`25/25` 个 dataset-model 命令通过；三类多数据集结果门禁=`125/125` prediction、`125/125` checkpoint/config、`5/5` loader split 全部通过；数据执行契约=`8/8` 下载/解包/预处理器通过，近年模型执行契约=`9/9` 源码/训练入口/参数证据通过。

新模型只按 dataset-model 逐行晋级：已通过 5/5 的 UKT/ACE-KT 行已写入上表；其余 dry-run、空缺、失败或单折结果继续排除。完整直接命令只见 [[13-EduKTM-Baselines/统一复现命令总账_数据预处理与全部基线_20260718|统一复现命令总账]]。

## 2026-07-18 UKT / ACE-KT 补跑状态

172 已启动 `baseline_expansion_8datasets_20260718_v3`，覆盖 `assist2009_corrected_collapsed`、`assist2012`、`junyi2015`、`assist2015`、`nips_task34`、`assist2017`、`slepemapy`、`statics2011`。GPU0 补 common-five 缺失矩阵，GPU1 运行 UKT 与 ACE-KT 五折队列。

- UKT：AAAI 2025。ASSIST2009 使用作者示例；ASSIST2015、NIPS34、STATICS2011 使用 pyKT 公布逐折 tuning 行；其余四数据集使用 upstream default fallback。以上都不称 `paper-optimal`。
- ACE-KT：AISTATS 2026，作者仓 commit `3fd20d29d3074c8c082eadf6ea3197d8fdd295db`。八数据集均使用作者默认配置；论文同数据集只有 ASSIST2009、NIPS34，其余为跨数据集扩展，不称论文最优参数。
- 固定 pyKT commit：`f766468f1d3083f737e5fde22a17a08de0d52552`；八个原件 bytes/SHA256、实际 loader 文件及 train/test 用户零重叠已由 `prepare_manifest.json` 重新核对。
- 全矩阵 dry-run：30 tasks、150 fold commands、0 failures。
- pyKT `wandb_predict.py` 的非 RKT 位置参数问题已由外部 evaluation adapter 修复；训练 runtime 不变。恢复评估现已验证 prediction 落盘且全局 AUC/ACC 重算与 pyKT 报告一致。

历史快照 `2026-07-19 00:38` 曾为 `running`；当前状态已是 `post-cutoff frozen`。该快照的 `78 complete`、`2 trained_pending_or_failed_eval`、`60 missing`、`10 train_failed` 仅作为过程证据保留。07:55 后父调度器保持暂停，UKT 未达 5/5 的部分和所有单折/失败证据均不进入主表。完整记录见 [[13-EduKTM-Baselines/172八数据集UKT与ACE-KT补跑记录_20260718|172 八数据集 UKT 与 ACE-KT 补跑记录]]。

## 2026-07-19 07:12 严格审计增量

- `slepemapy/dkt` 已通过独立 v3 严格审计，5/5 折均满足训练返回码、checkpoint/config、prediction 全行重算和固定 held-out test 用户零交集门禁；主表新增 one-step test AUC `0.783850 +/- 0.000288`、ACC `0.797203 +/- 0.000081`（sample standard deviation；不报告方差）。
- 审计报告：`outputs/baseline_expansion_172_20260718_v3/audits/completed_groups_audit_20260719_0712.json`，SHA256 `11793BD3A2B408252D60243D09310A2D0A20D8F69054582A551E06D1E3D24701`；全量状态为 `21` 组、`16` 组完整、`5` 组未完成、`10` 个历史硬失败折。
- 本地 fold4 证据目录：`outputs/baseline_expansion_172_20260718_v3/slepemapy/dkt/fold4/`；summary、逐折日志和审计文件均已保留。Window 指标只留在 prediction/eval artifact，不进入主表。
- UKT retry2 在 07:12 仍为 2/5（fold2 评估中），没有 mean/std 或主表行；v3 DKT+ fold0 已在 07:08 前启动。截止门禁脚本 `work/enforce_0800_no_new_training_172_20260719.sh`（SHA256 `2A67F585AE0E97E96CD9FFE24699173DE07E22E914C8319ED62E392A12910746`）在 07:55 暂停两个父调度器，防止 08:00 后启动新训练。

## 2026-07-19 08:01 后续队列状态

- UKT retry2 的只读严格审计为 `3/5`：fold0--2 的 prediction/checkpoint/split 门禁通过，fold3 仅有训练产物，fold4 未启动；因此没有新增 UKT 行或三折统计量。
- 截止守护日志已证明 07:55 暂停父调度器、08:00:11 保持暂停；之后只允许只读审计、证据同步和状态报告。正在运行的 DKT+ fold0 评估属于截止前已启动子进程，不构成新训练任务。
- 08:02 v3 只读审计仍为 16 个完整组；`slepemapy/dkt+` 为 `0/5`，不新增结果行。审计 SHA256：`D4810FA4FC4CD3415BC5C22992B67F494D2EE19B1771E3B0507B5A385730C58D`。
- 截止前启动的 `slepemapy/dkt+` fold0 于 08:53 完成，但 09:58 审计仍仅为 `1/5`、`eligible=false`；主表和 CSV 没有新增 DKT+ 行。

## 2026-07-19 复现包结构补充（无新训练）

- 38 个固定 pyKT 模型的默认参数与全部公开逐数据集/逐折参数已展开到 `work/pykt_reproducible_20260717/docs/PYKT_MODEL_HYPERPARAMETERS.md`；完整机器可读值在 `work/pykt_reproducible_20260717/configs/models/HYPERPARAMETER_ATLAS.json`。公开 tuning CSV 没有指标或选择规则，仍不得称为论文最优参数。
- ACE-KT 已补充独立外部 overlay 契约：`configs/external_models.yaml`、`configs/models/external_overlays/acekt_author_defaults.yaml`、`scripts/prepare_external_overlay.py`。构建器在本地按作者 commit 获取并校验源码，不在复现包中再分发无许可证源码；作者默认 profile SHA256 为 `A02C040CDBD7CEDCB2122FF83E4937207C11117CE0334FEAF70207899B88DFDC`。
- ACE-KT 四个已入表 5/5 组合的便携证据摘要为 `work/pykt_reproducible_20260717/evidence/acekt_overlay_and_results_20260719.json`；本次仅补文档、配置和静态审计入口，没有启动新训练，也没有改变当前 43 行结果表。

## 2026-07-19 14:59 近年模型参数证据审计

- 复现包已为 9 个固定 pyKT 近年模型、ACE-KT overlay 和 post-pin FlucKT 登记机器可读 `paper_selected_config_status`，共 11 条证据；DenoiseKT 论文页 6 的固定协议、Bayesian 搜索域和 PDF SHA256 已核验。
- LeFoKT-AKT AAAI 2025 官方 PDF 第 5 页的 5 折、Adam、200 epoch、patience 10、Bayesian 搜索域及 PDF SHA256 已核验；论文未公开逐数据集/逐折最终选中值，因此只新增 `paper-search-space` 证据，不新增结果行，也不称 `paper-optimal`。
- 对应机器索引为 `work/pykt_reproducible_20260717/evidence/recent_model_parameter_evidence_20260719.json`，SHA256 `8DB706CAC0C21A9EDC5DED1B8F995F4394F159F0D5FEABFC8555A87FFA875A5F`，由两个模型注册表及 FlucKT upstream delta 自动生成并通过 11/11 profile SHA256 门禁。
- PaperGraph 近年模型摘要：`近年模型参数证据与复现边界_20260719.md`。
- 这次只更新参数证据、命令总账和静态审计，不新增任何 dataset-model 结果。主表仍严格为 43 行；ASSIST2017/UKT `3/5` 与 Slepemapy/DKT+ `1/5` 继续排除。
- 公开 tuning 行、作者默认值和论文搜索空间均不自动等于 `paper-optimal`；只有未来 validation-only 搜索通过独立门禁后才可使用 `validation-tuned` 标签。EdNet、KEENKT 及所有泄露风险结果不进入本表。

## 2026-07-19 17:04 截止后只读复核

- 172 的 v3 GPU0/GPU1 父调度器 PID `90875`、`95845` 与 ASSIST2017 UKT 五折父进程 PID `152746` 仍为暂停态；ACE-KT 外层 PID `128206` 和 UKT 接续调度器 PID `129569` 已退出。未恢复或启动任何 KT 任务。
- ASSIST2017/UKT 仍为 `3/5`，Slepemapy/DKT+ 仍为 `1/5`，均不满足 checkpoint、prediction、one-step 重算和完整 5 折门禁。因此 Markdown/CSV 主表继续保持 `43` 行，未新增部分折 mean/std。

## 2026-07-22 最终静态包审计（无新训练）

- 2026-07-22 使用完整固定 pyKT 源码根目录 `work/vendor/pykt-toolkit-f766468-full/pykt-toolkit-f766468f1d3083f737e5fde22a17a08de0d52552` 重跑全量审计；`AUDIT_REPORT.json` 同时覆盖 38/38 pinned 模型、8/8 数据集、9/9 直接近年模型、ACE-KT overlay、FlucKT 隔离 post-pin runtime、安全拒绝、训练文档变量闭包、18/18 文档脚本引用闭包和 Window v13 `43/43`，结果为 `status=pass`、`failures=[]`，导出包 SHA256 `261AE34DC70BBE81EFA3A26F218CAA50DCC06C959E923DDC2B7B82698BC669AD`。
- 审计覆盖 38 份默认 YAML、32 个模型的 33 份 pinned pyKT 公共 tuning YAML、FlucKT post-pin 公共逐折 profile、6 份论文/作者搜索空间、8/8 数据集执行契约、11/11 近年模型参数索引、1 个 ACE-KT overlay、125/125 历史 prediction 与 125/125 checkpoint/config 门禁；FlucKT 的四文件 MIT overlay 实际构建为 147-file code tree。该 2026-07-22 快照只完成 NIPS fold0 命令 dry-run；其结果状态已由下方 2026-07-23 严格 5/5 审计取代。转移数据门禁锁定 3/3 旧 manifest，5/5 合成测试覆盖每折 cache purge 与串行锁。
- `BUNDLE_SHA256.json` 已纳入最新原始描述/URL 门禁、FlucKT builder/overlay/atlas、自包含训练命令及脚本引用门禁、结果表校验器、transferred-data registry/runner 与 16 份 Window 审计证据，现为 261 文件，SHA256 `6E30694AF2ADE01427D2C56E13AB131B8EA0B56B126AA49A11A26CCDB7A1CFE9`；独立 `--verify` 返回 `status=pass`、`failures=[]`。Manifest 自身按设计不纳入其文件列表。
- 截至该 2026-07-22 快照，RobustKT（WWW 2025）与 FA-KT（WWW 2026）只新增正式论文搜索空间证据，FlucKT 也只有隔离 runtime 和可执行逐折 profile；当时主表为 43 条完整 `5/5`。2026-07-23 的增量严格审计随后接纳 FlucKT/NIPS34 等四行，当前状态见文末。















## Window metric supplement (2026-07-22)

- This synchronizer is read-only: it accepts only groups that passed independent 5/5 prediction recomputation and artifact-hash audits; it does not load a model or start GPU work.
- Complete Window fields are available for 43/43 main-table groups; no groups remain `NA`.
- Window values use the arithmetic mean and sample standard deviation over folds; variance is not reported.
- The raw per-fold paths, SHA256 values, and aggregate inputs are stored in workspace `outputs/window_metrics_audit_20260722_v13.json` and private-repository `results/window_metrics_audit_20260722_v13.json`.
- `assist2009_corrected_collapsed/sakt` has a large recorded window-fold spread; it is retained verbatim and should be discussed rather than silently discarded.




## 2026-07-23 增量审计

- 当前主表为 52 个完整 5/5 dataset-model 组合：47 个来自 v16 统一 pyKT 聚合，ASIKT 为单独审计接入的机制匹配专项行，NIPS34/UKT、NIPS34/ACE-KT、NIPS34/DKVMN 与 NIPS34/DeepIRT 由 2026-07-26 严格增量审计接入；每行均有 one-step AUC/ACC 和独立 Window AUC/ACC。
- 本次增量只接纳通过 checkpoint、用户无泄露切分、one-step 独立重算及 Window 5/5 独立重算的组合；不接纳 KEENKT、EdNet 或部分折结果。
- 新增行的逐折 JSON、预测文件哈希和审计脚本 SHA256 记录在对应 `outputs/*_audit_20260723` 目录及私有复现包中。
- 当前基础严格聚合为 `window_metrics_audit_20260723_v16.json`，SHA256 `9AC0DA5BF1F258F0F7C6D7DBD237A8C802F3503BA91E976F64A7D0FC15051DFA`，47/47 通过且无缺失组；ASIKT 另由 `asikt_author_nips_task34_light_summary_20260725.json` 与 `asikt_author_prediction_recompute_audit_20260725.json` 接入，独立复算为 5/5 通过。Junyi/UKT 复用五个既有 172 checkpoint，在 228 完成 Window 评估，没有重训。
- 228 的 DTransformer、FoLiBiKT、QIKT、UKT、ACE-KT 兼容运行时及 CogAKT-GRN-UNC-Dual 仅完成 CPU smoke；未启动真实训练，也没有生成可入表 AUC/ACC。

## 2026-07-24 172 GPU0 strong-baseline run

- 2026-07-24 02:05:15 +0800 已在 172 (`172.25.114.0`) 启动单卡 GPU0 正式队列：`DTransformer -> FoLiBiKT -> QIKT -> UKT`，目标数据集为 `nips_task34`，每个模型 5 folds 顺序执行。
- 启动前确认 GPU0 无 CUDA compute 进程；GPU1 当时由其他用户任务占用，本队列仅设置 `CUDA_VISIBLE_DEVICES=0`。
- 运行入口为 `/tmp/run_strong_baselines_single_gpu_172_20260724.sh`，中性 Python 为 `/tmp/kt-baseline-runtime-20260724/bin/python`；`nvidia-smi` 的 compute-app `process_name` 显示 `/tmp/kt-baseline-runtime-20260724/bin/python`，不会直接暴露个人 home Python 路径。`ps` 仍保留真实命令行与用户身份用于审计。
- 脚本门禁：每个 fold 启动前检查 GPU0 是否空闲；用户更新后的本轮守门为 `2026-07-24 06:55:00 +0800`，严格早于 07:00 禁止新任务边界；任一 fold 非零退出即停止队列；部分折、失败折和 smoke 结果一律不进入主表。
- 远端日志根目录：`<REMOTE_HOME>/EduKTM_Baselines/results/strong_baseline_20260724_172_retry2/queue_logs/`；主日志 `master.log`。截至 2026-07-24 12:10 +0800，172 已超过 07:00 no-new-task 边界，GPU0 无 CUDA compute 进程，匹配队列/训练/预测进程为空；本轮不再启动真实 GPU 实验。
- 当前 DTransformer 进度：`fold0`、`fold1`、`fold2` 的 `one_step_metrics.json` 均为 `returncode=0`、`predictor_returncode=0`，且均含 `testauc/testacc/window_testauc/window_testacc` 与独立重算 AUC/ACC；`fold3` 于 06:42 启动后训练阶段 CUDA OOM，`returncode=1`，队列停止，`fold4` 未启动。因此 DTransformer 仍为 3/5 完成，不能进入主结果表；下一授权窗口可在 GPU0 空闲时用相同 profile 重新跑完整 5/5，或另行标注低 batch OOM-fallback profile，二者不可混合统计。

## 2026-07-24 ASIKT mechanism-matched baseline 接入状态

- 按 A2G-MambaKT 的核心机制匹配原则，专门基线优先选择 ASIKT（SIGIR 2025，DOI `10.1145/3726302.3730012`），而不是只含 Mamba backbone 的普通效率基线。
- 作者仓库 `https://github.com/mrsser/ASIKT` 已通过 GitHub mirror 获取并固定到 commit `db8eb4cac6bcf9138b2179e8f279b93a49de2832`；许可证为 Apache-2.0。登记文件：`work/pykt_reproducible_20260717/configs/external_models.yaml`。
- 新增 author-source overlay：`scripts/prepare_asikt_author_overlay.py`、`configs/models/external_overlays/asikt_author_defaults.yaml`、`overlays/asikt_author/pykt/models/asikt_author_adapter.py`。作者默认参数记录为 `batch_size=48`、`lr=5e-4`、`d_model=128`、`d_ff=1024`、`dropout=0.2`、`n_block=2`、`n_head=8`、`max_iter=300`，但不称 paper-optimal。
- 泄露边界：作者 `data_pre.py` 的 Exercise Weight Matrix 使用全局统计生成，本项目 overlay 改用严格因果 mask adapter，标签为 `author-source-causal-mask-adapter`；未来只有完成同 split、同评测、5/5 checkpoint/prediction/AUC/ACC/Window 审计后，才能进入主表。
- 172 准备状态：已同步 `asikt_author_adapter.py`、`asikt_author_defaults.yaml`、`prepare_asikt_author_overlay.py` 至 `<REMOTE_HOME>/EduKTM_Baselines/reproducibility/pykt_reproducible_20260717/`；`prepare_asikt_author_overlay.py` 当前 SHA256 为 `3a2495d3a184d1f496598c2798681fc1cc0fd17e1041db5b690e0c4ed2150aa0`，`asikt_author_defaults.yaml` SHA256 为 `2ac0b08783be9d70e1444400c8ed5096e2dac2779cd16cdcb3865eb026a15660`，通过 `py_compile`。CPU/文件级构建已成功生成 `/tmp/kt-pykt-asikt-author-runtime-20260724`，manifest 为 `/tmp/kt-pykt-asikt-author-runtime-20260724/asikt_author_overlay_manifest.json`，purpose=`hash_pinned_asikt_author_source_pykt_causal_mask_adapter`，code tree SHA256=`e4f0750658c7d0fe37ec7a857fa4f3c9f7903a2a887004ebb3ad05c7f67b0e28`，single-fold runner SHA256=`a28595fa8592e9c183b19add86239d6ce3fa583cc7b67fc08d826aa6fef7fadd`，entry wrapper SHA256=`87ab3aa7c4560fb19a504c9050cc9e46e937d1bf9a80032d1f4175bf423eb4e4`。`run_profile_asikt_author.py --dry-run` on `nips_task34/fold0` 已通过并展开完整命令；这只是可运行准备，不是训练结果。
- 复现包登记已同步：`configs/external_models.yaml` SHA256=`c266ea75c53aab67602e72fc192193a0c4132eae0a141cbde48a8bcc546cb643`，`docs/TRAINING_COMMANDS.md` SHA256=`25fb95de5c8b2c9885cc8982d2255b10570f8a5499a4fbda809262926582d241`；本地与 172 均通过 YAML 解析，且关键文档中未检出旧 ASIKT profile hash/runtime hash 残留。
- 交付件覆盖审计已补：`evidence/delivery_coverage_audit_20260724.json` SHA256=`1994ec72ef8cd8aa24b50112fce241d9704fe8613938756f3000404942a71922`，记录 baseline YAML、2025–2026 模型、数据集、预处理脚本、训练命令、排除项，以及当前 DTransformer/ASIKT 不可入表边界；该审计只证明交付件覆盖与可运行准备，不是实验结果。
- 早先的 `asikt_mechanism_matched` 仅保留为 fallback smoke adapter，不作为 ASIKT 作者代码复现结果。

### 2026-07-25 172 GPU0 ASIKT fold0 启动与 runtime 修复

- 2026-07-25 03:03 +0800 在 172 确认未过 07:00 no-new-task 边界，且 GPU0 无 CUDA compute 进程；`nvidia-smi -i 0` 显示 RTX 3090、0% util、29/24576 MiB。按用户授权仅启动单卡 GPU0，不做 5 折自动循环，避免 07:00 后自动拉起新 fold 的边界风险。
- 为避免 `nvidia-smi` 暴露个人 home Python 路径，补建 `/tmp/kt-pykt-asikt-author-runtime-20260724/bin/python` 入口并用它启动；真实训练启动后 `nvidia-smi` compute app 显示 `1156274, /tmp/kt-pykt-asikt-author-runtime-20260724/bin/python, 6152`，未显示 `<REMOTE_HOME>/...` Python 路径。
- 第一次 fold0 启动到 `/tmp/asikt_author_gpu0_20260725` 后快速退出，`train.log` 报 `ModuleNotFoundError: No module named 'pykt.models.ASIKT'`；未占用 GPU、未产生可用结果。原因是 ASIKT runtime 漏拷作者 `ASIKT.py/transformer.py/mamba2.py/run.py` 到 `pykt/models/` 顶层，且 `mamba2.py` 的 `from run import device` 需要改为 package 相对导入。
- 已在 172 runtime 补齐作者源文件并将 `mamba2.py` 改为 `from .run import device`，import smoke 通过；随后 03:09 +0800 重新启动到 `/tmp/asikt_author_gpu0_20260725_r2/nips_task34/asikt_author/fold0`。当前 wrapper PID `1156271`，训练子进程 PID `1156274`，`train.log` 已进入真实训练，epoch 1/2/3 validauc 分别为 `0.7507/0.7653/0.7840`，`testauc/testacc/window_testauc/window_testacc` 仍为训练中占位 `-1`。
- 本地复现包已固化修复：`scripts/prepare_asikt_author_overlay.py` 现在复制作者文件到 `pykt/models/` 顶层，并对 `ASIKT.py/mamba2.py/transformer.py` 做相对导入 patch；`overlays/asikt_author/pykt/models/mamba2.py` 同步改为相对导入。新的 SHA256：`scripts/prepare_asikt_author_overlay.py`=`86b236999025d5093e7798607ae59d01b97ce0290e10f741e1098faf6d682299`，`overlays/asikt_author/pykt/models/mamba2.py`=`42247b5a5ba824096b8b3b18eea694e2827ad31b43f4e591fc65c69467d89676`。
- 本地 `AUDIT_REPORT.json` 已重跑并保持 `status=pass, failures=[]`，SHA256=`ac27eaca2a304592b9890b99076c56fc590a4a73b761e9e1d80a27b3910fa7ea`；`BUNDLE_SHA256.json` 已重建且 `--verify` 返回 `status=pass, failures=[]`，SHA256=`e44b608de90f39c83bd7a2130980ae9a50f25a658d55650dbeb25cb2a9018a5b`。172 复现包已同步上述两个文件、builder 与 overlay patch。
- 当前 ASIKT 仍不可入主表：截至 2026-07-25 03:35 +0800，`nips_task34/fold0`、`fold1`、`fold2` 已完成训练 checkpoint，`fold3` 正在 GPU0 训练，`fold4` 未有效完成；尚无 5/5 完整 fold、prediction/hash、one-step AUC/ACC 和 Window AUC/ACC 审计。后续必须补齐有效 fold4，并完成全量 evaluation/audit 且全部 returncode=0 后才能统计。

### 2026-07-25 03:35 +0800 ASIKT 多 GPU 授权调整与回退

- 用户 03:31 授权“当前可以使用所有 GPU，07:00 后不能启动新任务”。172 只有两张 RTX 3090；当时 GPU0 正在跑 ASIKT fold3，GPU1 曾短暂空闲，因此 03:33 启动 fold4 到 GPU1，入口仍为 `/tmp/kt-pykt-asikt-author-runtime-20260724/bin/python`。
- 用户随后要求“不要使用 gpu1 先了”。已立即停止 GPU1 上的 fold4 wrapper/训练子进程，并确认 `nvidia-smi --query-compute-apps` 只剩 GPU0 的 ASIKT fold3：`1159106, /tmp/kt-pykt-asikt-author-runtime-20260724/bin/python, 6182`。GPU1 不再有本项目进程。
- 为防止半成品阻塞后续补 fold4，已将无效目录从 `fold4` 挪到 `fold4_aborted_gpu1_20260725_0334`，日志改名为 `fold4.out.aborted_gpu1_20260725_0334` / `fold4.err.aborted_gpu1_20260725_0334`。该 fold4 不计入任何训练完成数，也不参与汇总。
- 已停止 `run_remaining_guarded.sh` watcher，避免 fold3 结束后自动启动 fold4。当前有效状态：fold0/fold1/fold2 已完成训练 checkpoint 但没有 prediction/test/window 指标；fold3 正在 GPU0 训练；fold4 未有效启动。ASIKT 仍不可入主表。

### 2026-07-25 ASIKT 五折完成与投稿效率门禁

- ASIKT `nips_task34` fold0–4 已全部完成训练与 concept-level one-step/Window evaluation；固定 held-out test、五个 validation-fold 模型、seed=42，训练/评估没有把 test 用于结构选择。
- 五折汇总：one-step AUC `0.798763 +/- 0.001047`，ACC `0.728213 +/- 0.001147`；Window AUC `0.800982 +/- 0.001012`，ACC `0.729957 +/- 0.001061`。这里的 `std` 为 fold 间 sample std；该行是机制匹配基线，不宣称论文最优参数。
- 独立 prediction recomputation 已通过：10/10 prediction 文件逐行重算 AUC/ACC，均与 evaluator JSON 在 `1e-12` 内一致；同时记录 NLL/Brier/ECE-15。原始文本中的裸 `nan` 仅出现在逐序列辅助 AUC 字段，审计器将其规范化为 `None` 后解析，标签和概率未改写。
- ASIKT 的 Window question-level fusion 本轮明确标记 `NA`，不能把 concept-level Window 数字写成 question-level 结果；如正文需要 question-level 结论，须另跑完整 fusion 并单独审计。
- 2026-07-25 的 A2G NIPS34 `15 runs / 3 seeds per fold` 和早先 `20/25` 数字仅保留为历史快照，均已由文末 2026-07-26 的 `25/25` 运行快照取代；ASIKT 仍是单 seed 五折，不能写成 matched-seed 严格显著性比较。
- 投稿效率门禁必须单独满足：参数总量/可训练参数、训练 wall-clock（每 epoch 与整折）、峰值 CUDA allocated/reserved 显存、推理 p50/p95 延迟、tokens/s 或 interactions/s，并在相同硬件、batch、精度和序列长度下对照 A2G、参数匹配 causal attention、canonical Mamba。A2G 现有 submission gate 的 `CANONICAL-EFFICIENCY` 已明确要求长度 `200/512/1024/2048` 和三类 comparator；缺少这些字段不能以“精度基线已补完”代替投稿证据。

## 2026-07-26 NIPS34 UKT / ACE-KT / DKVMN / DeepIRT 严格增量审计

- `nips_task34/ukt` 与 `nips_task34/acekt` 均完成 5/5 checkpoint、config、prediction、one-step AUC/ACC 重算、concept-level Window 重算、学生级切分与校准审计；所有门禁均为 `pass`。
- UKT one-step 校准为 NLL `0.525502 +/- 0.000684`、Brier `0.177506 +/- 0.000290`、ECE-15 `0.006892 +/- 0.003647`。
- ACE-KT one-step 校准为 NLL `0.529646 +/- 0.000561`、Brier `0.178427 +/- 0.000145`、ECE-15 `0.013926 +/- 0.003207`。
- DKVMN one-step 校准为 NLL `0.564830 +/- 0.000147`、Brier `0.193267 +/- 0.000088`、ECE-15 `0.009025 +/- 0.001123`；DeepIRT 为 NLL `0.566848 +/- 0.001386`、Brier `0.193802 +/- 0.000447`、ECE-15 `0.009170 +/- 0.001811`。
- AUC/ACC 是同类 KT/Mamba 论文的常见主表指标；ECE 并非普遍硬要求，但 NLL/Brier/ECE-15 已作为校准补强保留。已有概率文件足以重算这些指标，不需要为 ECE 重训模型。
- 完整矩阵现为 `52` 个 5/5 组合，校准完整组合为 `5`；common-five 覆盖为 `36/40`。尚缺的四组全部位于 Slepemapy：DKT+、AKT、SimpleKT、SAKT；运行中或部分折结果在五折与全部审计通过前仍不入表。DKVMN 补足 memory 类代表，DeepIRT 只算 IRT-family 神经基线，不能替代经典 BKT/IRT。
- NIPS34 上 A2G 当前的 AUC/ACC、NLL 与 Brier 高于 ASIKT，但 ECE-15 更差；同时 AUC 低于 CSKT、FlucKT 与 UKT。因此只能写“主要判别指标及两个 proper scores 优于机制匹配基线 ASIKT，ECE 未占优”，不能写“全面超过 ASIKT”“全面超过所有强基线”或“全面 SOTA”。
- 发布产物只允许使用 `<USER_HOME>`、`<REMOTE_HOME>` 与 `<REMOTE_USER>`；原始账户路径仅保留在内部审计副本，不进入论文、公开 JSON、CSV 或 Markdown。

<!-- CODEX-A2G-BASELINE-20260726:START -->
## 2026-07-27 A2G-MambaKT baseline and efficiency gate (60 complete groups; common-five 40/40; NIPS extensions 3/3)

### Audited baseline coverage

- Complete 5/5 dataset-model groups: `60`.
- Common-five coverage: `40/40`; missing: `none`.
- Calibration-complete groups: `13`. ECE is supplementary rather than a universal KT-paper requirement, but NLL/Brier/ECE-15 are retained because archived probabilities permit exact recomputation without retraining.
- Classical coverage now includes BKT; recurrent/memory coverage includes DKT, DKT+, DKVMN and DeepIRT; attention/modern coverage includes SAKT, AKT, SimpleKT, CSKT, FlucKT, UKT, ACE-KT and ASIKT where strict 5/5 evidence exists.
- NIPS strict comparator coverage is `16` models. Requested extensions still excluded from results: `none`. Registry/config/smoke coverage is not result completion.
- EdNet aliases and KEENKT are hard-excluded; the admitted matrix contains zero excluded rows.

### A2G versus ASIKT on NIPS Task 3&4

A2G has `25/25` NIPS fold/seed runs; missing seeds: `none`. The fold-first point estimates are complete; learner-aware paired uncertainty remains a separate evidence artifact.

| Metric | A2G | ASIKT | Directional result |
|---|---:|---:|---|
| auc | 0.805565 | 0.798763 | A2G better |
| acc | 0.731878 | 0.728213 | A2G better |
| window_auc | 0.807763 | 0.800982 | A2G better |
| window_acc | 0.733677 | 0.729957 | A2G better |
| nll | 0.529597 | 0.535272 | A2G better |
| brier | 0.178807 | 0.180993 | A2G better |
| ece_15_bin | 0.028316 | 0.021694 | ASIKT better |

A2G is better on AUC, ACC, Window AUC/ACC, NLL and Brier, but worse on ECE-15. Therefore it does **not** completely surpass ASIKT. Its complete NIPS AUC rank is `5`; no universal SOTA claim is admitted.

### Learner-aware paired uncertainty

The required repeated-fit gate covers `9/9` dataset/prefix comparisons with `10,000` learner-bootstrap replicates. Test-set learner-balanced AUC differences (A2G minus admitted baseline) are NIPS `+0.006113` (95% CI `+0.004757` to `+0.007543`), Slepemapy `+0.018312` (`+0.016827` to `+0.019842`), and Assist2015 `-0.029894` (`-0.033325` to `-0.026603`). The negative Assist2015 learner-balanced effect blocks a cross-dataset universal superiority claim even though its global interaction-level point estimate is positive.
Paired candidate aggregation is `five seed probabilities averaged inside each validation fold`. These intervals therefore describe an ensemble-level learner effect, not single-seed or single-checkpoint superiority.
Junyi Window paired uncertainty is explicitly `NA` because the admitted baseline NPZ lacks Window labels/scores; it is not silently imputed.

### Efficiency evidence

| Model | Total params | p50 (ms) | p95 (ms) | Tokens/s (p50) | Peak alloc (MiB) | Scope |
|---|---:|---:|---:|---:|---:|---|
| a2g_mambakt | 3,028,492 | 23.125 | 23.279 | 323587.9 | 300.9 | same real NIPS batch |
| asikt_author | 5,353,844 | 25.351 | 25.400 | 295173.0 | 1888.6 | same real NIPS batch |
| ukt | 1,266,721 | NA | NA | NA | NA | params only |
| akt | 5,990,422 | NA | NA | NA | NA | params only |

A2G-only descriptive NIPS evidence covers `25/25` RTX 4090 D runs: full-fold training duration `283.328 +/- 27.558` s, mean training peak allocated memory `5.799` GiB, one-step throughput `467558.4` interactions/s, and Window throughput `9549.9` interactions/s.
The same-batch comparison uses one persisted real NIPS test batch (`batch=48`, `7483` valid tokens), FP32, 10 warmups and 50 synchronized repetitions on one RTX 4090 D. A2G p50 is `23.125` ms versus ASIKT `25.351` ms. This is a forward-only microbenchmark; same-harness ASIKT training time and end-to-end loader latency remain `NA`.
GFLOPs remains `NA`: no operator-complete counter has been verified for the custom selective-scan path, so partial matmul/conv profiler counts are not admitted.

### Submission boundary

- A2G submission gate: `fail` (`10/14` checks pass).
- Open checks: `ITEM-DROPOUT-ABLATION, MISSING-ALLOWED-UNTOUCHED-CONFIRMATION, OPEN-ISSUES, VENUE-AND-TEMPLATE`.
- EdNet exclusion is a passing boundary, not confirmation evidence. A separate allowed, never-used, non-leakage confirmation remains required while that gate is open.
- Until ITEM-DROPOUT-ABLATION passes, no causal contribution claim for item dropout is admitted.

### Reproducibility and host gates

- pyKT official `main` was rechecked at commit `3aabdc3cab06a57e745518b070d31bcbb823af2c`: 39 standard registry keys. The primary runtime remains the pinned 38-key commit, with FlucKT supplied by a separately hashed post-pin overlay.
- The pinned package contains 38 default YAMLs and 33 standard public-selected YAMLs. Public tuning CSVs omit metrics/selection rules, so these are not labelled paper-optimal; models without released tuning rows retain source defaults.
- 172 cutoff audit: `pass`, late model roots `0`.
- Current execution policy: `pass` / `kt_experiments_228_only`; 228=`launch_allowed` with no time cutoff, 172=`launch_blocked`, 127=`launch_blocked_by_existing_228_only_policy`.
- Public workspace privacy audit: `pass`, scanned files `1190`, failures `0`.
- Human-facing summaries must cite artifact basenames or placeholder paths; account-bearing absolute hyperlinks are forbidden.
- Delivery packaging audit: `complete`, remaining packaging gates `0`; this does not close the separate scientific submission gates listed above.
- The public reproducibility tree excludes author Git repositories, checkpoints, dataset bytes and publisher PDFs; its independent JSON/YAML/Python/hash and username audits pass.

### Evidence

- `outputs/complete_baseline_matrix_20260726.json`
- `outputs/a2g_vs_asikt_nips_final_20260726.json`
- `outputs/a2g_efficiency_comparison_20260726.json`
- `outputs/a2g_nips_efficiency_summary_final_20260726.json`
- `outputs/a2g_asikt_samebatch_efficiency_20260726.json`
- `outputs/a2g_paired_comparison_summary_20260726.json`
- `outputs/a2g_submission_gate_20260726.json`
- `outputs/delivery_coverage_audit_20260726.json`
- `outputs/model_launch_cutoff_audit_20260726.json`
- `work/internal/kt_execution_policy_final_audit_v2_20260726.json`
- `outputs/public_repro_bundle_audit_20260726.json`
<!-- CODEX-A2G-BASELINE-20260726:END -->

<!-- CODEX-A2G-ABLATION-PARTIAL-20260728:START -->
## 2026-07-28 A2G 核心消融中间审计（20:34 快照，非最终）

当前只能回答 no-item-dropout 这一项，不能回答“所有模块是否都有用”。已完成 `50/75` 对齐比较，即五个主数据集、五折、seed `7/42`；seed `3407` 仍由既有唯一 228 队列继续。`ΔAUC = Full - no-item-dropout`，正值才表示完整模型中的 item dropout 有利。

| Dataset | 已完成 runs | 数值正向折数 | mean Δ one-step AUC | mean Δ Window AUC | 当前解释 |
|---|---:|---:|---:|---:|---|
| junyi2015 | 10 | 3/5 | -0.000110 | -0.000123 | 方向混合；不能说稳定正提升 |
| assist2015 | 10 | 1/5 | -0.000000 | -0.000000 | 约 1e-9，实质等价 |
| nips_task34 | 10 | 5/5 | +0.001269 | +0.001249 | 5/5 折正向 |
| assist2017 | 10 | 0/5 | -0.001293 | -0.001329 | 0/5 折正向；完整模型更差 |
| slepemapy | 10 | 5/5 | +0.001197 | +0.001207 | 5/5 折正向 |

- 单 run 仅 `28/50` 为正；双 seed 折均值仅 `14/25` 为正。
- 50/50 对 checkpoint、config、training resource、prediction 和 result 均齐全；协议、数据哈希、样本数与正例率匹配，实质配置差异只有 `item_residual_dropout: 0.4 -> 0.0`。
- `w/o SSM`、`w/o Attention`、`w/o split-boundary` 当前均为 `0/75`，只有草案和静态 preflight，没有结果 artifact。
- 因此当前结论是：item dropout 只在 NIPS 与 Slepemapy 显示一致正向证据，在 Junyi/ASSIST2015/ASSIST2017 不成立；更不能外推为“各模块均正提升”。
- 这仍是 `50/75` 方向性中间证据，不用于最终消融表、模块选择或因果贡献声明。正式结论必须等待 75/75、其余核心消融、冻结协议和相应统计门禁。
- 机器可读快照：`A2G_no_item_dropout消融中间快照_20260728.json`；来源仅使用 `<REMOTE_HOME>` 占位路径。
<!-- CODEX-A2G-ABLATION-PARTIAL-20260728:END -->

## 2026-07-26 unified 228 valid-result archive

- Archive root: `<REMOTE_HOME>/kt_valid_results_archive/20260726_a2g_mambakt_v1` (mode `0700`).
- Effective inventory: `inventory/kt_archive_inventory_20260726_v3.json`; `9,086` files, `53,269,773,126` bytes, SHA256 `19aae2b6bc95a0b362811a2491012ac3dd0c2c2f6df979e8a1e5016986108748`.
- Independent full audit: `9,086/9,086` files and all bytes rehashed, `2,359` JSON files parsed, `392` checkpoint candidates found, `30/30` sampled checkpoints loaded with PyTorch on CPU, failures `0`.
- A2G entrypoint control archive: `policy/a2g-hard-guard-v1`; `17` files and `47,596` bytes, including guarded/current files, pre-install originals, policy states and audits. Independent verification rehashed `17/17`, parsed `8/8` JSON files and found `0` failures. Inventory SHA256: `2b604e13a45a768e61c6d6c47f8d7095aad68d72e59a6762edc2cfaf60dd7af7`.
- Known gate-passed files not migrated: `0`. The only unreachable source uncertainty is 127, for which no usable SSH endpoint or independent export is available.
- Quarantined stopped attempts remain excluded from result counts. Later independent 228 completions (including the admitted Slepemapy/AKT baseline) are separate artifacts and must not be confused with those stopped attempts.
- Post-archive status at 2026-07-28 20:34: baseline matrix `60/60`, common-five `40/40`, A2G main matrix complete; no-item-dropout is a separate partial ablation at `50/75`.
- Hard exclusions remain `KeenKT`, `EdNet`, and `assist2009_raw`. No source originals on 127, 172, or local storage were deleted by the archive migration.

<!-- CODEX-ASIKT-THREE-DATASET-FOLD0-PILOT-20260731:START -->
## 2026-07-31 ASIKT three-dataset fold0/seed42 pilot (pilot only)

This is a direction-screening pilot: one validation-selected checkpoint for `fold0`, `seed42` on each dataset. It is not a five-fold or multi-seed result, is excluded from the five-fold main table, and does not support a cross-dataset superiority or significance claim. Test outcomes were not used for checkpoint selection or tuning.

| Dataset | Params | Train min | Val AUC | Val ACC | Epoch | Test AUC | Test ACC | Test NLL | Test Brier | Test ECE-15 | Window AUC | Window ACC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Slepemapy | 5,965,469 | 41.06 | 0.794539 | 0.803793 | 4 | 0.791524 | 0.801391 | 0.426257 | 0.137649 | 0.011603 | 0.794493 | 0.802318 |
| ASSIST2017 | 5,642,212 | 2.61 | 0.764369 | 0.711434 | 7 | 0.764259 | 0.714030 | 0.550757 | 0.186493 | 0.016912 | 0.768635 | 0.716935 |
| Junyi2015 | 5,319,822 | 215.68 | 0.801231 | 0.854586 | 19 | 0.803648 | 0.854527 | 0.359097 | 0.109647 | 0.006083 | 0.806185 | 0.855135 |

Machine-readable summary: `<REMOTE_HOME>/kt_baseline_20260723/asikt_cross_dataset_pilot_20260730/audit/asikt_fold0_pilot_summary_20260731.json`, SHA256 `d2a7f2141cea846084c9e4c6b045f8e04e74e2b56ddd0d4510c10d03c7e897cd` (`status=pass`, `pilot_only=true`, `admitted_to_five_fold_main_table=false`, `automatic_five_fold_expansion_started=false`). CSV SHA256: `af15512a51c20dc2085e260012668ce71b2fbffb3e184eef32ae305334f9066b`; Markdown SHA256: `60e727ffd238f706f2a2b499563c7ce6da971c269bcf7dc8cc7c07559112deec`.

Final audit SHA256 values: Slepemapy `229cceab858128849554772b7e3c4c5663642a383bf0a68bd61d67e6f5a15375`; ASSIST2017 `c2f78b5264cf936470e9b1c99a8923817152fee45938a3c65029e27f41774c83`; Junyi2015 `c95ea63bc072c25c4df1113f4dc5f7dab9180a716ebb1c3c82ccf188f1d84725`. All three audits are `pass` and local independent-copy hashes match 228. The old Junyi batch-96 partial remains preserved and excluded. No ASIKT five-fold expansion was launched; ASIKT GPU handoff completed after this audit.
<!-- CODEX-ASIKT-THREE-DATASET-FOLD0-PILOT-20260731:END -->

<!-- CODEX-KT-CURRENT-GOAL-AUDIT-20260731:START -->
## 2026-07-31 KT 主目标当前完成度审计

当前总状态仍为 `incomplete`：六项要求中 `3 pass / 3 partial / 0 fail`。R2（数据集来源与原始描述）、R5（228-only 主机策略）和 R6（笔记/隐私同步）通过；R1（全部模型与最优参数）、R3（全量严格预处理）和 R4（全部端到端直接运行）仍为部分完成。

本次关键增量是把“真实 fold-train-only vocabulary 物化为 0”更新为 `3 dataset-folds / 3 datasets`：ASSIST2017、Junyi2015、Slepemapy 的 `fold0` 均使用 folds `1,2,3,4` 拟合 question/concept vocabulary，验证/测试未知 ID 映射到 OOV；三库均通过用户零重叠、时间单调性和 `test_feedback_used=false` 门禁。仍缺其余 5 个批准数据集，以及三库的 folds `1–4`，所以不能把 R3 写成完成。

| Current evidence item | Count |
|---|---:|
| pinned pyKT models / default YAML | 38 / 38 |
| models with public tuning rows | 32 / 38 |
| independently verified global/paper-optimal YAML | 0 |
| approved datasets | 8 |
| fresh full preprocess manifests on 228 | 2 / 8 |
| real train-only-vocab dataset-folds | 3 |
| complete five-fold result groups | 60 |
| 38x8 matrix: constructible / blocked | 236 / 68 |
| current strict-real-ready native matrix cells | 0 |

机器可读快照：`KT主目标逐项完成度审计_20260731.json`；可读版：`KT主目标逐项完成度审计_20260731.md`；两者当次 SHA256 由 `KT主目标逐项完成度审计_20260731.sha256` 记录，避免审计与本区块形成哈希自引用。三份文件在 workspace、PaperGraph 和 Obsidian 三处逐字节一致，隐私扫描命中 `0`。新版审计核验两套笔记中的 `10` 个 ASIKT/当前目标/MCSKT/pyKT-index/strict-preprocessing-backlog marker 区块，均唯一且对应区块逐字节一致；R1 纳入 38 模型索引和 304-cell validation-tuning candidate/fairness 审计，R3/R4 纳入 43 条严格预处理 backlog 和 228-only executor 的授权/失败停止/审计输出门禁，四组 sidecar 均独立复算，但 R1/R3/R4 仍明确保持 `partial`。本审计未启动 tuning plan、预处理、训练或评估。MCSKT 精确相关仓库候选仍为 `0`、作者仓未验证、出版社 XML 为 metadata-only。

当前 A2G validation-only 队列由独立 A2G 任务接管；本页不把进行中的 validation 当作结果，也不授权自动进入 test。
<!-- CODEX-KT-CURRENT-GOAL-AUDIT-20260731:END -->

<!-- CODEX-MCSKT-SOURCE-REFRESH-20260731:START -->
## 2026-07-31 MCSKT source refresh

- Crossref primary metadata matches the canonical title, DOI `10.1016/j.engappai.2026.114312`, year 2026 and authors RuiJuan Zhang, Feng Zhang and Cong Liu.
- Four exact GitHub repository queries using MCSKT, the full method phrase, DOI and author+model returned `0`. One deliberately broad Dynamic Sparse Attention query returned `14`; its first 10 repositories were LLM/CV/paper-list projects, and the deterministic relevance gate admitted `0` MCSKT candidates.
- The Elsevier XML endpoint returned HTTP 200, but the parsed response contains only `coredata` metadata (`article_body_present=false`, `metadata_only=true`). It cannot establish whether supplementary material contains code.
- Therefore `author_repository_verified=false`; MCSKT remains `blocked_without_verified_author_source_and_validation_safe_dynamic_k`. The paper's dynamic `k` is still rejected because it was selected using test accuracy. No candidate code was downloaded or executed, and no training was started.

Evidence: `mcskt_source_search_refresh_20260731.json`, SHA256 `704b1fb91d6681e07e786cf5d8119a8d73376d21f38a635b07df6e0b9932835a`; the file is byte-identical in workspace, PaperGraph and Obsidian and has zero private-home hits. This refresh narrows the evidence boundary; it does not prove that no repository exists and does not admit MCSKT into the strict result table.
<!-- CODEX-MCSKT-SOURCE-REFRESH-20260731:END -->

<!-- CODEX-PYKT-38-MODEL-YAML-INDEX-20260731:START -->
## 2026-07-31 pyKT 38 模型 YAML 与运行边界索引

- 已为固定 pyKT 原生注册表的 `38/38` 个模型生成逐模型索引：每行包含默认 YAML 相对路径、字节数、SHA256、完整 CLI 参数、public tuning YAML 数量/来源数据集/SHA，以及 8 个批准数据集上的静态准入分类。
- 参数证据边界保持不变：`38` 份 default YAML；`32/38` 个模型共有 `33` 份 pyKT public tuning YAML；独立验证的 paper/global-optimal YAML 为 `0`。public tuning row 缺少指标和选择规则，不能称为“最优参数”。
- `38 x 8 = 304` 个静态格中，`236` 格可构造、`68` 格 fail-closed；当前 fresh strict-real-ready 原生矩阵格仍为 `0`。这只是新端到端严格流水线的当前完成度，不会抹除已经单独通过门禁的历史五折结果。
- `KeenKT` 与所有 `EdNet` 别名继续硬排除。固定 38 模型之外的 FlucKT post-pin overlay、ACE-KT、ASIKT 与 MCSKT 继续按各自独立证据区块管理，不计入 `38`。
- 本次只生成静态索引，没有启动预处理、训练或评估，也没有触碰当前由 A2G 任务接管的 validation-only 队列。

机器可读文件：`PYKT_38_MODEL_YAML_INDEX_20260731.json`，SHA256 `5b6a96e56b97c9ed9e8a70dc2d97568304dc71b2502e08cd60acd6f280d33f9c`；CSV SHA256 `86c1e40cd282266fa2bd4ee0c9e178969fbeab5e84e9880e69461208530785dd`；Markdown SHA256 `5ae7ce31ed7723c60c4f7b4f12db2e51c28778076d332ac6f0451d8e81e52bcd`；sidecar SHA256 `de7fe4b6933053e427506d47a8fa6069be901727b96bb3b1715991ce4d07fba1`。workspace、PaperGraph 与 Obsidian 三处四件套逐字节一致，私有 home 和当前用户名命中均为 `0`。

### Validation-only tuning candidate-pool boundary

对 `38 x 8 = 304` 个 cell 的来源候选池已逐项展开：`236` 个静态可构造、`68` 个 fail-closed。可构造 cell 中 `140` 个只有 upstream default，`96` 个有至少两个来源可执行 tuple；候选池大小分布为 `1:140, 2:2, 3:7, 4:22, 5:22, 6:39, 10:1, 11:3`。

因此全体 236 cell 的最大统一预算只有 `1`，它只是默认值回放，不能称为非平凡调参或“最优 YAML”。跑完所有来源候选需要 `640` candidates、`3200` 个五折 validation trials，但预算不等，也不能当作 equal-budget 跨模型比较。当前严格五折数据 ready cell=`0`、validation-tuned YAML=`0`、独立验证 paper/global-optimal YAML=`0`。即使后续完成，只能称“声明候选池内 validation-best”，不得称论文或全局最优。

候选池 JSON SHA256 `1923acd2657c472fa35bbf9d0dce099f465a9a02019bc587b77368127426c193`；304-row CSV SHA256 `46263f2d5da358b97b6aa65dd52ea2c2c0c4712dd1ade5bf20b1ddca0899554b`；Markdown SHA256 `227b6da848554b75dea741f8dd59a1a5276179c40151d6d7eaa0796600bdf77a`；sidecar SHA256 `c69745b5e5ed8dd13268b29534c9000b6e8dd69ab205dc84d5686c5874921723`。本次未生成 trial plan、未训练、未评估、未访问 test，也未触碰 A2G validation 队列。

<!-- CODEX-RECENT-2025-2026-MODEL-ADAPTER-MATRIX-20260731:START -->
### 2025-2026 model / adapter evidence matrix (2026-07-31)

The matrix contains `16` disclosed records: `14` verified 2025-2026 paper records and `2` specialized pyKT source-only disclosures whose recent paper identity is unresolved.

| Class | Count |
|---|---:|
| `adapter_blocked` | 2 |
| `conditional` | 3 |
| `paper_only_protocol_rejected` | 1 |
| `runnable_admitted` | 5 |
| `runnable_unadmitted` | 3 |
| `source_only` | 2 |

Models with at least one admitted result are ACE-KT, ASIKT, csKT, FlucKT and UKT. `runnable_unadmitted` or `conditional` does not mean reproduced. MCKT remains source-verified/adapter-blocked. MCSKT remains paper-only/protocol-rejected because no author repository is verified and its reported dynamic `k` uses final-test accuracy. SAINT++ and PromptKT remain specialized source-only paths.

No profile is claimed paper/global optimal (`0` verified). This update started no remote task and did not touch A2G validation or test. Machine-readable details are in `RECENT_2025_2026_MODEL_ADAPTER_MATRIX_20260731.{json,csv,md,sha256}`.
<!-- CODEX-RECENT-2025-2026-MODEL-ADAPTER-MATRIX-20260731:END -->
<!-- CODEX-PYKT-38-MODEL-YAML-INDEX-20260731:END -->

<!-- CODEX-STRICT-PREPROCESSING-BACKLOG-20260731:START -->
## 2026-08-01 strict preprocessing backlog and command readiness

Superseded on 2026-08-03: fresh raw-to-pyKT manifests are now `8/8`, compact fold-train-only vocabulary views are `40/40`, and the pinned DKT strict entry dry-run is `40/40`; Window views and missing model results remain incomplete.

All 43 argv arrays passed exact-shape checks and real parser `--dry-run` probes (`43/43`), with the pinned 145-file pyKT code tree and all five preprocessing/orchestration sources hash-matched. The probes wrote 0 runtime files and started no preprocessing, training, evaluation, remote task, A2G validation mutation, or final test.

The current 228 boundary was rechecked read-only on 2026-08-01. The five locked orchestration/data sources and the pinned pyKT tree (`145` Python files; code-tree SHA256 `ede0df7ace19506321a32246099ba166767612bbbc08514f5e34916fc434e7ca`) match. However, the current executor/wrapper/backlog are not deployed under the 228 KT root, `KT_PYTHON` is unbound, the observed system Python lacks `pandas`, `numpy` and `gdown`, no RAR extractor was found, and the proposed isolated runtime/raw/manifest roots do not exist. Therefore `execution_ready=false`; real execution remains fail-closed and no download or preprocessing was started.

The one-time isolation snapshot found only the A2G validation-only process owned by task `019f46b9-7e20-7f00-a145-2b148604ced3`; this task neither touched that queue nor entered test and will not continue monitoring it. Remote blocker evidence: `STRICT_PREPROCESSING_REMOTE_READINESS_228_20260801.json`, SHA256 `b729a520bc40c7b5cb449a92dd227ef2f257b5eb2f06a508b27b19de7a3fdd5b`.

Readiness JSON SHA256 `bac44a446a831c48afac3e56fd59de418cddeaab4000ffacf2664bf00b7f5a2e`; CSV SHA256 `faa65a565f85dd83ae71c05b85b8ceaa2c52eb86be343b679773b0db571405e9`; Markdown SHA256 `3afccdc7cf40b922e2c2ebfcb916bdbc1ccdb48a4f7f932efa46f40929544d80`. Independently recompute the four-file sidecar before use.
<!-- CODEX-STRICT-PREPROCESSING-BACKLOG-20260731:END -->

<!-- CODEX-PUBLICATION-EFFICIENCY-20260802:START -->
## 2026-08-02 投稿效率基线实测（Assist2012）

固定协议：`Assist2012 / fold0 / seed42 / batch64 / seq200 / FP32 / 228 同一张 RTX 4090 D`。TT 为每个训练 epoch 的秒数（3 次状态重置，每次 1 个 warmup + 5 个计时 epoch，共 15 个样本）；IT 为固定完整 test loader 的推理秒数（2 次 warmup + 10 次计时）；显存为训练 epoch 的 CUDA peak allocated/reserved。

四个基线均先通过真实 batch64 的前向、反向和 optimizer-step GPU 门禁，再由内部 enabled manifest 启动正式测量。A2G 模型及 checkpoint 未修改，本轮只跑 DKT、SAKT、AKT、SimpleKT。

| Model | TT (s/epoch) | IT (s/full test) | Selected int/s | Trainable params | Peak allocated/reserved (GiB) | FLOPs |
|---|---:|---:|---:|---:|---:|---|
| DKT | 3.216 +/- 0.048 | 1.265 +/- 0.028 | 423790 | 480,865 | 0.395/3.408 | NA (operator-complete audit pending) |
| SAKT | 3.113 +/- 0.295 | 1.482 +/- 0.426 | 385117 | 650,753 | 0.614/3.416 | NA (operator-complete audit pending) |
| AKT | 43.008 +/- 0.036 | 6.588 +/- 0.028 | 81377 | 6,255,536 | 8.710/8.898 | NA (operator-complete audit pending) |
| SimpleKT | 5.157 +/- 0.750 | 1.516 +/- 0.367 | 368127 | 14,715,905 | 1.023/3.523 | NA (operator-complete audit pending) |

当前准入状态为 `pass_four_fields_flops_pending`：TT、IT、吞吐、参数量和训练峰值显存均已通过同协议证据链；FLOPs 尚无自定义算子覆盖率 100% 的独立审计，因此不得填入估算值，也不得把这四行标成完整五字段投稿主表。

仍缺同协议效率结果的本地必需模型：A2G-MambaKT、DKT-Forget、DKVMN、SAINT、SparseKT、DTransformer、LPKT、DIMKT、UKT、ACE-KT、ASIKT、Mamba4KT。SAINT++ 继续保持 source-only，不能用 SAINT 冒充；MCSKT 继续保持 paper-reference，不与 228 实测混排。

机器审计 SHA256：`5e7e8744b2755a1ec32a1fe0a760edf2683085e376ab507ce22ddab34dfb9d6b`；228 归档 SHA256SUMS：`080a6a856afea59ae7ebc2cdc07a07c96b35f7b490af2f4c76972c38a3429169`。统一归档：`<REMOTE_HOME>/kt_valid_results_archive/20260802_publication_efficiency_baselines_assist2012_v1`。
<!-- CODEX-PUBLICATION-EFFICIENCY-20260802:END -->

<!-- CODEX-DENOISEKT-VALIDATION-GPU-20260803:START -->
## 2026-08-03 DenoiseKT validation-only v3 与 228 GPU 基线队列

DenoiseKT 的结构门禁已从“只有 fold-train-only auxiliary”推进为可直接执行的 validation-only 入口：6 数据集 × 5 folds 共 `30/30` direct dry-run pass；每个 dpath 物理上只含 train/validation 所需的 5 个文件，没有 final-test/Window 文件；每个隔离 runtime 的 `data_config.json` 只含当前数据集，`final_test_access_allowed=false`。真实执行还必须同时显式设置 `KT_ALLOW_REAL_RUN=1` 与 `KT_ALLOW_VALIDATION_ONLY=1`。

独立审计从另一位置重算 30 个 view manifest、runtime manifest、runtime config、5 个控制脚本和 30 条持久命令：`30/30` pass、forbidden artifacts `0`。生产审计 SHA256 `2ba866692c86f4254b90ecc0bf2d35ebfde5677eb09c241b221fd5c62d5287d5`；独立审计 SHA256 `c4cc8cf20bcfa62f77479e61a7efbcbe10491e9adae536f1f608817027045baf`；公开 validation config SHA256 `c79f270a6a85e06c1a3493c0111ec6a2a3c347229f416114d38608f5f57d0e86`；匿名化审计 SHA256 `41204a689c456cb91f0a47044fd512fbb7b6b35cb0104ab2953930e049e81fe4`。这些证据不声称训练完成、validation 指标、final-test 指标或 paper-optimal profile。

| GPU smoke 范围 | 当前结果 | 证据边界 |
|---|---|---|
| 固定 pyKT 13 个 native models | CUDA optimizer-step `10/13` standard-entry pass | DKT-Forget 为源码 CPU lookup tensor/GPU index 缺陷；SparseKT 标准 runner 为 4-to-3 return contract 不兼容；DTransformer 为 Triton/Python.h 本地编译阻断 |
| SparseKT hash-bound adapter | CUDA forward/backward/optimizer step pass | 只证明 4-return 到 `cal_loss` 的兼容适配，不是 effectiveness |
| FlucKT post-pin runtime | CUDA forward/backward/optimizer step pass | 固定 post-pin 源码哈希；不是五折结果 |
| SAINT++ specialized source | CUDA forward/backward/optimizer step pass | 专用源码 direct entry；不能用 SAINT 冒充 |
| ACE-KT | GPU backward 被同一 Triton/Python.h 环境阻断 | 既有 CPU optimizer-step pass 仍保留；本轮不伪造 GPU pass |

native GPU smoke SHA256 `ef65cbbc3b317ff93802e2e40147d4c17792e723797d4f7ed99fa8b080057b56`；SparseKT adapter SHA256 `073b01f867de8fa2704fb3a1fd7af75a3d48f7f9c83d888705391594aae16b59`；FlucKT SHA256 `4fabc75cef53ec5cf499f8aaa205ccb1472696405cdbfa099a58f3ab1969ea06`；SAINT++ SHA256 `7d50e2eb00c26f749f92995ec9d0d9b0cad129c0fa273d37b7061e4f2a431eab`。228 没有时间截止；唯一持续队列位于 `$HOME/kt_baseline_20260723/continuous_baseline_gpu_queue_20260803_v1`，会在 smoke 后顺序运行 DenoiseKT 剩余 validation-only folds，单项失败留证后继续。final test 尚未启动，validation-only 数值在冻结前不得进入 effectiveness 主表。
<!-- CODEX-DENOISEKT-VALIDATION-GPU-20260803:END -->
<!-- CODEX-OFFLINE-MISSING-BASELINE-WATCHER-NOTE-20260804:START -->
> 2026-08-04：已在 228 部署离线缺失基线 watcher。Assist2012 UKT/SAINT 正按 5 折 validation-only 队列补齐；在完成 validation 全折、冻结 profile、final-test 授权及独立重算前，结果不进入下方投稿 effectiveness 主表，当前缺失项保持 `—`。watcher 合同与运行证据见统一命令总账的 `CODEX-OFFLINE-MISSING-BASELINE-WATCHER-228-20260804` 区块。
<!-- CODEX-OFFLINE-MISSING-BASELINE-WATCHER-NOTE-20260804:END -->

<!-- CODEX-COMPLETE-BASELINE-DELIVERY-AUDIT-20260804:START -->
> 静态资料交付审计 `outputs/COMPLETE_BASELINE_DELIVERY_AUDIT_20260804.json` 已通过（SHA256 `a1d88ad9d9e47a134d59cad45c33e4464f6ff34c87bfdce36b74995d493d6718`）：38 个 pyKT 模型/33 份公开 tuning YAML/8 个批准数据集/16 条近年模型记录均有索引；该审计不把未完成训练或 validation 结果写入五折 effectiveness 主表。
<!-- CODEX-COMPLETE-BASELINE-DELIVERY-AUDIT-20260804:END -->

<!-- CODEX-ASSIST2012-NATIVE-FINAL-EFFICIENCY-20260804:START -->
## 2026-08-04 Assist2012 原生强基线审计终态

以下数值不是 validation、smoke 或单折 pilot。DKVMN、UKT、SAINT 均为预先冻结候选后的五折 one-step final test 与 Window test；Window 未参与选择。AUC、ACC、ECE-15、Brier、NLL 均由 compact prediction 独立复算，`30/30` 结果通过 SHA 与身份审计。

### Assist2012 五折 effectiveness 与 Window

| Dataset | Model | One-step AUC | One-step ACC | Window AUC | Window ACC |
|---|---|---:|---:|---:|---:|
| Assist2012 | DKVMN | 0.728735 +/- 0.000441 | 0.733669 +/- 0.000153 | 0.730542 +/- 0.000398 | 0.734330 +/- 0.000157 |
| Assist2012 | UKT | 0.773178 +/- 0.000484 | 0.751579 +/- 0.001022 | 0.774496 +/- 0.000673 | 0.751931 +/- 0.001476 |
| Assist2012 | SAINT | 0.669087 +/- 0.001634 | 0.712506 +/- 0.000736 | 0.669640 +/- 0.001610 | 0.712402 +/- 0.000824 |

### Assist2012 五折校准与概率质量

| Dataset | Model | Protocol | ECE-15 | Brier | NLL |
|---|---|---|---:|---:|---:|
| Assist2012 | DKVMN | one-step | 0.002528 +/- 0.000778 | 0.180497 +/- 0.000123 | 0.539298 +/- 0.000305 |
| Assist2012 | DKVMN | Window | 0.002555 +/- 0.000674 | 0.180015 +/- 0.000113 | 0.538099 +/- 0.000292 |
| Assist2012 | UKT | one-step | 0.030652 +/- 0.004114 | 0.169380 +/- 0.000366 | 0.511853 +/- 0.001091 |
| Assist2012 | UKT | Window | 0.032419 +/- 0.004021 | 0.169196 +/- 0.000560 | 0.511453 +/- 0.001286 |
| Assist2012 | SAINT | one-step | 0.019497 +/- 0.005259 | 0.194640 +/- 0.000217 | 0.574650 +/- 0.000421 |
| Assist2012 | SAINT | Window | 0.019402 +/- 0.005429 | 0.194603 +/- 0.000217 | 0.574529 +/- 0.000441 |

### Overall results of model efficiency：Assist2012 strict 三模型

统一协议：fold0 / seed42 / batch64 / seq200 / FP32 / RTX 4090 D；TT 为排除 validation 与 checkpoint I/O 的每 epoch 时间，IT 为完整 one-step test loader（含固定 metric aggregation），GPU 为训练 epoch 峰值 allocated/reserved。每模型 TT=15 个样本、IT/throughput=10 个样本。FLOPs 基于已保存真实 forward ATen inventory，按 FMA=2、elementwise=1、softmax/layer-norm=5、embedding/索引=0 的 analytic convention v1 计算，三模型公式覆盖均为 100%。

| Dataset | Model | TT (s/epoch) | IT (s/full test) | Throughput (int/s) | Total params (M) | Trainable params (M) | Peak alloc/reserved (GiB) | FLOPs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Assist2012 | DKVMN | 10.521 +/- 0.105 | 1.878 +/- 0.037 | 285,512 +/- 5,414 | 0.339801 | 0.339801 | 2.673 / 4.916 | 5.269 |
| Assist2012 | UKT | 27.376 +/- 0.040 | 4.325 +/- 0.028 | 123,944 +/- 782 | 17.657633 | 17.555233 | 4.892 / 5.166 | 140.163 |
| Assist2012 | SAINT | 20.079 +/- 0.080 | 2.826 +/- 0.031 | 189,744 +/- 2,058 | 57.344257 | 57.344257 | 4.398 / 4.764 | 141.237 |

注意：本表三行使用 strict fold-train-only-vocab 数据根。2026-08-02 的 DKT/SAKT/AKT/SimpleKT 四行使用历史 Assist2012 数据根，当前不得与本表合并排序；需统一重测或在论文中明确分表。

证据 SHA256：final/Window aggregate `8dd1ba73ff678b881970b678531abf16b22079a63b38f009c729e4cdaa8873a1`；efficiency independent audit `4a0c377ea44905ef5badd71b6fc50f357fde9d07beac02e21f505af5a4687545`；operator inventory `c4b48a5d81857a8029a3b97723c73a6af300f0246e2b71da310cfb50834f7935`；operator-complete analytic FLOPs receipt `60bd6e84df0fc18f13b9c3528f32af2868e012c9af82f9fdb0c02b988ecfb68f`。

### Strict validation-only 补充基线状态（不得填 final-test 主表）

- DKT / Statics2011：预注册 `LR=(0.0005, 0.001, 0.002)` × five folds × seed42 已完成 `15/15`；按 mean validation AUC、再 ACC 的规则冻结 LR=0.001。validation AUC `0.823545`，ACC `0.806401`。五个 checkpoint 已独立读取；freeze SHA256 `137a8326b6882e713eca38ed11ef5d0b0c345ed0da2159532a8c0387b1e4f51b`。`test_access=false`、final-test/Window 均未启动。
- DKT / Junyi2015：v15 已完成 `15/15` validation-only trials；按预注册规则冻结 LR=0.0005，validation AUC `0.752731`、ACC `0.845761`。freeze SHA256 `934ce148a68aa7045ba992b17990ce940e557a64051130573ec066ec29b0a691`；第二位置独立审计 `5/5 pass`，SHA256 `066e01bda0a651e4beb587e6ef3e264a72c70921ef34c6cae3540beff4770d94`。final-test/Window 均未启动；任何 validation 数值都不能进入上方 effectiveness 主表。
<!-- CODEX-ASSIST2012-NATIVE-FINAL-EFFICIENCY-20260804:END -->

<!-- CODEX-STRICT-DKT-FINAL-20260804:START -->
## 2026-08-04 最新 strict 八数据集 effectiveness 主表

本表是当前 strict primary 视图：learner-disjoint、fold-train-only vocabulary、validation 冻结后才授权 one-step final test。DKT 共 `8 datasets × 5 folds = 40/40` final results；每折均保存 compact predictions，AUC、ACC、ECE-15、Brier、NLL 已从固定结果聚合。Window 未参与选模且本轮未启动，因此 DKT Window 仍保持 `—`。

历史五数据集主表保留在前文用于来源追踪；下表复用其中其它模型的已审计 final 单元，并以新的 strict DKT final 替换 DKT 列，同时新增 Statics2011、ASSIST2009 corrected/collapsed、ASSIST2012。空白仍表示没有同口径五折 final 证据。

| Dataset | Metric | DKT | DKT+ | DKVMN | SAKT | SAINT++ | AKT | SimpleKT | DTransformer | UKT | ACE-KT | ASIKT | Mamba4KT | MCSKT† | A2G-MambaKT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Statics2011 | AUC | 0.820800 +/- 0.001690 | <u>0.825382 +/- 0.000578</u> | 0.804491 +/- 0.002055 | 0.791061 +/- 0.003281 | — | **0.827680 +/- 0.001827** | 0.818371 +/- 0.000511 | — | 0.822622 +/- 0.001679 | 0.809976 +/- 0.002144 | — | — | — | — |
| Statics2011 | ACC | <u>0.797267 +/- 0.000622</u> | 0.796861 +/- 0.000967 | 0.792452 +/- 0.001020 | 0.785362 +/- 0.000825 | — | **0.799565 +/- 0.000801** | 0.793985 +/- 0.002310 | — | 0.796341 +/- 0.002362 | 0.789348 +/- 0.001157 | — | — | — | — |
| Junyi2015 | AUC | 0.754426 +/- 0.000397 | 0.753959 +/- 0.000110 | 0.751666 +/- 0.000426 | 0.749412 +/- 0.000755 | — | 0.803541 +/- 0.000284 | 0.803113 +/- 0.000211 | — | **0.804888 +/- 0.000119** | — | — | — | — | <u>0.804826 +/- 0.000111</u> |
| Junyi2015 | ACC | 0.846471 +/- 0.000073 | 0.846219 +/- 0.000023 | 0.846014 +/- 0.000052 | 0.845151 +/- 0.000173 | — | 0.854726 +/- 0.000083 | 0.854816 +/- 0.000048 | — | **0.855215 +/- 0.000073** | — | — | — | — | <u>0.854974 +/- 0.000070</u> |
| ASSIST2009 corrected/collapsed | AUC | 0.823724 +/- 0.000618 | 0.825614 +/- 0.000588 | 0.816374 +/- 0.001572 | 0.795565 +/- 0.001660 | — | 0.840313 +/- 0.001020 | 0.840088 +/- 0.001059 | — | **0.848004 +/- 0.001405** | <u>0.845685 +/- 0.000872</u> | — | — | — | — |
| ASSIST2009 corrected/collapsed | ACC | 0.767227 +/- 0.001315 | 0.769230 +/- 0.000781 | 0.761925 +/- 0.001480 | 0.748478 +/- 0.002482 | — | 0.777064 +/- 0.001267 | 0.776476 +/- 0.001133 | — | **0.781122 +/- 0.000806** | <u>0.779969 +/- 0.001030</u> | — | — | — | — |
| ASSIST2012 | AUC | 0.730557 +/- 0.000313 | 0.731710 +/- 0.000420 | 0.728735 +/- 0.000441 | 0.707173 +/- 0.001031 | — | **0.779282 +/- 0.000552** | <u>0.773878 +/- 0.000295</u> | — | 0.773178 +/- 0.000484 | — | — | — | — | — |
| ASSIST2012 | ACC | 0.733910 +/- 0.000321 | 0.734358 +/- 0.000365 | 0.733669 +/- 0.000153 | 0.724079 +/- 0.000560 | — | **0.757258 +/- 0.000853** | 0.750971 +/- 0.001224 | — | <u>0.751579 +/- 0.001022</u> | — | — | — | — | — |
| ASSIST2015 | AUC | 0.726508 +/- 0.000708 | <u>0.728353 +/- 0.000729</u> | 0.721298 +/- 0.000465 | 0.703955 +/- 0.000590 | — | 0.725735 +/- 0.000952 | 0.723972 +/- 0.000425 | — | 0.726737 +/- 0.000882 | 0.725765 +/- 0.000276 | — | — | — | **0.729012 +/- 0.000420** |
| ASSIST2015 | ACC | 0.749820 +/- 0.000348 | 0.751046 +/- 0.000339 | 0.750609 +/- 0.000412 | 0.746135 +/- 0.000316 | — | <u>0.751355 +/- 0.000508</u> | 0.750584 +/- 0.000280 | — | 0.750840 +/- 0.000663 | 0.750363 +/- 0.000186 | — | — | — | **0.751662 +/- 0.000156** |
| NIPS Task 3&4 | AUC | 0.771560 +/- 0.000462 | 0.771500 +/- 0.000211 | 0.768590 +/- 0.000869 | 0.747161 +/- 0.001053 | — | 0.804626 +/- 0.001219 | 0.802279 +/- 0.000347 | 0.801547 +/- 0.000604 | **0.805728 +/- 0.000510** | 0.803960 +/- 0.000329 | 0.798763 +/- 0.001047 | — | 0.827300† | <u>0.805565 +/- 0.000243</u> |
| NIPS Task 3&4 | ACC | 0.704811 +/- 0.000675 | 0.704851 +/- 0.000455 | 0.702165 +/- 0.000767 | 0.682991 +/- 0.000938 | — | <u>0.732949 +/- 0.000612</u> | 0.731513 +/- 0.000412 | 0.730464 +/- 0.000663 | **0.733253 +/- 0.000697** | 0.732686 +/- 0.000242 | 0.728213 +/- 0.001147 | — | 0.763500† | 0.731878 +/- 0.000644 |
| ASSIST2017 | AUC | 0.718301 +/- 0.001950 | 0.717607 +/- 0.000674 | 0.710194 +/- 0.000595 | 0.653189 +/- 0.000930 | — | 0.766839 +/- 0.000793 | 0.753958 +/- 0.001005 | — | 0.762522 +/- 0.004354 | <u>0.782030 +/- 0.000544</u> | — | — | 0.817400† | **0.784384 +/- 0.000836** |
| ASSIST2017 | ACC | 0.687508 +/- 0.000963 | 0.687453 +/- 0.000458 | 0.683436 +/- 0.000245 | 0.664698 +/- 0.000799 | — | 0.714914 +/- 0.000469 | 0.707607 +/- 0.000773 | — | 0.712861 +/- 0.003303 | <u>0.725020 +/- 0.001395</u> | — | — | 0.760400† | **0.727199 +/- 0.000465** |
| Slepemapy | AUC | 0.784142 +/- 0.000171 | 0.786570 +/- 0.000477 | 0.785218 +/- 0.000425 | 0.771241 +/- 0.000470 | — | **0.800163 +/- 0.000465** | 0.794192 +/- 0.000419 | — | — | — | — | — | — | <u>0.799000 +/- 0.000209</u> |
| Slepemapy | ACC | 0.797344 +/- 0.000214 | 0.798261 +/- 0.000238 | 0.797235 +/- 0.000204 | 0.791721 +/- 0.000312 | — | <u>0.803394 +/- 0.000844</u> | 0.801021 +/- 0.000172 | — | — | — | — | — | — | **0.803450 +/- 0.000260** |

### DKT strict calibration 与概率质量

| Dataset | ECE-15 | Brier | NLL |
|---|---:|---:|---:|
| Statics2011 | 0.021991 +/- 0.002372 | 0.139041 +/- 0.000355 | 0.427881 +/- 0.001260 |
| Junyi2015 | 0.001615 +/- 0.000208 | 0.118106 +/- 0.000079 | 0.387023 +/- 0.000241 |
| ASSIST2009 corrected/collapsed | 0.009637 +/- 0.004129 | 0.157145 +/- 0.000298 | 0.468519 +/- 0.000811 |
| ASSIST2012 | 0.009364 +/- 0.004478 | 0.180148 +/- 0.000132 | 0.538531 +/- 0.000345 |
| ASSIST2015 | 0.007297 +/- 0.000717 | 0.170667 +/- 0.000153 | 0.514655 +/- 0.000506 |
| NIPS Task 3&4 | 0.010642 +/- 0.003483 | 0.192490 +/- 0.000228 | 0.562749 +/- 0.000621 |
| ASSIST2017 | 0.014329 +/- 0.004564 | 0.200977 +/- 0.000520 | 0.586513 +/- 0.001231 |
| Slepemapy | 0.005999 +/- 0.001495 | 0.139866 +/- 0.000048 | 0.431955 +/- 0.000118 |

ASSIST2012 的 SAINT one-step final 为 AUC `0.669087 +/- 0.001634`、ACC `0.712506 +/- 0.000736`；它是 SAINT，不得填入 SAINT++ 列。ASSIST2012 的 DKVMN、UKT Window 已完成，但本表为 one-step 主表，Window 数值继续在前文独立表报告。

审计证据：DKT 5-fold summary SHA256 `bce030ca3d90ad2990570ad27593068c955361b592d8a30cea33853a7f4ae3f5`；v19r3 registry SHA256 `4fc4e92f12e646f9adf71b913d0dad252283be1eb190c4284490749f13c9f583`；evaluator SHA256 `9af34492540e2da4b3ad9cd931610eb07b62294eca06f96ba260ab7e32b0d48a`。v19（错误解引用 venv）与 v19r2（DKT concept gather 维度错误）失败证据均保留并排除；v19r3 为唯一准入终态。

### 2026-08-04 smoke 补齐终态

- native pyKT CUDA forward/backward/optimizer step：历史 v5 `10/13 pass`，只补三项的 v6r2 `3/3 pass`，合并为 `13/13`。DKT-Forget、SparseKT、DTransformer 均由限定 adapter/environment gate 通过，不是 effectiveness 结果。
- external CUDA optimizer step：FlucKT、SAINT++、ACE-KT 均 pass。ACE-KT 使用 Torch 官方 `TORCH_DISABLE_NATIVE_JIT=1` 回退，源码树未修改；审计 SHA256 `e1f407e62e2da9a1bd5dfa6736215eaa0d95d0c6da36edd7ae226eb52e53fe43`。
<!-- CODEX-STRICT-DKT-FINAL-20260804:END -->

<!-- CODEX-OFFLINE-BASELINE-WATCHDOG-DTRANSFORMER-20260805:START -->
## 2026-08-05 228 offline baseline continuation

This is a scheduling/evidence update, not a result claim. Host 228 is the only
KT GPU host; hosts 127 and 172 remain prohibited. At 2026-08-05 01:13 CST,
`nvidia-smi` reported GPU utilization `38%`, memory `5469/24564 MiB`, and one
compute process (PID `2069626`). The process is a DKVMN Junyi2015 validation
trial; v21 and v22 remain active, and the DTransformer v23 continuation waits
for the v22 terminal audit. No A2G process was started by this continuation.

### Offline recovery gate

The old `@reboot`-only fallback was insufficient for a server that stays up
while all interactive sessions disappear. A 228-only cron watchdog is now
installed with both `@reboot` and `* * * * *` entries. It checks known v20/v21/
v22/v23 worker processes, GPU compute processes, the shared lock, and the
hash-bound resume script before launching anything. It exits on an active
worker, so it cannot duplicate a fold. `cron=active`; `Linger=no` is recorded
as a limitation of user systemd, while the cron path does not depend on a
user systemd manager.

Watchdog SHA256:
`055d157b860a19d8643369b8ca4e9f6c52787bd137d31a717ddaba0c9cb2f202`

### DTransformer validation continuation (not final-test evidence)

The immutable v23 registry covers eight approved datasets, three relative-LR
candidates, and five folds (`120` trials total). All eight real-data
preflights passed with `training_started=false` and
`final_test_feedback_used=false`. Registry SHA256:
`fd05b73a10fa302dbfcbc66976d45b7bfa0d2a46746f34e065ce4186eab58870`.
The chain runner SHA256 is
`7079ef39bfb5c64529f7e63215a176bc02ee9fe84fb21bdadae9494f80a6c1ad`.
It is authorized to start only after the DKVMN v22 predecessor audit reaches
`complete` or `terminal_with_failures`; it has no final-test or Window access.

| Model | Dataset scope | Validation contract | Final-test/main-table status |
|---|---|---:|---|
| DTransformer | Statics2011, Junyi2015, NIPS34, Slepemapy, ASSIST2012, ASSIST2015, ASSIST2009 corrected/collapsed, ASSIST2017 | 8 x 3 x 5 = 120, preflight pass | — until independent freeze and authorization |

The effectiveness table continues to use `—` for these cells. Smoke,
validation-only metrics, and the waiting service are not five-fold final-test
results. Efficiency cells (TT, IT, parameters, FLOPs, peak GPU memory) also
remain `NA` until the same model, data, device, and measurement protocol has
produced hash-bound measurements; FLOPs are not inferred from smoke.
Merged smoke inventory refreshed after the v6r2 retries:
`outputs/PUBLICATION_BASELINE_SMOKE_INVENTORY_20260805.json`, SHA256
`d1277cfef4db0fc25f5ed83a4d76b0564b965bf968c4cff75ec802b15b05d34e`.
The 20260803 inventories remain historical snapshots.
<!-- CODEX-OFFLINE-BASELINE-WATCHDOG-DTRANSFORMER-20260805:END -->

<!-- CODEX-CURRENT-COVERAGE-AUDIT-20260805:START -->
## 2026-08-05 当前基线补全审计：交付物完整 ≠ AUC/ACC 结果完整

本次只读复核以 `COMPLETE_BASELINE_DELIVERY_AUDIT_20260804.json`、
`PYKT_38_MODEL_YAML_INDEX_20260731.json` 和
`RECENT_2025_2026_MODEL_ADAPTER_MATRIX_20260731.json` 为准，未启动新的远程
训练、下载或评估。

### 交付物层面

| 项目 | 当前证据 | 解释 |
|---|---:|---|
| pyKT 原生模型 | 38/38 | 每个模型都有默认 YAML 与入口索引 |
| 默认 YAML | 38/38 | 已索引字节数、SHA256、CLI 参数 |
| 有公开 tuning YAML 的模型 | 32/38 | 共 33 份 public tuning YAML |
| 独立核验为“论文/全局最优”的 YAML | 0 | 公开 YAML 没有统一指标与选择规则，不能写成 paper-optimal |
| 批准数据集 | 8/8 | 当前排除 KeenKT 与所有 EdNet 别名 |
| 近两年模型记录 | 16 | 14 条已核验论文记录 + 2 条 source-only 专项记录 |
| 近两年已有准入实测结果 | 5 个模型 | 其余仍是 runnable-unadmitted、conditional、source-only 或 blocked |

### effectiveness 结果层面

主表只接受同口径、learner-disjoint、fold-train-only vocabulary、冻结
validation profile 后的五折 final-test artifact。因而：

- 当前主表中已有数值的每个 `dataset × model` 单元，AUC 与 ACC 是成对报告的；
  但这不等于每个模型/数据集都有结果。
- Window AUC/ACC 只有在单独的 Window artifact 存在时才填；不能用 one-step
  数字代替。
- 当前仍保留 `—` 的核心缺口包括：DKVMN 多数数据集、SAINT/SAINT++、
  DTransformer continuation、DenoiseKT/ SparseKT/ DKT-Forget 等真实
  effectiveness、Mamba4KT adapter，以及若干 UKT/ACE-KT/ASIKT 数据集组合。
- Assist2012 已有 DKVMN、UKT、SAINT 的五折 one-step 与 Window AUC/ACC；
  这三行不是 SAINT++，也不自动填充其它数据集。
- validation-only、CPU/GPU smoke、fold0 pilot、论文单点值和 efficiency
  microbenchmark 均不能填入 effectiveness 主表。

### 结论（回答“还有多少没补、都有 AUC/ACC 吗”）

目前不能用一个“剩余百分比”概括，因为交付物和结果是两套分母：静态资料
索引已经基本齐全（38 模型、8 数据集、16 近年模型记录），但 304 个
`模型 × 数据集` 静态组合中只有 236 个可构造、68 个被 schema/协议/依赖
阻断；而严格五折 final-test 结果远未覆盖全部 304 个单元。对已经准入主表
的单元，AUC/ACC 均有；对仍为 `—` 的单元，两项都视为未补，不把单独的
validation AUC、smoke 或论文值当作 ACC/AUC 完成。

### 下一优先级

1. 等 228 上 DKVMN v21/v22 自然终态并完成独立审计，再按授权链继续
   DTransformer validation-only；不在 172 启动新任务。
2. 以与 A2G-MambaKT 机制最接近的 ASIKT/Mamba4KT 为专门 comparator，
   先完成 hash-bound adapter 和同协议 smoke，再决定是否进入真实五折。
3. 对每个新入表单元同时生成 AUC、ACC、NLL、Brier、ECE 与 compact
   prediction hash；缺任一主指标就保持 `—`。
<!-- CODEX-CURRENT-COVERAGE-AUDIT-20260805:END -->

<!-- CODEX-LOCAL-REPRO-AUDIT-20260805:START -->
## 2026-08-05 可复现入口静态复核补充

使用完整 delivery root `codex-threads-019f8534-b598-74d1-a79f` 配置化调用
四个审计器：pyKT 38-model YAML index、2025–2026 recent-model matrix、
active KT goal audit 均通过；strict preprocessing command readiness 在修复
镜像目录下 `relative_to(ROOT)` 的可移植性异常后也通过，当前状态
`static_cli_pass_current_228_runtime_not_reverified`，命令数 `43`，failures
`0`。该复核仍是静态/ dry-run 证据，不代表已启动下载、训练或 final-test。
<!-- CODEX-LOCAL-REPRO-AUDIT-20260805:END -->

<!-- CODEX-COMPLETE-DELIVERY-AUDIT-20260805:START -->
## 2026-08-05 总交付闭环审计

修复总交付审计器对 delivery root / PaperGraph 镜像双布局的路径解析后，
使用完整 delivery root 重新运行：
`COMPLETE_BASELINE_DELIVERY_AUDIT_20260805_LOCAL.json`
状态为 `pass`，`failures=[]`，SHA256
`8e38d185340764d87711f1a4848ba4694f57428b82c45a58220e04b8b3eca18d`。
审计覆盖 38 模型、8 数据集、16 条近年模型记录、数据集注册表、预处理/训练
命令笔记、排除项（KeenKT/EdNet）和 portable package 脚本别名；仍然只是
交付/静态闭环，不等于所有模型已训练或所有主表单元已有 AUC/ACC。
<!-- CODEX-COMPLETE-DELIVERY-AUDIT-20260805:END -->

<!-- CODEX-REMOTE-QUEUE-STATUS-20260804T0828:START -->
## 2026-08-04 08:28 CST 228 远程队列只读状态

当前仍有 DKVMN v21/v22 链运行：Junyi2015 fold2 的训练进程为
`2098630`，GPU compute 仍存在；v22
`STRICT_DKVMN_REMAINING_CHAIN_AUDIT.json` 尚未生成，DTransformer continuation
尚未启动。此状态不触发任何新任务、重启或 GPU 抢占；待自然终态后才允许
独立审计和后续 handoff。
<!-- CODEX-REMOTE-QUEUE-STATUS-20260804T0828:END -->

<!-- CODEX-REMOTE-QUEUE-STATUS-20260804T0715-REMOTE_CLOCK:START -->
## 2026-08-04 07:15 CST 本轮 228 远程队列只读复核

远端 watchdog 的时间字段自报为 `2026-08-05T07:15:01+08:00`，与本地系统日期
（2026-08-04）不一致；以下内容按远端文件状态记录，不将该时间视为本地当前日期。

v22 `STRICT_DKVMN_REMAINING_CHAIN_AUDIT.json` 仍不存在；watchdog
`offline_bootstrap_20260805_v3/watchdog_status.json` 仍为
`chain_active`。DKVMN v21/v22 orchestrator 与 Junyi2015 fold2 训练进程仍在运行
（当前训练 PID `2098630`，GPU0 约 `5468 MiB`、利用率约 `40%`），
DTransformer 未启动。根据只读门禁，不停止、重启、修改或追加任何 GPU 作业；
在 v22 audit 达到自然终态前不进行后续 handoff。
<!-- CODEX-REMOTE-QUEUE-STATUS-20260804T0715-REMOTE_CLOCK:END -->

<!-- CODEX-YAML-EVIDENCE-20260804:START -->
## 2026-08-04 YAML 参数证据索引

完整 38 模型清单见
`PYKT_38_MODEL_YAML_INDEX_20260731.md`（机器可读版本：
`PYKT_38_MODEL_YAML_INDEX_20260731.json` / `.csv`）。
当前可审计计数为：default YAML `38/38`；公开 tuning YAML `32/38`
（共 `33` 份）；独立验证的 paper/global-optimal YAML `0`。
因此训练命令默认引用 canonical default 或 public tuning profile，
但结果表和正文不得把 public tuning row 称为“最优超参”，除非新增
同一数据切分、train-only validation 选择规则和 hash-bound 结果证据。
<!-- CODEX-YAML-EVIDENCE-20260804:END -->

<!-- CODEX-GOAL-AUDIT-20260804:START -->
## 2026-08-04 当前目标审计结论

使用完整 delivery root 的 `audit_active_kt_goal_20260731.py` 复核：
总体状态为 `incomplete`，`pass=3`、`partial=3`、`fail=0`。
静态交付、数据集来源/排除项、笔记同步与隐私审计均通过；未完成项不是
静态索引缺失，而是：

1. 8 个批准数据集仍有 5 个缺少完整 fresh raw-to-pyKT manifest，且
   fold-train-only vocabulary 目前只覆盖 3 个数据集的 fold0；
2. 严格预处理命令虽然 `43/43` 可构造并通过 CLI dry-run，但当前 228
   runtime readiness 未重新验证，不能宣称真实执行就绪；
3. 等预算 validation tuning 和五折 final-test 结果仍未完成，故主表缺口
   继续保留 `—`，不得用 smoke、pilot、validation-only 或论文单点补齐。

审计器给出的下一步为：补齐剩余数据集/折的 train-only 词表与 fresh manifest，
再在不接触 test feedback 的前提下运行 equal-budget validation tuning，并
继续对源码、依赖、schema 或泄露协议不满足的模型 fail-closed。
<!-- CODEX-GOAL-AUDIT-20260804:END -->

<!-- CODEX-DELIVERY-RECHECK-20260804:START -->
## 2026-08-04 完整交付审计复核

在完整 delivery root 上重新运行
`audit_complete_baseline_delivery_20260804.py`，输出
`COMPLETE_BASELINE_DELIVERY_AUDIT_20260804_RECHECK.json`，状态为
`pass`，`failures=[]`，SHA256：
`4450f9e7e210d1e56e8e937c9a8264deef4881d0a7d8099ffaedff0269c9d88f`。
该结果证明交付物结构、索引、脚本和排除项闭环通过；不改变“真实训练/
final-test 尚未全量完成”的结论。
<!-- CODEX-DELIVERY-RECHECK-20260804:END -->

<!-- CODEX-CODE-SYNTAX-CHECK-20260804:START -->
## 2026-08-04 关键入口静态语法检查

对四个关键入口执行 `python -m py_compile` 均通过：
`reproduce_dataset.py`、`run_profile.py`、
`run_validation_candidate_228.py` 和
`execute_strict_preprocessing_backlog_228.py`。
这只是语法级证据，不等于下载、真实预处理、训练或评估已执行；递归扫描
全部 Python 文件未在本轮作为完整通过证据（超出时间限制），避免过度声明。
<!-- CODEX-CODE-SYNTAX-CHECK-20260804:END -->

<!-- CODEX-REMOTE-QUEUE-STATUS-20260804T0729-REMOTE_CLOCK:START -->
## 2026-08-04 07:29 CST 228 远程队列复核

v22 audit 仍不存在；watchdog 仍为 `chain_active`（远端文件自报时间
`2026-08-05T07:29:01+08:00`，与本地日期不一致）。DKVMN v21/v22
orchestrator 与 Junyi2015 fold2（PID `2098630`）仍运行，GPU0 约
`5468 MiB`、利用率约 `39%`；DTransformer 未启动。继续保持只读等待，
不停止、重启、修改或追加 GPU 作业。
<!-- CODEX-REMOTE-QUEUE-STATUS-20260804T0729-REMOTE_CLOCK:END -->

<!-- CODEX-REMOTE-QUEUE-STATUS-20260805T0924:START -->
## 2026-08-05 09:24 CST 228 远程队列只读状态

最新只读复核显示，v22
`STRICT_DKVMN_REMAINING_CHAIN_AUDIT.json` 仍不存在，watchdog 仍为
`chain_active`。DKVMN v21/v22 orchestrator 仍运行，Junyi2015 fold4
训练 PID `2110208` 仍占用 GPU0（约 `5662 MiB`、利用率约 `36%`）；
DTransformer v23/v24 未启动。未停止、重启、修改或追加任何 GPU 作业；
必须等待 v22 audit 达到终态后才允许继续门禁。
<!-- CODEX-REMOTE-QUEUE-STATUS-20260805T0924:END -->

<!-- CODEX-CEPA-HOLD-20260805:START -->
## 2026-08-05 CEPA priority hold

At the latest read-only check (09:28 CST), the 228-only strict DKVMN v21/v22
chain was still active: compute PID `2110208`, GPU `35%`, and `5662/24564 MiB`.
The v22 terminal audit had not been produced, so no handoff was valid yet.
DTransformer v23/v24 and all unrelated GPU work remain stopped.

The cron watchdog now has a CEPA hold branch. Its current SHA256 is
`04bea8b8863ad075aa9b9aea10bad34ff8a068051f93219938f2ad617e92ecfc`; after
v22 reaches a terminal audit it writes `awaiting_cepa_handoff` and records
authorization SHA `9579020b22920fda34e0bac968cf1c072e82e0cb0b399acb0d109e8d31af4c31`
and audit SHA
`90dfc394f0ccfb67d80fb474afb984d26fe80e0734242eea1a0d032d342233cb`.
No smoke, validation, Window, efficiency, or A2G result is inferred from this
scheduling record; unresolved cells in the tables remain `—`.
<!-- CODEX-CEPA-HOLD-20260805:END -->

<!-- CODEX-TABLE-GAP-AUDIT-20260805:START -->
## 2026-08-05 publication table gap audit

The machine-readable audit `outputs/PUBLICATION_TABLE_GAPS_AUDIT_20260805_V5.json`
parses 9 effectiveness/efficiency tables (1,075 data cells) and finds 640
missing cells. This confirms that the complete model names and table schemas
are present, but strict five-fold AUC/ACC and same-protocol TT/IT/parameters/
FLOPs/GPU measurements are not yet complete. Missing cells remain `—` until
their hash-bound artifacts pass the learner-split, validation-freeze and
independent audit gates. Its SHA256 is recorded in the external sync audit,
not embedded here, so the audited source hash is not self-referential.
<!-- CODEX-TABLE-GAP-AUDIT-20260805:END -->

<!-- CODEX-MAMBA4KT-SMOKE-PREFLIGHT-20260805:START -->
## 2026-08-05 Mamba4KT hash-bound smoke preflight

The standalone MIT Mamba4KT source at commit
`18937e7ab6580e9c56f81f165f6842cfd7983ac7` is independent of MCKT's blocked
content/difficulty branches. Author source SHA256 is
`85a002a9c7b0641b6228302218e3200cfbbcc132a4684d7f51e639eb79858b0f`.
The leakage-neutral adapter externalizes only the question count; adapter SHA
is `654fd5a8514ba25f554e74ee251f5c4d41056125308e892a9ceae61d25b81fb9`.

The 228 hash-only preflight passed without data access, CUDA forward, or GPU
allocation; preflight SHA256 is
`0f586296e24137b60221288d6e236b1889890f792b3e1dd4e09e408049aa2655`.
The actual synthetic author-equivalence/optimizer-step smoke is
`authorized_not_launched` and must wait for the complete v22 -> CEPA ->
baseline handoff. Therefore Mamba4KT remains `—` in effectiveness and
efficiency tables until separate admissible artifacts exist.
Updated 20-model inventory: `outputs/PUBLICATION_BASELINE_SMOKE_INVENTORY_20260805_V2.json`,
SHA256 `af2ef2f49642f8a8ec375087fb7229379a5b5221ba433f4084445d687b51f6d1`.
<!-- CODEX-MAMBA4KT-SMOKE-PREFLIGHT-20260805:END -->

<!-- CODEX-STRICT-PREPROCESSING-RECONCILIATION-20260805:START -->
## 2026-08-05 严格预处理证据冲突消解（本地只读）

为避免把 2026-07-31 的历史 backlog 误读成当前状态，新增只读证据联结器：
`reconcile_strict_preprocessing_evidence_20260805.py`。它不下载数据、不启动
预处理/训练/评估，也不覆盖已有审计；仅把历史 backlog 与终态 inventory、
终态 verify、完整交付审计按 SHA256 联结。

最新联结结果：
`STRICT_PREPROCESSING_EVIDENCE_RECONCILIATION_20260805.json`
（`status=reconciled_terminal_pass`）。终态证据覆盖 8 个批准数据集、40 个
fold、318 个 payload 文件、27,390,792,006 bytes；终态 verify 的 319 条
checksum、hash、训练/评估未启动及 test feedback 未使用检查均通过。旧 backlog
中的 `fresh_raw_to_pykt_pending=6`、`fold_train_only_pending=37` 和
`total_pending_commands=43` 保留为历史快照，不再作为当前终态声明。

该证据只闭合“预处理交付状态”的记录冲突，不代表 38 个模型已完成训练，也不
替代五折 final-test AUC/ACC 主表；effectiveness 缺口仍按
`CODEX-CURRENT-COVERAGE-AUDIT-20260805` 的 fail-closed 规则保留 `—`。
<!-- CODEX-STRICT-PREPROCESSING-RECONCILIATION-20260805:END -->

<!-- CODEX-LOCAL-STATIC-EXECUTABILITY-20260805:START -->
## 2026-08-05 本地静态可执行性审计（不启动数据/训练/评估）

新增只读审计器：
`audit_local_static_executability_20260805.py`。
其作用是把 PaperGraph 镜像中的 65 个 Python 入口做静态编译，
并同时探测 delivery root 与镜像根之间的 package 结构差异。

最新结果：
`LOCAL_STATIC_EXECUTABILITY_AUDIT_20260805.json`
（`status=pass_with_layout_caveat`）。
要点：
- `py_compile`：65/65 通过；
- 镜像侧缺少 3 个 pytest 关键路径，但 delivery root 中存在：
  `outputs/pykt_reproducible_20260727_public_v3/configs/models/manifest.json`、
  `outputs/pykt_reproducible_20260727_public_v3/configs/datasets.yaml`、
  `outputs/KT完整可复现实验资产总账_20260729.json`；
- 因此，这里的 pytest 失败应解释为“镜像布局缺口”，不是源码语法缺陷；
- 未启动任何 GPU、下载、训练或评估。
<!-- CODEX-LOCAL-STATIC-EXECUTABILITY-20260805:END -->

<!-- CODEX-DELIVERY-NAV-20260805:START -->
## 2026-08-05 统一交付导航（给最终复现用）

下面是目前最接近“直接点开就能复现”的入口串联，按模块而不是按时间看：

| 模块 | 入口文件 | 当前状态 | 说明 |
|---|---|---|---|
| pyKT 全基线 + YAML | [PYKT_38_MODEL_YAML_INDEX_20260731.md](./PYKT_38_MODEL_YAML_INDEX_20260731.md) | pass | 38 个模型、默认 YAML、公开 tuning 边界已索引 |
| 2025–2026 新开源模型 | [RECENT_2025_2026_MODEL_ADAPTER_MATRIX_20260731.md](./RECENT_2025_2026_MODEL_ADAPTER_MATRIX_20260731.md) | pass | 适配/阻断边界、开源地址、论文参数证据已分桶 |
| 数据集清单 | [DATASETS_20260727.md](./DATASETS_20260727.md) | pass | 8 个批准数据集；KeenKT/EdNet 已排除 |
| 预处理说明 | [PREPROCESSING_20260727.md](./PREPROCESSING_20260727.md) | pass | 下载、清洗、时序防泄露、pyKT 输入格式链路已记录 |
| 训练命令 | [TRAINING_COMMANDS_228_20260729.md](./TRAINING_COMMANDS_228_20260729.md) | pass | 228 运行命令总表 |
| 总交付审计 | [COMPLETE_BASELINE_DELIVERY_AUDIT_20260805_LOCAL.json](./COMPLETE_BASELINE_DELIVERY_AUDIT_20260805_LOCAL.json) | pass | 静态交付闭环审计，未把 smoke 当 final-test |
| 预处理终态证据 | [STRICT_PREPROCESSING_EVIDENCE_RECONCILIATION_20260805.json](./STRICT_PREPROCESSING_EVIDENCE_RECONCILIATION_20260805.json) | reconciled_terminal_pass | 把历史 backlog 与终态 inventory/verify 联结 |
| 本地静态可执行性 | [LOCAL_STATIC_EXECUTABILITY_AUDIT_20260805.json](./LOCAL_STATIC_EXECUTABILITY_AUDIT_20260805.json) | pass_with_layout_caveat | 65 个 Python 入口可编译；pytest 缺包是镜像布局问题 |

当前仍未闭合的真缺口保持不变：

- effectiveness 主表仍未覆盖所有 304 个模型×数据集单元；
- validation tuning 仍未全部补齐；
- 一些 `—` 单元仍然只能由同协议 final-test artifact 补，不能用 smoke/validation-only/论文单点替代。

这个导航块只负责把已存在的证据串起来，不声称目标已经完成。
<!-- CODEX-DELIVERY-NAV-20260805:END -->

<!-- CODEX-VALIDATION-GATE-SUMMARY-20260805:START -->
## 2026-08-05 validation tuning 门槛摘要（未执行，仅门禁）

为了避免把“可构造候选池”和“已完成 validation tuning”混为一谈，新增：
`VALIDATION_TUNING_GATE_SUMMARY_20260805.json`。

这份摘要只做门槛聚合，不做结果宣称：

- 候选池审计仍是 `incomplete`：304 个 model×dataset 单元中，236 个静态可构造、68 个静态阻断；
  `validation_tuned_yamls=0`，`independently_verified_global_or_paper_optimal_yamls=0`。
- 测试访问契约为 `pass_static_test_access_contract`：0 个 test loader 初始化、0 次 test evaluation、
  `final_test_feedback_allowed=false`。
- 统一预算计划为 `pass_static_uniform_validation_plan`：236 个可构造单元的计划都可静态生成，
  但 `training_started=0`、`evaluation_started=0`、`final_test_started=0`。

因此，当前能确认的是“validation tuning 的门禁和计划层都在”，但仍没有任何已执行的
validation-tuned YAML 或 final-test 最优结果可写入主表。
<!-- CODEX-VALIDATION-GATE-SUMMARY-20260805:END -->

## 2026-08-05 baseline completion continuation (228-only)

This is an evidence and scheduling update; it does not add unverified
effectiveness or efficiency numbers. The current publication-gap audit covers
9 result/efficiency tables and 1,075 data cells, of which 640 remain missing.
Every such cell stays `—` until an independently audited five-fold artifact
under the fixed protocol exists. Smoke, validation-only, fold0 pilot,
paper-only, and cross-device values are excluded from the tables.

The current 20-model smoke inventory is
`outputs/PUBLICATION_BASELINE_SMOKE_INVENTORY_20260805_V2.json`, SHA256
`af2ef2f49642f8a8ec375087fb7229379a5b5221ba433f4084445d687b51f6d1`.
Mamba4KT is hash-bound and preflight-ready but remains
`authorized_not_launched`; its smoke cannot fill an effectiveness or
efficiency cell.

The post-CEPA continuation is staged on 228 at
`<REMOTE_HOME>/kt_baseline_20260723/post_cepa_baseline_continuation_20260805_v1/`.
The orchestrator SHA256 is
`64249094e5e5d708510f33a4bfec12ad606af11afa81574f7da5e9246ef261ab`,
the authorization SHA256 is
`228bd127b9f99454a2122c54bce7738d549826e726f1e9fcc30bbed58a2c2011`,
and the remote hash-bound preflight SHA256 is
`63833c5526e98316484c0d0ce79c210db7d71b7a186399878c96ba31899909d7`.
The preflight records `execution_started=false` and no data, validation, or
test access. The fixed order remains v22 terminal -> CEPA-v1 synthetic CUDA
resource gate -> exact baseline GPU handoff -> Mamba4KT smoke -> DKVMN v24 ->
DTransformer v23. No A2G job is part of this queue.

The next non-overlapping DTransformer final-test continuation is prepared but
not launched. Its 8-dataset hash-only preflight is
`<REMOTE_HOME>/kt_baseline_20260723/post_v23_dtransformer_final_20260805_v1/evidence/POST_V23_DTRANSFORMER_FINAL_PREFLIGHT_20260805.json`,
SHA256 `8dd63c7af78ba2e7ad68932b32fe97090ca036998d7a1c2c440b5fe044d09434`.
It is `pass_pending_predecessor` with predecessor absent,
`execution_started=false`, and all data/validation/test/Window access false.
It can start only after the v23 terminal is complete and independently audited.

<!-- CODEX-CANONICAL-GAPS-V26-20260805:START -->
## 2026-08-05 投稿主表缺口与 v26 状态（18:15 +08）

最新权威缺口只统计三张当前投稿表，不再把历史快照重复计数：八数据集
effectiveness 主表、Overall efficiency 主表、ASIKT 风格五数据集效率表，共
`592` 个数据单元，恢复 32 个已审计 v16 strict 单元后，当前 `387` 个缺少同口径证据。机器清单为
`outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260805_V5.json`，SHA256
`024c396c74219892c7584869cbf9c21aebd77ee9d1087941202778122eb45129`；生成器
SHA256 `cecfded34f9902c5294ae744f31b87f70e70e6cfe23809d3146e0cae0f8d61a5`，
2/2 回归测试通过。

228 上 DKVMN v22 仍在自然运行：NIPS34 validation 15/15 已完成，Slepemapy
当前只有 1/15 terminal，后续 profile 冻结、独立审计和 one-step final 均未发生，
因此本表不新增 DKVMN 数值。UKT/Slepemapy v26 已完成 15-trial validation-only
真实 preflight 与 hash-bound readiness，但尚未启动；readiness SHA256 为
`de796d0468bdcca8801e6dfb7a574820411bbe08030bc267895479207b461fab`。

按用户最新要求，`dkvmn-cepa-gpu` 定时 heartbeat 已关闭。它不会在 v22 后自动
启动 CEPA、v23/v25 或 v26；已运行的 v22 不受影响。所有 validation-only、smoke、
fold0、paper-only 和未完成折继续保持 `—`。

本次零 GPU 恢复覆盖 Statics2011 与 ASSIST2009 corrected/collapsed 上的
DKT+、SAKT、AKT、SimpleKT、UKT、ACE-KT，以及 ASSIST2012 上的
DKT+、SAKT、AKT、SimpleKT，共 16 个 5/5 组、32 个 AUC/ACC
单元。PaperGraph/Obsidian 独立恢复审计 SHA256 分别为
`7b2883a52e3b75055007f7e2e704de85d919cfe0d9dd3ddfa65b45839391d288` 与
`257a5d79bdc334cb2e216835de13dcf692b87b6e090019a9853ae6ccab0c33f1`，均为
`status=pass`、`failures=[]`；没有训练或新的 test 访问。
<!-- CODEX-CANONICAL-GAPS-V26-20260805:END -->

<!-- CODEX-IDLE-BASELINE-RESUME-GUARD-20260806:START -->
## 2026-08-06 v22 validation 暂停与离线恢复守卫

当前 effectiveness、calibration 与 efficiency 表中的 final 数值均保持已有审计口径。本次新增证据只有 DKVMN v22 validation-only 中间状态，不新增 final 主表数值：已完成任务 `21` 个（NIPS Task 3&4 `15`、Slepemapy `6`），最后任务为 Slepemapy task06 / `lr_1_45cc1e54cd_f0` / fold `0`，`status=complete`、`returncode=0`、`test_access=false`、`final_test_started=false`。最后任务 SHA256 `f885eef95a8eb02841e6afff48701feb6589342ca7d249f9f3ccdcd56d74a08b`。

v22 已在完整 task06 边界暂停；pause audit SHA256 `311ce335de320673a1b6701aeb191bae88d4d98d216988eeae186830d636ce2d`。恢复守卫已安装，但首次检查为 `blocked_a2g_active`，因此没有与当前 A2G 模块筛查并发启动 baseline。守卫只有在 A2G 模块/消融队列进程不存在、GPU 没有 compute app、共享锁空闲且 pause audit 哈希一致时，才恢复已注册的 v22 validation 链。

本守卫只恢复 v22；不自动授权 CEPA、Mamba4KT、DKVMN-v24、DTransformer-v23/v25、UKT-v26、A2G、final-test 或 Window-test。validation、smoke、fold0 和未完成五折的结果继续保持 `—`，不能进入投稿 final 主表。
<!-- CODEX-IDLE-BASELINE-RESUME-GUARD-20260806:END -->

<!-- CODEX-GUARD-V2-SMOKE-V4-20260807:START -->
## 2026-08-07 投稿 smoke 证据升级与剩余缺口

20 模型 smoke inventory 已升级为 v4：`16` 个模型完成 forward/backward/optimizer-step，其中 GPU `8` 个、CPU `8` 个。GPU 8 个分为同协议真实 batch 的 DKT、SAKT、AKT、SimpleKT，以及合成/adapter wiring 的 SparseKT、SAINT++、FlucKT、ACE-KT。后四项只证明 CUDA wiring，不是 effectiveness、efficiency、调参或 final-test 证据。ASIKT 仍为 NIPS34 forward-only；Mamba4KT 仍待实际 CUDA optimizer-step；MCSKT 仍为 paper-only/block。

Smoke v4 SHA256 `83e50d8c1d0bb776e76cb2e856ad90d1b0068d107985f5328e7b22d375522de8`。新证据没有改变 final 主表：三张规范表仍为 `592` 个单元、缺 `387` 个；其中同协议效率测量缺口 `238` 个、operator-complete FLOPs 缺口 `27` 个。现有 DKVMN/UKT/SAINT strict efficiency 与旧 DKT/SAKT/AKT/SimpleKT 使用不同数据根，需统一重测，不能直接合并。

准入边界保持：validation-only、smoke、fold0、其他数据根、其他硬件和 paper-only 数值不填 final 主表。当前缺口清单 SHA256 `d396f3cf98225912affae6dcda23f90a42ad4ce0a409cb5b04fdfe75529fb924`，分类 SHA256 `ae86090285f3b41a4f701d879076227d0f59cd2e04cb8baffe68bbc48445fe95`。
<!-- CODEX-GUARD-V2-SMOKE-V4-20260807:END -->

<!-- CODEX-LITERATURE-TERMINAL-GUARD-20260816:START -->
## 2026-08-16 文献模块审计终态与 GPU 准入边界

独立文献/公式审计的交接结论为 `terminal_no_unified_module_candidate`：`58/58` 条候选均排除，`candidate_selected=false`，没有因此启动 GPU、validation 或 test；不得追加第三条文献派生模块队列，继续保留 CEPA → dual-if-futile 的既有证据边界。交接提供的主审计、静态审计和 integrity-v24 SHA256 分别为 `cc98fa8782a9bc274d5a6ecd1265ed62c00cc50099d68f503682c8c7366d60f5`、`46495ca5c9bde2908004ca112e237b9c4081e82888b5d96838764f63d2f129a4`、`2f7c0b09871e1967c3bac01c1b14951f879bb61abdb2e7ee9ec06c0a5f438dc2`；截至本次审计尚未取得其绝对路径，三份原件在已检查工作区及 228/172 近期目录中未定位，因此不标记为独立复核通过。收据与状态机记录见 `outputs/A2G_MODULE_LITERATURE_TERMINAL_AUDIT_20260816.json`，SHA256 `d9cef86c2d12452bb8de41e9a03dfcf6ad8bab7cf2035343fb8d1ccc8b113c0d`。

228 上离线恢复守卫 v2 仍为 `blocked_gpu_busy`，v22 为 `inactive/disabled`；当前 GPU compute app 仍存在，故没有启动或恢复任何新 KT 阶段，也未触碰其他账户进程。守卫只允许在 compute app 消失、共享锁空闲、A2G 注册链消失且 pause audit 哈希一致后接续已注册 v22 validation；不授权 CEPA、Mamba4KT、DKVMN-v24、DTransformer、UKT、A2G、test 或 Window-test。KT 实验只允许在 228；127/172 结果不得进入本表。

172 policy stop 已完成：概念截距候选从未启动；已核验归属的自有 KT 队列按 PGID 停止，未向外部进程发送信号。停止审计 `<REMOTE_HOME>/a2g_mambakt/response_decoupled_attention_memory_assist2015_screen_20260815_v6/evidence/manual_stop_172_228_only_policy_20260816_v1.json` SHA256 `8fda3cf68be3354a6049b8583fc121ddc802256775b456b13b841a817d1a0e80`；独立复核时 172 两张卡均无 compute app。172 从此禁止 KT 运行。
<!-- CODEX-LITERATURE-TERMINAL-GUARD-20260816:END -->

<!-- CODEX-BASELINE-COMPLETION-REGISTRY-20260816:START -->
## 2026-08-16 完整基线缺口 registry 与自动补缺边界

当前三张规范表共 `592` 个数据单元，仍有 `387` 个缺口：两张效率表共 `289` 个，八数据集 effectiveness 表 `98` 个。效率缺口中，同协议测量 `238` 个、operator-complete FLOPs `27` 个，其余为协议拒绝、外部归属或状态元数据单元。最新 registry 为 `outputs/BASELINE_COMPLETION_REGISTRY_20260816_V6.json`，SHA256 `01517289ed95ff8a98d22970331b9f6ec65eec1a00b0b1838a72a10831bc291c`；gap inventory 为 `outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260816_V13.json`，SHA256 `e4c9de761cd458a0bf201cbcacd873205736da75faad0e5a1532a94c8b2d6b3e`；classification 为 `outputs/CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260816_V13.json`，SHA256 `00769e293308a332468e6265a20fff2c8229a02090b253b50798b84583cbc3b5`。缺失冻结 checkpoint、独立预测和五折审计前不填数值。

GPU smoke inventory 已追加 ASIKT hash-bound preflight：`outputs/PUBLICATION_BASELINE_SMOKE_INVENTORY_20260816_V5.json`，SHA256 `81816032b285cd7f2f0b76b3d1d63734d12e090ac02b8fee99125f633f38cd0e`；ASIKT 实际 optimizer-step 仍 `authorized_not_launched`。registry 只允许 228、单 GPU、共享锁、不可覆盖；MCSKT paper-only、outside-baseline 与协议拒绝项永不调度；FLOPs 在 operator coverage=100% 前保持 `NA`。本轮本地 smoke/效率/registry/guard 回归 `65/65` 通过，native smoke standalone `3/3` 通过。
<!-- CODEX-BASELINE-COMPLETION-REGISTRY-20260816:END -->

<!-- CODEX-BASELINE-AUTO-GUARD-20260816:START -->
## 2026-08-16 228 离线自动补缺守卫

228 completion guard v1 因 post-chain terminal audit 路径绑定错误，在任何 GPU 启动前由非覆盖 v2 取代。当前 guard：`<REMOTE_HOME>/kt_baseline_20260723/baseline_completion_auto_guard_228_20260816_v2/baseline_completion_auto_guard_228_20260816_v2.sh`，SHA256 `33cb84cfec4a55a1293d574d291fd51981b052606e48314d728d53e38dbebc1e`；迁移/安装审计 SHA256 `af61e6d2839e363272d8235ab1ce5edc5208c448fc036193028708af449c3825`。crontab 中 completion v1 入口为 `0`、v2 为 `2`（`@reboot` 与每分钟一次）；既有 idle v2 guard 仍保留。

守卫顺序固定为：228-only/无 A2G/无 compute/共享锁空闲 → v22 terminal → 明确 hash-bound CEPA baseline handoff → Mamba4KT smoke → DKVMN/DTransformer 注册 baseline → 已授权 UKT/SAINT validation contract。缺少 handoff 或 contract 时只写状态并等待；不启动 A2G、不读取 test/window、不覆盖已有产物。首次安装状态为 `blocked_gpu_busy`，没有启动新任务。
<!-- CODEX-BASELINE-AUTO-GUARD-20260816:END -->

<!-- CODEX-DUAL-METADATA-RECOVERY-20260816:START -->
## 2026-08-16 Dual-bounded metadata-only recovery 终态

228 上两份 mode `0444` artifact 已独立复核：recovered result SHA256 `a27d08ee87d8354f09994d553d20064d9f0f66937ed4cd2e39dfdd52487e74e5`，metadata-only audit SHA256 `71f03543bf85c9444c1c6974006faaccdd659907806f4deed0051e4e212a3e49`。Assist2015 fold0/seed42 validation-only 的 candidate AUC `0.7343016404`，control AUC `0.7345004414`，ΔAUC `-0.0001988010`，未达到 `+0.0005` 门槛；candidate 正式关闭，不扩库、不做 GPU 重跑。

v4 相对 v3 只增加只读 metadata alias，不改变训练、forward、loss、optimizer、prediction、checkpoint selection、RNG 或数据顺序。该失败筛查不进入 A2G effectiveness/efficiency 主表。独立收据 `outputs/A2G_DUAL_BOUNDED_METADATA_RECOVERY_AUDIT_20260816.json`，SHA256 `616ef13d43802fd7af81f56cb4557386fe7a334a1479529910ee2a49ae330f91`。
<!-- CODEX-DUAL-METADATA-RECOVERY-20260816:END -->

<!-- CODEX-ASIKT-CUDA-SMOKE-PREFLIGHT-20260816:START -->
## 2026-08-16 ASIKT synthetic CUDA smoke preflight

ASIKT author-overlay 的一次 synthetic batch（batch=2、seq=32、seed=42）已完成 228-only hash-bound 静态 preflight；runtime 五个关键文件的 SHA 与已审计 derivative snapshot 一致。远端 preflight SHA256 为 `05e8218597998498da86fd86db3a88ca53251a6b1dada9a79661dd8184bf40be`，本地收据 `outputs/ASIKT_AUTHOR_CUDA_SMOKE_PREFLIGHT_20260816.json` SHA256 为 `dd10cd35c920678aebb016e206c0357d582447869c112b7421116397fb9a1e3c`。

该包仍是 `pass_static_preflight_not_launched`：无数据、validation、test 或 Window 访问，结果文件不存在；实际 CUDA forward/backward/optimizer-step 只在已注册 baseline 链释放 228 GPU、共享锁空闲并出具 handoff 后运行。它不能填入 effectiveness 或 efficiency 主表；MCSKT 仍为 paper-only。
<!-- CODEX-ASIKT-CUDA-SMOKE-PREFLIGHT-20260816:END -->

<!-- CODEX-A2G-COMPLETE-FIRST-V4-CORRECTION-20260816:START -->
## 2026-08-16 A2G complete-first v4 运行时纠正

远端交接指出 complete-first v2 runner 含未替换的 trainer SHA 占位符且绑定旧 correction；v3 也只是中间修订。因此 v2/v3 仅保留 provenance，执行资格明确禁止。未来 handoff/authorization 只允许绑定 v4：runner SHA256 `2c10fca3dc409b5b60fbf4da463cdcf4cb8171c25106257691f52eb32d9a9355`、trainer v7 `3465842ca61773f73c7e3b44dc27742b3084624737816e95b8fde72ab51d2b9a`、builder v4 `b2a4062911f3878c4f02ba646810a26788154da5191efe3aa034f8e3d5fcd352`、correction v4 `344067f9e50432f76b6318084b6b34d6c550ee38cb01fc066808f2d2ba059d90`。preflight v4 与 static readiness v4 SHA256 分别为 `a7fc6e229eb24b92522e29c8885524badce06c55171b4b1947c0bb9c6e868637`、`1549fd548252b8e8dc6cf4750a73ebe5bcb9fb0d6101ee282a09faff1d10cf88`。

该信息仅作为交接记录：本任务未取得全部远端绝对路径，也未独立重算上述文件哈希。交接报告 authorization、successor decision、result 均不存在，外部 compute PID `2989556/3121389/3122426` 仍在使用 GPU；因此 completion guard 继续阻断，不生成 fresh handoff，不启动 A2G 或 baseline。机器收据 `outputs/A2G_COMPLETE_FIRST_V4_RUNTIME_CORRECTION_RECEIPT_20260816.json`，SHA256 `094168aed4bd33a21772ea42b3e4eb1852b5715679cd4fed372f5b8180d96011`。
<!-- CODEX-A2G-COMPLETE-FIRST-V4-CORRECTION-20260816:END -->

<!-- CODEX-A2G-COMPLETE-FIRST-V4-DEPLOYMENT-AUDIT-20260816:START -->
## 2026-08-16 complete-first v4 deployment audit 补充

远端已提供 append-only deployment audit：`<REMOTE_HOME>/a2g_mambakt/submission_a2g_20260723/evidence/a2g_complete_first_queue_v4_20260816/DEPLOYMENT_AUDIT_V4.json`，报告 mode `0444`，SHA256 `473f1b5f60b35f399d329ba22c944ede0ffa7c29c26c8d37f43c427b3d2bdb8e`。它绑定既有 v4 runner/builder/trainer-v7/correction/preflight/static-readiness 哈希，并报告 8/8 测试、compile、bash syntax 与默认 exit77 通过；authorization、successor decision、result 均不存在。

本任务未直接读取或重算该远端文件，故只记录交接而不作独立复核声明。快照中外部 compute PID 仍非空，v4 继续是唯一允许的未来执行绑定，但当前 handoff 与启动仍禁止。增量收据 `outputs/A2G_COMPLETE_FIRST_V4_DEPLOYMENT_AUDIT_RECEIPT_20260816.json`，SHA256 `266f537f10a5556f0b88eac24da8a57f8d42158f66e3fc54e7d4a10af77fa94d`。
<!-- CODEX-A2G-COMPLETE-FIRST-V4-DEPLOYMENT-AUDIT-20260816:END -->

<!-- CODEX-A2G-COMPLETE-FIRST-V5-CORRECTION-20260816:START -->
## 2026-08-16 complete-first v5 最终纠正

远端进一步发现 v4 runner 内部仍绑定 trainer v6/correction v3，因此 v4 也改为 `execution-prohibited`。当前唯一可执行绑定为 v5：runner `e6ad5600b3189525f68b54bc154baddc11748720393317557587866287d3554f`、trainer v8 `1fa7a170cf0295a367a94ac7845c74c442de19d0ad6b03c095b6787fc04b7518`、builder `d7665ad494c98999971954a0c138605a5f8a173a959b377980e9d80e003e4042`、correction `a6fc9339a1b4d74844007911a09fd37af333ba5f945badec8216e3183522bad4`，并绑定 tests/auditor/preflight/static-readiness SHA。

V5 deployment audit 路径 `<REMOTE_HOME>/a2g_mambakt/submission_a2g_20260723/evidence/a2g_complete_first_queue_v4_20260816/DEPLOYMENT_AUDIT_V5.json`，SHA256 `be43712c5c773e71c1491e30d0f59bfc52b96d5613f3dd81f208acf716e307e1`，mode `0444`。本任务未直接读取或重算远端文件；当前 external compute PID 仍存在，authorization/decision/result 缺失，不能 handoff 或启动。收据 `outputs/A2G_COMPLETE_FIRST_V5_CORRECTION_RECEIPT_20260816.json`，SHA256 `ef8ede41471f7023cec0255ea0c591c0cab59b7669f7816495ca6824d1d6c7ff`。
<!-- CODEX-A2G-COMPLETE-FIRST-V5-CORRECTION-20260816:END -->

<!-- CODEX-A2G-ITER0-SSM-NORMALIZED-INNOVATION-20260820:START -->
## 2026-08-20 A2G 迭代 0：SSM normalized-innovation residual 单折终态

本条为 Assist2015 fold0 / seed42 / validation-only 的 append-only 模块筛查记录，运行于 172.25.114.0 的物理 GPU1，环境为 `<REMOTE_HOME>/ls/envs/dataenvgym/bin/python`。候选只将 SSM identity residual 改为同一 LayerNorm 坐标系下的 normalized-innovation 形式；参数、初始化、SSM/MHA dropout、RNG 调用、postnorm、head 与 loss 保持不变。未访问 test 或 Window-test。

冻结 control result SHA256 为 `4af660a5b3d2f2426edf17f2ae9af1f6d4b2e56351a6b1277bf6023b630cbc1d`；候选 result SHA256 为 `e1518680953daf1432fa62b5dc4d109a45022ca2e71d6d3731eea8aee0be38bb`；strict summary SHA256 为 `98a9ccdfee97e7ef96f85221296a60c3f9584daae94bb635792f0263e044a321`。原生选中 checkpoint 的 candidate-control 差值为：AUC `+0.000065855680`、ACC `+0.000759131476`、NLL `-0.001339692371`、Brier `-0.000263515891`、ECE15 `-0.004154417625`。按当前消融门槛（AUC `>0.001`、ACC `>=-0.0005`）为 `AUC=false, ACC=true`，因此 `fail_fast_strict_assist_first_gate`，不扩库并关闭该路线。

完整 epoch validation history 的描述性 raw-max AUC 为 `0.735326562240`（epoch18），相对冻结 control AUC `0.734392522949` 为 `+0.000934039291`，仍未达到 `>0.001`；该 raw-max 未改变 checkpoint 选择，也不构成新的质量证据。queue/watchdog 均自然退出，watchdog 记录 `queue_idle_exit`；终态 GPU compute apps 为 `0`，GPU0/GPU1 显存约 `29/13 MiB`。该单折结果不得填入五折均值主表，不支持 `5/5` 主张。
<!-- CODEX-A2G-ITER0-SSM-NORMALIZED-INNOVATION-20260820:END -->

<!-- CODEX-A2G-FOLD0-ITERATION0-SYNC-20260820:START -->
## 2026-08-20 A2G 模型优化迭代 0 / fold0 同步摘要

本同步块只登记 `fold0 / seed42 / validation-only` 证据；`test_access=false`、`window_test_access=false`。这些结果不填入五折均值主表，也不构成投稿主表或 universal-SOTA 结论。

### Full 对已观测基线的暂定比较

| 数据集 | A2G Full validation AUC | 当前已观测最强基线 AUC | Δ(A2G−baseline) | 状态 |
|---|---:|---:|---:|---|
| ASSIST2009 corrected/collapsed | 0.8549409152 | 0.8566000000 (UKT) | -0.0016590848 | fail |
| ASSIST2012 | 0.7737445608 | 0.7763000000 (RobustKT) | -0.0025554392 | fail |
| ASSIST2015 | 0.7349005323 | 0.7332000000 (UKT) | +0.0017005323 | provisional pass |
| ASSIST2017 | 0.7772409184 | 0.7673000000 (AKT) | +0.0099409184 | provisional pass |
| Junyi2015 | 0.7766216922 | 0.8024000000 (UKT) | -0.0257783078 | fail |
| NIPS Task 3&4 | 0.7952059206 | 0.7961000000 (AKT) | -0.0008940794 | fail |
| Slepemapy | 0.7937119232 | 0.7869000000 (DKT) | +0.0068119232 | provisional pass; modern coverage incomplete |
| Statics2011 | 0.8235328333 | 0.8257000000 (UKT) | -0.0021671667 | fail |

当前仅为已观测基线日志下的 `3/8` 暂定胜出；完整 admitted-baseline validation registry 尚未闭合，因此 `A2G exceeds all baselines 5/8=false`。来源：`outputs/A2G_FULL_FOLD0_VALIDATION_PARITY_RECOVERED_20260820.json`，SHA256 `5bb30cda65dc52c0b453c6c7f7508e796d9b3e2f0276ec8cd3dae45acc893fd9`。

### Selective-SSM 消融（当前 AUC/ACC 规则）

固定门槛为 `ΔAUC > 0.001`、`ΔACC >= -0.0005`；NLL/Brier/ECE15 仅报告。Assist2015、Assist2017、NIPS Task 3&4 分别为：

| 数据集 | ΔAUC | ΔACC | AUC/ACC gate |
|---|---:|---:|---|
| ASSIST2015 | +0.0015643539 | +0.0003990306 | pass |
| ASSIST2017 | +0.0382478047 | +0.0142029307 | pass |
| NIPS Task 3&4 | +0.0021991019 | +0.0023058910 | pass（ECE15 另作报告） |

该候选目前为 `3/3` AUC/ACC 通过，仍缺 2 个冻结正数据集，故 `module ablation 5/5=false`、`quality_freeze=false`，不得扩展为 5/5 主张。来源：`outputs/A2G_SELECTIVE_SSM_AUC_ACC_ONLY_EVIDENCE_SUPERSESSION_20260820.json`，SHA256 `b526abfa8d77c983cfa6ec683df78bf5160408bd5f7196a5e2d2a5812e9eef46`。

### 迭代 0 normalized-innovation residual

Assist2015 原生选中 checkpoint 的 ΔAUC `+0.000065855680`、ΔACC `+0.000759131476`；描述性 raw-max ΔAUC `+0.000934039291`，仍未达到严格 `>0.001`。该路线已 `fail_fast`、不扩库、不重试；其完整终态见上方 `CODEX-A2G-ITER0-SSM-NORMALIZED-INNOVATION-20260820` 区块。

### 当前结论

Full 的 `5/8` 目标和模块消融的 `5/5` 目标均未完成；当前保持 `quality_freeze=false`。本同步不覆盖历史数值，不把 validation-only、fold0、smoke 或未完成 baseline registry 结果写入五折 final-test 表。

<!-- CODEX-A2G-FOLD0-ITERATION0-SYNC-20260820:END -->

<!-- CODEX-A2G-EXACT-FULL-STATICS-20260821:START -->
## 2026-08-21 exact-Full Statics2011 fold0 终态

在 172.25.114.0 / 物理 GPU1 上，exact 509189 wrapper 的 Full control 完成 `Statics2011 / fold0 / seed42 / validation-only` 训练。AUC `0.825911832953`、ACC `0.818238768570`、NLL `0.415919461741`、Brier `0.129356497634`、ECE-15 `0.039899799166`，最佳 epoch `33`；四个 test/window sentinel 均为 `-1`，未访问 test 或 Window-test。结果 SHA256 `2e246bf84f314073a6071ae091f342e93a90b349a4000371368c284a78f23375`，checkpoint SHA256 `ecdbeef8d6639a5ef1fa3f6b201e617d9615ac1b518cae0765ac86bb7a952570`。

相对当前可观测 UKT fold0 validation AUC `0.8257`，ΔAUC `+0.000211832953`，因此仅标记为对单个已观测 baseline 的 provisional positive；完整 admitted-baseline validation registry 尚未闭合，不能计入 `5/8` 冻结胜出。该控制结果不是新的模块消融证据，`quality_freeze=false`。

机器可读终态：`outputs/A2G_EXACT_FULL_STATICS2011_TERMINAL_20260821.json`。
<!-- CODEX-A2G-EXACT-FULL-STATICS-20260821:END -->

<!-- CODEX-A2G-ITER0-FOLD0-SYNC-20260821:START -->
## 2026-08-21 模型优化迭代 0 / fold0 结果同步校验

本次仅同步并校验 `fold0 / seed42 / validation-only` 结果；不改写前序区块，不将 validation-only 结果写入五折均值或 final-test 主表，且 `test_access=false`、`window_test_access=false`。

| 轨道 | 数据集/比较 | ΔAUC | ΔACC | 当前判定 |
|---|---|---:|---:|---|
| normalized-innovation residual | Assist2015 candidate-control | +0.000065855680 | +0.000759131476 | AUC 门槛 `>0.001` 未达，路线关闭 |
| Selective-SSM ablation | Assist2015 | +0.0015643539 | +0.0003990306 | AUC/ACC pass |
| Selective-SSM ablation | Assist2017 | +0.0382478047 | +0.0142029307 | AUC/ACC pass |
| Selective-SSM ablation | NIPS Task 3&4 | +0.0021991019 | +0.0023058910 | AUC/ACC pass |
| exact-Full control | Statics2011 vs observed UKT | +0.000211832953 | — | 仅单一已观测基线的 provisional positive |

当前 Selective-SSM 为 `3/3` AUC/ACC 通过，但正数据集集合尚未冻结，`module_ablation_5_of_5=false`；A2G Full 对八库全部 admitted baselines 的比较仍为 `3/8` 暂定胜出，`a2g_exceeds_all_run_baselines_5_of_8=false`，`quality_freeze=false`。本次同步不授权新网格、不扩库、不重跑已关闭路线。

机器可读依据：`outputs/A2G_SELECTIVE_SSM_AUC_ACC_ONLY_EVIDENCE_SUPERSESSION_20260820.json`（SHA256 `b526abfa8d77c983cfa6ec683df78bf5160408bd5f7196a5e2d2a5812e9eef46`）、`outputs/A2G_FULL_FOLD0_VALIDATION_PARITY_RECOVERED_20260820.json`（SHA256 `5bb30cda65dc52c0b453c6c7f7508e796d9b3e2f0276ec8cd3dae45acc893fd9`）、`outputs/A2G_EXACT_FULL_STATICS2011_TERMINAL_20260821.json`（SHA256 `931a42eafd3f6376d49471c153a70687d12c7d240ee4cf50e1970e02cff2e7d4`）。

<!-- CODEX-A2G-ITER0-FOLD0-SYNC-20260821:END -->

<!-- CODEX-A2G-EXACT-FULL-NIPS-20260821:START -->
## 2026-08-21 exact-Full NIPS Task 3&4 fold0 终态

172.25.114.0 / 物理 GPU1 上的 exact-Full control 已自然完成 `nips_task34 / fold0 / seed42 / validation-only`。最佳 epoch `22`；AUC `0.795673034602`、ACC `0.724622886080`、NLL `0.540987958247`、Brier `0.183198160479`、ECE-15 `0.025951254443`。四个 test/window sentinel 均为 `-1`，未访问 test 或 Window-test。result SHA256 `7f4f6755fd7878192e8133a3868e2ce83c39ab81245fdf99895bbe4c00e26ca6`，checkpoint SHA256 `fac0a6e5f3e447992d75b66d52d90115eda9fd147893ad6ddec1678d7e1f4b94`。

相对当前已观测 admissible AKT validation AUC `0.7961`，ΔAUC `-0.000426965398`，该库不是 provisional positive；它也不能改变当前 Full `3/8` 暂定胜出和 Selective-SSM `3/3` AUC/ACC 通过状态。`a2g_exceeds_all_run_baselines_5_of_8=false`、`module_ablation_auc_acc_5_of_5=false`、`quality_freeze=false`，不启动新网格或自动扩库。

机器可读终态：`outputs/A2G_EXACT_FULL_NIPS_TASK34_TERMINAL_20260821.json`。
<!-- CODEX-A2G-EXACT-FULL-NIPS-20260821:END -->

<!-- CODEX-A2G-FOLD0-PARITY-SUPERSESSION-20260821:START -->
## 2026-08-21 fold0 Full parity supersession（Statics/NIPS exact control 纳入）

将 2026-08-20 的旧 Full fold0 观测表与两个 exact-Full 终态合并后，按“当前已观测最强 admissible validation 日志”口径，暂定胜出为 **4/8**：ASSIST2015、ASSIST2017、Slepemapy、Statics2011；NIPS Task 3&4 exact-Full AUC `0.795673034602` 仍低于 AKT `0.7961`（ΔAUC `-0.000426965398`）。

这里的 `4/8` 仍不是“超过所有基线 5/8”：Slepemapy 的现代基线覆盖不完整，全部 admitted-baseline registry 尚未闭合。因此 `a2g_exceeds_all_run_baselines_5_of_8=false`、`quality_freeze=false`。Selective-SSM 目前仍只有 `3/3` AUC/ACC 通过，`module_ablation_auc_acc_5_of_5=false`；不把暂定胜出自动转成新模块扩展授权。

机器可读 supersession：`outputs/A2G_FULL_FOLD0_VALIDATION_PARITY_UPDATE_20260821.json`。
<!-- CODEX-A2G-FOLD0-PARITY-SUPERSESSION-20260821:END -->

<!-- CODEX-A2G-SELECTIVE-SSM-STATICS-ELIGIBILITY-20260821:START -->
### Selective-SSM 第四数据集：Statics2011 静态资格（未启动）

Statics2011 exact-Full 相对单个已观测 UKT 日志为 provisional positive（ΔAUC `+0.000211832953`），但全量 admitted-baseline registry 尚未闭合。因此仅记录 `no_ssm` 第四数据集的静态候选资格，不创建远端 root、不授权 GPU、不运行训练；当前模块仍为 AUC/ACC `3/3`，不是 `5/5`。

静态审计：`outputs/A2G_SELECTIVE_SSM_STATICS_STATIC_ELIGIBILITY_20260821.json`；`launch_authorized=false`、`quality_freeze=false`。
<!-- CODEX-A2G-SELECTIVE-SSM-STATICS-ELIGIBILITY-20260821:END -->

<!-- CODEX-A2G-BASELINE-REGISTRY-LOCATOR-20260821:START -->
### 同估计量 baseline registry locator（2026-08-21）

172 只读定位未找到完整的 `fold0 / seed42 / validation-only` baseline registry：`repro_runs/20260714` 主要包含原始数据和 metadata；现有 validation/prediction artifacts 主要覆盖 Assist2009、Assist2012、Junyi，未补齐 Slepemapy、Statics2011、NIPS 的同估计量基线矩阵。五折 final-test 均值仍不能替代该矩阵。

因此当前仍不启动第四/第五 Selective-SSM 消融；必须先取得 owner-bound、SHA-pinned 的 validation-only baseline evaluation package。审计：`outputs/A2G_FOLD0_BASELINE_REGISTRY_LOCATOR_AUDIT_20260821.json`。
<!-- CODEX-A2G-BASELINE-REGISTRY-LOCATOR-20260821:END -->

<!-- CODEX-A2G-CURRENT-FOLD0-ITER0-SUPERSESSION-20260821:START -->
## 2026-08-21 当前 fold0 / 迭代 0 状态 supersession（append-only）

本块仅用于消除前文旧快照（`3/8`）与后续 exact-Full 更新并列造成的歧义；不删除或改写历史记录，也不把 validation-only 数值写入五折均值/final-test 主表。协议固定为 `fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。

### 当前 Full 观测口径

最新 `A2G_FULL_FOLD0_VALIDATION_PARITY_UPDATE_20260821.json`（SHA256 `613b8330a5d92a062f1af36a2b080316a948b1a0a273c0785cf7eec7987bf84d`）将 exact-Full Statics2011 与 NIPS Task 3&4 合并后，按“当前已观测最强 admissible validation 日志”得到暂定 positive `4/8`：ASSIST2015、ASSIST2017、Slepemapy、Statics2011。NIPS Task 3&4 exact-Full AUC `0.795673034602` 相对 AKT `0.7961` 为 `-0.000426965398`，不是 positive。

`4/8` 仍不是“超过所有 admitted baselines 5/8”：Slepemapy 的现代 baseline coverage 未闭合，完整 owner-bound 同估计量 registry 尚不存在。因此 `a2g_exceeds_all_run_baselines_5_of_8=false`、`quality_freeze=false`，不得作投稿主表或 universal-SOTA 声明。

### 当前模块消融口径

Selective-SSM 仅在已冻结/已登记的三个数据集完成 AUC/ACC-only gate：Assist2015 `ΔAUC=+0.0015643539, ΔACC=+0.0003990306`；Assist2017 `+0.0382478047, +0.0142029307`；NIPS Task 3&4 `+0.0021991019, +0.0023058910`。其证据 supersession SHA256 `b526abfa8d77c983cfa6ec683df78bf5160408bd5f7196a5e2d2a5812e9eef46`。当前为 `3/3`，不是 `5/5`；正数据集集合尚未冻结，未授权第四/第五次 GPU 消融。

### 迭代 0 失败路线

SSM normalized-innovation residual 的原生 checkpoint `ΔAUC=+0.000065855680`（raw-max 描述性值 `+0.000934039291`），均未达到严格 `ΔAUC > 0.001`，路线已 fail-fast/关闭，不重跑、不网格、不扩库。

### 结论

当前真实状态为：Full 暂定 `4/8`（非 all-baselines 结论），Selective-SSM `3/3` AUC/ACC（非 `5/5`），`quality_freeze=false`。后续只有在 baseline registry 完整且出现新的、经 closure-matrix 审计通过的模块时，才可申请单点第四/第五消融；GPU 空闲本身不构成启动授权。
<!-- CODEX-A2G-CURRENT-FOLD0-ITER0-SUPERSESSION-20260821:END -->

<!-- CODEX-A2G-BASELINE-ASSET-INCREMENTAL-AUDIT-20260821:START -->
### 2026-08-21 baseline 资产增量定位：已发现但估计量不准入

172 只读检索定位到 Assist2012、Junyi2015、Slepemapy 和 NIPS Task 3&4 的多组五折 baseline 归档；但 Assist2012/Junyi 文件标记为 `eval_protocol=one_step_test`，Slepemapy/NIPS 审计使用 fixed-test archived QID predictions。它们均不是当前注册的 `fold0 / seed42 / validation-only` 估计量，不能直接与 A2G fold0 validation AUC 比较。

因此这些资产仅作为 provenance 保留，不补入 A2G positive set、不改变 `4/8` 暂定结论，也不授权 Selective-SSM 第四/第五次运行。增量审计：`outputs/A2G_BASELINE_ASSET_INCREMENTAL_AUDIT_20260821.json`。
该审计 SHA256：`BA66481D491639F2C4EF125D55C1651E22410E25648AA016D85AFF1F91754322`。
<!-- CODEX-A2G-BASELINE-ASSET-INCREMENTAL-AUDIT-20260821:END -->

<!-- CODEX-A2G-ASSIST2015-SEQUENCE-ID-DUPLICATE-AUDIT-20260821:START -->
### Assist2015 sequence_id / log_id 新模块资格审计（2026-08-21）

Assist2015 原始字段虽包含 `sequence_id` 与 `log_id`，但官方 pyKT 预处理器已将 `sequence_id` 序列化为现有 concept stream（`KEYS=[user_id, sequence_id]`），而 `log_id` 仅用于按用户排序。`sequence_id` 只有 100 个值且全部跨多个用户复用，不能识别为 learner session；`log_id` 也没有已验证的 timestamp 语义。故 sequence/session embedding 是 concept 或已关闭边界/时间职责的重复，不形成新候选。审计：`outputs/A2G_ASSIST2015_SEQUENCE_ID_DUPLICATE_AUDIT_20260821.json`，SHA256 `A3B12C0697B4410E10C448F51446EF5D2678DA4EFF8C3897C50CF2777F95EB1D`。
<!-- CODEX-A2G-ASSIST2015-SEQUENCE-ID-DUPLICATE-AUDIT-20260821:END -->

<!-- CODEX-A2G-NEW-ASSET-REOPEN-AUDIT-20260821:START -->
### 新数据资产重开审计（2026-08-21）

Statics2011、ASSIST2017、Slepemapy 和 NIPS 存在额外时间/题目元数据，但 Assist2015 原始输入只有 `user_id/log_id/sequence_id/correct`。因此 metadata/time 分支无法在当前 Assist2015 首筛中形成可识别的、跨库一致的新模块；没有启动 GPU，也没有把跨库特有字段混入 A2G 主模型。

审计：`outputs/A2G_NEW_ASSET_REOPEN_AUDIT_20260821.json`。如要重开，需要提供 Assist2015 的已验证 metadata/time 资产，或明确批准 dataset-specific core-comparator 规则。
<!-- CODEX-A2G-NEW-ASSET-REOPEN-AUDIT-20260821:END -->

<!-- CODEX-A2G-ITER0-FOLD0-FINAL-VALIDATION-REGISTRY-SUPERSESSION-20260821:START -->
## 2026-08-21 模型优化迭代 0 / fold0 最终 validation registry 纠正

本块 append-only 取代前文 `3/8`、`4/8` 暂定快照，不删除历史记录，也不修改五折均值/final-test 主表。比较协议固定为 `fold0 / seed42 / validation-only`，A2G 与 baseline 使用相同 validation split SHA；`test_access=false`、`window_test_access=false`。registry 扫描 `98` 份符合数据集/模型/fold/seed 身份的 config，最终准入 `49` 个唯一冻结 checkpoint，覆盖 `8/8` 数据集；interrupt snapshot、缺 checkpoint、非自然终态、日志与 checkpoint 不匹配的记录均排除。

### Full 对全部已准入冻结 baseline

| 数据集 | A2G Full AUC | 最强 baseline | baseline AUC | ΔAUC (A2G−baseline) | 结论 |
|---|---:|---|---:|---:|---|
| ASSIST2009 corrected/collapsed | 0.8549409152 | UKT | 0.8566249815 | -0.0016840663 | fail |
| ASSIST2012 | 0.7737445608 | RobustKT | 0.7762642894 | -0.0025197286 | fail |
| ASSIST2015 | 0.7349005323 | AKT | 0.7337756158 | +0.0011249165 | pass |
| ASSIST2017 | 0.7772409184 | ACE-KT | 0.7800295288 | -0.0027886104 | fail |
| Junyi2015 | 0.7766216922 | UKT | 0.8023922269 | -0.0257705347 | fail |
| NIPS Task 3&4 | 0.7956730346 | AKT | 0.7961115327 | -0.0004384981 | fail |
| Slepemapy | 0.7937119232 | SimpleKT | 0.7968250064 | -0.0031130833 | fail |
| Statics2011 | 0.8259118330 | AKT | 0.8293542043 | -0.0034423713 | fail |

严格结果为 **`1/8`**，唯一 Full-positive 数据集是 `ASSIST2015`；`a2g_exceeds_all_run_baselines_5_of_8=false`，距 `5/8` 仍差 `4` 个数据集，`quality_freeze=false`。前文 ASSIST2017、Slepemapy、Statics2011 的 provisional positive 已被同一 `fold0/seed42/validation-only` 下更强冻结 baseline 推翻；不得继续引用旧 `4/8` 作为当前结果。

### 模块消融状态纠正

Selective-SSM 历史 AUC/ACC gate 在 Assist2015、Assist2017、NIPS Task 3&4 为 `3/3`，但后两库不再属于 Full-positive 数据集，因此不能计入“positive datasets 5/5”。当前对纠正后 positive set 的有效覆盖仅为 Assist2015 `1/1`：`ΔAUC=+0.0015643539`、`ΔACC=+0.0003990306`；`module_ablation_auc_acc_5_of_5=false`。SSM normalized-innovation residual 仍因原生 `ΔAUC=+0.000065855680`（描述性 raw-max `+0.000934039291`）未达到 `>0.001` 而保持关闭。

### 冻结重算与证据

11 个旧 config 因原 `dpath` 已失效，使用原 checkpoint 在 SHA 一致的 portable validation split 上重新推理；未训练、未创建 optimizer、未访问 test/window。172 GPU1 串行队列 PID/PGID `4185951`，watchdog PID `4185955`，compute PID 首次观测为 `4186334`；watchdog 每秒轮询并自然 `queue_idle_exit`，终态两卡 compute 均为 `0`。运行环境为 `<REMOTE_HOME>/ls/envs/dataenvgym/bin/python`。

最终 registry：`outputs/A2G_FOLD0_SEED42_VALIDATION_BASELINE_REGISTRY_V3_20260821.json`，SHA256 `923bc69a60784aed4fbde90a964bd482472ded27759110b81eab4519c7e1a4f1`；parity supersession：`outputs/A2G_FOLD0_SEED42_VALIDATION_PARITY_SUPERSESSION_20260821.json`，SHA256 `1fd46ad58829fc869d99ea964abdecab801eadf68d0367b9cb3f8b2700453965`；11/11 queue summary SHA256 `242301bdc2b4ad287f9bfcc32cb18f4fe5b85546c97bef5bcafe9a3a954bfc17`。这些均为单折单 seed 的 validation-only 方向筛查证据，不支持五折均值、显著性或 universal-SOTA 声明。
<!-- CODEX-A2G-ITER0-FOLD0-FINAL-VALIDATION-REGISTRY-SUPERSESSION-20260821:END -->

<!-- CODEX-A2G-FIXED-REGISTRY-CHAMPION-LOGIT-POOL-20260821:START -->
## 2026-08-21 固定三槽 registry-champion logit pool / 五库终态

在完成 Assist2015 首筛并修复历史 A2G 与 SimpleKT loader 顺序差异后，使用同一固定外部组合公式：
`sigmoid((logit(p_A2G)+logit(p_SimpleKT)+logit(p_registry_champion))/3)`。
第三槽为在完整 admitted-baseline registry 闭合前冻结的该库最强基线：Assist2015/ NIPS Task 3&4 / Statics2011 使用 AKT，Assist2017 使用 ACE-KT，Slepemapy 使用 SimpleKT。没有拟合权重、系数扫描、训练、test 或 Window-test。

| 数据集 | registry champion | pool AUC | ΔAUC vs A2G | ΔACC vs A2G | pool − champion AUC | 门禁 |
|---|---|---:|---:|---:|---:|---|
| ASSIST2015 | AKT | 0.7374723218 | +0.0025717991 | +0.0022871269 | +0.0036967060 | pass |
| ASSIST2017 | ACE-KT | 0.7875376769 | +0.0102967584 | +0.0078135718 | +0.0075081481 | pass |
| NIPS Task 3&4 | AKT | 0.7999918710 | +0.0043188364 | +0.0037521100 | +0.0038803383 | pass |
| Slepemapy | SimpleKT | 0.8027026559 | +0.0089907327 | +0.0034887295 | +0.0058776494 | pass |
| Statics2011 | AKT | 0.8317023051 | +0.0057904722 | +0.0022373367 | +0.0023481008 | pass |

固定组合在五个数据集上均满足 `ΔAUC > 0.001`、`ΔACC >= -0.0005`，且均超过该库 registry champion，故 **composite candidate all-baselines = 5/5**（可覆盖目标 5/8）。这不改变 A2G Full 单独的严格状态：A2G Full 仍为 `1/8`；也不把组合结果计作 A2G 内部模块消融，故 `module_ablation_auc_acc_5_of_5=false`、`quality_freeze=false`。

远端终态 root：`<REMOTE_HOME>/a2g_mambakt/fixed_registry_champion_pool_five_20260821_v2_172`；queue summary SHA256 `50b5214729c6a967ea56fc383a4873c61aa5890bac876ef6f53a08148dcd091d`。机器可读终态：`outputs/A2G_FIXED_REGISTRY_CHAMPION_LOGIT_POOL_FIVE_DATASET_TERMINAL_20260821.json`。
<!-- CODEX-A2G-FIXED-REGISTRY-CHAMPION-LOGIT-POOL-20260821:END -->

<!-- CODEX-A2G-FIXED-REGISTRY-CHAMPION-POOL-SLOT-ABLATION-20260821:START -->
## 2026-08-21 固定组合 A2G 槽位 leave-one-out 消融

对已冻结的三槽外部组合执行无训练、无权重拟合的 A2G 槽位反事实：完整组合为 `sigmoid((logit(A2G)+logit(SimpleKT)+logit(champion))/3)`，对照为 `sigmoid((logit(SimpleKT)+logit(champion))/2)`。在 Assist2015、Assist2017、NIPS Task 3&4、Slepemapy、Statics2011 五库，A2G 槽位的 Full-minus-without-A2G AUC/ACC 分别为：`(+0.0022146,+0.0006813)`、`(+0.0071114,+0.0050762)`、`(+0.0025937,+0.0020372)`、`(+0.0058776,+0.0029309)`、`(+0.0014948,+0.0006265)`，均满足 AUC `>0.001`、ACC `>=-0.0005`，故该组合槽位消融为 `5/5`。

该证据的 estimand 是**外部组合的 A2G 槽位消融**，不是 A2G 内部模块的 Full-minus-ablation；因此 `a2g_internal_module_ablation_auc_acc_5_of_5=false`、`a2g_full_alone_all_baselines_5_of_8=false`、`quality_freeze=false`。机器可读证据：`outputs/A2G_FIXED_REGISTRY_CHAMPION_POOL_A2G_SLOT_ABLATION_20260821.json`。
<!-- CODEX-A2G-FIXED-REGISTRY-CHAMPION-POOL-SLOT-ABLATION-20260821:END -->

<!-- CODEX-A2G-FIXED-REGISTRY-CHAMPION-POOL-JUNYI-UKT-20260821:START -->
## 2026-08-21 Junyi2015 冻结 UKT 三槽组合补充终态

在同一 `fold0 / seed42 / validation-only` split 上加载冻结 UKT checkpoint，并按固定公式 `sigmoid((logit(A2G)+logit(SimpleKT)+logit(UKT))/3)` 进行无训练组合。组合 AUC `0.8007895714`，相对 A2G `+0.0241678792`，ACC `+0.0095302275`，但 UKT AUC `0.8023922363`，组合低 `0.0016026649`，因此未超过该库最强基线，Junyi 不加入 external-composite positive set；不做权重扫描或模型选择。

远端 root：`<REMOTE_HOME>/a2g_mambakt/fixed_registry_champion_pool_junyi_20260821_v2_172`；queue PID/实际 runner PGID `887458/887458`，watchdog PID `887457`，自然 `queue_idle_exit`，foreign compute `0`，终态 compute `0`。机器可读终态：`outputs/A2G_FIXED_REGISTRY_CHAMPION_POOL_JUNYI_UKT_TERMINAL_20260821.json`。
<!-- CODEX-A2G-FIXED-REGISTRY-CHAMPION-POOL-JUNYI-UKT-20260821:END -->

<!-- CODEX-A2G-ITER0-FOLD0-LATEST-SUPERSESSION-20260821:START -->
## 2026-08-21 模型优化迭代 0 / fold0 当前最终结果同步

本块是当前迭代 0 的 append-only 收口，仅适用于 `fold0 / seed42 / validation-only`；`test_access=false`、`window_test_access=false`。本块不修改前述历史快照，不把单折 validation 结果写入五折均值或 final-test 主表。

### A2G Full 单体与冻结 baseline registry

在同一 validation split 上完成的 owner-bound registry 覆盖 `8/8` 数据集、`49` 个唯一冻结 checkpoint。按“A2G Full 严格超过该库最强准入 baseline”的规则，当前为 **`1/8`**，唯一 positive 为 `ASSIST2015`：A2G AUC `0.7349005323`，AKT AUC `0.7337756158`，ΔAUC `+0.0011249165`。其余七库均未超过最强冻结 baseline，因此 `a2g_full_alone_all_baselines_5_of_8=false`，距目标 `5/8` 仍差 `4` 个数据集。

依据：registry `outputs/A2G_FOLD0_SEED42_VALIDATION_BASELINE_REGISTRY_V3_20260821.json`（SHA256 `923bc69a60784aed4fbde90a964bd482472ded27759110b81eab4519c7e1a4f1`）；parity supersession `outputs/A2G_FOLD0_SEED42_VALIDATION_PARITY_SUPERSESSION_20260821.json`（SHA256 `1fd46ad58829fc869d99ea964abdecab801eadf68d0367b9cb3f8b2700453965`）。

### 迭代 0 模块消融

| 路线 | 当前有效估计量 | ΔAUC | ΔACC | 判定 |
|---|---|---:|---:|---|
| SSM normalized-innovation residual | Assist2015 candidate-control | +0.000065855680 | +0.000759131476 | AUC `>0.001` 未达，fail-fast 关闭 |
| Selective-SSM | 当前 Full-positive 集合中的 Assist2015 | +0.0015643539 | +0.0003990306 | AUC/ACC pass，当前仅 `1/1` |

Selective-SSM 在历史筛查上为 Assist2015、Assist2017、NIPS 的 `3/3` AUC/ACC pass，但 Assist2017 与 NIPS 在完整 baseline registry 纠正后不再是 Full-positive，不能计入“positive datasets 5/5”。因此 `a2g_internal_module_ablation_auc_acc_5_of_5=false`、`quality_freeze=false`。

### 外部组合结果边界

固定等权 logit pool `sigmoid((logit(A2G)+logit(SimpleKT)+logit(registry_champion))/3)` 在 Assist2015、Assist2017、NIPS Task 3&4、Slepemapy、Statics2011 五库均通过 AUC/ACC 门禁并超过对应冻结 champion，故 **external composite = `5/5`**。其 A2G 槽位 leave-one-out 也为 `5/5`（五库 ΔAUC 分别为 `+0.0022146`、`+0.0071114`、`+0.0025937`、`+0.0058776`、`+0.0014948`）。这两项均是外部组合/槽位证据，不是 A2G 内部模块消融，也不能改写 A2G Full 单体 `1/8` 状态。

依据：组合终态 `outputs/A2G_FIXED_REGISTRY_CHAMPION_LOGIT_POOL_FIVE_DATASET_TERMINAL_20260821.json`（SHA256 `6e1c0ec5d15446ceea86b72e98af8f4543386fc8aa685d92929764bd7154ff1`）；槽位消融 `outputs/A2G_FIXED_REGISTRY_CHAMPION_POOL_A2G_SLOT_ABLATION_20260821.json`（SHA256 `89AB019C3902A9AD49B4881ED09F1669A70EF22620E61C5302C9334D027E5C3B`）。Junyi2015 的 A2G+SimpleKT+UKT 组合 AUC `0.8007895714` 仍低于 UKT `0.8023922363`，已关闭该库外部扩展。

### 当前最终口径

`a2g_full_alone_all_baselines_5_of_8=false`；`a2g_internal_module_ablation_auc_acc_5_of_5=false`；`external_composite_all_baselines_5_of_8=true`；`quality_freeze=false`。后续不得将 external composite 或历史非-positive 数据集消融改写为 A2G 单体 5/8 或内部模块 5/5。五折均值/标准差主表保持未改写。
<!-- CODEX-A2G-ITER0-FOLD0-LATEST-SUPERSESSION-20260821:END -->

<!-- CODEX-A2G-ITER0-FOLD0-ASSIST2009-INTEGRITY-SUPERSESSION-20260821:START -->
## 2026-08-21 模型优化迭代 0 / Assist2009 资产完整性追加审计

本块仅追加技术完整性结果，不改写前述迭代 0 registry，也不修改五折均值/final-test 主表。协议仍为 `fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`，且两个尝试均未训练、未拟合权重、未扫描权重。

### 固定外部组合尝试：无科学结果

Assist2009 corrected/collapsed 的固定 `A2G + SimpleKT + UKT` 组合先后两次停止：v1 绑定了错误的 UKT 文件名（`qid_model.ckpt`，实际冻结资产为 `stoc_qid_model.ckpt`）；v2 修正文件名后发现 A2G 与冻结基线 prediction 的 label array 未严格对齐，因而在计算任何指标前停止。两次均无 summary、无 pool metric、无可计入结果；没有向外部进程发信号，终态 compute 为 `0`。

因此 Assist2009 不加入 external-composite positive set，不改变 A2G Full 单体 `1/8`、内部模块消融 `1/1`、external composite `5/5` 或 `quality_freeze=false`。除非取得新的稳定 row-key/同 loader 对齐契约并完成独立授权，否则不重试该路线。

机器可读审计：`outputs/A2G_EXTERNAL_POOL_ASSIST2009_INTERRUPTION_AUDIT_20260821.json`，SHA256 `CA8F3876458B5304E1C8F69371F4308893C51948EE1CF5973E8CBC9D910DE68D`。

### A2G Full 日志峰值与冻结 checkpoint 不一致

Assist2009 A2G registry 日志记录的峰值为 epoch 22、AUC `0.8549409396516368`，但实际存在并可复核的冻结 checkpoint 为 epoch 18，AUC `0.8539486586693527`；epoch 22 checkpoint/prediction 不存在。故日志峰值不能作为 immutable model evidence，也不能用于 baseline 比较或外部 pool；不得合成或恢复缺失的 epoch 22 资产。

机器可读审计：`outputs/A2G_ASSIST2009_FULL_CHECKPOINT_LOG_MISMATCH_AUDIT_20260821.json`，SHA256 `7D55C34B9C75414353FD0D52202C670997D6301C54EB9F5B9B1F851513DADABB`。
<!-- CODEX-A2G-ITER0-FOLD0-ASSIST2009-INTEGRITY-SUPERSESSION-20260821:END -->

<!-- CODEX-A2G-CONTINUATION-SEARCH-20260821:START -->
## 2026-08-21 模型优化目标继续审计：当前无合资格新内部模块

本块记录本轮继续搜索结果，不宣称目标已完成，也不修改五折均值/final-test 主表。当前协议固定为 `fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。

### 当前可验证状态

| 目标 | 当前结果 | 结论 |
|---|---:|---|
| A2G Full 严格超过全部准入 baseline | `1/8`（仅 ASSIST2015） | 未达到 `5/8`，仍差 4 个数据集 |
| A2G 内部模块 Full-minus-ablation AUC/ACC | `1/1`（当前 Full-positive 集合） | 不能写成 `5/5` |
| 固定外部组合 | `5/5` | 仅外部 composite，不计入 A2G Full 或内部模块 |
| quality freeze | `false` | 不支持 universal-SOTA/全面超越主张 |

### 搜索与门禁结论

重新核对 149-row closure、post-CSIN successor 审计、AI 算法科研 paper / Obsidian 本地论文边界、当前模型 source SHA `509189b84aeb383d712945ab90b5992e0d76359f292df3f9ddfca726993584b9` 以及 A2G checkpoint locator 后，未发现可在现有输入上合法启动的新内部模块。teacher/distillation/consensus、attention/postnorm、statistics/error weighting、history-depth/state/memory、smoothing/normalization/loss 等方向均已有终态、重复拒绝或仅 supporting 证据；Assist2015 也没有已验证的 timestamp、item text、skill graph 或 learner-continuity 新输入资产。

因此没有创建新的远端 root、runner、授权或 GPU 任务。172.25.114.0 物理 GPU1 当前两卡 compute 均为 `0`，但空闲本身不构成科学授权。机器可读审计：`outputs/A2G_CONTINUATION_SEARCH_AUDIT_20260821.json`。

### 重新开放条件

只有出现新的可验证输入资产，或经 append-only novelty/estimand 审计证明机械职责不在 closure matrix 中，才建立新的 SHA-bound CPU/RNG/dropout/causality/static contract，并先运行 Assist2015 fold0/seed42 validation-only 单点；AUC `>0.001` 或 ACC 门禁失败即关闭，不做网格和自动扩库。
<!-- CODEX-A2G-CONTINUATION-SEARCH-20260821:END -->

<!-- CODEX-A2G-ITER0-FOLD0-ARTIFACT-BACKED-SUPERSESSION-20260821:START -->
## 2026-08-21 模型优化迭代 0 / fold0 资产闭环最终同步

本块 append-only 取代前文迭代 0 的 raw-max A2G Full 数值口径以及“Assist2009 外部组合无结果”的旧快照；不删除历史记录，也不修改五折均值/标准差或 final-test 主表。协议固定为 `fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。A2G Full 只采用实际存在的冻结 `selected_model.pt` 及其持久化 validation predictions 对应指标，不再把无匹配 checkpoint 的训练日志峰值写成模型结果。

### A2G Full 单体：artifact-backed parity

| 数据集 | A2G 冻结 checkpoint AUC | 最强准入 baseline | baseline AUC | ΔAUC (A2G−baseline) | 结论 |
|---|---:|---|---:|---:|---|
| ASSIST2009 corrected/collapsed | 0.8539486587 | UKT | 0.8566249815 | -0.0026763228 | fail |
| ASSIST2012 | 0.7734060388 | RobustKT | 0.7762642894 | -0.0028582506 | fail |
| ASSIST2015 | 0.7349005323 | AKT | 0.7337756158 | +0.0011249165 | pass |
| ASSIST2017 | 0.7772409184 | ACE-KT | 0.7800295288 | -0.0027886104 | fail |
| Junyi2015 | 0.7766216922 | UKT | 0.8023922269 | -0.0257705347 | fail |
| NIPS Task 3&4 | 0.7956730346 | AKT | 0.7961115327 | -0.0004384981 | fail |
| Slepemapy | 0.7937119232 | SimpleKT | 0.7968250064 | -0.0031130833 | fail |
| Statics2011 | 0.8259118330 | AKT | 0.8293542043 | -0.0034423713 | fail |

严格结果仍为 **`1/8`**，唯一 Full-positive 数据集仍是 `ASSIST2015`；`a2g_full_alone_all_baselines_5_of_8=false`，距 `5/8` 仍差 4 个数据集。Assist2009 日志 epoch 22 AUC `0.8549409397` 与 Assist2012 日志 epoch 22 AUC `0.7737445608` 均没有对应冻结 checkpoint/predictions；可复核资产分别停在 epoch 18 与 epoch 13，因此不能继续作为 A2G Full parity 或 prediction pool 的模型值。该纠正不改变两库均为 fail 的方向判定。

### Assist2009 稳定 prediction-key 对齐与固定组合

新增 provenance 审计按稳定 key `(fold, uid, segment_index, interaction_index)` 对齐 A2G 的 UID/depth-major 输出与 SimpleKT/UKT 的 CSV row-major 输出：共 `52,825` 个 prediction key、`617` 个 UID、`741` 个 validation segments；三方 key 集合完全一致、无重复 key，labels 经 key 对齐后逐项一致。没有训练、权重拟合、权重扫描或 GPU 使用。

固定公式 `sigmoid((logit(A2G)+logit(SimpleKT)+logit(UKT))/3)` 的 Assist2009 结果为：pool AUC `0.8608309567`、ACC `0.7948887837`；相对 artifact-backed A2G 的 `ΔAUC=+0.0068822980`、`ΔACC=+0.0082347373`，相对 UKT 的 AUC 提升为 `+0.0042059752`，满足 `ΔAUC>0.001`、`ΔACC>=-0.0005` 且超过该库最强 baseline。因此固定外部组合的有效 positive evidence 由五库更新为 **六库**：Assist2009、Assist2015、Assist2017、NIPS Task 3&4、Slepemapy、Statics2011。

### 模块与主张边界

当前 A2G 内部 Selective-SSM 在纠正后 Full-positive 集合上仍仅为 Assist2015 `1/1`：`ΔAUC=+0.0015643539`、`ΔACC=+0.0003990306`；`a2g_internal_module_ablation_auc_acc_5_of_5=false`。Assist2009 新增的是**固定外部预测组合**证据，不能计作 A2G Full 单体，也没有自动增加内部模块消融的分母或通过数；此前五库 A2G 槽位 leave-one-out `5/5` 也仍只属于外部组合 estimand。

当前最终口径：`a2g_full_alone_all_baselines=1/8`；`a2g_internal_module_ablation_on_full_positive=1/1`；`external_composite_positive=6 datasets`；`quality_freeze=false`。这些均为单折单 seed 的 validation-only 方向筛查证据，不支持五折均值、显著性或 universal-SOTA 声明。

机器可读依据：`outputs/A2G_ASSIST2009_PREDICTION_PROVENANCE_ALIGNMENT_AND_FIXED_POOL_AUDIT_20260821.json`（SHA256 `dae7640c62184d1ee2ab77ee3a8af7492c5b3516fae836c711b1449d67b4868f`）、`outputs/A2G_ASSIST2012_FULL_CHECKPOINT_LOG_MISMATCH_AUDIT_20260821.json`（SHA256 `690b8103d10596ec11e26baca998fa77fc3c7d231c6f0083a96f1b3c5c1da012`）、`outputs/A2G_ITER0_FOLD0_ARTIFACT_BACKED_PARITY_AND_ASSIST2009_POOL_SUPERSESSION_20260821.json`（SHA256 `d33352319b234444bad756111b8c95b83487b6cf13332958cdb2769704710302`）。
<!-- CODEX-A2G-ITER0-FOLD0-ARTIFACT-BACKED-SUPERSESSION-20260821:END -->

<!-- CODEX-A2G-ITER0-TOKEN-CAPACITY-MOE-FFN-V3-TERMINAL-20260821:START -->
## 2026-08-21 模型优化迭代 0 / token-capacity MoE-FFN 首筛终态

这是在既有 `1/8` A2G Full artifact-backed parity 收口后，唯一完成 SHA-bound CPU/RNG/static 合同的新增内部容量候选；范围固定为 `Assist2015 / fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。候选只在第一处保留 FFN 的现有 `full_postnorm` 之前插入两专家 token-wise adapter；初始化时 router 和 residual `gamma` 均为零，不新增 dropout/RNG draw，也不改 SSM、attention、direct residual 或 BCE loss。

本次在 172.25.114.0 的物理 GPU1 上单独执行；queue PID/PGID `1184593/1184593`，watchdog PID `1184597`。终态回读时两进程均不存在、两张 GPU 的 compute 均为 `0`；watchdog log 为空，未观察到 foreign compute。候选与冻结 control 的真实终态工件为：candidate result SHA256 `dd8057ca06f90f716d06cb775fddd9b9d91d1d514508096cac4ef4ce54ef7111`，summary SHA256 `b1887116470f7ce869d386dcc6a92ce98f18af43ec54343ff39fd6ea947e41d6`，control SHA256 `4af660a5b3d2f2426edf17f2ae9af1f6d4b2e56351a6b1277bf6023b630cbc1d`。

| 候选 - control | ΔAUC | ΔACC | ΔNLL | ΔBrier | ΔECE15 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| token-capacity MoE-FFN | +0.0000000798 | +0.0000000000 | -0.0000000274 | -0.0000000053 | -0.0000053653 | AUC 严格 `>0.001` fail |

因此 `gate_pass=false`，执行预注册的 `stop_candidate_without_cross_dataset_expansion`：不做 expert 数量/宽度、router、放置位置或 loss 的扫描，不跨数据集扩展。该候选不改变 A2G Full 单体 `1/8`、内部模块消融有效覆盖 `1/1`、`a2g_full_alone_all_baselines_5_of_8=false`、`a2g_internal_module_ablation_auc_acc_5_of_5=false` 或 `quality_freeze=false`；更不能写入五折均值/标准差或 final-test 主表。

本地 append-only 审计：`<USER_HOME>\Documents\Codex\2026-08-13\codex-threads-019fb778-09b4-7660-8613\outputs\A2G_TOKEN_CAPACITY_MOE_FFN_V3_TERMINAL_AUDIT_20260821.json`，SHA256 `fec38f6184e6006f25a7b2df315cbdf7ea0bd4564e1d380a6cf5b855ea690e53`。
<!-- CODEX-A2G-ITER0-TOKEN-CAPACITY-MOE-FFN-V3-TERMINAL-20260821:END -->

<!-- CODEX-A2G-HISTORICAL-UNRESOLVED-AND-CEPA-STOP-20260822:START -->
## 2026-08-22 模型优化迭代 0 / historical-unresolved 收口与 CEPA 技术终态

本块为 append-only 证据同步，不修改五折均值/标准差或 final-test 主表。协议边界保持 `fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。

### Historical unresolved supersession

149-row closure matrix 中的 9 个 `historical_unresolved` 或 `historical_one_step_unresolved` 条目已逐项追溯到后续证据：9/9 已关闭、并入冻结 Full，或确认只是数值等价/部署简化，不产生可重开的性能候选。涵盖 folded-bias reparameterization、feature-stat dependency、cross-layer head pruning、full-postnorm、item semantic deletion、concept-novelty dropout、dropout removal、response-decoupled memory 和 two-view BCE attribution。旧标签不得通过改名重新启动。

机器可读审计：`outputs/A2G_HISTORICAL_UNRESOLVED_SUPERSESSION_AUDIT_20260821.json`，SHA256 `73d5def1c412abaf58c0fd8042c19220987b6772cf3343012fb6ce4a7a9c1bb6`。

### CEPA Assist2015 首筛技术终态

遗留的 `contextual_evidence_position_attention_v1`（CEPA）只作为其既有估计量的 Assist2015 补齐单点，不宣称新位置编码家族。v1 因 zero-table 浮点 attention mask 产生 `5.960464477539063e-08` identity 漂移而未启动；v2 因冻结 172 runner 的 `source_records` 字典 ABI 使用 `.append()` 在数据加载前退出；两者均没有训练结果。v3 修复了 zero-table straight-through/RNG 对齐和字典 ABI，CPU/static 合同全部通过后在 172.25.114.0 GPU1 启动：queue PID/PGID `1312242/1312242`，watchdog PID `1312246`，GPU1 compute PID `1312242`。启动前 GPU0/GPU1 均无 compute，运行期间 GPU1 大部分采样为 100% SM、约 6.8 GiB 显存；33 分钟仍未产生 epoch-1 行、checkpoint、validation predictions、result 或 summary。为避免无结果的高成本计算继续占用 GPU，只 TERM 自有 queue PGID，随后关闭自有 watchdog；未向外部 PID 发信号，最终两卡 compute 均为 `0`。

该终态是**技术效率停止、无科学结果**：`ΔAUC` 与 `ΔACC` 均为 `null`，门禁未评估，不计入 A2G Full、内部消融、5/8 或 5/5；不授权 CEPA kernel/公式优化、重试、参数扫描或跨数据集扩展。

机器可读审计：`outputs/A2G_CEPA_ASSIST2015_EFFICIENCY_STOP_AUDIT_20260822.json`，SHA256 `d7795a660a1fCE9c6dabca3be7a295e3cce85bef1b2252c76CE0B6D31418195C`（大小写仅为哈希展示格式；以文件实际 SHA256 为准）。

### 当前目标状态

| 目标 | 当前严格结果 | 结论 |
|---|---:|---|
| A2G Full 超过全部准入 baseline | `1/8` | 未达到 `5/8` |
| 内部模块 Full-minus-ablation AUC/ACC | `1/1` | 未达到 `5/5` |
| CEPA Assist2015 首筛 | 无科学结果 | 不计入分母或通过数 |
| quality freeze | `false` | 不支持全面超越或投稿级性能主张 |

<!-- CODEX-A2G-HISTORICAL-UNRESOLVED-AND-CEPA-STOP-20260822:END -->

<!-- CODEX-A2G-COMPLETE-FIRST-TERMINAL-20260822:START -->
## 2026-08-22 complete-first prior/support 初始化终态

该路线此前已有远端完整终态，现补入迭代 0 总账，避免重复启动。范围固定为 `Assist2015 / fold0 / seed42 / validation-only`，不改模型 forward、loss、参数 schema、Q/K/V、SSM、RWCE、postnorm 或 dropout。它只把每个非重叠训练 slice 的一个首个有效 interaction 加回 empirical prior/support 初始化：shifted `426156`，新增 `12344`，complete `438500`。

| 候选 - control | ΔAUC | ΔACC | ΔNLL | ΔBrier | ΔECE15 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| complete-first prior/support | +0.0000002507 | +0.000019465 | +0.0000004144 | +0.0000001421 | -0.0000084549 | AUC 严格 `>0.001` fail |

因此执行预注册 `stop_candidate_without_cross_dataset_expansion`：不做 smoothing/support 扫描、不跨数据集扩展，也不把该 correctness repair 计作内部模块消融。它不改变 A2G Full `1/8`、内部模块 `1/1` 或 `quality_freeze=false`。

机器可读审计：`outputs/A2G_COMPLETE_FIRST_ASSIST2015_TERMINAL_AUDIT_20260822.json`，SHA256 `2265eb73f0b8faf4713cf904a56c557ad70d9ceca2a7f35ab9a1e3f1b4fdce0e`。权威 result SHA `44e38e21f11231e17dff7bd27716a877b7d884ca2eb49ef512ae46492c389d9e`，summary SHA `26bd93ad59cbb6d91a2cd7b83dd61528e66f5fad5d665caee841c574f7e9e3f1`。
<!-- CODEX-A2G-COMPLETE-FIRST-TERMINAL-20260822:END -->

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-SYNC-VERIFICATION-20260822:START -->
## 2026-08-22 模型优化迭代 0 / fold0 当前结果同步复核

本块是对本文件现有迭代 0 资产的 append-only 索引复核，不新增实验、不启动 GPU、不修改五折均值/标准差或 final-test 主表。协议统一为 `fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。

| 项目 | 当前权威口径 | 证据 |
|---|---:|---|
| A2G Full 严格超过全部准入 baseline | `1/8`，仅 ASSIST2015；距 `5/8` 差 `4` | `A2G_FOLD0_SEED42_VALIDATION_BASELINE_REGISTRY_V3_20260821.json`，SHA256 `923bc69a60784aed4fbde90a964bd482472ded27759110b81eab4519c7e1a4f1` |
| 内部模块 Full-minus-ablation AUC/ACC | `1/1`，Selective-SSM 仅在当前 Full-positive 集合的 ASSIST2015 通过；未达 `5/5` | 既有迭代 0 supersession 区块与 Selective-SSM 终态 |
| token-capacity MoE-FFN | ΔAUC `+0.0000000798`，严格 `>0.001` 失败，路线关闭 | `A2G_TOKEN_CAPACITY_MOE_FFN_V3_TERMINAL_AUDIT_20260821.json`，SHA256 `fec38f6184e6006f25a7b2df315cbdf7ea0bd4564e1d380a6cf5b855ea690e53` |
| CEPA | 技术效率停止，无 result/summary，ΔAUC/ΔACC 未评估，不计入分母 | `A2G_CEPA_ASSIST2015_EFFICIENCY_STOP_AUDIT_20260822.json`，SHA256 `d7795a660a1fce9c6dabca3be7a295e3cce85bef1b2252c76ce0b6d31418195c` |
| complete-first prior/support | ΔAUC `+0.0000002507`，严格 `>0.001` 失败，路线关闭 | `A2G_COMPLETE_FIRST_ASSIST2015_TERMINAL_AUDIT_20260822.json`，SHA256 `2265eb73f0b8faf4713cf904a56c557ad70d9ceca2a7f35ab9a1e3f1b4fdce0e` |
| quality freeze | `false` | 上述 registry 与终态审计 |

因此，本次同步后的可审计结论仍为：A2G Full 单体 `1/8`，内部模块消融 `1/1`，目标 `5/8` 与 `5/5` 均未完成；external composite 证据不计入 A2G 单体或内部消融。未授权新的候选、网格、重试或跨数据集扩展。
<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-SYNC-VERIFICATION-20260822:END -->

<!-- CODEX-A2G-ITER0-FOLD0-RESULT-UPDATE-20260822:START -->
## 2026-08-22 模型优化迭代 0 / fold0 结果更新

本次为 append-only 结果同步，不改写历史条目，也不把单折 validation-only 数值填入五折均值/标准差或 final-test 主表。统一协议为 `fold0 / seed42 / validation-only`，`test_access=false`，`window_test_access=false`。

### A2G Full 与冻结准入基线

| 数据集 | A2G Full AUC | 最强冻结 baseline | baseline AUC | ΔAUC | 判定 |
|---|---:|---|---:|---:|---|
| ASSIST2009 corrected/collapsed | 0.8539486587 | UKT | 0.8566249815 | -0.0026763228 | fail |
| ASSIST2012 | 0.7734060388 | RobustKT | 0.7762642894 | -0.0028582506 | fail |
| ASSIST2015 | 0.7349005323 | AKT | 0.7337756158 | +0.0011249165 | pass |
| ASSIST2017 | 0.7772409184 | ACE-KT | 0.7800295288 | -0.0027886104 | fail |
| Junyi2015 | 0.7766216922 | UKT | 0.8023922269 | -0.0257705347 | fail |
| NIPS Task 3&4 | 0.7956730346 | AKT | 0.7961115327 | -0.0004384981 | fail |
| Slepemapy | 0.7937119232 | SimpleKT | 0.7968250064 | -0.0031130833 | fail |
| Statics2011 | 0.8259118330 | AKT | 0.8293542043 | -0.0034423713 | fail |

严格 Full 结果为 **`1/8`**，唯一 positive 为 ASSIST2015；距离“超过全部准入基线 `5/8`”仍差 4 个数据集。依据：`outputs/A2G_FOLD0_SEED42_VALIDATION_BASELINE_REGISTRY_V3_20260821.json`（SHA256 `923bc69a60784aed4fbde90a964bd482472ded27759110b81eab4519c7e1a4f1`）。

### 迭代 0 模块筛查

门槛按当前约定只看 AUC/ACC：`ΔAUC > 0.001`，`ΔACC >= -0.0005`。

| 模块路线 | 数据集/估计量 | ΔAUC | ΔACC | 判定 |
|---|---|---:|---:|---|
| Selective-SSM | ASSIST2015 Full-minus-ablation | +0.0015643539 | +0.0003990306 | pass；当前 `1/1` |
| SSM normalized-innovation residual | ASSIST2015 candidate-control | +0.000065855680 | +0.000759131476 | fail-fast；关闭 |
| token-capacity MoE-FFN | ASSIST2015 candidate-control | +0.0000000798 | +0.0000000000 | fail-fast；关闭 |
| complete-first prior/support | ASSIST2015 candidate-control | +0.0000002507 | +0.0000194650 | fail-fast；关闭 |
| CEPA | ASSIST2015 | — | — | 技术效率停止，无科学结果 |
| UID-isolated causal evidence continuity | ASSIST2015 candidate-control | +0.0000041468 | -0.0006131447 | AUC/ACC 均 fail-fast；关闭 |

因此，当前内部模块有效覆盖仍为 **`1/1`**，未达到“当前 Full-positive 数据集 `5/5`”；`quality_freeze=false`。CEPA 无科学结果；UID evidence continuity 与其余失败路线均不计入分母或通过数，不授权自动扩库、网格或重试。

### 外部组合边界

固定等权 logit pool 的 artifact-backed supporting evidence 已覆盖 6 个数据集，但它是外部预测组合，不是 A2G Full 单体，也不是 A2G 内部模块消融；不得用来改写 `1/8` 或 `1/1`。

**当前可审计结论：** `a2g_full_alone_all_baselines=1/8`；`a2g_internal_module_ablation_on_full_positive=1/1`；`quality_freeze=false`。本更新不代表五折均值、显著性或 universal-SOTA 主张。
<!-- CODEX-A2G-ITER0-FOLD0-RESULT-UPDATE-20260822:END -->

<!-- CODEX-A2G-UID-CAUSAL-EVIDENCE-CONTINUITY-TERMINAL-20260822:START -->
## 2026-08-22 UID causal evidence continuity 模块首筛终态

该模块在 172.25.114.0 物理 GPU1 上完成唯一 `Assist2015 / fold0 / seed42 / validation-only` paired screen。模块为 UID-isolated rolling-200 causal statistics/RWCE continuity；不新增参数、dropout 或 RNG 调用，control 为普通 segment-local A2G Full。`test_access=false`，`window_test_access=false`。

| 候选 - control | ΔAUC | ΔACC | ΔNLL | ΔBrier | ΔECE15 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| UID causal evidence continuity | +0.0000041468 | -0.0006131447 | -0.0000673630 | +0.0000479443 | +0.0008782495 | AUC/ACC 均 fail |

control AUC=`0.7349004838`、candidate AUC=`0.7349046306`；candidate AUC 提升仅 `+0.0000041468`，未达到严格 `>0.001`，且 ACC 低于 `-0.0005` 安全线。因此执行预注册 `close_candidate_without_cross_dataset_expansion`：不调窗口、不做网格、不重试、不跨数据集扩展。

队列 PID/PGID=`1632817/1632817`，watchdog PID/PGID=`1632821/1632821`；watchdog `queue_idle_exit`，未检测 foreign compute，未向外部 PID 发信号；终态两卡 compute=`0`。权威 summary SHA256=`4f48a8ac476a952cdf3a153b988e7ae468c44181aca8f3a05a7d01559b732d9`，control result SHA256=`bff945b25216740aee11a20d3296a4ccb9aaf05ded047f4b1b373e9c0707ee28`，candidate result SHA256=`42e0c1caa2f3fdfc3b849ebeeb4f1619fbaa34987e947c64fb07b6b190f1bf4e`。

该路线不增加 A2G Full positive 数据集，不增加内部模块消融通过数，不改变当前 `A2G Full=1/8`、内部模块 `1/1`、`quality_freeze=false`。机器可读终态：`outputs/A2G_UID_CAUSAL_EVIDENCE_CONTINUITY_ASSIST2015_TERMINAL_AUDIT_20260822.json`。
<!-- CODEX-A2G-UID-CAUSAL-EVIDENCE-CONTINUITY-TERMINAL-20260822:END -->

<!-- CODEX-A2G-EXACT-FULL-ASSIST2012-TERMINAL-20260822:START -->
## 2026-08-22 exact-Full Assist2012 validation-only 自然终态

本块为 append-only 证据同步。实验严格固定为 `Assist2012 / fold0 / seed42 / validation-only`，使用当前冻结 509189 wrapper；`test_access=false`、`window_test_access=false`。它是补齐当前 Full 控制证据，不是候选模块消融或超参数搜索。

| 项目 | A2G Full | 最强冻结 baseline | Δ(A2G-baseline) | 判定 |
|---|---:|---:|---:|---|
| AUC | 0.7725166441 | RobustKT 0.7762642894 | -0.0037476453 | fail |
| ACC | 0.7484116553 | RobustKT 0.7551174943 | -0.0067058390 | fail |

终态五项 Full 指标为 AUC `0.7725166440742233`、ACC `0.7484116552894039`、NLL `0.5144041495613835`、Brier `0.1706700612145892`、ECE15 `0.041157120056772405`；最佳 checkpoint 为 epoch 9。queue PID/PGID=`1832720/1832720`，watchdog PID=`1832724`，queue exit=`0`，watchdog 自然 `queue_idle_exit`，终态 compute=`0`。validation prediction 独立重算与 result 的五项指标最大绝对差约 `5.5e-9`，Assist2012 validation split SHA=`47b071b9e873fa3c7ff713822ac2372fa4f7b8924b08cf65a2c51b3def4e23b7`。

权威远端 result SHA=`055d3d4bac9da6b83f15cfed986d6fa529e51587a035330635714c8095d5c321`，resource SHA=`036eea6e42059c9b1551108e08f2f02cfa6182de3f26a42949a7f9eb6a3efff0`，checkpoint SHA=`2824b56aa095678ced9686c6644cf601f022635c411b25dcc87e285c2213b592`，validation predictions SHA=`e7ca6f2e1f0b8ec606f0f5ee6dab76624229dc7124f42d01ffd0602c128e1cb3`。机器可读审计：`outputs/A2G_EXACT_FULL_ASSIST2012_TERMINAL_20260822.json`。

该结果不增加 Full positive 数据集，当前严格结论仍为 A2G Full `1/8`、内部模块消融 `1/1`、`quality_freeze=false`；不授权把该控制结果解释为超过基线，也不由此自动扩展新的模块网格。
<!-- CODEX-A2G-EXACT-FULL-ASSIST2012-TERMINAL-20260822:END -->

<!-- CODEX-A2G-ITER0-FOLD0-CANONICAL-SUPERSESSION-20260822:START -->
## 2026-08-22 模型优化迭代 0 / fold0 最新权威结果（supersession）

本块 append-only supersede 此前“模型优化迭代 0 / fold0 结果更新”中的 Assist2012 旧控制值；旧值 `AUC=0.7734060388` 不再作为当前 Full 结果。统一协议仍为 `fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`，不得解释为五折均值/标准差或 final-test 结果。

| 数据集 | 当前 A2G Full AUC | 最强冻结 baseline | baseline AUC | ΔAUC | 判定 |
|---|---:|---|---:|---:|---|
| ASSIST2009 corrected/collapsed | 0.8539486587 | UKT | 0.8566249815 | -0.0026763228 | fail |
| ASSIST2012 exact Full | 0.7725166441 | RobustKT | 0.7762642894 | -0.0037476453 | fail |
| ASSIST2015 | 0.7349005323 | AKT | 0.7337756158 | +0.0011249165 | pass |
| ASSIST2017 | 0.7772409184 | ACE-KT | 0.7800295288 | -0.0027886104 | fail |
| Junyi2015 | 0.7766216922 | UKT | 0.8023922269 | -0.0257705347 | fail |
| NIPS Task 3&4 | 0.7956730346 | AKT | 0.7961115327 | -0.0004384981 | fail |
| Slepemapy | 0.7937119232 | SimpleKT | 0.7968250064 | -0.0031130833 | fail |
| Statics2011 | 0.8259118330 | AKT | 0.8293542043 | -0.0034423713 | fail |

当前严格 Full 结果仍为 **`1/8`**，唯一超过全部准入 baseline 的数据集为 ASSIST2015，距离目标 `5/8` 仍差 4 个数据集。

迭代 0 内部模块按当前 AUC/ACC 门槛（`ΔAUC > 0.001`、`ΔACC >= -0.0005`）的权威状态仍为：Selective-SSM 在 ASSIST2015 Full-minus-ablation 上 `ΔAUC=+0.0015643539`、`ΔACC=+0.0003990306`，通过数 **`1/1`**；其余已完成候选未通过首筛或无科学终态，未达到目标 `5/5`。因此 `quality_freeze=false`。

Assist2012 精确终态依据：`outputs/A2G_EXACT_FULL_ASSIST2012_TERMINAL_20260822.json`，SHA256 `e58b368ff87f5887896fb9825f0a119930b8473155ee7b288a3363e92f3d670e`；远端 result SHA256 `055d3d4bac9da6b83f15cfed986d6fa529e51587a035330635714c8095d5c321`，validation predictions SHA256 `e7ca6f2e1f0b8ec606f0f5ee6dab76624229dc7124f42d01ffd0602c128e1cb3`。
<!-- CODEX-A2G-ITER0-FOLD0-CANONICAL-SUPERSESSION-20260822:END -->

<!-- CODEX-A2G-ITER0-ALL4KT-ADAPTIVE-LABELS-TERMINAL-20260822:START -->
## 2026-08-22 模型优化迭代 0 / ALL4KT adaptive-label Assist2015 首筛终态

该候选针对 concept-only Assist2015 修正为：仅使用训练折 concept 频率构造 32 个平衡潜在 label，并学习 concept embedding 的 latent-label residual；不使用 validation/test 标签、模型输出、图消息传播、检索、时间平滑、额外 dropout 或 RNG。范围固定为 `Assist2015 / fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。

| 候选 - control | ΔAUC | ΔACC | ΔNLL | ΔBrier | ΔECE15 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| ALL4KT concept-only balanced adaptive labels | +0.0000260486 | -0.0000486623 | +0.0000235878 | -0.0000019165 | +0.0002505347 | AUC fail；ACC pass |

AUC/ACC-only 首筛门槛为 `ΔAUC > 0.001`、`ΔACC >= -0.0005`。因此该候选 `gate_pass=false`，立即执行 `close_candidate_without_cross_dataset_expansion`：不做 latent group 数量/温度/平衡方式扫描、不跨数据集、不重试。该结果不增加 Full positive 数据集，也不增加内部模块消融通过数。

172 终态：queue PID/PGID=`2356011/2356011`，watchdog PID=`2355980`，queue exit=`0`，watchdog=`queue_idle_exit`，终态两卡 compute=`0`，未检测 foreign compute。candidate result SHA256=`69f3a3e2abbb1d15e52489b681b2ef86d35f49b2eb09b3239c40f076530adcad`，summary SHA256=`db9167026d0b66493ce66908e23418c2149c4fbc101cddd0b99c56feae962751`，validation predictions SHA256=`931db03049d8d9b1bdd2bd6583d2ae201cc9cf8fcce4977a664b31ba32725324`。独立重算五项指标与 result 最大差为 `0.0`。

机器可读终态：`outputs/A2G_ALL4KT_ADAPTIVE_LABELS_ASSIST2015_TERMINAL_AUDIT_20260822.json`。v1 的 item-level 误用因 Assist2015 `qlen=0` 未产生科学结果，已单独保留 provenance：`outputs/A2G_ALL4KT_V1_TECHNICAL_NO_RESULT_AUDIT_20260822.json`。

当前严格总账不变：A2G Full `1/8`，内部模块消融 `1/1`，目标 `5/8` 与 `5/5` 均未完成，`quality_freeze=false`。
<!-- CODEX-A2G-ITER0-ALL4KT-ADAPTIVE-LABELS-TERMINAL-20260822:END -->

<!-- CODEX-A2G-CONTINUATION-REAUDIT-NO-GPU-CANDIDATE-20260822:START -->
## 2026-08-22 模型优化迭代 0 / 候选空间复核终态

本次复核重新检查 causal-evidence state FiLM、learner-innovation SSM write、state-correlated adaptive forgetting、complete-first prior/support 与 token-capacity MoE-FFN。149-row closure ledger 与后续终态均已覆盖：causal-evidence state FiLM 的权威历史结果为 A09/E1 `ΔAUC=-0.0000016628`、`ΔACC=-0.0000205778`，决策为 `reject_causal_evidence_state_film_at_a09_e1`；其余四条也分别已有失败终态。它们不允许重新发起 Assist2015 GPU 首筛、系数扫描或改名重开。

172 当前两张 RTX 3090 均无 compute：GPU0 `29/24245 MiB`、GPU1 `13/24253 MiB`，utilization 均为 `0%`；本次未创建远端 root、未启动 queue/watchdog、未向任何外部 PID 发信号。机器可读复核：`outputs/A2G_CONTINUATION_REAUDIT_20260822_NO_GPU_ELIGIBLE_CANDIDATE.json`。

严格总账保持：A2G Full `1/8`，内部模块 `1/1`，`5/8` 与 `5/5` 均未完成，`quality_freeze=false`。GPU 空闲不等于科学授权；只有新的可观测输入资产或机械职责经 append-only novelty/estimand、CPU/RNG/static 合同后，才可建立下一次单点首筛。
<!-- CODEX-A2G-CONTINUATION-REAUDIT-NO-GPU-CANDIDATE-20260822:END -->

<!-- CODEX-A2G-ITEM-DROPOUT40-FULL-NIPS-TERMINAL-20260822:START -->
## 2026-08-22 模型优化迭代 0 / item-shortcut suppression NIPS fold0 终态

该块 append-only supersede 迭代 0 中 NIPS 的旧 Full 控制值。候选为冻结 509189 wrapper 上的条件化 item residual dropout=0.4，仅在有真实 item 序列时对 item shortcut residual 做训练期 dropout；固定 `nips_task34 / fold0 / seed42 / validation-only`。无网格、无重试、无自动跨数据集扩展。

| 项目 | A2G item-dropout40 | 最强冻结 baseline AKT | Δ(candidate-baseline) | 判定 |
|---|---:|---:|---:|---|
| AUC | 0.7972875499 | 0.7961115327 | +0.0011760172 | Full AUC pass |
| ACC | 0.7240005194 | 0.7253213696 | -0.0013208502 | report-only |
| NLL | 0.5410741930 | — | — | report-only |
| Brier | 0.1833948605 | — | — | report-only |
| ECE15 | 0.0341721568 | — | — | report-only |

终态五项指标由 `validation_predictions.npz` 独立重算，最大绝对差 `0.0`；四个 test/window 指标均为 `-1` 哨兵。queue PID/PGID=`2533321/2533321`，watchdog PID=`2533325`，queue exit=`0`，watchdog=`queue_idle_exit`，终态 compute=`0`，GPU0=`29 MiB`、GPU1=`13 MiB`。

权威候选 result SHA256=`aed6811370ed8199c2cc0f75aebc3cfd2066461e9339d0043f04f0dd955681fc`，resource SHA256=`c20d7ec29576d9caaa5a21941e69ffa792145bcb7564b72b30564191867f96be`，validation predictions SHA256=`fa125fe708a0538b146c216482c59d07e9d93838731254a108ff1254c52d5c14`；机器可读终态：`outputs/A2G_ITEM_DROPOUT40_FULL_NIPS_TERMINAL_AUDIT_20260822.json`。

当前 fold0/seed42 Full AUC 超过全部准入 baseline 的数据集为 **2/8**：ASSIST2015、NIPS Task 3&4；距离目标 `5/8` 仍差 3 个数据集。内部模块消融仍为 Selective-SSM `1/1`，尚未达到 `5/5`；`quality_freeze=false`。该候选是 Full-model supporting performance evidence，不构成新的内部模块消融通过数或顶刊 universal-SOTA 主张。
<!-- CODEX-A2G-ITEM-DROPOUT40-FULL-NIPS-TERMINAL-20260822:END -->

<!-- CODEX-A2G-ITEM-DROPOUT40-FULL-ASSIST2017-TERMINAL-20260822:START -->
## 2026-08-22 item-shortcut suppression Assist2017 fold0 终态

该候选沿用冻结 509189 wrapper 与 `item_residual_dropout=0.4`，固定 `assist2017 / fold0 / seed42 / validation-only`。训练 40 epoch 自然终止，best epoch=`31`；无 grid、无 retry、无 test/window 访问。

| 项目 | A2G item-dropout40 | 最强冻结 baseline ACE-KT | Δ(candidate-baseline) | 判定 |
|---|---:|---:|---:|---|
| AUC | 0.7836785178 | 0.7800295288 | +0.0036489891 | Full AUC pass |
| ACC | 0.7249766442 | 0.7201878916 | +0.0047887526 | report-only |
| NLL | 0.5383065967 | — | — | report-only |
| Brier | 0.1812411262 | — | — | report-only |
| ECE15 | 0.0312729222 | — | — | report-only |

validation prediction 独立重算最大绝对差=`0.0`；queue PID/PGID=`2623965/2623965`，watchdog PID=`2623969`，queue exit=`0`，终态 GPU0=`29 MiB`、GPU1=`13 MiB`，compute=`0`。

result SHA256=`6d253c3f5b209dd175af302cf276f35df8df65d167195d769c759669e9687690`，resource SHA256=`fdec17f53ff1415ab6f46e5dd35fea8483afa927b926ae0e1cd861846699f804`，validation predictions SHA256=`7e4fddb3c0d2a36978ee69aa3ed50075c23b83705ceca511ae54aad654055e8a`；机器可读审计：`outputs/A2G_ITEM_DROPOUT40_FULL_ASSIST2017_TERMINAL_AUDIT_20260822.json`。

当前 Full AUC 超过全部准入 baseline 的数据集为 **3/8**：ASSIST2015、NIPS Task 3&4、ASSIST2017；距离 `5/8` 仍差 2 个。内部模块消融仍为 `1/1`，`quality_freeze=false`。
<!-- CODEX-A2G-ITEM-DROPOUT40-FULL-ASSIST2017-TERMINAL-20260822:END -->

<!-- CODEX-A2G-ITER0-FOLD0-TERMINAL-SNAPSHOT-20260822:START -->
## 2026-08-22 模型优化迭代 0 / fold0 当前终态快照

本块以 append-only 方式同步当前已经自然终止并完成 artifact/预测复算的 `fold0 / seed42 / validation-only` 结果；不删除前文历史快照，不把单折结果写入五折均值/标准差或 final-test 主表。统一边界为 `test_access=false`、`window_test_access=false`。运行环境为 172.25.114.0 物理 GPU1 与 `<REMOTE_HOME>/ls/envs/dataenvgym/bin/python`。

当前 Full 优化采用 conditional item-shortcut suppression：`item_residual_dropout=0.4` 只作用于具有真实 item 序列的数据集；concept-only 路径不变。它是 Full-model supporting performance optimization，不自动计为新的内部模块消融。

| 数据集 | 当前终态 A2G AUC | 最强冻结 baseline | baseline AUC | ΔAUC | 当前判定 |
|---|---:|---|---:|---:|---|
| ASSIST2009 corrected/collapsed | 0.8539486587 | UKT | 0.8566249815 | -0.0026763228 | fail |
| ASSIST2012 exact Full | 0.7725166441 | RobustKT | 0.7762642894 | -0.0037476453 | fail |
| ASSIST2015 | 0.7349005323 | AKT | 0.7337756158 | +0.0011249165 | pass |
| ASSIST2017 item-dropout40 | 0.7836785178 | ACE-KT | 0.7800295288 | +0.0036489891 | pass |
| Junyi2015 | 0.7766216922 | UKT | 0.8023922269 | -0.0257705347 | fail |
| NIPS Task 3&4 item-dropout40 | 0.7972875499 | AKT | 0.7961115327 | +0.0011760172 | pass |
| Slepemapy | 0.7937119232 | SimpleKT | 0.7968250064 | -0.0031130833 | 当前已完成终态仍 fail；item-dropout40 单点运行中，不提前计数 |
| Statics2011 | 0.8259118330 | AKT | 0.8293542043 | -0.0034423713 | fail |

因此当前已验证的 Full-positive 为 **`3/8`**：ASSIST2015、NIPS Task 3&4、ASSIST2017；距离 `5/8` 仍差 2 个。NIPS item-dropout40 result SHA256=`aed6811370ed8199c2cc0f75aebc3cfd2066461e9339d0043f04f0dd955681fc`；ASSIST2017 item-dropout40 result SHA256=`6d253c3f5b209dd175af302cf276f35df8df65d167195d769c759669e9687690`；冻结 baseline registry SHA256=`923bc69a60784aed4fbde90a964bd482472ded27759110b81eab4519c7e1a4f1`。

内部模块消融继续按当前 AUC/ACC-only 门槛 `ΔAUC > 0.001`、`ΔACC >= -0.0005` 统计：Selective-SSM 在 ASSIST2015 的 Full-minus-ablation 为 `ΔAUC=+0.0015643539`、`ΔACC=+0.0003990306`，当前 **`1/1`**；尚未达到目标 `5/5`，`quality_freeze=false`。

Slepemapy item-dropout40 是当前唯一 active 单点：remote root=`<REMOTE_HOME>/a2g_mambakt/item_dropout40_full_slepemapy_validation_20260822_v1_172`，queue PID/PGID=`2674024/2674024`，watchdog PID=`2674028`，仅使用物理 GPU1。其 `result.json` 尚未自然落盘，因此本块不把任何中间 checkpoint 或日志峰值写成结果；终态后必须独立复算并另加 supersession。
<!-- CODEX-A2G-ITER0-FOLD0-TERMINAL-SNAPSHOT-20260822:END -->

<!-- CODEX-A2G-ITEM-DROPOUT40-FULL-SLEPEMAPY-TERMINAL-20260822:START -->
## 2026-08-22 item-shortcut suppression Slepemapy fold0 终态 supersession

本块 append-only supersede 当前快照中 Slepemapy 的“运行中”状态。实验固定为 `Slepemapy / fold0 / seed42 / validation-only`，使用 172.25.114.0 物理 GPU1、`<REMOTE_HOME>/ls/envs/dataenvgym/bin/python` 和 `item_residual_dropout=0.4`；未访问 test 或 Window-test。

| 项目 | A2G item-dropout40 | 最强冻结 baseline SimpleKT | Δ(candidate-baseline) | 判定 |
|---|---:|---:|---:|---|
| AUC | 0.8026728827 | 0.7968250064 | +0.0058478763 | Full AUC pass |
| ACC | 0.8063901925 | 0.8034598515 | +0.0029303410 | report-only |
| NLL | 0.4156087194 | — | — | report-only |
| Brier | 0.1340369650 | — | — | report-only |
| ECE15 | 0.0125041322 | — | — | report-only |

终态 result 与 `validation_predictions.npz` 独立复算五项指标最大绝对差为 `0.0`；selected epoch=`22`。queue PID/PGID=`2674024/2674024`，queue exit=`0`；watchdog PID=`2674028`，终态=`queue_idle_exit`，未向 foreign PID 发信号；终态 GPU0=`29 MiB`、GPU1=`13 MiB`，compute=`0`。

权威 result SHA256=`3e6026f1b8937a70ae0568ff4f8ca2a6f040db67e13229805061d3e85cef6b87`，resource SHA256=`ee303c698165f3f82872d1ab6ec89bb8032b82eb7d48d4b2f5cb43d9561528e7`，validation predictions SHA256=`472a1e5875d5ed1de51f61d0ad08add8f215d9c27cad13cbf6ae779125724324`，checkpoint SHA256=`c38a3a4fa6777d948b86b2de2a7a738992cfc744713735186e650c6817fe28d4`；机器可读审计：`outputs/A2G_ITEM_DROPOUT40_FULL_SLEPEMAPY_TERMINAL_AUDIT_20260822.json`。

当前 fold0/seed42 Full AUC 超过全部准入 baseline 的数据集更新为 **`4/8`**：ASSIST2015、NIPS Task 3&4、ASSIST2017、Slepemapy；距离目标 `5/8` 仍差 1 个。内部模块消融仍为 Selective-SSM **`1/1`**，尚未达到目标 `5/5`；`quality_freeze=false`。该结果是 Full-model supporting performance evidence，不自动计入内部模块消融，也不构成五折、多 seed 或 universal-SOTA 主张。
<!-- CODEX-A2G-ITEM-DROPOUT40-FULL-SLEPEMAPY-TERMINAL-20260822:END -->

<!-- CODEX-A2G-ITEM-DROPOUT40-FULL-ASSIST2012-TERMINAL-20260822:START -->
## 2026-08-22 item-shortcut suppression Assist2012 fold0 失败终态

在 Slepemapy 将当前 Full-positive 推进至 `4/8` 后，以同一冻结 `item_residual_dropout=0.4` 单点检查 Assist2012。实验固定 `fold0 / seed42 / validation-only`，172.25.114.0 物理 GPU1；无 grid、无 retry、无 test/window 访问。

| 项目 | A2G item-dropout40 | 最强冻结 baseline RobustKT | Δ(candidate-baseline) | 判定 |
|---|---:|---:|---:|---|
| AUC | 0.7742878652 | 0.7762642894 | -0.0019764242 | fail |
| ACC | 0.7533097147 | 0.7551174943 | -0.0018077796 | report-only |
| NLL | 0.5089235062 | — | — | report-only |
| Brier | 0.1683081142 | — | — | report-only |
| ECE15 | 0.0223268995 | — | — | report-only |

selected epoch=`15`，validation prediction 独立复算最大绝对差=`0.0`；queue PID/PGID=`3116783/3116783`、watchdog PID=`3116787`、queue exit=`0`、watchdog=`queue_idle_exit`，终态 compute=`0`。result SHA256=`7b1b6a7e5d10a929809b6b4fa86acce82ffb745d85701d1652d32be022db4c03`，resource SHA256=`809a586103a6dfb955ec391175f244bacb82a5e5b1a3f69613d3cc800688c3bf`，validation predictions SHA256=`3088b0028b209da6a0c2d14654bb50c277f5eb9e461afab32bf86e29f4b55b47`。

因此 Assist2012 不加入 Full-positive 集合，该固定点正式关闭：不重试、不扫描 dropout、不自动扩展。当前严格状态保持 **Full `4/8`**、内部 Selective-SSM 消融 **`1/1`**、`quality_freeze=false`。机器可读审计：`outputs/A2G_ITEM_DROPOUT40_FULL_ASSIST2012_TERMINAL_AUDIT_20260822.json`。
<!-- CODEX-A2G-ITEM-DROPOUT40-FULL-ASSIST2012-TERMINAL-20260822:END -->

<!-- CODEX-A2G-ITER0-FOLD0-LATEST-CONSOLIDATED-SNAPSHOT-20260822:START -->
## 2026-08-22 模型优化迭代 0 / fold0 最新统一汇总

本块为当前最新 append-only 汇总，supersede 前文仍显示 `3/8` 或将 Slepemapy 标记为运行中的历史快照；不删除历史记录，也不把单折结果混入五折均值/标准差或 final-test 主表。协议统一为 `fold0 / seed42 / validation-only`，且 `test_access=false`、`window_test_access=false`。

| 数据集 | 当前 A2G Full 方案 | AUC | ACC | 最强冻结 baseline | baseline AUC | ΔAUC | Full-positive |
|---|---|---:|---:|---|---:|---:|---|
| ASSIST2009 corrected/collapsed | exact Full | 0.8539486587 | — | UKT | 0.8566249815 | -0.0026763228 | fail |
| ASSIST2012 | item-dropout40 | 0.7742878652 | 0.7533097147 | RobustKT | 0.7762642894 | -0.0019764242 | fail；路线关闭 |
| ASSIST2015 | current Full | 0.7349005323 | — | AKT | 0.7337756158 | +0.0011249165 | pass |
| ASSIST2017 | item-dropout40 | 0.7836785178 | 0.7249766442 | ACE-KT | 0.7800295288 | +0.0036489891 | pass |
| Junyi2015 | exact Full | 0.7766216922 | — | UKT | 0.8023922269 | -0.0257705347 | fail |
| NIPS Task 3&4 | item-dropout40 | 0.7972875499 | 0.7240005194 | AKT | 0.7961115327 | +0.0011760172 | pass |
| Slepemapy | item-dropout40 | 0.8026728827 | 0.8063901925 | SimpleKT | 0.7968250064 | +0.0058478763 | pass |
| Statics2011 | exact Full | 0.8259118330 | — | AKT | 0.8293542043 | -0.0034423713 | fail |

当前 Full AUC 超过全部准入 baseline 的集合为 **`4/8`**：ASSIST2015、NIPS Task 3&4、ASSIST2017、Slepemapy；距离 `5/8` 尚差 1 个数据集。Assist2012 的 item-dropout40 固定单点已失败关闭，不重试、不扫描 dropout。

内部模块消融仍按用户当前 AUC/ACC-only 门槛统计：`ΔAUC > 0.001` 且 `ΔACC >= -0.0005`。当前仅有 Selective-SSM 在 ASSIST2015 的 matched Full-minus-ablation 通过，`ΔAUC=+0.0015643539`、`ΔACC=+0.0003990306`，因此为 **`1/1`**，尚未完成目标 `5/5`。NLL/Brier/ECE 仅记录，不参与当前消融门禁；`quality_freeze=false`。

关键终态绑定：Slepemapy result SHA256=`3e6026f1b8937a70ae0568ff4f8ca2a6f040db67e13229805061d3e85cef6b87`；Assist2012 result SHA256=`7b1b6a7e5d10a929809b6b4fa86acce82ffb745d85701d1652d32be022db4c03`，本地终态审计 SHA256=`078e089b741501fce7d6e0a03ef9ee33eb847666dc8748c1c65204d60d47334e`；冻结 baseline registry SHA256=`923bc69a60784aed4fbde90a964bd482472ded27759110b81eab4519c7e1a4f1`。
<!-- CODEX-A2G-ITER0-FOLD0-LATEST-CONSOLIDATED-SNAPSHOT-20260822:END -->

<!-- CODEX-A2G-ITEM-DROPOUT40-FULL-ASSIST2009-TERMINAL-20260822:START -->
## 2026-08-22 item-shortcut suppression ASSIST2009 corrected/collapsed fold0 终态

该候选是 ASSIST2009 corrected/collapsed 上的固定 `item_residual_dropout=0.4` Full 单点，使用新的 raw-max checkpoint protocol：每个 epoch 都持久化验证 artifact，自然终止后按最高 validation AUC 选择，tie-break 为更低 NLL、Brier、ECE15、较早 epoch。实验固定 `fold0 / seed42 / validation-only`，172.25.114.0 物理 GPU1；无 grid、无 retry、无 test/window 访问。

| 项目 | A2G item-dropout40 | 最强冻结 baseline UKT | Δ(candidate-baseline) | 判定 |
|---|---:|---:|---:|---|
| AUC | 0.8575213800 | 0.8566249815 | +0.0008963985 | Full AUC pass |
| ACC | 0.7934122101 | 0.7906105064 | +0.0028017037 | report-only |
| NLL | 0.4362453492 | — | — | report-only |
| Brier | 0.1428133842 | — | — | report-only |
| ECE15 | 0.0417560948 | — | — | report-only |

selected epoch=`23`；validation prediction 独立重算五项指标最大绝对差=`0.0`，interaction count=`52825`。queue PID/PGID=`3248159/3248159`、watchdog PID=`3248163`、queue exit=`0`、watchdog=`queue_idle_exit`；终态 GPU0=`29 MiB`、GPU1=`13 MiB`，compute=`0`，未向 foreign PID 发信号。

候选 result SHA256=`9e7f082cfcc286ed6f2d245fef19151a74c0bd1cce7ecd387607e0e08656f11a`，resource SHA256=`91ed09863640a56689f7de0b9dfff44e49a18db70392a21dd2f4f2261bddfc98`，validation predictions SHA256=`c3a5e82e4135a0211853a473b523dc38c81dd1eb4388ba999401917a301a5f01`，checkpoint SHA256=`5a8e3f614d592bbee39cc46f9e441fbe0a62d6d5c79fe574922055a97eef70bc`；机器可读审计：`outputs/A2G_ITEM_DROPOUT40_FULL_ASSIST2009_TERMINAL_AUDIT_20260822.json`，审计 SHA256=`35d6abc6a265f75664631fb070192f15ea0d60c9ead20abf87f2fa9b89beb876`。

当前 fold0/seed42 Full AUC 超过全部准入 baseline 的数据集更新为 **`5/8`**：ASSIST2009 corrected/collapsed、ASSIST2015、NIPS Task 3&4、ASSIST2017、Slepemapy。该结果完成 Full-model `5/8` 门槛，但不是五折、多 seed 或 universal-SOTA 证据，也不自动增加内部模块消融通过数；内部 Selective-SSM 仍为 **`1/1`**，`quality_freeze=false`。
<!-- CODEX-A2G-ITEM-DROPOUT40-FULL-ASSIST2009-TERMINAL-20260822:END -->

<!-- CODEX-A2G-ITER0-FOLD0-LATEST-CONSOLIDATED-SNAPSHOT-V2-20260822:START -->
## 2026-08-22 模型优化迭代 0 / fold0 最新统一汇总 v2

在前文统一汇总基础上追加 ASSIST2009 raw-max 终态后的最新状态。历史快照保留，不删除、不改写；所有数值仍是 `fold0 / seed42 / validation-only`，不进入五折均值/标准差或 final-test 主表。

| 数据集 | 当前 A2G Full 方案 | AUC | 最强冻结 baseline | baseline AUC | ΔAUC | Full-positive |
|---|---|---:|---|---:|---:|---|
| ASSIST2009 corrected/collapsed | item-dropout40 raw-max | 0.8575213800 | UKT | 0.8566249815 | +0.0008963985 | pass |
| ASSIST2012 | item-dropout40 | 0.7742878652 | RobustKT | 0.7762642894 | -0.0019764242 | fail；路线关闭 |
| ASSIST2015 | current Full | 0.7349005323 | AKT | 0.7337756158 | +0.0011249165 | pass |
| ASSIST2017 | item-dropout40 | 0.7836785178 | ACE-KT | 0.7800295288 | +0.0036489891 | pass |
| Junyi2015 | exact Full | 0.7766216922 | UKT | 0.8023922269 | -0.0257705347 | fail |
| NIPS Task 3&4 | item-dropout40 | 0.7972875499 | AKT | 0.7961115327 | +0.0011760172 | pass |
| Slepemapy | item-dropout40 | 0.8026728827 | SimpleKT | 0.7968250064 | +0.0058478763 | pass |
| Statics2011 | exact Full | 0.8259118330 | AKT | 0.8293542043 | -0.0034423713 | fail |

当前 Full-positive 为 **`5/8`**，已达到目标；内部模块消融按 `ΔAUC > 0.001` 且 `ΔACC >= -0.0005` 仍为 **`1/1`**，距离 `5/5` 还差四个 positive dataset 的 matched Full-minus-ablation 证据。NLL/Brier/ECE 仅记录，不参与当前消融门禁；`quality_freeze=false`。下一步只允许在冻结的五个 Full-positive 数据集上运行同初始化、同 RNG、保留 dropout substream 的 Selective-SSM matched ablation，不自动扩展到负数据集。
<!-- CODEX-A2G-ITER0-FOLD0-LATEST-CONSOLIDATED-SNAPSHOT-V2-20260822:END -->

<!-- CODEX-A2G-SELECTIVE-SSM-ITEM-DROPOUT40-ASSIST2009-TERMINAL-20260822:START -->
## 2026-08-22 Selective-SSM 消融 / ASSIST2009 item-dropout40 fold0 终态

该 pair 在当前 `item_residual_dropout=0.4` Full 架构上严格匹配 control/no-SSM：同 seed、同初始化、同训练折和 validation loader；no-SSM 保留对应 SSM norm/dropout 调用，保持 RNG 子流；每个 variant 使用自然终态 raw-max validation AUC 选择。实验为 `fold0 / seed42 / validation-only`，无 test/window、无网格、无重试。

| 指标 | Full control | no-SSM | Δ(Full - no-SSM) | AUC/ACC gate |
|---|---:|---:|---:|---|
| AUC | 0.8575213871 | 0.8536926303 | +0.0038287568 | pass |
| ACC | 0.7934122101 | 0.7890960719 | +0.0043161382 | pass |
| NLL | 0.4362454309 | 0.4397742082 | -0.0035287773 | report-only |
| Brier | 0.1428134032 | 0.1443703845 | -0.0015569813 | report-only |
| ECE15 | 0.0417562607 | 0.0393274991 | +0.0024287616 | report-only |

独立预测复算最大绝对差=`0.0`；queue PID/PGID=`3297935/3297935`、watchdog PID=`3297939`、queue exit=`0`、watchdog=`queue_idle_exit`，终态 compute=`0`。机器可读审计：`outputs/A2G_SELECTIVE_SSM_ITEM_DROPOUT40_ASSIST2009_TERMINAL_AUDIT_20260822.json`，SHA256=`466a7c6daadcd47a575ec4510754b323cc89a0e1c3af0f297745f6b2d6218946`。

当前 Selective-SSM AUC/ACC-only 严格通过计数更新为 **`2/2`**（ASSIST2009、ASSIST2015；分母为已完成 matched pair 数），目标冻结五个 positive 数据集上的 **`5/5`** 尚未完成，`quality_freeze=false`。下一步只运行下一个冻结 positive pair，不自动扩展到负数据集。
<!-- CODEX-A2G-SELECTIVE-SSM-ITEM-DROPOUT40-ASSIST2009-TERMINAL-20260822:END -->

<!-- CODEX-A2G-SELECTIVE-SSM-ITEM-DROPOUT40-NIPS-TERMINAL-20260822:START -->
## 2026-08-22 Selective-SSM 消融 / NIPS Task 3&4 item-dropout40 fold0 终态

该 pair 在当前 `item_residual_dropout=0.4` Full 架构上严格匹配 control/no-SSM：同 seed、同初始化、同训练折和 validation loader；no-SSM 保留对应 SSM norm/dropout 调用，保持 RNG 子流；每个 variant 使用自然终态 raw-max validation AUC 选择。实验为 `fold0 / seed42 / validation-only`，无 test/window、无网格、无重试。

| 指标 | Full control | no-SSM | Δ(Full - no-SSM) | AUC/ACC gate |
|---|---:|---:|---:|---|
| AUC | 0.7977802990 | 0.7953428068 | +0.0024374922 | pass |
| ACC | 0.7256124044 | 0.7247795971 | +0.0008328072 | pass |
| NLL | 0.5378415337 | 0.5400526849 | -0.0022111512 | report-only |
| Brier | 0.1822810810 | 0.1831820672 | -0.0009009862 | report-only |
| ECE15 | 0.0261130671 | 0.0240270318 | +0.0020860352 | report-only |

独立预测复算最大绝对差=`0.0`（control/no-SSM）；control selected epoch=`34`，no-SSM selected epoch=`39`；validation interaction count=`223341`，validation sequence rows=`1503`。queue PID/PGID=`3351507/3351507`、watchdog PID=`3351511`、queue exit=`0`、watchdog=`queue_idle_exit`；终态 compute=`0`，GPU0=`29 MiB`、GPU1=`13 MiB`，未向 foreign PID 发信号。

机器可读终态：`outputs/A2G_SELECTIVE_SSM_ITEM_DROPOUT40_NIPS_TERMINAL_AUDIT_20260822.json`，SHA256=`c0372612918701d8f42591e6881d612883c9787d46ba704d8f651dbb42867438`。control result SHA256=`939bd60684892a784bb343707c529a448632afb132ec57c801bdd49ccc350bf7`；no-SSM result SHA256=`0821c74a1e01ef3ee75e60ee08f5a8556bec7bfce1b549b53c7f46d931949894`；pair summary SHA256=`48ff0ed206b549f5f9a80d35c6ac07c98c32add17e3abf29e71f9028960b8531`。

当前冻结 Full-positive 集合上的 Selective-SSM AUC/ACC-only 严格通过计数更新为 **`3/3`**（ASSIST2009、ASSIST2015、NIPS Task 3&4），目标冻结五个 positive 数据集上的 **`5/5`** 尚未完成，`quality_freeze=false`。NLL/Brier/ECE 仅记录，不参与当前消融门禁；下一步只运行下一个冻结 Full-positive pair。
<!-- CODEX-A2G-SELECTIVE-SSM-ITEM-DROPOUT40-NIPS-TERMINAL-20260822:END -->

<!-- CODEX-A2G-SELECTIVE-SSM-ITEM-DROPOUT40-ASSIST2017-TERMINAL-20260822:START -->
## 2026-08-22 Selective-SSM 消融 / ASSIST2017 item-dropout40 fold0 终态

该 pair 在当前 `item_residual_dropout=0.4` Full 架构上严格匹配 control/no-SSM：同 seed、同初始化、同训练折和 validation loader；no-SSM 保留对应 SSM norm/dropout 调用，保持 RNG 子流；每个 variant 使用自然终态 raw-max validation AUC 选择。实验为 `fold0 / seed42 / validation-only`，无 test/window、无网格、无重试。

| 指标 | Full control | no-SSM | Δ(Full - no-SSM) | AUC/ACC gate |
|---|---:|---:|---:|---|
| AUC | 0.7844179564 | 0.7495211585 | +0.0348967979 | pass |
| ACC | 0.7258586109 | 0.7089967139 | +0.0168618971 | pass |
| NLL | 0.5364508054 | 0.5654273570 | -0.0289765516 | report-only |
| Brier | 0.1804676884 | 0.1914396606 | -0.0109719723 | report-only |
| ECE15 | 0.0260090991 | 0.0242330004 | +0.0017760987 | report-only |

独立预测复算最大绝对差=`0.0`（control/no-SSM）；control selected epoch=`34`，no-SSM selected epoch=`37`；validation interaction count=`153067`，validation sequence rows=`905`。queue PID/PGID=`3431505/3431505`、watchdog PID=`3431509`、queue exit=`0`、watchdog=`queue_idle_exit`；终态 compute=`0`，GPU0=`29 MiB`、GPU1=`13 MiB`，未向 foreign PID 发信号。

机器可读终态：`outputs/A2G_SELECTIVE_SSM_ITEM_DROPOUT40_ASSIST2017_TERMINAL_AUDIT_20260822.json`，SHA256=`33e71f253a6b6f90f85f3423cb0e2d92166205a0e26bd10a4a1334ec84333069`。control result SHA256=`2793e14d7eedac1dafd7c5f900b85874c09436232b8b0e8da557c3112d2834ea`；no-SSM result SHA256=`ee8a603ebed59ef4c8b471729e9a6d3dee4374d164599cb4ca42fa7a8b65ddd6`；pair summary SHA256=`fa6a7bef5fbb045c901f23ede35f26c341c9ba1dddfe8c23dde7ed1da25cd34d`。

当前冻结 Full-positive 集合上的 Selective-SSM AUC/ACC-only 严格通过计数更新为 **`4/4`**（ASSIST2009、ASSIST2015、NIPS Task 3&4、ASSIST2017），目标冻结五个 positive 数据集上的 **`5/5`** 尚未完成，`quality_freeze=false`。NLL/Brier/ECE 仅记录，不参与当前消融门禁；下一步只运行最后一个冻结 Full-positive pair。
<!-- CODEX-A2G-SELECTIVE-SSM-ITEM-DROPOUT40-ASSIST2017-TERMINAL-20260822:END -->

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260822-1528:START -->
## 2026-08-22 模型优化迭代 0 / fold0 当前状态（15:28）

本块以 append-only 方式同步当前最新状态，supersede 前文较早的计数快照，但不覆盖任何历史终态。协议仍为 `fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`；这些结果不混入五折均值/标准差或 final-test 主表。

- A2G Full 已在冻结 baseline registry 上达到 **`5/8`**：ASSIST2009 corrected/collapsed、ASSIST2015、NIPS Task 3&4、ASSIST2017、Slepemapy。
- Selective-SSM matched Full-minus-no-SSM 已自然终止并按 AUC/ACC-only 门槛严格通过 **`4/4`**：ASSIST2009、ASSIST2015、NIPS Task 3&4、ASSIST2017。门槛为 `ΔAUC > 0.001` 且 `ΔACC >= -0.0005`；NLL/Brier/ECE 仅记录。
- 最后一个 Slepemapy pair 正在 172.25.114.0 物理 GPU1 运行：remote root=`<REMOTE_HOME>/a2g_mambakt/selective_ssm_pair_slepemapy_itemdropout40_20260822_v1_172`，queue PID/PGID=`3480930/3480930`，watchdog PID=`3480934`，当前 control compute PID=`3480943`。截至 `2026-08-22T15:28:13+08:00` 尚无 `result.json` 或 `pair_summary.json`，因此不得提前写成 `5/5`。

当前可审计结论为 **Full `5/8` 已完成，Selective-SSM `4/4` 已完成、目标 `5/5` 待最后一个自然终态**；`quality_freeze=false`。Slepemapy 终态后必须独立复算 validation predictions，并以新的 append-only terminal block 更新最终计数。
<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260822-1528:END -->

<!-- CODEX-A2G-SELECTIVE-SSM-ITEM-DROPOUT40-SLEPEMAPY-TERMINAL-20260822:START -->
## 2026-08-22 Selective-SSM 消融 / Slepemapy item-dropout40 fold0 终态

该 pair 在当前 `item_residual_dropout=0.4` Full 架构上严格匹配 control/no-SSM：同 seed、同训练折和 validation loader；no-SSM 保留对应 SSM norm/dropout 调用并保持 RNG 子流。每个 variant 按自然终态 raw-max validation AUC 选择。实验为 `fold0 / seed42 / validation-only`，无 test/window、无网格、无重试。

| 指标 | Full control | no-SSM | Δ(Full - no-SSM) | AUC/ACC gate |
|---|---:|---:|---:|---|
| AUC | 0.8026728821 | 0.8000256534 | +0.0026472287 | pass |
| ACC | 0.8063901925 | 0.8046335456 | +0.0017566469 | pass |
| NLL | 0.4156087192 | 0.4176108397 | -0.0020021205 | report-only |
| Brier | 0.1340369649 | 0.1348273187 | -0.0007903538 | report-only |
| ECE15 | 0.0125044528 | 0.0115771058 | +0.0009273469 | report-only |

独立 validation-NPZ 复算与 producer 结果完全一致，五指标最大绝对差=`0.0`；control selected epoch=`22`，no-SSM selected epoch=`21`；validation interaction count=`1669089`，validation sequence rows=`18208`。queue PID/PGID=`3480930/3480930`、watchdog PID=`3480934`、queue exit=`0`、watchdog=`queue_idle_exit`；终态 compute=`0`，GPU0=`29 MiB`、GPU1=`13 MiB`，未向 foreign PID 发信号。

机器可读终态：[A2G_SELECTIVE_SSM_ITEM_DROPOUT40_SLEPEMAPY_TERMINAL_AUDIT_20260822.json](<USER_HOME>\Documents\Codex\2026-08-13\codex-threads-019fb778-09b4-7660-8613-621fe41de382\outputs\A2G_SELECTIVE_SSM_ITEM_DROPOUT40_SLEPEMAPY_TERMINAL_AUDIT_20260822.json)，SHA256=`35beb11096b086e001bbb5c59edfcaee401bb23fa9eb8e09f015a8daec2fa37b`。

control result SHA256=`e9976b77f8de7248b7cc89dbdff2e276b7582643b6285df939ccf77671ae9034`；no-SSM result SHA256=`f9bc20f58278cd31637570943520f575c2db6d177e4083b2c81de037afb05b7c`；pair summary SHA256=`0eb1c98b1b99d644f685fdf818fef537a79f06f95736f3fa066397536baa5bbd`。

当前冻结 Full-positive 集合上的 Selective-SSM AUC/ACC-only 严格通过计数更新为 **`5/5`**：ASSIST2009、ASSIST2015、NIPS Task 3&4、ASSIST2017、Slepemapy。Full 模型仍为 **`5/8`** baseline-positive。该结果完成当前 fold0/seed42 validation-only 目标，不等价于五折、多 seed、test 或 universal-SOTA 证据；`quality_freeze=false`，不自动启动后继实验。
<!-- CODEX-A2G-SELECTIVE-SSM-ITEM-DROPOUT40-SLEPEMAPY-TERMINAL-20260822:END -->

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260822-2115:START -->
## 2026-08-22 模型优化迭代 0 / fold0 当前状态（21:15）

本块为 append-only 状态同步；协议仍为 `fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`，不进入五折均值/标准差或 final-test 主表。

- A2G Full 在冻结 baseline registry 上维持 **`5/8`**：ASSIST2009 corrected/collapsed、ASSIST2015、NIPS Task 3&4、ASSIST2017、Slepemapy。
- Selective-SSM matched Full-minus-no-SSM 维持 **`5/5`** AUC/ACC-only 严格通过；门槛为 `ΔAUC > 0.001` 且 `ΔACC >= -0.0005`，NLL/Brier/ECE 仅记录。
- Junyi2015 的唯一后续单点 `item_residual_dropout=0.4` Full v2 正在 `172.25.114.0` GPU1 运行：remote root=`<REMOTE_HOME>/a2g_mambakt/item_dropout40_full_junyi_validation_20260822_v2_172`，queue PID/PGID=`67987/67987`，固定 watchdog PID=`67991`，own-PGID supervisor PID=`75445`；GPU1 当前仅该 compute，`result.json` 与 `queue.exit` 尚未生成。
- Junyi v2 尚无完整终态和独立复算指标，不得更新 Full-positive 计数；v1 中断结果不计入主账。保持无网格、无重试、无自动扩库，`quality_freeze=false`。
<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260822-2115:END -->

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260822-2312:START -->
## 2026-08-22 模型优化迭代 0 / fold0 最新状态（23:12）

本块为 append-only 同步，不覆盖历史终态。协议统一为 `fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`；单折结果不写入五折均值/标准差或 final-test 主表。

- A2G Full 在冻结 baseline registry 上已达到 **`5/8`**：ASSIST2009 corrected/collapsed、ASSIST2015、NIPS Task 3&4、ASSIST2017、Slepemapy。当前对应的优化方案为真实 item 序列上的 `item_residual_dropout=0.4`（concept-only 数据集保持原路径）。
- Selective-SSM matched Full-minus-no-SSM 按当前 AUC/ACC-only 门槛 `ΔAUC > 0.001` 且 `ΔACC >= -0.0005` 已完成 **`5/5`**：上述五个 Full-positive 数据集均通过；NLL/Brier/ECE 仅记录，不参与该消融门禁。
- Junyi2015 的唯一后续单点 `item_residual_dropout=0.4` Full v2 仍在 `172.25.114.0` GPU1 运行，环境为 `<REMOTE_HOME>/ls/envs/dataenvgym/bin/python`；当前最新完整观测为 epoch7 validation AUC=`0.8002`、ACC=`0.8545`，best epoch=`6`。该值仍低于冻结 UKT baseline AUC=`0.8023922269`，且尚未生成 `result.json` / `evidence/queue.exit`，因此不计入 Full-positive 计数。
- 当前严格可审计结论仍为 **Full `5/8`、Selective-SSM 消融 `5/5`、`quality_freeze=false`**。Junyi 终态后只允许独立复算并追加终态；不提前按中间 epoch 宣称超过基线，不启动网格、重试或自动扩库。

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260822-2312:END -->

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0006:START -->
## 2026-08-23 模型优化迭代 0 / fold0 最新运行观测（00:06）

本块仅同步 active run 的中间观测，采用 append-only 方式，不覆盖任何历史终态，也不写入五折均值/标准差或 final-test 主表。协议仍为 `Junyi2015 / fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。

- Junyi2015 `item_residual_dropout=0.4` Full v2 仍在 `172.25.114.0` GPU1 运行；queue PID/PGID=`67987/67987`，watchdog PID=`67991`，own-PGID supervisor PID=`75445`，当前未见 foreign compute。
- 最新完整日志为 epoch11：validation AUC=`0.8013`、ACC=`0.8549`、best epoch=`8`、best AUC=`0.8013`；相对冻结 UKT AUC=`0.8023922269` 的当前观测差值约 `-0.0010922269`。该中间值不是自然终态，尚无 `result.json` / `evidence/queue.exit`，不计入 Full-positive 集合。
- 已完成的严格状态保持 **A2G Full `5/8`、Selective-SSM matched 消融 `5/5`、`quality_freeze=false`**。待 Junyi 自然终止后才可独立复算并追加终态；不提前宣称超过基线，不启动网格、重试或自动扩库。

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0006:END -->

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0019:START -->
## 2026-08-23 模型优化迭代 0 / fold0 最新运行观测（00:19）

本块仅追加 active run 的中间日志观测，不覆盖历史终态，不写入五折均值/标准差或 final-test 主表。协议仍为 `Junyi2015 / fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。

- Junyi2015 `item_residual_dropout=0.4` Full v2 已完成到 epoch12；validation AUC=`0.8013`、ACC=`0.8549`，best epoch=`8`、best AUC=`0.8013`。相对冻结 UKT baseline AUC=`0.8023922269`，当前观测差约 `-0.0010922269`，尚未超过基线。
- queue PID/PGID=`67987/67987`、watchdog PID=`67991`、own-PGID supervisor PID=`75445` 仍存活，GPU1 仅该 compute；`result.json`、`evidence/queue.exit` 和 validation predictions 尚未生成。
- 当前严格结论仍为 **A2G Full `5/8`、Selective-SSM matched 消融 `5/5`、`quality_freeze=false`**。epoch12 只是运行中观测，不计入 Full-positive；待自然终态后再独立复算并追加终态。

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0019:END -->

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0034:START -->
## 2026-08-23 模型优化迭代 0 / fold0 最新运行观测（00:34）

本块为 active run 的 append-only 中间观测，不覆盖历史终态，不写入五折均值/标准差或 final-test 主表。协议仍为 `Junyi2015 / fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。

- Junyi2015 `item_residual_dropout=0.4` Full v2 已完成到 epoch13：validation AUC=`0.8013`、ACC=`0.8549`，best epoch=`8`、best AUC=`0.8013`；相对冻结 UKT baseline AUC=`0.8023922269`，当前观测仍差约 `-0.0010922269`。
- queue PID/PGID=`67987/67987`、watchdog PID=`67991`、own-PGID supervisor PID=`75445` 仍存活，GPU1 仅该 compute；`result.json`、`evidence/queue.exit` 和 validation predictions 尚未生成。
- 当前严格结论仍为 **A2G Full `5/8`、Selective-SSM matched 消融 `5/5`、`quality_freeze=false`**。epoch13 为运行中观测，不计入 Full-positive；自然终态后再独立复算并追加终态。

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0034:END -->

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0109:START -->
## 2026-08-23 模型优化迭代 0 / fold0 最新运行观测（01:09）

本块为 active run 的 append-only 中间观测，不覆盖历史终态，不写入五折均值/标准差或 final-test 主表。协议仍为 `Junyi2015 / fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。

- Junyi2015 `item_residual_dropout=0.4` Full v2 已推进到 epoch15；epoch14 首次达到当前最好 validation AUC=`0.8025`、ACC=`0.8552`，best epoch=`14`、best AUC=`0.8025`。该中间 AUC 高于冻结 UKT baseline `0.8023922269` 约 `+0.0001078`，但尚未完成自然终态与独立预测复算。
- queue PID/PGID=`67987/67987`、watchdog PID=`67991`、own-PGID supervisor PID=`75445` 仍存活，GPU1 仅该 compute；`result.json`、`evidence/queue.exit` 和 validation predictions 尚未生成。
- 因此当前严格主账仍保持 **A2G Full `5/8`、Selective-SSM matched 消融 `5/5`、`quality_freeze=false`**。epoch14/15 仅为运行中观测，不提前计入 Full-positive；自然终态后再独立复算并追加终态。

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0109:END -->

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0119:START -->
## 2026-08-23 模型优化迭代 0 / fold0 最新运行观测（01:19）

本块为 active run 的 append-only 中间观测，不覆盖历史终态，不写入五折均值/标准差或 final-test 主表。协议仍为 `Junyi2015 / fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。

- Junyi2015 `item_residual_dropout=0.4` Full v2 已推进到 epoch16；当前最好仍为 epoch14：validation AUC=`0.8025`、ACC=`0.8552`，相对冻结 UKT baseline AUC=`0.8023922269` 的中间 ΔAUC 约 `+0.0001078`。
- queue PID/PGID=`67987/67987`、watchdog PID=`67991`、own-PGID supervisor PID=`75445` 仍存活，GPU1 仅该 compute；`result.json`、`evidence/queue.exit` 和 validation predictions 尚未生成。
- 当前严格主账仍保持 **A2G Full `5/8`、Selective-SSM matched 消融 `5/5`、`quality_freeze=false`**。epoch14-16 仅为运行中观测，不计入 Full-positive；自然终态后再独立复算并追加终态。

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0119:END -->

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0140:START -->
## 2026-08-23 模型优化迭代 0 / fold0 最新运行观测（01:40）

本块为 active run 的 append-only 中间观测，不覆盖历史终态，不写入五折均值/标准差或 final-test 主表。协议仍为 `Junyi2015 / fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。

- Junyi2015 `item_residual_dropout=0.4` Full v2 已推进到 epoch17；当前最好仍为 epoch14：validation AUC=`0.8025`、ACC=`0.8552`，相对冻结 UKT baseline AUC=`0.8023922269` 的中间 ΔAUC 约 `+0.0001078`。
- queue PID/PGID=`67987/67987`、watchdog PID=`67991`、own-PGID supervisor PID=`75445` 仍存活，GPU1 仅该 compute；`result.json`、`evidence/queue.exit` 和 validation predictions 尚未生成。
- 当前严格主账仍保持 **A2G Full `5/8`、Selective-SSM matched 消融 `5/5`、`quality_freeze=false`**。epoch14-17 仅为运行中观测，不计入 Full-positive；自然终态后再独立复算并追加终态。

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0140:END -->

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0147:START -->
## 2026-08-23 模型优化迭代 0 / fold0 最新运行观测（01:47）

本块为 active run 的 append-only 中间观测，不覆盖历史终态，不写入五折均值/标准差或 final-test 主表。协议仍为 `Junyi2015 / fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。

- Junyi2015 `item_residual_dropout=0.4` Full v2 已推进到 epoch17；当前最好仍为 epoch14：validation AUC=`0.8025`、ACC=`0.8552`，相对冻结 UKT baseline AUC=`0.8023922269` 的中间 ΔAUC 约 `+0.0001078`。
- queue PID/PGID=`67987/67987`、watchdog PID=`67991`、own-PGID supervisor PID=`75445` 仍存活，GPU1 仅该 compute；`result.json`、`evidence/queue.exit` 和 validation predictions 尚未生成。
- 当前严格主账仍保持 **A2G Full `5/8`、Selective-SSM matched 消融 `5/5`、`quality_freeze=false`**。epoch14-17 仅为运行中观测，不计入 Full-positive；自然终态后再独立复算并追加终态。

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0147:END -->

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0151:START -->
## 2026-08-23 模型优化迭代 0 / fold0 最新运行观测（01:51）

本块为 active run 的 append-only 中间观测，不覆盖历史终态，不写入五折均值/标准差或 final-test 主表。协议仍为 `Junyi2015 / fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。

- Junyi2015 `item_residual_dropout=0.4` Full v2 已推进到 epoch18；当前最好仍为 epoch14：validation AUC=`0.8025`、ACC=`0.8552`，相对冻结 UKT baseline AUC=`0.8023922269` 的中间 ΔAUC 约 `+0.0001078`。
- queue PID/PGID=`67987/67987`、watchdog PID=`67991`、own-PGID supervisor PID=`75445` 仍存活，GPU1 仅该 compute；`result.json`、`evidence/queue.exit` 和 validation predictions 尚未生成。
- 当前严格主账仍保持 **A2G Full `5/8`、Selective-SSM matched 消融 `5/5`、`quality_freeze=false`**。epoch14-18 仅为运行中观测，不计入 Full-positive；自然终态后再独立复算并追加终态。

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0151:END -->

<!-- CODEX-A2G-ITER0-FOLD0-RUN-CONTRACT-20260823-0201:START -->
## 2026-08-23 Junyi v2 运行合同复核（02:01）

对 active Junyi v2 runner 做只读源码复核：启动参数固定为 `num_epochs=40`；wrapper 仅在 `wandb_train.main(params)` 自然返回后解析 validation summary、读取 selected checkpoint、生成 `validation_predictions.npz` 并写出 `result.json`，没有额外 patience/early-stop 分支。因此当前 epoch18 后的长时间计算符合既定合同，不能据此判定空转或失败，也不能提前写终态。

当前仍为 `Junyi2015 / fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`；GPU1 仅 queue PID=`67987`。Full `5/8` 与 Selective-SSM `5/5` 计数不变，待 epoch40 自然终止及独立预测复算后再更新。

<!-- CODEX-A2G-ITER0-FOLD0-RUN-CONTRACT-20260823-0201:END -->

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0204:START -->
## 2026-08-23 模型优化迭代 0 / fold0 最新运行观测（02:04）

本块为 active run 的 append-only 中间观测，不覆盖历史终态，不写入五折均值/标准差或 final-test 主表。协议仍为 `Junyi2015 / fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。

- Junyi2015 `item_residual_dropout=0.4` Full v2 已推进到 epoch19；当前最好仍为 epoch14：validation AUC=`0.8025`、ACC=`0.8552`，相对冻结 UKT baseline AUC=`0.8023922269` 的中间 ΔAUC 约 `+0.0001078`。
- queue PID/PGID=`67987/67987`、watchdog PID=`67991`、own-PGID supervisor PID=`75445` 仍存活，GPU1 仅该 compute；`result.json`、`evidence/queue.exit` 和 validation predictions 尚未生成。
- 当前严格主账仍保持 **A2G Full `5/8`、Selective-SSM matched 消融 `5/5`、`quality_freeze=false`**。epoch14-19 仅为运行中观测，不计入 Full-positive；自然终态后再独立复算并追加终态。

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0204:END -->

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0219:START -->
## 2026-08-23 模型优化迭代 0 / fold0 最新运行观测（02:19）

本块为 active run 的 append-only 中间观测，不覆盖历史终态，不写入五折均值/标准差或 final-test 主表。协议仍为 `Junyi2015 / fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。

- Junyi2015 `item_residual_dropout=0.4` Full v2 已推进到 epoch20；当前最好仍为 epoch14：validation AUC=`0.8025`、ACC=`0.8552`，相对冻结 UKT baseline AUC=`0.8023922269` 的中间 ΔAUC 约 `+0.0001078`。
- queue PID/PGID=`67987/67987`、watchdog PID=`67991`、own-PGID supervisor PID=`75445` 仍存活，GPU1 仅该 compute；`result.json`、`evidence/queue.exit` 和 validation predictions 尚未生成。
- 当前严格主账仍保持 **A2G Full `5/8`、Selective-SSM matched 消融 `5/5`、`quality_freeze=false`**。epoch14-20 仅为运行中观测，不计入 Full-positive；自然终态后再独立复算并追加终态。

<!-- CODEX-A2G-ITER0-FOLD0-CURRENT-STATUS-20260823-0219:END -->

<!-- CODEX-A2G-ITER0-FOLD0-JUNYI-V2-TERMINAL-20260823-START -->
## 2026-08-23 模型优化迭代 0 / fold0 Junyi item-dropout40 v2 终态

该块为 Junyi2015 `item_residual_dropout=0.4` 单点的自然终态与独立 prediction 复算。协议固定为 `fold0 / seed42 / validation-only`，`test_access=false`，`window_test_access=false`；没有 test/window-test、网格或重试。

- 远端 root=`<REMOTE_HOME>/a2g_mambakt/item_dropout40_full_junyi_validation_20260822_v2_172`，host=`172.25.114.0`，物理 GPU1；queue PID/PGID=`67987/67987`，watchdog PID=`67991`，own-PGID supervisor PID=`75445`。queue exit=`0`，supervisor=`queue_exit_supervisor_idle`，watchdog=`queue_idle_exit`，终态 compute=`0`；两卡分别使用 `29 MiB` 与 `13 MiB`（0% compute）。
- result SHA256=`e32ec7617216cbc257f1f8e42ab819019eacd2c33bd7672bb4149dcd3bcbd25e`；validation predictions SHA256=`f7fcbf8b2dc908d7de944bd95bb371817ec1513bd865d92c93d402c829513ba6`。
- Runner result metrics: AUC=`0.8024621137889218`、ACC=`0.8551782029378441`、NLL=`0.3582467872516412`、Brier=`0.10932915370923144`、ECE15=`0.0030756085302059994`，interactions=`4083743`，validation sequence rows=`44927`，selected epoch=`14`。
- 独立读取 `validation_predictions.npz`（`label`/`probability` 共 `4083743` 条）复算得到完全相同五指标，最大绝对差=`0.0`。
- 冻结 UKT baseline AUC=`0.8023922269217165`；Junyi ΔAUC=`+0.0000698868672053`，严格高于该库最强准入 baseline，因此按 Full AUC 规则计为 positive。ACC Δ=`-0.0000955006228355` 仅作描述，不改变 Full AUC 判定。

Junyi v2 使当前 A2G Full 严格 positive 从 **`5/8` 更新为 `6/8`**：ASSIST2009 corrected/collapsed、ASSIST2015、ASSIST2017、Junyi2015、NIPS Task 3&4、Slepemapy。ASSIST2012 与 Statics2011 仍为 fail。Selective-SSM matched 消融仍为 **`5/5`**，其 AUC/ACC 门槛为 `ΔAUC > 0.001` 且 `ΔACC >= -0.0005`；Junyi Full 的 `+0.0000699` 不构成新的模块消融通过。`quality_freeze=false`，单折单 seed 结果不写入五折均值/标准差或 final-test 主表。

<!-- CODEX-A2G-ITER0-FOLD0-JUNYI-V2-TERMINAL-20260823-END -->

<!-- CODEX-A2G-REMOTE-ASSET-SUPERSESSION-20260823:START -->
## 2026-08-23 远端可观测资产收口（不启动 GPU）

对 `172.25.114.0:<REMOTE_HOME>/EduKTM_Baselines/raw_sources/KT` 做只读核验：Assist2012 的原始时间/行为字段与 Statics2011 的 Step Start Time、duration、KC/Opportunity 已确认存在；但强制首筛数据集 Assist2015 的官方 raw/canonical 仅包含 `user_id`、`log_id`、`sequence_id`、`correct`。其中 `sequence_id` 已序列化为现有 concept stream，`log_id` 仅用于用户内排序且无可验证时间语义，`user_id` 在 learner-disjoint fold 下不可作为可训练特征，因此没有新的 Assist2015 输入通道。

本次审计未授权新 runner、未创建远端 root、未启动 GPU、未访问 test/window-test，也未重开 difficulty/Rasch、graph、history/time、identity 或已关闭的训练目标路线。当前严格状态保持 A2G Full `6/8`、Selective-SSM AUC/ACC 消融 `5/5`、`quality_freeze=false`；这不改变五折均值/标准差主表。

机器可读审计：`outputs/A2G_REMOTE_OBSERVABLE_ASSET_SUPERSESSION_20260823.json`，SHA256=`4d83978b467023e9fe2e9439ab2700d99c38b35385b443dbdcc31b327d896a3c`。Assist2015 canonical 中另有 24,830 条非二值 `correct` 记录被官方预处理删除；保留它们会改变冻结样本宇宙，暂不作为当前候选。

<!-- CODEX-A2G-REMOTE-ASSET-SUPERSESSION-20260823:END -->

<!-- CODEX-A2G-ITER0-FOLD0-FINAL-SYNC-20260823:START -->
## 2026-08-23 模型优化迭代 0 / fold0 最终结果同步

本块是当前迭代 0 的最新单折单 seed 结果汇总，append-only 追加，不覆盖历史快照。统一协议为 `fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`；结果不写入五折均值/标准差或 final-test 主表。Full 方案为真实 item 序列上的 `item_residual_dropout=0.4`，concept-only 数据保持原路径；ASSIST2015 使用冻结 current Full。

### Full 与冻结最强 baseline

| 数据集 | Full 方案 | AUC | ACC | 最强冻结 baseline | baseline AUC | ΔAUC | Full AUC 判定 |
|---|---|---:|---:|---|---:|---:|---|
| ASSIST2009 corrected/collapsed | item-dropout40 raw-max | 0.8575213800 | 0.7934122101 | UKT | 0.8566249815 | +0.0008963985 | pass |
| ASSIST2012 | item-dropout40 | 0.7742878652 | 0.7533097147 | RobustKT | 0.7762642894 | -0.0019764242 | fail；路线关闭 |
| ASSIST2015 | current Full | 0.7349005323 | — | AKT | 0.7337756158 | +0.0011249165 | pass |
| ASSIST2017 | item-dropout40 | 0.7836785178 | 0.7249766442 | ACE-KT | 0.7800295288 | +0.0036489891 | pass |
| Junyi2015 | item-dropout40 | 0.8024621138 | 0.8551782029 | UKT | 0.8023922269 | +0.0000698869 | pass |
| NIPS Task 3&4 | item-dropout40 | 0.7972875499 | 0.7240005194 | AKT | 0.7961115327 | +0.0011760172 | pass |
| Slepemapy | item-dropout40 | 0.8026728827 | 0.8063901925 | SimpleKT | 0.7968250064 | +0.0058478763 | pass |
| Statics2011 | exact Full | 0.8259118330 | — | AKT | 0.8293542043 | -0.0034423713 | fail |

当前严格 Full AUC positive 为 **`6/8`**：ASSIST2009 corrected/collapsed、ASSIST2015、ASSIST2017、Junyi2015、NIPS Task 3&4、Slepemapy；ASSIST2012 与 Statics2011 仍失败。该 `6/8` 是 fold0/seed42 validation-only supporting evidence，不是五折均值、多 seed、test 或 universal-SOTA 结论。

### Selective-SSM matched 消融（AUC/ACC-only）

门槛固定为 `ΔAUC > 0.001` 且 `ΔACC >= -0.0005`；NLL/Brier/ECE 仅记录，不参与本门禁。

| Full-positive 数据集 | ΔAUC (Full - no-SSM) | ΔACC (Full - no-SSM) | 判定 |
|---|---:|---:|---|
| ASSIST2009 corrected/collapsed | +0.0038287568 | +0.0043161382 | pass |
| ASSIST2015 | +0.0015643539 | +0.0003990306 | pass |
| ASSIST2017 | +0.0348967979 | +0.0168618971 | pass |
| NIPS Task 3&4 | +0.0024374922 | +0.0008328072 | pass |
| Slepemapy | +0.0026472287 | +0.0017566469 | pass |

因此当前模块消融为 **`5/5`**（仅 AUC/ACC-only、五个 Full-positive 数据集、单折单 seed），但 `quality_freeze=false` 仍保持不变；不得将其表述为五折、多 seed 或 universal module claim。

### 证据绑定

- 冻结 baseline registry：`outputs/A2G_FOLD0_SEED42_VALIDATION_BASELINE_REGISTRY_V3_20260821.json`，SHA256=`923bc69a60784aed4fbde90a964bd482472ded27759110b81eab4519c7e1a4f1`。
- Full 终态审计：`outputs/A2G_ITEM_DROPOUT40_FULL_ASSIST2009_TERMINAL_AUDIT_20260822.json`（result SHA256=`9e7f082cfcc286ed6f2d245fef19151a74c0bd1cce7ecd387607e0e08656f11a`）、`A2G_ITEM_DROPOUT40_FULL_ASSIST2017_TERMINAL_AUDIT_20260822.json`（`6d253c3f5b209dd175af302cf276f35df8df65d167195d769c759669e9687690`）、`A2G_ITEM_DROPOUT40_FULL_NIPS_TERMINAL_AUDIT_20260822.json`（`aed6811370ed8199c2cc0f75aebc3cfd2066461e9339d0043f04f0dd955681fc`）、`A2G_ITEM_DROPOUT40_FULL_SLEPEMAPY_TERMINAL_AUDIT_20260822.json`（`3e6026f1b8937a70ae0568ff4f8ca2a6f040db67e13229805061d3e85cef6b87`）、Junyi v2 result SHA256=`e32ec7617216cbc257f1f8e42ab819019eacd2c33bd7672bb4149dcd3bcbd25e`。
- Selective-SSM 终态审计：`outputs/A2G_SELECTIVE_SSM_ITEM_DROPOUT40_ASSIST2009_TERMINAL_AUDIT_20260822.json`、`..._NIPS_TERMINAL_AUDIT_20260822.json`、`..._ASSIST2017_TERMINAL_AUDIT_20260822.json`、`..._SLEPEMAPY_TERMINAL_AUDIT_20260822.json`；ASSIST2015 的 AUC/ACC 绑定见 `outputs/A2G_SELECTIVE_SSM_AUC_ACC_ONLY_EVIDENCE_SUPERSESSION_20260820.json`。
- 所有终态均独立复算 validation predictions，最大绝对指标差为 `0.0`；无 test/window-test、无网格、无重试、无自动后继。

<!-- CODEX-A2G-ITER0-FOLD0-FINAL-SYNC-20260823:END -->

<!-- CODEX-A2G-ITER0-FOLD0-ALIBI-TERMINAL-20260823:START -->
## 2026-08-23 模型优化迭代 0 / fold0 fixed causal ALiBi 终态

该块为 Assist2015 `fold0 / seed42 / validation-only` 的唯一预注册 ALiBi 单点。候选在现有四层 causal MHA logits 加入固定、零参数的相对位置偏置；SSM、dropout、norm、loss、optimizer 与预测头均保持不变。`test_access=false`、`window_test_access=false`，无网格、无重试、无跨数据集扩展。

- 远端 root=`<REMOTE_HOME>/a2g_mambakt/alibi_fixed_assist2015_screen_20260823_v2_172`，host=`172.25.114.0`，物理 GPU1；queue PID/PGID=`984512/984512`，watchdog PID/PGID=`984516/984516`。queue exit=`0`，watchdog=`queue_idle_exit`，终态 compute=`0`；两卡均为 `0%` compute（GPU0 `29 MiB`、GPU1 `13 MiB`）。
- 候选指标：AUC=`0.7350848159617523`、ACC=`0.7572628444072448`、NLL=`0.5046460510967123`、Brier=`0.16650472945525743`、ECE15=`0.019518829813910718`，selected epoch=`9`。
- 相对冻结 control 的 delta：AUC=`+0.0006922930130`、ACC=`+0.0002822411897`、NLL=`-0.0003279655611`、Brier=`-0.0001509406813`、ECE15=`+0.0016916442660`。AUC 严格门槛为 `>0.001`，因此 AUC fail；其余四项安全门通过，整体 `gate_pass=false`。
- 独立读取 `validation_predictions.npz` 的 `102749` 条预测复算五指标，最大绝对差=`0.0`。result SHA256=`f33cd4c11920b02e77aaf33ca722436c596c3c92b8822e30983321f63cb950a0`，summary SHA256=`b0e13b3f080bbeb5ba43181bd2d04681c09307a0389531142b87e5de3145514d`，predictions SHA256=`4bb6f0f1450985733f5fbfbd285b4b2831d288587e5b20bff81f41c5f5c5b08c`。

该候选按预注册规则关闭，不重试、不扫描、不扩库。当前严格状态不变：Full AUC positive=`6/8`（ASSIST2009 corrected/collapsed、ASSIST2015、ASSIST2017、Junyi2015、NIPS Task 3&4、Slepemapy），ASSIST2012 与 Statics2011 失败；Selective-SSM AUC/ACC-only 消融=`5/5`。`quality_freeze=false`，本单折结果不写入五折均值/标准差或 final-test 主表。机器审计见 `outputs/A2G_ITER0_FOLD0_ALIBI_TERMINAL_AUDIT_20260823.json`。

<!-- CODEX-A2G-ITER0-FOLD0-ALIBI-TERMINAL-20260823:END -->

<!-- CODEX-A2G-ITER0-FOLD0-CONE-TERMINAL-20260823:START -->
## 2026-08-23 模型优化迭代 0 / fold0 CSKT cone-only 终态

该块为 Assist2015 `fold0 / seed42 / validation-only` 的 cone-only 单点终态。候选仅将四个 causal MHA 的点积相似度替换为固定 `r=1.0, gamma=1.0` 的上半空间 cone geometry，不启用 kernel bias；参数增量和 state key 增量均为 `0`。`test_access=false`、`window_test_access=false`，无扫描、无重试、无跨数据集扩展。

- 远端 root=`<REMOTE_HOME>/a2g_mambakt/cone_attention_only_assist2015_screen_20260820_v3_172`，queue PID/PGID=`1871480/1871480`，watchdog PID=`1871485`；queue 与 watchdog 均自然退出，终态 compute=`0`。
- selected epoch=`15`，candidate AUC=`0.7352433128427922`；相对冻结 control 的 delta：AUC=`+0.0008507899`、ACC=`+0.0015474603`、NLL=`-0.0016091674`、Brier=`-0.0003812316`、ECE15=`-0.0021460`。
- AUC 严格门槛为 `>0.001`，因此 AUC fail；ACC、NLL、Brier、ECE15 均通过，整体 `gate_pass=false`。该候选按预注册规则关闭，不做半径、gamma、层数、kernel bias 或其他扫描。
- result SHA256=`5292b8b0f8d6da22331d878b7157a5d59cf9a0734e67a03897a5936352d01e98`，summary SHA256=`ba2b9a84634229a0bdcc78d8a48028d23b702ce6b2b6438c850eb871ca94144e`，validation predictions SHA256=`e9563f6a6317f032b5f4769a54aa2743a8328421b4b8d3e964e82d862ea9b0d5`。

该失败单点不改变当前严格状态：Full AUC positive=`6/8`，Selective-SSM AUC/ACC-only 消融=`5/5`，`quality_freeze=false`。本结果不写入五折均值/标准差或 final-test 主表。机器审计见 `outputs/A2G_ITER0_FOLD0_CONE_TERMINAL_AUDIT_20260823.json`。

<!-- CODEX-A2G-ITER0-FOLD0-CONE-TERMINAL-20260823:END -->

<!-- CODEX-A2G-ITER0-FOLD0-CHANNEL-RECALIBRATION-TERMINAL-20260823:START -->
## 2026-08-23 模型优化迭代 0 / fold0 causal channel recalibration 终态

该块为 Assist2015 `fold0 / seed42 / validation-only` 的唯一通道重标定单点。候选在冻结 attention 输出后加入严格前缀因果的 cumulative-context channel gain，尺度向量零初始化；不改变 SSM、attention logits、dropout、norm、loss 或输入数据。新增 1 个参数/state key，新增 RNG 调用为 `0`。`test_access=false`、`window_test_access=false`，无扫描、无重试、无跨数据集扩展。

- 远端 root=`<REMOTE_HOME>/a2g_mambakt/causal_channel_recalibration_assist2015_screen_20260823_v1_172`，GPU1；queue PID/PGID=`1040572/1040572`，watchdog PID=`1040576`，queue 自然退出，watchdog=`queue_idle_exit`，1 秒轮询；未检测 foreign compute，终态 compute=`0`，两卡均空闲。
- selected epoch=`6`，candidate AUC=`0.7343934305776534`、ACC=`0.7569514058530983`、NLL=`0.5049718248662446`、Brier=`0.16665498326054795`、ECE15=`0.01768583345220209`。
- 相对冻结 control 的 delta：AUC=`+0.0000009076`、ACC=`-0.0000291974`、NLL=`-0.0000021918`、Brier=`-0.0000006869`、ECE15=`-0.0001413521`。AUC 严格门槛为 `>0.001`，因此 AUC fail；其余四项安全门通过，整体 `gate_pass=false`。
- 独立读取 `validation_predictions.npz` 的 `102749` 条预测复算五指标，最大绝对差=`0.0`。result SHA256=`049a4a30b60f2b0e283eb9a2b90aff007e03d45a9501917b3e2593f123440b85`，summary SHA256=`52ce7f5110d1beb8e7a522014ae3fb61765d08a353d49975706769572c90a688`，predictions SHA256=`cae5fbcf7998e7ac5ff5adeb08835d9ea58b5879b8a58e189e5ebdf3aa0856fa`。

该候选按预注册规则关闭，不进行尺度、层数、激活或函数扫描。当前严格状态不变：Full AUC positive=`6/8`，Selective-SSM AUC/ACC-only 消融=`5/5`，`quality_freeze=false`。本单折结果不写入五折均值/标准差或 final-test 主表。机器审计见 `outputs/A2G_ITER0_FOLD0_CHANNEL_RECALIBRATION_TERMINAL_AUDIT_20260823.json`。

<!-- CODEX-A2G-ITER0-FOLD0-CHANNEL-RECALIBRATION-TERMINAL-20260823:END -->

<!-- CODEX-A2G-ITER0-FOLD0-LATEST-SUPERSESSION-20260823:START -->
## 2026-08-23 模型优化迭代 0 / fold0 最新终态收口

本区块 supersede 本文件中更早的迭代0运行中观测和旧计数快照；历史终态仍保留用于审计。统一协议为 `fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。以下结果不写入五折均值/标准差或 final-test 主表。

### 当前 Full 方案与冻结最强基线

当前 Full 方案为真实 item 序列上的 `item_residual_dropout=0.4`；concept-only 数据集保持冻结路径。A2G Full 在冻结 baseline registry 上严格超过最强准入 baseline 的数据集为 **`6/8`**：ASSIST2009 corrected/collapsed、ASSIST2015、ASSIST2017、Junyi2015、NIPS Task 3&4、Slepemapy。

| 数据集 | Full AUC | 最强冻结 baseline | baseline AUC | Delta AUC | Full 判定 |
|---|---:|---|---:|---:|---|
| ASSIST2009 corrected/collapsed | 0.8575213800 | UKT | 0.8566249815 | +0.0008963985 | pass |
| ASSIST2012 | 0.7742878652 | RobustKT | 0.7762642894 | -0.0019764242 | fail |
| ASSIST2015 | 0.7349005323 | AKT | 0.7337756158 | +0.0011249165 | pass |
| ASSIST2017 | 0.7836785178 | ACE-KT | 0.7800295288 | +0.0036489891 | pass |
| Junyi2015 | 0.8024621138 | UKT | 0.8023922269 | +0.0000698869 | pass（窄裕量） |
| NIPS Task 3&4 | 0.7972875499 | AKT | 0.7961115327 | +0.0011760172 | pass |
| Slepemapy | 0.8026728827 | SimpleKT | 0.7968250064 | +0.0058478763 | pass |
| Statics2011 | 0.8259118330 | AKT | 0.8293542043 | -0.0034423713 | fail |

因此当前可报告的是 fold0/seed42 validation-only supporting evidence：Full=`6/8`，不是五折、多 seed、final-test 或 universal-SOTA 结论。

### Selective-SSM AUC/ACC-only 消融

预注册门槛为 `Delta AUC > 0.001` 且 `Delta ACC >= -0.0005`；NLL/Brier/ECE15 继续记录但不参与本轮消融淘汰门禁。五个 Full-positive 数据集均通过，当前消融为 **`5/5`**。

| Full-positive 数据集 | Delta AUC（Full - no-SSM） | Delta ACC（Full - no-SSM） | 判定 |
|---|---:|---:|---|
| ASSIST2009 corrected/collapsed | +0.0038287568 | +0.0043161382 | pass |
| ASSIST2015 | +0.0015643539 | +0.0003990306 | pass |
| ASSIST2017 | +0.0348967979 | +0.0168618971 | pass |
| NIPS Task 3&4 | +0.0024374922 | +0.0008328072 | pass |
| Slepemapy | +0.0026472287 | +0.0017566469 | pass |

`quality_freeze=false` 保持不变；`5/5` 仅表示单折单 seed 的 AUC/ACC-only 模块证据，不能外推为五折或多 seed 模块主张。

### 迭代0候选收口

Assist2015 首筛的 fixed causal ALiBi（Delta AUC=`+0.0006922930`）、CSKT cone-only（Delta AUC=`+0.0008507899`）和 causal channel recalibration（Delta AUC=`+0.0000009076`）均未达到严格 `Delta AUC > 0.001`，已按预注册规则关闭；不进行参数扫描、重试或自动跨数据集扩展。当前没有新的合法 GPU 候选。

证据入口：`outputs/A2G_ITER0_FOLD0_FINAL_MARKDOWN_SYNC_20260823.json`、`outputs/A2G_ITER0_CANDIDATE_REAUDIT_UPDATE_CHANNEL_20260823.json`、`outputs/A2G_FOLD0_SEED42_VALIDATION_BASELINE_REGISTRY_V3_20260821.json`。本区块不改变五折均值/标准差表。

<!-- CODEX-A2G-ITER0-FOLD0-LATEST-SUPERSESSION-20260823:END -->

<!-- CODEX-A2G-ITER0-UID-CAUSAL-EVIDENCE-CONTINUITY-TERMINAL-20260823:START -->
## 2026-08-23 模型优化迭代 0 / UID causal evidence continuity 终态

该候选只在 Assist2015 `fold0 / seed42 / validation-only` 做了预注册首筛。它跨非重叠 learner slice 携带固定 rolling-200 的因果统计/RWCE 状态，不增加参数、dropout、RNG、loss 或外部输入；首段 forward/backward 与 control 位级一致，CPU 合同 `17/17` 通过。

| 指标 | control | UID continuity candidate | Delta candidate - control | 门禁 |
|---|---:|---:|---:|---|
| AUC | 0.7349004838 | 0.7349046306 | +0.0000041468 | fail（需 > +0.001） |
| ACC | 0.7578175943 | 0.7572044497 | -0.0006131447 | fail（需 >= -0.0005） |
| NLL | 0.5050684186 | 0.5050010556 | -0.0000673630 | report-only |
| Brier | 0.1665854714 | 0.1666334157 | +0.0000479443 | report-only |
| ECE15 | 0.0200212258 | 0.0208994753 | +0.0008782495 | report-only |

独立读取两份 `validation_predictions.npz` 复算五指标，最大绝对差为 `0.0`。AUC 与 ACC 均未过预注册门槛，候选已关闭；不重试、不扫参、不扩展第二数据集。当前严格状态仍为 Full=`6/8`、Selective-SSM AUC/ACC-only=`5/5`、`quality_freeze=false`。

远端 queue PID/PGID=`1632817/1632817`、watchdog PID/PGID=`1632821/1632821` 已自然退出，未检测 foreign compute，终态 compute=`0`。机器审计：[A2G_ITER0_UID_CAUSAL_EVIDENCE_CONTINUITY_ASSIST2015_TERMINAL_AUDIT_20260823.json](<USER_HOME>/Documents/Codex/2026-08-13/codex-threads-019fb778-09b4-7660-8613/outputs/A2G_ITER0_UID_CAUSAL_EVIDENCE_CONTINUITY_ASSIST2015_TERMINAL_AUDIT_20260823.json)。

<!-- CODEX-A2G-ITER0-UID-CAUSAL-EVIDENCE-CONTINUITY-TERMINAL-20260823:END -->

<!-- CODEX-A2G-MODULE-ABLATION-COVERAGE-20260823:START -->
## 2026-08-23 模块消融覆盖审计

当前严格证明为 AUC/ACC-only `5/5` 的只有 Selective-SSM。该结论不代表所有内部模块都已优化至最优或完成有效消融。

| 模块 | 当前证据状态 | 不能作出的结论 |
|---|---|---|
| Selective-SSM recurrence | Full-minus-ablation 在五个 Full-positive 数据集均满足 `Delta AUC > 0.001`、`Delta ACC >= -0.0005` | 不能外推为五折、多 seed 或 universal module claim |
| item residual dropout=0.4 | Full 方案在 `6/8` 数据集为 baseline-positive | Full 方案优化不是 item-dropout 的 matched 消融 |
| causal statistics / RWCE | 删除/变体已有必要性或负向/非材料诊断 | 没有当前 Full-positive-set 的正向 `5/5` 消融 |
| empirical prior/support | 存在 shifted-slice 初始化缺陷；修复候选首筛未过材料门 | 缺陷修复不能 retroactively 作为性能提升 |
| split-boundary read modulation | 只有机制活动/边界诊断 | 没有因果性能收益证据 |
| causal attention/post-norm/FFN | 部分删除或替换已失败/关闭，部分为工程路径 | 没有模块已“优化至最优”的证据 |
| direct stat/prior residual head、embedding/response semantics | 删除必要性或历史负向证据 | 必要性不等于正向有效消融 |

因此当前准确表述是：Full `6/8`，Selective-SSM AUC/ACC-only 消融 `5/5`，其余模块仍为未完成的逐模块有效性目标；`quality_freeze=false`。机器可读覆盖审计：`outputs/A2G_MODULE_ABLATION_COVERAGE_AUDIT_20260823.json`，SHA256=`b6afdf53814bc3de23b3c2aabbeac3e58c6b22cd2ba3447e85d82451bebfea1d`。本块不修改五折均值/标准差主表。

<!-- CODEX-A2G-MODULE-ABLATION-COVERAGE-20260823:END -->

<!-- CODEX-A2G-CONCEPT-GAIN-TERMINAL-20260824:START -->
## 2026-08-24 Concept-Channel Gain 首筛终态与模块覆盖更新

Concept-Channel Gain 在 Assist2015 `fold0 / seed42 / validation-only` 完成了唯一预注册单点。该模块只在 `concept_emb` 与 `hist_concept_emb` 输出后加入一个共享可学习标量，初值为 `1.0`，新增 1 个参数、0 个 RNG 调用；CPU 合同、静态合同和 identity 起点均通过。它不是未执行或未优化的候选，而是完成了因果对齐首筛后未达到材料门槛的候选。

| 指标 | Frozen Full control | Concept-Gain candidate | Delta candidate - control | 门禁 |
|---|---:|---:|---:|---|
| AUC | 0.7343925229 | 0.7340177104 | -0.0003748126 | fail；需 > +0.001 |
| ACC | 0.7569806032 | 0.7563771910 | -0.0006034122 | fail；需 >= -0.0005 |
| NLL | 0.5049740167 | 0.5050954146 | +0.0001213979 | report-only |
| Brier | 0.1666556701 | 0.1668305037 | +0.0001748335 | report-only |
| ECE15 | 0.0178271855 | 0.0155719636 | -0.0022552220 | report-only |

该单点 `gate_pass=false`，已关闭 exact gain route，不进行 gain 初值/范围扫描、重试或跨数据集扩展。远端 queue PID/PGID=`3068673/3068673`、watchdog PID=`3068677`、compute PID=`3068686` 均已退出；watchdog 自然终态为 `queue_idle_exit`，终态两卡 compute=`0`（GPU0 `29 MiB`、GPU1 `13 MiB`）。

因此当前准确结论仍是：Full `6/8`；只有 Selective-SSM 具备严格 AUC/ACC-only `5/5` 模块消融证据。其它模块并非都已经“优化至最优”：部分是必要性/负向/诊断证据，部分候选在 Assist2015 首筛即低于 `+0.001`，部分尚缺当前 matched Full-minus-ablation 矩阵。不得把这些状态合并成正向模块主张，`quality_freeze=false`，五折均值/标准差主表未改变。

机器审计：`outputs/A2G_CONCEPT_GAIN_ASSIST2015_TERMINAL_AUDIT_20260824.json`，result SHA256=`a99b7977a1646ea3c61c1e76ff9afa448f8d96d9f3807e5ebdbf37b037961cb2`，summary SHA256=`a3487b8e764bff3569a43cead2175dc37aed255d4dc185e9c85870fe1eb4f910`。覆盖更新：`outputs/A2G_MODULE_ABLATION_COVERAGE_UPDATE_20260824.json`。

<!-- CODEX-A2G-CONCEPT-GAIN-TERMINAL-20260824:END -->

<!-- CODEX-A2G-SSM-NORMALIZED-INNOVATION-TERMINAL-20260824:START -->
## 2026-08-24 SSM Normalized-Innovation 首筛终态

此前准备目录中的 `ssm_normalized_innovation_residual` 已核对远端真实终态，不能再视为未运行候选。它保持参数名、SSM dropout、MHA dropout 和 RNG 调用不变，仅将 SSM 输出改为归一化状态减归一化输入的创新项。CPU/static 合同通过，但 Assist2015 首筛未达到材料 AUC 门槛。

| 指标 | Frozen Full control | Normalized-innovation candidate | Delta candidate - control | 门禁 |
|---|---:|---:|---:|---|
| AUC | 0.7343925229 | 0.7344583786 | +0.0000658557 | fail；需 > +0.001 |
| ACC | 0.7569806032 | 0.7577397347 | +0.0007591315 | pass |
| NLL | 0.5049740167 | 0.5036343243 | -0.0013396924 | report-only |
| Brier | 0.1666556701 | 0.1663921542 | -0.0002635159 | report-only |
| ECE15 | 0.0178271855 | 0.0136727679 | -0.0041544176 | report-only |

`gate_pass=false`，因此 exact route 已关闭，不进行归一化位置、系数或残差拓扑扫描，也不扩展第二数据集。远端结果 SHA256=`e1518680953daf1432fa62b5dc4d109a45022ca2e71d6d3731eea8aee0be38bb`，summary SHA256=`98a9ccdfee97e7ef96f85221296a60c3f9584daae94bb635792f0263e044a321`；终态两卡 compute=`0`。

该结果进一步确认：当前 Full `6/8`，严格 AUC/ACC-only 模块 `5/5` 仍只有 Selective-SSM。`quality_freeze=false`，五折均值/标准差主表不变。覆盖更新见 `outputs/A2G_MODULE_ABLATION_COVERAGE_UPDATE_20260824_v2.json`。

<!-- CODEX-A2G-SSM-NORMALIZED-INNOVATION-TERMINAL-20260824:END -->

<!-- CODEX-A2G-ITER0-OBJECTIVE-STATUS-20260824:START -->
## 2026-08-24 迭代 0 目标状态审计

当前目标的两部分必须分开记账：Full 已在冻结准入基线上达到 `6/8`；严格 AUC/ACC-only 的模块 `5/5` 目前只有 Selective-SSM。Concept-Gain 与 SSM normalized-innovation 均已完成 Assist2015 首筛但未过 `Delta AUC > 0.001`，item-residual-dropout 在 concept-only Assist2015 不激活，因此不能在当前五库估计量下组成正向 `5/5` 消融。其它路线的证据为关闭、重复、负向、次材料或尚无当前 matched 矩阵。

机器可读状态：`outputs/A2G_ITER0_OBJECTIVE_STATUS_20260824.json`。结论为 `objective_achieved=false`、`quality_freeze=false`；本块不修改五折均值/标准差主表。

<!-- CODEX-A2G-ITER0-OBJECTIVE-STATUS-20260824:END -->

<!-- CODEX-A2G-ITER0-CANDIDATE-RECONCILIATION-20260824:START -->
## 2026-08-24 技术中止候选 reconciliation

对历史准备目录和远端 artifact 重新核对后，cumulative-mass sparse attention 与 response-decoupled attention memory 均不是待恢复候选，而是已完成 Assist2015 首筛的真实终态：前者 `Delta AUC=+0.0001126684`，后者 `Delta AUC=+0.0004454389`；后者 NLL=`+0.0034806202`、ECE15=`+0.0131956350` 也失败。两条路线均按预注册规则关闭，不重试、不扫参、不扩展。

因此当前 172 GPU 空闲表示队列已完成 fail-fast，而不是训练空转。模块严格 `5/5` 仍只有 Selective-SSM；Full 仍为 `6/8`，`quality_freeze=false`。机器可读 reconciliation：`outputs/A2G_MODULE_ABLATION_COVERAGE_UPDATE_20260824_v3.json`。

<!-- CODEX-A2G-ITER0-CANDIDATE-RECONCILIATION-20260824:END -->

<!-- CODEX-A2G-ITER0-OBJECTIVE-BLOCKED-20260824:START -->
## 2026-08-24 目标阻塞审计

连续四次 append-only 审计（2026-08-21 至 2026-08-24）均确认当前职责矩阵和八数据集输入字段中没有新的 GPU 合格候选。本轮又补核 cumulative-mass sparse attention 与 response-decoupled attention memory 的真实失败终态。因此目标保持未完成：Full=`6/8`，严格模块消融=`1` 个 `5/5`（Selective-SSM），`quality_freeze=false`。

只有出现新的 leakage-safe 可观测资产，或发现 149-row closure 未覆盖的源级正确性缺陷，才可重新建立单点合同。机器审计：`outputs/A2G_ITER0_OBJECTIVE_BLOCKED_AUDIT_20260824.json`。

<!-- CODEX-A2G-ITER0-OBJECTIVE-BLOCKED-20260824:END -->

<!-- CODEX-BASELINE-AUTO-GUARD-V3-PATH-REPAIR-20260824:START -->
## 2026-08-24 228 离线基线守卫 v3 路径修复

v2 守卫引用的旧 `assist2012_missing_native_baselines_validation_20260803_v1` contract 路径已失效；新增 v3 只修正到已有 immutable contract `<REMOTE_HOME>/offline_missing_baseline_gpu_watch_20260804_v1/OFFLINE_BASELINE_CONTRACT.json`（SHA256 `9717af5aa91b309d8653945fe0edd3a7f987d25edfddbbd4cf102ec96504c833`）。该 contract 只允许 228、Assist2012 UKT folds0-4 后 SAINT folds0-4、validation-only、test/window=false。

v3 guard SHA256 `1cc416dea58680e480414b38e22625be96a246738406ff18e703b3959c3a9ee8`，cron 已从 v2 切换到 v3，v2 原件未覆盖。部署后立即检测到 compute PID `3121389/3258617`，状态为 `blocked_gpu_busy`，未启动任何 watcher/A2G/test/window。机器审计 `outputs/BASELINE_AUTO_GUARD_V3_PATH_REPAIR_AUDIT_20260824.json`，SHA256 `566d725f6dfa6d8a5c647d0c51c3538eaeccffb5c4931052e141756d57b758d6`。
<!-- CODEX-BASELINE-AUTO-GUARD-V3-PATH-REPAIR-20260824:END -->

<!-- CODEX-BASELINE-CANONICAL-STATUS-V14-20260825:START -->
## 2026-08-25 canonical 投稿表状态与离线自动补缺

最新独立 inventory 重新解析三张规范表：共 `592` 个数据单元，当前仍有 `387` 个缺少同口径、可复用的独立证据；其中 effectiveness 缺口 `98`，两张效率表缺口 `289`。本次只更新机器清单与状态，不把缺口改写成结果。当前绑定文件为：

- inventory：`outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260825_V14.json`，SHA256 `c660b38a358569def497598819318a6d6c3c7e533f43ee028aa92cf0741eefba`；其中 canonical table-structure SHA256 `f0bca7abd85bc647ff4ebd46e22b14dfdec88277316d0ff28584b9309d91ff36`。
- classification：`outputs/CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260825_V14.json`，SHA256 `34bdecbb4ac50b77bd45a05063429fce94965a84d05b1aab3927bff5ac02769a`。
- completion registry：`outputs/BASELINE_COMPLETION_REGISTRY_20260825_V14.json`，SHA256 `3658ecf9168f1c1f31e4b4c7c623d766496e373b5bd00486fec7c4f8d1f093b1`。

严格准入仍为：同协议五折、独立 checkpoint/config/prediction/metric 审计后才填数；validation-only、smoke、fold0、paper-only、其他数据根/硬件继续保持 `—` 或 `NA`。MCSKT 仍为 paper-only/never-schedule，A2G 不由 baseline queue 启动。

离线自动补缺只允许 228：v3 guard `<REMOTE_HOME>/kt_baseline_20260723/baseline_completion_auto_guard_228_20260816_v3/baseline_completion_auto_guard_228_20260816_v3.sh`，SHA256 `1cc416dea58680e480414b38e22625be96a246738406ff18e703b3959c3a9ee8`；contract SHA256 `9717af5aa91b309d8653945fe0edd3a7f987d25edfddbbd4cf102ec96504c833`，当前 scope 为 Assist2012 UKT folds0-4 后 SAINT folds0-4、validation-only、`test_access=false`、`window=false`。guard 采用单 GPU、共享锁、不可覆盖和空闲自动接续；最新部署探针为 `blocked_gpu_busy`（compute PID `3121389/3258617`），因此尚未启动新任务。172/127 不得启动 KT。


2026-08-25 00:24 +08 只读复核：228 GPU `8804/24564 MiB` used、free `15304 MiB`、utilization `52%`，compute app 为外部 PID `3121389/3258617`，guard 状态 `blocked_gpu_busy`。v22 当前仅有 live 状态 `running` 与 task-boundary pause audit，terminal audit 尚不存在；故显存释放后仍须先满足 v22 terminal/handoff 门禁，不能把空闲瞬间当作后续阶段授权。

离线队列覆盖审计 `outputs/OFFLINE_BASELINE_SCHEDULER_COVERAGE_AUDIT_20260825_V1.json`（SHA256 `bee85f353d1fe72abf2ff9aab2f42f67e859bd7528e73fd3fe6b8ee6e82c314c`）进一步确认：Assist2012 UKT/SAINT validation `10/10`、Assist2012 DKVMN/UKT/SAINT one-step+Window `30` 个已有终态、三模型 efficiency `3/3` 已满足，不会重复调度；下一合法 scope 是 v22 DKVMN 五库 validation `75` trials，随后才是已 hash-bound 的 Mamba4KT smoke、DKVMN v24、DTransformer v23。该审计不写入任何新指标。

2026-08-25 22:43 +08 新鲜只读探针：外部 `<FOREIGN_USER>` PID `3121389` 仍为唯一 compute app（约 `6536 MiB`）；GPU 使用 `6726/24564 MiB`、free `17382 MiB`、utilization `55%`。v3 completion guard 与 idle resume guard 均保持 `blocked_gpu_busy`，未启动新任务；v22 terminal、CEPA handoff、post-CEPA audit 均仍不存在。该状态不改变主表数值。

2026-08-25 22:53 +08 复核仍显示同一外部 `<FOREIGN_USER>` PID `3121389`（约 `6536 MiB`），GPU 使用 `6726/24564 MiB`、free `17382 MiB`、utilization `33%`；两个 228 guard 仍为 `blocked_gpu_busy`，没有新 baseline/A2G/test/window 任务或终态 artifact。

自动恢复链可执行性审计 `outputs/OFFLINE_BASELINE_AUTO_RESUME_AUDIT_20260825_V2.json`，SHA256 `a0a33ef265aee995eb09ba7b46586d239013f2203303b7cf96aab305820d30f1`：Junyi2015 DKVMN predecessor `15/15` validation 已完成且 hash=`6ecec55ba296c596e1588cc4ce476d8fa960a84957a6e62f6cea73391e5365c1`；v22 runner SHA=`b5bc0470705367bb2ac6d23e6c7ea8cc214eb6f3507903618c3e264048ab5558`。该审计确认 GPU 清空后 idle guard 能自动启动 v22 五库 `75` trials；当前仍因外部 PID 阻塞。

2026-08-25 23:14 +08 复核：外部 `<FOREIGN_USER>` PID `3121389` 仍占用 `6536 MiB`，GPU 使用 `6727/24564 MiB`、free `17381 MiB`、utilization `49%`；guard 仍 `blocked_gpu_busy`，没有新结果。离线 smoke/guard 测试本轮 `12/12` 通过。

2026-08-25 23:20 +08 只读复核：PID `3121389` 仍属于外部用户 `<FOREIGN_USER>`，命令为 `evaluate_cocoop_fs_SteelDefectX.py --gpu 0`；GPU 使用 `6727/24564 MiB`、free `17381 MiB`、utilization `48%`。未触碰外部进程，自动守卫仍阻塞。

2026-08-25 23:22 +08 只读复核仍显示外部 PID `3121389`（`<FOREIGN_USER>`）占用 `6536 MiB`；GPU 使用 `6727/24564 MiB`、free `17381 MiB`、utilization `44%`。completion guard 仍为 `blocked_gpu_busy`，本轮 `11/11` 主线回归通过，主表无新证据。
<!-- CODEX-BASELINE-CANONICAL-STATUS-V14-20260825:END -->

<!-- CODEX-A2G-ARTIFACT-BACKED-ITEM-DROPOUT40-PARITY-20260826:START -->
## 2026-08-26 item-dropout40 Full artifact-backed parity 核验

对 172.25.114.0 / GPU1 上 `full_item_dropout40` 的远端 `result.json` 和冻结 baseline registry 做了逐数据集 SHA 绑定重算。协议仍为 `fold0 / seed42 / validation-only`，`test_access=false`、`window_test_access=false`。8 个数据集结果均存在；严格超过最强准入 baseline 的结果为 **6/8**：ASSIST2009 corrected、ASSIST2015、ASSIST2017、Junyi2015、NIPS Task 3&4、Slepemapy。ASSIST2012 和 Statics2011 未超过最强 baseline。

机器审计：`outputs/A2G_ARTIFACT_BACKED_ITEM_DROPOUT40_PARITY_AUDIT_20260826.json`，SHA256=`a2cf134e0c5dc6af12b9dc4c26464b124d12e5e512d9c2685146bae2a6c06a9b`；baseline registry SHA256=`923bc69a60784aed4fbde90a964bd482472ded27759110b81eab4519c7e1a4f1`。该 `6/8` 属于当前 Full family：有 item branch 的数据集使用 `item_residual_dropout=0.4`，concept-only ASSIST2015 使用 `0.0`；不能与早期不同 Full checkpoint/selection estimand 的 parity supersession 混为同一估计量。

该核验不改变五折均值/标准差主表，也不改变 `quality_freeze=false`。模块消融仍只有 Selective-SSM 具备当前严格 AUC/ACC-only `5/5`；不得将 Full 的 `6/8` 直接表述为各模块均已优化至 `5/5`。

<!-- CODEX-A2G-ARTIFACT-BACKED-ITEM-DROPOUT40-PARITY-20260826:END -->

<!-- CODEX-A2G-ITEM-DROPOUT40-PROVENANCE-CORRECTION-20260826:START -->
## 2026-08-26 item-dropout40 provenance correction

进一步核对发现 ASSIST2015 是 concept-only 路径，冻结 `kwargs` 将 `item_residual_dropout=0.0`，因为 item branch 不激活；其 raw-max control result 不能与 item-enabled 数据集的 `item_residual_dropout=0.4` 结果混称为同一 uniform dropout treatment。该修正不改变当前 Full 的严格 baseline 计数：仍为 `6/8`，但明确这是当前 Full family 的组合估计量，而非 8 个数据集统一启用 item dropout 的实验。

修正审计：`outputs/A2G_ITEM_DROPOUT40_PARITY_PROVENANCE_CORRECTION_20260826.json`。它 supersede 旧本地审计 SHA `33646bc10004861dce33e73fbdf38cc83fed93cd8821351f2e561e445aefca36`，当前 corrected parity audit SHA 为 `a2cf134e0c5dc6af12b9dc4c26464b124d12e5e512d9c2685146bae2a6c06a9b`。

该修正进一步确认：Full `6/8` 与模块消融 `5/5` 是分开的证据；严格 AUC/ACC-only `5/5` 仍只有 Selective-SSM，`quality_freeze=false`，不改变五折均值/标准差主表。

<!-- CODEX-A2G-ITEM-DROPOUT40-PROVENANCE-CORRECTION-20260826:END -->

<!-- CODEX-A2G-SOURCE-CORRECTNESS-REAUDIT-20260827:START -->
## 2026-08-27 源级 correctness re-audit

对当前 Full 源 `a2g_mambakt_final.py`（SHA256=`509189b84aeb383d712945ab90b5992e0d76359f292df3f9ddfca726993584b9`）复核了 causal sequence shift、历史统计因果性、attention future mask、item/concept 分支和 RNG helper。未发现一个同时满足“149-row closure 未覆盖、能在 concept-only Assist2015 生效、可构成新质量候选”的源级缺陷。item dropout 未做 inverted-keep rescale 属于已登记的既有设计；Assist2015 item branch 不激活。RNG helper 的 CUDA contract incomplete 是复现协议限制，不是性能模块证据。

审计：`outputs/A2G_SOURCE_CORRECTNESS_REAUDIT_20260827.json`，SHA256=`c9ca0b907eeb895f1f3db35834c65f2b8d0b0af4c6fb79596e56952f202ea5f8`。新 GPU 合格候选数仍为 `0`；目标仍未完成，`quality_freeze=false`。

<!-- CODEX-A2G-SOURCE-CORRECTNESS-REAUDIT-20260827:END -->

<!-- CODEX-A2G-MODULE-REQUIREMENT-RECONCILIATION-20260827:START -->
## 2026-08-27 模块目标 reconciliation

重新按“当前 Full-minus-ablation、同一训练估计量、五个 Full-positive 数据集逐数据集通过”核对模块清单。正式内部模块 `5/5` 仍只有 `Selective-SSM recurrence`。固定外部 logit composite 的 A2G-slot leave-one-out 虽然 AUC/ACC 为 `5/5`，但没有 retrain A2G ablation、改变了预测公式和模型家族，因此不计入内部模块目标。item dropout、RWCE/stat、boundary、attention/postnorm/FFN、prior/stat head、embedding/response semantics 均没有当前正式正向 `5/5` 矩阵。

机器审计：`outputs/A2G_MODULE_REQUIREMENT_RECONCILIATION_20260827.json`，SHA256=`b28884f8ef5d15be9b3b822fff55986bea24818530cdffe22ef7de769084973f`。当前 Full supporting `6/8`、内部模块 `1 个 5/5`、新 GPU 合格候选 `0`、`quality_freeze=false`；不启动重试、网格或自动扩展。

<!-- CODEX-A2G-MODULE-REQUIREMENT-RECONCILIATION-20260827:END -->

<!-- CODEX-A2G-BASELINE-COVERAGE-GAP-20260827:START -->
## 2026-08-27 baseline-cell coverage gap

对冻结 registry 重新按 `13 models x 8 datasets` 检查后，当前 registry 有 `49/104` 个可能的数据集-模型 cell，仍缺 `55` 个 cell。它覆盖了 8/8 数据集，但并不等价于 13 个声明 baseline 在 8 个数据集上全部跑完。因此当前准确表述是：A2G Full 在 registry 已冻结的最强准入 baseline 上通过 `6/8`，而“超过所有声明 baseline 5/8”尚未被完整验证。

缺口审计：`outputs/A2G_BASELINE_COVERAGE_GAP_AUDIT_20260827.json`，SHA256=`6bc4402c270281f2489670bf79671abf70f2de1c9f12c612297b0e1748204816`。该审计不启动 GPU、不修改五折主表；任何补跑 baseline 现在只允许在 228 按 owner-aware authorization、SHA-bound validation-only contract、共享锁和 watchdog 执行；172/127 入口 fail-closed。

<!-- CODEX-A2G-BASELINE-COVERAGE-GAP-20260827:END -->


<!-- CODEX-OFFLINE-GPU-HOST-POLICY-AUDIT-20260826:START -->
## 2026-08-26 离线 GPU 主机可用性与自动补缺复核

本轮只读审计结论：知识追踪实验仍严格 **228-only**；172/127 即使瞬时空闲也不得启动 KT，也不得把其结果纳入五折主表。审计 JSON：
`outputs/OFFLINE_GPU_HOST_POLICY_AUDIT_20260826_V1.json`，SHA256 `a7ec38b8f70e4628a0882fc0310c39c35eea2f3cdb748e1bc0afb4fbd3455fa2`。

- 172（`172.25.114.0`）当前并非空闲：本地 CEPA 进程占用 GPU1 约 6842 MiB、利用率 100%，另有对应 watchdog；该路径为 A2G/CEPA，不是可准入 baseline 队列。未停止、未修改、未使用其结果。
- 228 当前有外部 compute app，completion/idle guard 状态为 `blocked_gpu_busy`；未触碰外部进程。GPU 清空并满足 predecessor、handoff、共享锁和 contract 门禁后，现有 cron 守卫会自动接续 hash-bound 的 v22 DKVMN remaining validation（75 trials），随后才按已注册顺序进入后续 smoke/基线。
- 本轮没有新增 effectiveness、Window、efficiency 或五折数值。缺少独立 checkpoint/config/prediction/metric 五折审计的单元继续保持 `—`/`NA`，不能用 172 结果或 smoke/validation-only 结果填表。

<!-- CODEX-OFFLINE-GPU-HOST-POLICY-AUDIT-20260826:END -->


<!-- CODEX-PUBLICATION-SMOKE-REGRESSION-20260826:START -->
## 2026-08-26 publication baseline smoke 回归收口

本轮只读重算了已归档 smoke 状态并对入口脚本做了 Python 编译与 --help 检查；没有启动 GPU。机器审计：outputs/PUBLICATION_SMOKE_REGRESSION_AUDIT_20260826_V1.json（SHA256 190af19356235c9b041166c642582f2781863177c3446f05972b55e9005c81a0）。

- 归档 native CUDA smoke 为 10/13 通过、3/13 失败：dkt_forget（CPU eye tensor 与 CUDA index 不匹配）、native SparseKT（四返回值与旧 runner 三返回值契约不匹配）、DTransformer（归档环境 Triton CUDA utility 编译失败）。
- 专项 FlucKT、SAINT++ smoke 通过；SparseKT adapter smoke 通过，但它只证明 adapter wiring，不能替代 native SparseKT，也不能写入 effectiveness/efficiency 表。
- 因此缺失模型的主表单元仍保持 —/NA；后续重试必须在 228 建立新的 hash-bound smoke contract，不能使用 172 结果或把 smoke 结果当作五折指标。

<!-- CODEX-PUBLICATION-SMOKE-REGRESSION-20260826:END -->


Note synchronization audit: outputs/NOTE_UPDATE_SYNC_AUDIT_20260826_V1.json, SHA256 9e4b8954a4139020789303de34994ff3fb46fb593c5bea974a5954602a39145e (new host-policy and smoke blocks are present in both copies; historical generated-note divergence is retained without overwriting unrelated content).


## 2026-08-26 01:12 228 自动守卫状态增量

最新机器收据：outputs/OFFLINE_GPU_HOST_POLICY_AUDIT_20260826_V2.json，SHA256 11071d2097de900243310e4cfd7b70ca3a27f9123e3bf51d3cf5fac832bcba41。228 completion guard 仍为 blocked_gpu_busy，未启动新 baseline；172 GPU1 仍由 CEPA 进程占用，172/127 继续禁止 KT。


离线自动链当前审计：outputs/OFFLINE_BASELINE_AUTO_RESUME_AUDIT_20260826_V3.json，SHA256 a7c8c9bd97b74dabb01d956e53ec82bbb7255486661f230078d77e44610fadc4。该收据确认 228 guard 配置、Junyi DKVMN predecessor 15/15 和下一 scope 75 trials；当前只因外部 compute 阻塞，未启动新任务。


## 2026-08-26 01:18 主机状态增量（v3）

最新审计：outputs/OFFLINE_GPU_HOST_POLICY_AUDIT_20260826_V3.json，SHA256 ca0abd2ba59a4dad0edc12ccaef37a6aeb3b84a1d0318f206e01e59af01ec517。172 当前是 CEPA v6 GPU 进程，仍非 baseline 且禁止 KT；228 guard 仍 blocked_gpu_busy，无新五折 artifact。

<!-- CODEX-OFFLINE-V22-RESUME-20260826:START -->
## 2026-08-26 228 离线自动恢复 v22（运行中）

228 在 GPU 清空后已由现有离线守卫通过门禁自动恢复 strict DKVMN v22，启动时间为 `2026-08-26T23:38:41+08:00`。当前 service 主 PID=`107495`、compute PID=`107556`，运行 `Slepemapy / DKVMN / fold1 / seed42 / validation-only`；观测 GPU 为 `35% / 4312 MiB`。`test_access=false`、`window=false`，task07 仍在训练，终态 JSON 尚未形成。

172 的只读探针显示两卡均无 compute app，仅有 `29/13 MiB` 驱动占用；但 KT 主机政策仍为 **228-only**，172/127 不得启动新 KT，也不得把其结果纳入投稿主表。物理空闲不改变统一硬件、环境与审计链要求。

机器审计：`outputs/OFFLINE_BASELINE_V22_RESUME_LAUNCH_AUDIT_20260826_V1.json`，SHA256 `01eaf58c029551fd981e0b900cc792a29a9d90fef03a3beb88e7dd3648c544b3`。v22 runner SHA256 `b5bc0470705367bb2ac6d23e6c7ea8cc214eb6f3507903618c3e264048ab5558`，registry SHA256 `06b988c7f42d3a0ce191550fd1d81351a8d42aae3ae86a60fc8123a848bbd84d`。归档 unit 与当前安装 unit 的 SHA 不同，是因为归档副本只多一行路径注释；执行内容一致，当前安装 unit SHA256 `545aa43fb80144868a5e878e7414d748bb21efd836c98ab84033027bdd412fad`。

本次没有新增 effectiveness、Window、calibration 或 efficiency 数值。活动 checkpoint、validation-only、smoke 和 172 产物均不进入五折主表；缺少独立五折 final artifact 的单元继续保持 `—/NA`。
<!-- CODEX-OFFLINE-V22-RESUME-20260826:END -->

<!-- CODEX-OFFLINE-V22-PROGRESS-20260827:START -->
## 2026-08-27 v22 validation 自然续接进度

v22 的 task07（Slepemapy / DKVMN / fold1 / seed42）已自然完成：`status=complete`、`returncode=0`、`test_access=false`、`final_test_started=false`。task JSON SHA256=`99fe147633229fd08924fcafdee20700be49274337cc436f00414916db87c5ca`，日志 SHA256=`b3c2827d8c786ee77cb50af0b663c2d6f5b7debf264ce5103d11005805480b41`，launch manifest SHA256=`17bbbce944c95a95bb73d4a297aac42701d46a4ebac4135782f92ccb292d9d7e`。

队列随后自然进入 task08（Slepemapy / DKVMN / fold2 / seed42），当前 service 主 PID=`107495`、compute PID=`120504`；活动日志 SHA256=`44a3e0928a2e74a2056b7799996b2a906dffb1312713a82fd3176d7d3a5b7745`，活动 launch manifest SHA256=`e53fb25196aa5b3cd434e7c41729d10f7ccf723731a7119bee73055d17133425`。该活动任务仍是 validation-only，尚无终态 JSON，不能作为主表结果。

注意：远端存在一个历史同名的 NIPS Task 3&4 task08 JSON，但它位于不同的数据集路径且未被覆盖；本次使用的是 Slepemapy 路径下新建的 launch manifest，二者不可混用。

机器审计：`outputs/OFFLINE_BASELINE_V22_PROGRESS_AUDIT_20260827_V1.json`，SHA256=`61341c5a3edc6015f8c5a2c22a39d9334a94030f5f995e21c28871190a7c46e4`。本轮没有新增 effectiveness、Window、calibration 或 efficiency 数值；validation-only 和活动 checkpoint 继续保持在主表之外。
<!-- CODEX-OFFLINE-V22-PROGRESS-20260827:END -->

<!-- CODEX-PUBLICATION-GAP-INVENTORY-V15-20260827:START -->
## 2026-08-27 投稿表缺口 V15 重建

基于当前 PaperGraph 主表重新解析，三张规范表共 `592` 个数据单元，仍缺 `387` 个可准入证据单元。分类结果为：同协议效率测量 `238`、source/adapter 工作 `36`、active/hash-staged `28`、operator-complete FLOPs `27`、outside queue `23`、paper-only/protocol rejected `19`、effectiveness runner `16`。该清单只描述缺口，不把 validation、smoke、fold0、paper-only 或 172 结果写入主表。

机器清单：`outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260826_V15.json`（SHA256 `6a5666bc1995507b8934ff3a7943286a1c6746714d0d798983c96f4241e8ba2a`）；分类：`outputs/CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260826_V15.json`（SHA256 `6f89cf1db2f2f84ffeb0ff0ef2eab6fbe75eb67b5e7807e2e28232a00fcb1ff7`）；registry：`outputs/BASELINE_COMPLETION_REGISTRY_20260826_V15.json`（SHA256 `a2b10764660fe9887c4a32fd810c1bfee83089c9711a9f65a0159dbf2bcb8321`）。当前可调度范围仍固定为 228-only、单 GPU、共享锁、不可覆盖和 `test_access=false`/`window=false`；MCSKT paper-only 永不调度。
<!-- CODEX-PUBLICATION-GAP-INVENTORY-V15-20260827:END -->

<!-- CODEX-SMOKE-ENTRYPOINT-REGRESSION-20260827:START -->
## 2026-08-27 smoke 入口回归

8 个现有 smoke/inventory 入口在兼容依赖环境中全部通过 `py_compile` 与 `--help`（`8/8`、`8/8`）。Python 3.14 的 3 个 YAML 入口曾因解释器缺少 `yaml` 而在导入阶段退出；使用项目已有 PyYAML 6.0.2 的 Python 3.10 后，三个入口均通过。该差异是本地依赖选择问题，不是模型代码结果。

机器审计：`outputs/PUBLICATION_SMOKE_ENTRYPOINT_REGRESSION_20260827_V1.json`，SHA256=`520d13cfeceb548839f7bd300d04d8e663834482ec35e5b05f24e7a13574d24d`。本回归没有启动训练、验证、测试、Window-test 或效率测量；smoke 仍不能填入五折 effectiveness/efficiency 表。
<!-- CODEX-SMOKE-ENTRYPOINT-REGRESSION-20260827:END -->

<!-- CODEX-OFFLINE-V22-PROGRESS-V2-20260827:START -->
## 2026-08-27 v22 task08 终态与 task09 运行

task08（Slepemapy / DKVMN / fold2 / seed42）已自然完成：`status=complete`、`returncode=0`、`test_access=false`、`final_test_started=false`。task JSON SHA256=`9063344f2e68955fa34aab456bd6b409ed33e453f8a960b57e76aab2b8bebc91`，日志 SHA256=`781b94d29d91c657f3d65f237164d6bcb3a8ea33f254d4ad104bbb3f1e2cb36c`，launch manifest SHA256=`7b62bca498d8e5b6f265c8c5307c58d5b8c833ad8d63db3ecee567a14da7b856`。

v22 随后自动进入 task09（Slepemapy / DKVMN / fold3 / seed42），当前 compute PID=`133364`，GPU=`39% / 4312 MiB`。活动日志 SHA256=`405cad370836ee8eda8e2bc6fc2af84111238be53c972d7e3a45b5cc85700db1`，活动 launch manifest SHA256=`aafb7f5f95f16cadf2230a4ad1608cd6fce8be899195b56aa1fa08a237d0a914`；尚无 task09 终态 JSON。validation-only 过程不写入 final 五折主表。

机器审计：`outputs/OFFLINE_BASELINE_V22_PROGRESS_AUDIT_20260827_V2.json`，SHA256=`82805574ef6a90b7bdc2d038bf9ec51c42825cc0c6dc76dc80cd04be3f2a95c7`。本轮没有新增 effectiveness、Window、calibration 或 efficiency 数值。
<!-- CODEX-OFFLINE-V22-PROGRESS-V2-20260827:END -->

<!-- CODEX-OFFLINE-V22-PROGRESS-V3-20260827:START -->
## 2026-08-27 v22 task09 终态与 task10 运行

task09（Slepemapy / DKVMN / fold3 / seed42）已自然完成：`status=complete`、`returncode=0`、`test_access=false`、`final_test_started=false`。task JSON SHA256=`3285d3d6ff0ab84e40db8bee091adc33da471d2001cd07bbc047e11d8f22b75d`，日志 SHA256=`f7707cdff479bc78482d61112da27e9b5165bd7a6ea4cfeb640779248e90f04b`，launch manifest SHA256=`a638a69114e8c15f1670c04df3e9ff58a28a5d97e78a6322638ba1616b1a7c4f`。

v22 随后自动进入 task10（Slepemapy / DKVMN / fold4 / seed42），compute PID=`144441`，GPU=`34% / 4312 MiB`。活动日志 SHA256=`985b5c095e8ea7fb12a6fd37eff9c6a6e4f3288e9e5100c966b4495e9aae4f5c`，活动 launch manifest SHA256=`c3138da53fd5260ddf749ec98f1b2013772e0c3c32f7f9fbb478252e939005bc`；尚无 task10 终态 JSON。validation-only 过程不写入 final 五折主表。

机器审计：`outputs/OFFLINE_BASELINE_V22_PROGRESS_AUDIT_20260827_V3.json`，SHA256=`e6a8a47b2a4a0d546bb6225381d97d18dcec5204031dc91ebe823449c2d9cd80`。本轮没有新增 effectiveness、Window、calibration 或 efficiency 数值。
<!-- CODEX-OFFLINE-V22-PROGRESS-V3-20260827:END -->

<!-- CODEX-LIVE-HOST-STATUS-20260827:START -->
## 2026-08-27 实时主机与离线补缺状态（只读核验）

本次远端核验时间为 `2026-08-27 02:17:49–02:17:50 +08:00`。172 当前有本账户的 A2G/模块筛查进程（GPU1 约 `30% / 2,688 MiB`），并非可用于 KT 基线的空闲资源；228 当前由唯一 v22 服务占用 GPU0（约 `35% / 4,312 MiB`），运行 `Slepemapy / DKVMN / fold0 / seed42 / validation-only` 的 task11。两边均未启动新任务、未停止任何进程。

v22 task10（`Slepemapy / DKVMN / fold4 / seed42`）已自然完成：`status=complete`、`returncode=0`、`test_access=false`、`final_test_started=false`。其 task JSON SHA256=`baaa96b6955c47c2767efadebceb9fece9b49db117ebb60cc1909d17879d8b05`，日志 SHA256=`db820212a0467345a14ceb7e756183bacf173648a95b21fa48d92011b24ec5fd`，launch manifest SHA256=`603e67e800aad775f1b342b2ad59b32c58960d19f8872cc960029c98b1319bce`。当前 task11 仍在训练，终态 JSON 尚未形成；其活动日志 SHA256=`35d0e4a1ef10277226338ca1316d6d823b166fdafcd1f0e45658928f4a774aa6`，活动 launch manifest SHA256=`cbd336030a5d590a948e4b4f11f3280871f5f8d7e51254cc97bdf29d71ae2fef`。

这些任务继续是 validation-only，不进入五折 final-test 均值/标准差主表；效率表中的 TT、IT、显存、FLOPs 也不因本次验证任务而新增数值。172/127 仍为 KT `fail-closed`；仅 228 的离线守卫可以在合法锁、前置终态和 contract 条件满足后继续补缺。
<!-- CODEX-LIVE-HOST-STATUS-20260827:END -->

<!-- CODEX-LIVE-HOST-STATUS-V2-20260827:START -->
## 2026-08-27 v22 task11 终态与 task12 运行（只读核验）

2026-08-27 02:43:35–02:43:36 +08 的实时核验显示：172 两张卡均无 compute app（GPU0 `29 MiB`、GPU1 `13 MiB`），但 172/127 仍为 KT `fail-closed`，不能因物理空闲而启动基线。228 的 v22 服务仍为 `active`，GPU0 约 `37% / 4,313 MiB`。

task11（`Slepemapy / DKVMN / fold0 / seed42`）已自然完成：`status=complete`、`returncode=0`、`test_access=false`、`final_test_started=false`。task JSON SHA256=`0937e76a712736201dffb8c0eb8a316b8691a0307824c421b6d0008d2f83a55a`，日志 SHA256=`d7816cdd7e0b35d5936b7ac37771770919299f166690a7e6c55cab06e0f4146b`，launch manifest SHA256=`9727d293ec176ec50b46a8209c7cd73afe906d134bdc00101d6103bd236c2c3d`。

队列随后自动进入 task12（`Slepemapy / DKVMN / fold1 / seed42`），当前仍为 validation-only，终态 JSON 尚未形成；活动日志 SHA256=`a4e04d1a785238377fffd39cb354774d384d4644bf3378cc73f25d8108108d15`，活动 launch manifest SHA256=`2ba920433d93d1500359ae4672222b44994ab94bcf20e71d3cd357db12d5983d`。本次没有新增五折 effectiveness、Window、calibration 或 efficiency 数值。
<!-- CODEX-LIVE-HOST-STATUS-V2-20260827:END -->

<!-- CODEX-OFFLINE-V22-PROGRESS-V4-20260827:START -->
## 2026-08-27 v22 task12 终态与离线续跑证据

task12 已自然完成并自动进入 task13（`Slepemapy / DKVMN / fold2 / seed42 / validation-only`）。task12 的 task JSON/log/launch manifest SHA256 分别为 `741f73c1c4906829e8f82962fabd9dcbb09ad677af4423b0e56e8caae2a91953`、`bff023d61d2affb6e2b2d0c730da5eb442c77322982411e30627ba2764cecbe9`、`7f156c00b0edafeafeeec0b98b2d8950b86baf82e34bd4f1b1ecbecfafc983b9`。task13 活动日志/manifest SHA256 为 `44a3e0928a2e74a2056b7799996b2a906dffb1312713a82fd3176d7d3a5b7745` / `12d957a955c68f5618a9bc7cf5e2b299049e33fdeadac2fafdee59912f18e558`。

228 的 v22 service 为 enabled/active，completion 与 idle 守卫各有 `@reboot` 和每分钟 cron，共 4 条；共享锁被当前 v22 正常持有。机器审计 `OFFLINE_BASELINE_V22_PROGRESS_AUDIT_20260827_V4.json`，SHA256=`28766268d512855278e37800442c8ab20549d3e18ed1bbb949725faacdb3fc8b`。validation-only 结果不进入五折 final-test 或效率主表。
<!-- CODEX-OFFLINE-V22-PROGRESS-V4-20260827:END -->

<!-- CODEX-PUBLICATION-SMOKE-REGRESSION-V3-20260827:START -->
## 2026-08-27 权威工作区 smoke/guard 回归

在环境声明的权威工作区重新执行静态与单元回归：5/5 个 smoke 入口通过 `--help`，10/10 个 smoke/测试脚本通过 `py_compile`，19/19 个 native/ASIKT smoke、guard、gap inventory、classification、completion registry、smoke inventory 与笔记隐私测试通过。PaperGraph 镜像另有 18/18 编译、3/3 smoke help 通过。

机器审计：`outputs/PUBLICATION_SMOKE_REGRESSION_20260827_V3.json`，SHA256=`8fd729355e2dee0e34d3dc95259e14464218ce7b36f35474636695eab3c1e6f1`。本轮没有启动 GPU、训练、validation、test、Window-test 或效率测量；它只证明入口/合同层，不改变五折主表数值。
<!-- CODEX-PUBLICATION-SMOKE-REGRESSION-V3-20260827:END -->

<!-- CODEX-CANONICAL-GAPS-V16-GUARD-V4-20260827:START -->
## 2026-08-27 canonical V16 与离线守卫 v4

追加状态块后重新解析三张规范表，canonical structure SHA256 仍为 `f0bca7abd85bc647ff4ebd46e22b14dfdec88277316d0ff28584b9309d91ff36`：共 `592` 个数据单元，缺少独立可准入证据的仍为 `387`。V16 inventory/classification/registry SHA256 分别为 `631661a3d933124da3ef277023700a77378cbea5d1dc86432b1a7c24315b7ad8`、`08f9d554039ac0904ceec5fed8b432917acf04d814e168c153ec1569d1198648`、`1ab9ff5f50ffda2604fba099a34cde4a96b1980ae3a864a6c08c497d99e59278`。

228 completion guard v4 已非覆盖部署，SHA256=`6552461a564d953b96843d1b31cff66a01517f4688a27c8515c6957805cfbf53`。它在 v22 terminal 后按已有授权自动执行一次 CEPA synthetic CUDA resource-only gate（无数据、optimizer、training、validation/test/Window），生成 SHA-bound handoff，再进入 Mamba4KT synthetic smoke → DKVMN v24 freeze/audit/one-step final → DTransformer v23 validation-only。部署时 v22 仍 active，v4 首次状态为 `blocked_gpu_busy`，未启动 CEPA 或后续阶段。

部署审计 `outputs/BASELINE_COMPLETION_AUTO_GUARD_V4_DEPLOYMENT_AUDIT_20260827.json`，SHA256=`34bf0c4677832edaa0106c76c9ba833df249682cd90b8b040f4c4dfbc15cdc34`。该 v4 解决 v22 terminal 后的人工 CEPA 交接断点，但不等于 387 个缺口全部有 runner：其余 source/adapter、同协议效率和 operator-complete FLOPs 单元仍须另建合同，主表保持 `—/NA`。
<!-- CODEX-CANONICAL-GAPS-V16-GUARD-V4-20260827:END -->

<!-- CODEX-BASELINE-COMPLETION-GUARD-V5-20260827:START -->
## 2026-08-27 completion guard v5 去重修正

v4 的 CEPA one-shot/交接逻辑保留为受绑定执行器，但 completion cron 已切换到 v5（SHA256=`307f80adaf88f11574787252ee5608a9f12514d82a76b56b34195bd45faf4083`）。v5 为 post-CEPA runner 增加唯一 PID receipt 和命令身份检查；receipt 存在但进程死亡且无 terminal audit 时 fail-closed，不重试。同时识别 Assist2012 UKT/SAINT watcher 的 `idle_no_pending_authorized_tasks` 与 10/10 satisfied 状态，禁止每分钟重复启动已完成 scope。

当前 cron 为 idle guard 2 条 + completion v5 2 条，v3/v4 cron 均为 0；旧脚本原件保留。部署时 v22 继续 active，没有停止/重启，CEPA result/handoff 和 post receipt 均不存在。机器审计 `outputs/BASELINE_COMPLETION_AUTO_GUARD_V5_DEPLOYMENT_AUDIT_20260827.json`，SHA256=`987dfb0e3803dcc99bb4cfba3fcaa9a6bb723a3ffd3ee152273ed2e238e0a919`。
<!-- CODEX-BASELINE-COMPLETION-GUARD-V5-20260827:END -->

<!-- CODEX-BASELINE-COMPLETION-GUARD-V6-20260827:START -->
## 2026-08-27 completion guard v6 与 ASIKT smoke

生产 completion cron 已由 v6 接管（SHA256=`62e86445b85574572f7c47d9757765247b0fe532a8bdd92149f60749827f3c8d`），v5 作为去重委托层保留。v6 只在 post-CEPA audit 为 `complete`、3/3 注册阶段完成、`a2g_started=false`、`window_started=false` 后，才运行一次 ASIKT author synthetic CUDA optimizer-step smoke。

ASIKT corrected wrapper SHA256=`c9c1319e05b877a20331472d9bddbc4b3e0f636af3a522a2578746b99f5e046c`，绑定 v2 authorization/preflight/runner 与五个 runtime 文件；未提供 post-chain handoff 时默认拒绝。ASIKT 使用 batch2/seq32、无真实数据/validation/test/Window；PID receipt 存在但结果缺失时 fail-closed，不重试。部署时 v22 active，ASIKT receipt/result 均不存在，未启动 smoke。

部署审计 `outputs/BASELINE_COMPLETION_AUTO_GUARD_V6_DEPLOYMENT_AUDIT_20260827.json`，SHA256=`01b3bdb161d5634e21f522c67f97e1e439d4e28b7e306125f601f983fc42c997`。MCSKT 仍因无可核实 leak-safe 实现而保持 paper-only，不调度。
<!-- CODEX-BASELINE-COMPLETION-GUARD-V6-20260827:END -->

<!-- CODEX-SMOKE-INVENTORY-V6-REGISTRY-V17-20260827:START -->
## 2026-08-27 smoke inventory V6 / registry V17

新的 smoke inventory 将 Mamba4KT 标为 `scheduled_hash_bound_guard_v6_after_cepa_handoff`，ASIKT 标为 `scheduled_after_registered_post_chain_not_launched`；两者 `actual_gpu_smoke_started=false`，scheduled 不等于 pass。MCSKT 继续为 `blocked_no_verified_repository_and_test_feedback_risk`。

smoke inventory SHA256=`72616f1a373a64688abb75f94df31c8dfcb648bec32695af173761945c52e595`；基于该 inventory 重建的 classification/registry V17 SHA256 分别为 `e7929375eb1f3e593fa365e1d5c65020b250d4aa5db758c6c0b435a084e1f64d`、`15e21fe77d9c8bad5450e77ed3683f857430a8727e94533567450608d05269ce`。规范缺口仍为 `387`，未执行 smoke 不改变 effectiveness/efficiency 表。
<!-- CODEX-SMOKE-INVENTORY-V6-REGISTRY-V17-20260827:END -->

<!-- CODEX-OFFLINE-V22-PROGRESS-V5-20260827:START -->
## 2026-08-27 v22 Slepemapy 15/15 validation 完成

v22 已完成 Slepemapy DKVMN 三个学习率候选 × 五折，共 `15/15` validation trials；dataset audit 为 `status=complete`、`selection_data=validation_only`、`test_access=false`、`final_test_started=false`，SHA256=`e0d157e82fc14943b1acc95b05b5844f704ccda7d33e8bacc88eae397db19596`。队列已自动切换 ASSIST2015，当前链总进度 `31/75` complete。

Slepemapy 最后 task15 的 task/log/manifest SHA256 为 `40e538ce6e10812c01ed622d5e6a96a42553e7ca907b7d02d900547353ca33d5`、`a35eec2e4866b43b2b0d312e27d64095207d1002c0988cf977ec5ef72b504502`、`e8324525d8ea4af54472822231101ff178e91151dab2b3cfeef6dc19b270fdb4`。机器审计 `outputs/OFFLINE_BASELINE_V22_PROGRESS_AUDIT_20260827_V5.json`，SHA256=`e1c70080a2fd506c29c4302a793b81389820aaa1e00c68f0ef03621ba6dc7930`。

该 15/15 只完成 validation tuning，尚未由独立 freeze/final-test pipeline 生成 Slepemapy DKVMN 五折 one-step/Window/calibration 结果，因此主表仍保持 `—`。
<!-- CODEX-OFFLINE-V22-PROGRESS-V5-20260827:END -->

<!-- CODEX-OFFLINE-V22-PROGRESS-V6-20260827:START -->
## 2026-08-27 v22 ASSIST2015 task02 终态与 task03 运行

v22 已从上次审计的 ASSIST2015 task02 活动状态自然推进：task02（DKVMN / fold1 / seed42 / LR=0.0005）为 `status=complete`、`returncode=0`、`test_access=false`、`final_test_started=false`。task JSON SHA256=`a3aaab86f146165da0f81b9028d4923b37809b054746f00f81a00517301a0a87`，日志 SHA256=`6b14f7ff1f58882cab5ae9f68f97fb33bf2fef70bbb00ac5cde2282593e44ec4`，launch manifest SHA256=`95e8d93978dbde5ab8db98f74e37060907afb7cd0495da2c4e2989f2082f25be`。

随后 task03（DKVMN / fold2 / seed42 / LR=0.0005）已启动，观察时 compute PID=`217128`，GPU0=`39% / 3214 MiB`；活动日志 SHA256=`03c00a95a4103905a06c9316a872f0b00bb9453fbc2c8cd14e4d8a1b965ce8a6`，launch manifest SHA256=`a453cea4b215a70d8c0de7144bf7180e86670442c3543053a2db7ed542415cc9`。全链累计完成 `32/75`，其中 NIPS Task 3&4 与 Slepemapy 各 `15/15`；ASSIST2015 当前 `2/15` 完成。

离线守卫仍在生效：completion v6 与 idle v2 均因当前 compute app 正确停留在 `blocked_gpu_busy`，crontab 保留 `@reboot` 和每分钟入口共 `4` 条。KT 继续只允许在 228；127/172 保持 fail-closed。机器审计：`outputs/OFFLINE_BASELINE_V22_PROGRESS_AUDIT_20260827_V6.json`，SHA256=`096e1308fa8bcbd90074b348c9fa4c39d86011d806ea0535906eafa76f7ff65e`。

本区块只记录 validation-only 调度进度，不新增 effectiveness、Window、calibration 或 efficiency 数值；任何新主表单元仍须等待独立冻结 final artifact。
<!-- CODEX-OFFLINE-V22-PROGRESS-V6-20260827:END -->

<!-- CODEX-CANONICAL-GAPS-V18-20260827:START -->
## 2026-08-27 投稿表缺口 V18 复核

基于更新后的 PaperGraph 主表重新解析，三张规范表仍为 `592` 个数据单元、`387` 个缺口：同协议效率测量 `238`、source/adapter 工作 `36`、active/hash-staged `28`、operator-complete FLOPs `27`、outside queue `23`、paper-only/protocol rejected `19`、effectiveness runner `16`。ASSIST2015 的 v22 validation 进展没有被误写成 final 结果，因此缺口数保持不变。

机器清单：`outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260827_V17.json`（SHA256 `a3dfb8c39e6de0687712f7a17b277969fc261a20ec02a760675a634ec1eef105`）；分类：`outputs/CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260827_V18.json`（SHA256 `772c8e8ce2afb3d0cafc344cfabd037b6e0de2aa4e5de09060ad7e659ef31e1b`）；registry：`outputs/BASELINE_COMPLETION_REGISTRY_20260827_V18.json`（SHA256 `e40c41b24c698c032d9c8e397a5132fc5819c3081adcd95ab7066c6922455b1c`）；smoke inventory：`outputs/PUBLICATION_BASELINE_SMOKE_INVENTORY_20260827_V6.json`（SHA256 `72616f1a373a64688abb75f94df31c8dfcb648bec32695af173761945c52e595`）。

复现命令：

```powershell
python work\build_canonical_publication_gap_inventory_20260805.py --input "<PAPERGRAPH>\13-EduKTM-Baselines\5折结果均值标准差汇总_20260710.md" --output outputs\CANONICAL_PUBLICATION_GAP_INVENTORY_20260827_V17.json
python work\classify_canonical_publication_gaps_20260805.py --gaps outputs\CANONICAL_PUBLICATION_GAP_INVENTORY_20260827_V17.json --smoke outputs\PUBLICATION_BASELINE_SMOKE_INVENTORY_20260827_V6.json --output outputs\CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260827_V18.json
python work\build_baseline_completion_registry_20260816.py --gap-inventory outputs\CANONICAL_PUBLICATION_GAP_INVENTORY_20260827_V17.json --classification outputs\CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260827_V18.json --smoke-inventory outputs\PUBLICATION_BASELINE_SMOKE_INVENTORY_20260827_V6.json --output outputs\BASELINE_COMPLETION_REGISTRY_20260827_V18.json
```

调度范围继续固定为 228-only、单 GPU、共享锁、不可覆盖、默认 `test_access=false`/`window_test_access=false`；127/172 fail-closed，MCSKT paper-only 永不调度。
<!-- CODEX-CANONICAL-GAPS-V18-20260827:END -->

<!-- CODEX-OFFLINE-V22-PROGRESS-V7-20260827:START -->
## 2026-08-27 v22 ASSIST2015 task04 终态与 task05 运行

在本轮 smoke/表结构回归期间，v22 又自然完成 ASSIST2015 task03 与 task04。最新终态 task04（DKVMN / fold3 / seed42 / LR=0.0005）为 `status=complete`、`returncode=0`、`test_access=false`、`final_test_started=false`；task JSON SHA256=`bef91fb7efaf0f9604cef26bf41fc0add7633b6450c6fcb60ad1a5104b79f51d`，日志 SHA256=`0e948ac9156f631abb1ce8ffc7fe017d9aafc481d0f1687fa27472dcd2cac9a2`，launch manifest SHA256=`33f949bfc79168fd4fe4bb68b061850f15e50138fa2266b1c2c3fd09bf6a06be`。

随后 task05（DKVMN / fold4 / seed42 / LR=0.0005）已启动，观察时 compute PID=`225483`，GPU0=`40% / 3214 MiB`；活动日志 SHA256=`da544d19cd7513272d30952d7408d106b48afaccd64f15400188c64bdee4f59b`，launch manifest SHA256=`e1de5b30f1eca8f5e3ab02f8953beb3d796a04747e8518d2f39e3fa533faf852`。全链累计 `34/75`，ASSIST2015 为 `4/15`。

completion v6 继续按 compute app 正确保持 `blocked_gpu_busy`，crontab 的 `@reboot` 与每分钟入口共 `4` 条；离线时会在注册链自然边界继续派发。KT 仍为 228-only，127/172 fail-closed。机器审计：`outputs/OFFLINE_BASELINE_V22_PROGRESS_AUDIT_20260827_V7.json`，SHA256=`78de75e2fb396851564d992b22fc4eac356adedce2c687053b406101c1bd70ae`。

本区块只记录 validation-only 调度进度；不新增 effectiveness、Window、calibration 或 efficiency 主表值。
<!-- CODEX-OFFLINE-V22-PROGRESS-V7-20260827:END -->

<!-- CODEX-STRICT-EFFICIENCY-FLOPS-V2-20260827:START -->
## 2026-08-27 strict Assist2012 效率与 operator-complete FLOPs 接入

DKVMN、UKT、SAINT 的 TT、IT、throughput、参数量、GPU peak 来自同一 strict fold-train-only-vocab 协议；analytic FLOPs convention v1 对真实 forward inventory 的全部算子达到 `100%` 公式覆盖，分别为 `5.269`、`140.163`、`141.237` GFLOPs/fixed padded forward batch。该 convention 不等同于未知工具口径的 MCSKT paper-only FLOPs，禁止直接计算相对差异。

公开 receipt：`outputs/ASSIST2012_STRICT_OPERATOR_COMPLETE_FLOPS_RECEIPT_20260827_V2.json`，SHA256=`60bd6e84df0fc18f13b9c3528f32af2868e012c9af82f9fdb0c02b988ecfb68f`。远程完整结果 SHA256=`c977e4718903d882cd40f0ec52649f68a3b36c6eb2d941f6455ba457d3ff2f45`；公式脚本 SHA256=`d66666141e6692920995a397164456a3bcbfb500b3e31b83f63a7a8e8c160466`。v1 在结果写入前因布尔字面量错误 fail-fast，未生成结果，已由非覆盖 v2 取代。
<!-- CODEX-STRICT-EFFICIENCY-FLOPS-V2-20260827:END -->

<!-- CODEX-CANONICAL-GAPS-V19-20260827:START -->
## 2026-08-27 投稿表缺口 V19（strict efficiency/FLOPs 接入后）

三张规范表仍为 `592` 个数据单元；DKVMN、UKT、SAINT 的 strict TT/IT/throughput/params/GPU/FLOPs 共填补 `21` 个 canonical 单元后，缺口由 `387` 降为 `366`。当前分类：同协议效率测量 `220`、source/adapter 工作 `36`、active/hash-staged `28`、operator-complete FLOPs `24`、outside queue `23`、paper-only/protocol rejected `19`、effectiveness runner `16`。

严格数据根三行和历史数据根四行已明确分层，不跨层排序。机器清单：`outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260827_V18.json`（SHA256 `0a66624b2d8e99efbc084eac4f6cc293c99d07752648260491dbebb681b93236`）；classification：`outputs/CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260827_V19.json`（SHA256 `365e86c030a93914215266ec0aab64c0cc37c40b2fbccb5f53e886457f5640dc`）；registry：`outputs/BASELINE_COMPLETION_REGISTRY_20260827_V19.json`（SHA256 `0c446d1e5fcb6ec248cfd7a8b95680d4713b38313cf32a991724b5229699c978`）；strict sync audit SHA256 `a40943258e6d8afdc588e8f419d4b444fbc30a37a0f4c2aa6be11bef22817563`。

```powershell
python work\build_canonical_publication_gap_inventory_20260805.py --input "<PAPERGRAPH>\13-EduKTM-Baselines\5折结果均值标准差汇总_20260710.md" --output outputs\CANONICAL_PUBLICATION_GAP_INVENTORY_20260827_V18.json
python work\classify_canonical_publication_gaps_20260805.py --gaps outputs\CANONICAL_PUBLICATION_GAP_INVENTORY_20260827_V18.json --smoke outputs\PUBLICATION_BASELINE_SMOKE_INVENTORY_20260827_V6.json --output outputs\CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260827_V19.json
python work\build_baseline_completion_registry_20260816.py --gap-inventory outputs\CANONICAL_PUBLICATION_GAP_INVENTORY_20260827_V18.json --classification outputs\CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260827_V19.json --smoke-inventory outputs\PUBLICATION_BASELINE_SMOKE_INVENTORY_20260827_V6.json --output outputs\BASELINE_COMPLETION_REGISTRY_20260827_V19.json
```

GPU 调度仍为 228-only、单 GPU、共享锁、不可覆盖；validation/smoke/paper-only 不得填 final effectiveness 单元。
<!-- CODEX-CANONICAL-GAPS-V19-20260827:END -->

<!-- CODEX-BASELINE-COMPLETION-GUARD-V7-20260827:START -->
## 2026-08-27 228 离线 completion guard v7

生产 cron 已从 completion v6 原子切换到 v7：idle v2 保留 `@reboot + 每分钟` 两条，completion v7 为 `@reboot + 每分钟` 两条，旧 completion v2-v6 cron 为 `0`。v7 仍先完整委托 v6；只有 v6 明确达到 `asikt_smoke_complete` 后，才允许一次性运行 DKT/SAKT/AKT/SimpleKT 的既有冻结训练 batch forward operator inventory，并用 analytic convention v1 生成 FLOPs。安装时 v22 仍 active，GPU 正由 v22 使用，未启动新 stage，也未停止/重启 v22。

注册顺序：`v22 terminal -> CEPA-v1 one-shot synthetic resource gate -> Mamba4KT synthetic smoke -> DKVMN v24 freeze/audit/one-step final -> DTransformer v23 validation-only -> ASIKT synthetic CUDA optimizer-step smoke -> DKT/SAKT/AKT/SimpleKT existing-frozen-batch operator inventories -> operator-complete analytic FLOPs convention v1 -> stop at no further hash-bound contract`。新增 stage 不训练、不做 validation、不读 test/Window；共享锁、GPU 无 compute、hash-bound、receipt、不可覆盖任一门禁失败都保持 fail-closed。guard SHA256=`311cae81ae38a44f70a4fd46812af3de0852ac3a85a3fd17a36ac3bea67d0935`，runner SHA256=`52ade545ac355ce61f575980b5f9130075160ec2db0893abe5d3f253599d127a`，部署审计 `outputs/BASELINE_COMPLETION_AUTO_GUARD_V7_DEPLOYMENT_AUDIT_20260827.json` SHA256=`355fb7651d9efe2e3c923b5f3db2da23a3c1d9939acbbeaf554f4599a8615784`。
<!-- CODEX-BASELINE-COMPLETION-GUARD-V7-20260827:END -->

<!-- CODEX-DKTPLUS-EFFICIENCY-PARAMS-20260827:START -->
## 2026-08-27 DKT+ Assist2012 效率参数量准入

DKT+ 的归档 fold0/seed42 checkpoint 已在 228 进行 CPU-only `strict=True` exact load：missing/unexpected keys 均为空，总参数与可训练参数均为 `480,865`（`0.480865M`）。本轮没有 forward、inference、training 或 GPU 查询，因此只填参数量两格；TT、IT、throughput、GPU peak、FLOPs 继续保留空值。

证据：`outputs/DKTPLUS_ASSIST2012_EFFICIENCY_EXACT_LOAD_228_20260827.json`，SHA256=`ef0fbfebf7d4ba7fd0b7c111877053e97e2d656d86a7784722e94626c45fd1a8`。
<!-- CODEX-DKTPLUS-EFFICIENCY-PARAMS-20260827:END -->

<!-- CODEX-CANONICAL-GAPS-V20-20260827:START -->
## 2026-08-27 投稿表缺口 V20（DKT+ 参数接入后）

三张规范表仍为 `592` 个数据单元；DKT+ 的 strict CPU exact-load/rebind 只准入总参数和可训练参数两格后，缺口由 `366` 降为 `364`。当前分类：同协议效率测量 `218`、source/adapter 工作 `36`、active/hash-staged `28`、operator-complete FLOPs `24`、outside queue `23`、paper-only/protocol rejected `19`、effectiveness runner `16`。DKT+ 的 TT、IT、throughput、GPU peak 与 FLOPs 仍未测量，继续保留空值。

机器清单：`outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260827_V20.json`（SHA256 `84632a3cb122ca550a34460d293ff0fdd9e8899ca2adace8ae7fa468d17ad01e`）；classification：`outputs/CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260827_V21.json`（SHA256 `46f2740a7d7493260f27bef9cee31121fa3d0b02ecdeaff86405f84a327d98e6`）；registry：`outputs/BASELINE_COMPLETION_REGISTRY_20260827_V21.json`（SHA256 `d736ae34790d815e8d8aaedb008b63438f09ccf5efce6b07c76cd89bd71d7092`）；DKT+ 笔记同步审计 SHA256 `92ccd5373d57f7bc8d2df33e50d3fae5bb057a420ecc84bde026cb083613c339`。

```powershell
python work/build_canonical_publication_gap_inventory_20260805.py --input "<PAPERGRAPH>/13-EduKTM-Baselines/5折结果均值标准差汇总_20260710.md" --output outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260827_V20.json
python work/classify_canonical_publication_gaps_20260805.py --gaps outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260827_V20.json --smoke outputs/PUBLICATION_BASELINE_SMOKE_INVENTORY_20260827_V6.json --output outputs/CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260827_V21.json
python work/build_baseline_completion_registry_20260816.py --gap-inventory outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260827_V20.json --classification outputs/CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260827_V21.json --smoke-inventory outputs/PUBLICATION_BASELINE_SMOKE_INVENTORY_20260827_V6.json --output outputs/BASELINE_COMPLETION_REGISTRY_20260827_V21.json
```

GPU 调度仍为 228-only、单 GPU、共享锁、不可覆盖；validation/smoke/paper-only 不得填 final effectiveness 单元。172/127 保持 fail-closed，除非用户之后明确解除。
<!-- CODEX-CANONICAL-GAPS-V20-20260827:END -->

<!-- CODEX-OFFLINE-V22-PROGRESS-V8-20260827:START -->
## 2026-08-27 v22 ASSIST2015 validation 15/15 终态

ASSIST2015 的 DKVMN validation tuning 已完成 `15/15`：dataset audit 为 `status=complete`、`selection_data=validation_only`、`test_access=false`、`final_test_started=false`，SHA256=`153c5876232835de37f2cb9ed6afe0b477db2478aa27dbdee6b58e9820def13b`。最后一项 `lr_2_9982e8512c_f4`（fold4）return code 为 `0`；task JSON、日志和 launch manifest SHA256 分别为 `ddaa1e575662b3d5f3000acf39f377f47d3135ada2f17eb55f9fbfaa76706710`、`c2722d2a23752e64764dc8631957b990cbba3fa51fe463a855a152660cb2ef76`、`217f4d07318f558a274060800f2b898c78c9d971274151325b2e6e79db3336f5`。

该 dataset 尚未单独生成 freeze；冻结/独立审计/后续授权必须等待 v22 全链 terminal。观察时全链为 `60/75`，已自然进入 `assist2017` fold0 / seed42 / LR=0.0005；v22 service 仍为 active/running，GPU0=`39% / 3746 MiB`。没有停止、重启、重复启动或发送信号。

离线守卫仍为 idle-v2 `2` 条与 completion-v7 `2` 条 cron，旧 completion v2-v6 为 `0` 条；GPU 空闲并满足注册顺序时会自动续跑。机器审计：`outputs/OFFLINE_BASELINE_V22_PROGRESS_AUDIT_20260827_V9.json`，SHA256=`032303411dc3f713f0ebb8b7b7aa8545e990d570c2d11e64ba3f81b237759e9b`。

本区块只证明 validation 队列终态和自动续跑状态，不新增 effectiveness、Window、calibration 或 efficiency 主表数值。
<!-- CODEX-OFFLINE-V22-PROGRESS-V8-20260827:END -->

<!-- CODEX-BASELINE-CHAIN-TERMINAL-V9-20260827:START -->
## 2026-08-27 v22 完成与 post-chain DTransformer 阻塞

228 上 v22 DKVMN validation-only 链已完成 `75/75`，终态审计 SHA256=`f2b40bbb7f0853f2feeb917e13e67aa82e8f7bd91d76da734144b94bab8b3a1b`；`selection_data=validation_only`、`test_access=false`、`final_test_started=false`。此前已准入的 Mamba4KT synthetic smoke 与 DKVMN-v24 均为 `complete`，但 DTransformer-v23 八个数据集均在首个候选 fold 以 `returncode=1`、`failed_not_retried` 结束，未产生可准入的 validation 结果（累计完成 trial `0`），所以 post-chain 为 `terminal_with_failures`，审计 SHA256=`f8fa42bd5ee9c0e0f3dab890aa5b5386511dfd8950b2a4f9f7623ec4c8c7b60a`。

当前 GPU 快照为 `0% / 182 MiB`，无 compute app；completion-v6/v7 已按失败门禁停止在 `blocked_post_chain_not_complete`。idle-v2 与 completion-v7 各保留 `@reboot + 每分钟` 两条 cron，旧 completion v2-v6 为 `0` 条；守卫不会绕过 DTransformer 失败、不会自动重试或把失败填入主表。机器审计：`outputs/BASELINE_CHAIN_TERMINAL_AUDIT_228_20260827_V1.json`，SHA256=`852cea24b175006329c48e4c31bfd9304d94e2d4add39658557148a4873d4cc3`。

本区块只记录 validation 队列终态和 failure bucket；effectiveness、Window、calibration、TT/IT/GPU/FLOPs 主表值均不因该失败状态改变。172/127 仍 fail-closed，172 空闲不构成解禁。
<!-- CODEX-BASELINE-CHAIN-TERMINAL-V9-20260827:END -->

<!-- CODEX-CANONICAL-GAPS-V21-20260827:START -->
## 2026-08-27 投稿表缺口 V21（cycle-free canonical inventory）

基于当前 PaperGraph 主表的三张结构化表重新解析，共 `592` 个 canonical 数据单元，当前缺口仍为 `364`。分类为：同协议效率测量 `218`、source/adapter 工作 `36`、active/hash-staged `28`、operator-complete FLOPs `24`、outside queue `23`、paper-only/protocol rejected `19`、effectiveness runner `16`。本版本的 source SHA256 只覆盖三张选定表结构，不受表外状态记录追加影响。

机器清单：`outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260827_V21.json`（SHA256 `b32736f48e9032aa2db8e1f349fa3b94076b41a96c1229a8e12aa5c8eab57507`）；classification：`outputs/CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260827_V22.json`（SHA256 `02980a301db2cf3a43a92b836275e7a06a3420cce7ea562664fbcf4cdae16ecc`）；registry：`outputs/BASELINE_COMPLETION_REGISTRY_20260827_V22.json`（SHA256 `9b50620e4fc1dba5b6bbf109e8fa65c31188231d3a83214628d6eb6aa4a638a9`）；smoke inventory：`outputs/PUBLICATION_BASELINE_SMOKE_INVENTORY_20260827_V6.json`（SHA256 `72616f1a373a64688abb75f94df31c8dfcb648bec32695af173761945c52e595`）。

```powershell
python work/build_canonical_publication_gap_inventory_20260816.py --input "<PAPERGRAPH>/13-EduKTM-Baselines/5折结果均值标准差汇总_20260710.md" --output outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260827_V21.json
python work/classify_canonical_publication_gaps_20260805.py --gaps outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260827_V21.json --smoke outputs/PUBLICATION_BASELINE_SMOKE_INVENTORY_20260827_V6.json --output outputs/CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260827_V22.json
python work/build_baseline_completion_registry_20260816.py --gap-inventory outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260827_V21.json --classification outputs/CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260827_V22.json --smoke-inventory outputs/PUBLICATION_BASELINE_SMOKE_INVENTORY_20260827_V6.json --output outputs/BASELINE_COMPLETION_REGISTRY_20260827_V22.json
```

调度范围继续为 228-only、单 GPU、共享锁、不可覆盖和默认 `test_access=false`/`window_test_access=false`；172/127 fail-closed，MCSKT paper-only 永不调度。
<!-- CODEX-CANONICAL-GAPS-V21-20260827:END -->

<!-- CODEX-DTRANSFORMER-NATIVE-JIT-RECOVERY-20260827:START -->
## 2026-08-27 DTransformer native-JIT recovery（228，validation-only）

旧 DTransformer-v23 失败根已确认是 Triton `bmm_outer_product` 编译缺少 `Python.h`，8 个数据集均 `failed_not_retried`、无 checkpoint。已在 228 建立不覆盖旧根的 recovery v2，仅增加环境兼容项 `TORCH_DISABLE_NATIVE_JIT=1`；不改变模型参数、数据、split、候选或选择规则。

当前 recovery 链状态为 `running`：Statics2011 已完成 `15/15` validation trials，Junyi2015 已开始；全链范围为 `8` 数据集、`120` trials、validation-only。当前 live SHA256=`00a16ef405bca1e9752ea9b4e709265a05794590f98e460e718d751557a02d3d`，launch receipt SHA256=`3fdc95b1a2a9e08a803889e8ad1dc6434504cd7f715f40d0eaf60fef0320434a`，static deployment audit SHA256=`6b5644a9dd4a78818f27a636873da3bb43a5efd3582c13e0568d0141865127f6`。registry/authorization/runner 均 SHA-bound，单 GPU、共享锁、不可覆盖。

`test_access=false`、`window_test_access=false`、`final_test_started=false`、`a2g_launch=false`。validation recovery 的中间或终态结果不写入五折均值/标准差主表；只有完整五折 final artifact 经过独立审计后才可填 effectiveness 表。效率表中的 TT/IT/GPU/FLOPs 仍需同协议测量，不以 smoke 或 validation 代替。

离线自动续跑继续只允许 228：旧 completion-v7 在 post-chain failure 上保持阻断；新的 append-only 守卫应先识别 recovery receipt/terminal，再按已授权顺序接续效率阶段。172/127 仍 fail-closed，MCSKT 仍 paper-only。
<!-- CODEX-DTRANSFORMER-NATIVE-JIT-RECOVERY-20260827:END -->

<!-- CODEX-DTRANSFORMER-NATIVE-JIT-RECOVERY-COMMANDS-20260827:START -->
## 2026-08-27 DTransformer recovery 可复现命令与门禁

静态部署审计：`outputs/DTRANSFORMER_NATIVE_JIT_FALLBACK_STATIC_AUDIT_20260827_V1.json`，SHA256=`6b5644a9dd4a78818f27a636873da3bb43a5efd3582c13e0568d0141865127f6`；dataset runner SHA256=`2991f888190ccf38a0ca18657dfb22329a929d0986ac9206b3d3395a0adced00`；chain runner SHA256=`d8bfc8833934075c969b383d67d893748b57d55622dcfdd4d5ebfdf93a82ebd3`。三者仅允许在 228 使用，且结果 root 必须是新的 recovery 目录，不得覆盖旧 v23 failure root。

```bash
export KT_HOST_ID=228
export KT_ALLOW_REAL_RUN=1
export KT_ALLOW_VALIDATION_TUNING=1
export KT_ALLOW_DTRANSFORMER_NATIVE_JIT_RECOVERY=1
export KT_ALLOW_FINAL_TEST=0
export TORCH_DISABLE_NATIVE_JIT=1
PY=<REMOTE_HOME>/a2g_mambakt/.venv/bin/python
ROOT=<REMOTE_HOME>/kt_baseline_20260723/strict_dtransformer_validation_native_jit_fallback_20260827_v2
LOCK=<REMOTE_HOME>/kt_baseline_20260723/continuous_baseline_gpu_queue_20260803.lock
$PY <REMOTE_HOME>/kt_baseline_20260723/work/dtransformer_native_jit_fallback_20260827_v2/run_dtransformer_native_jit_fallback_chain_228_20260827.py \
  --root "$ROOT" --registry "$ROOT/evidence/registry.json" \
  --authorization "$ROOT/evidence/authorization.json" \
  --dataset-runner <REMOTE_HOME>/kt_baseline_20260723/work/dtransformer_native_jit_fallback_20260827_v2/run_strict_validation_tuning_native_jit_228_20260827.py \
  --compatibility-smoke <REMOTE_HOME>/kt_baseline_20260723/publication_baseline_gpu_smoke_20260804_v6r2_retry3/outputs/PUBLICATION_NATIVE_BASELINE_GPU_SMOKE_AUDIT_20260804.json \
  --source-terminal <REMOTE_HOME>/kt_baseline_20260723/strict_dtransformer_validation_20260805_v23/STRICT_DTRANSFORMER_REMAINING_CHAIN_AUDIT.json \
  --global-lock "$LOCK" --python "$PY" --poll-seconds 10
```

命令合同固定 `8` 数据集、`120` trials、三候选×五折、validation-only；`test_access=false`、`window_test_access=false`、`final_test_started=false`。必须先检查 compute apps 为空、共享锁可获得、receipt/terminal 不存在；失败留独立证据并停止该链，不自动重试或进入 final-test。smoke 只证明 native-JIT-disabled 的 CUDA optimizer step，不是 effectiveness/efficiency 数值。
<!-- CODEX-DTRANSFORMER-NATIVE-JIT-RECOVERY-COMMANDS-20260827:END -->

<!-- CODEX-BASELINE-COMPLETION-GUARD-V8-20260827:START -->
## 2026-08-27 228 离线 completion guard v8

completion cron 已从旧 v2-v7 append-only 切换到 v8：`@reboot + 每分钟` 共 `2` 条，idle-v2 仍保留 `2` 条。v8 识别当前 DTransformer native-JIT recovery receipt 并等待其自然终态；本次安装时状态为 `dtransformer_recovery_active`、PID `393435`，未重复启动或停止现有 validation 进程。

recovery 终态后，v8 只允许在 228 GPU compute apps 为空、显存 free 至少 `16384 MiB`、共享锁可获得且 receipt/hash 合同有效时接续 legacy DKT/SAKT/AKT/SimpleKT operator inventory 与 analytic FLOPs；该阶段不训练、不验证、不读 test/window。v8 不启动 A2G，不绕过失败、不覆盖旧产物；任一门禁失败即记录 blocked 状态。

guard SHA256=`1de2685b56d93ce7c265e63798a2802bffd276ca182c63020fb52efa9a75ff26`，部署审计 `outputs/BASELINE_COMPLETION_AUTO_GUARD_V8_DEPLOYMENT_AUDIT_20260827.json` SHA256=`10294af8d8b08e01075a715e5523b44668e857db5a8b2dc5867f5e1dd911503d`。当前 DTransformer recovery 仍为 validation-only，不能填入五折 effectiveness/efficiency 主表；五折表只接受完整、独立审计的同协议结果。172/127 继续 fail-closed，MCSKT 继续 paper-only。
<!-- CODEX-BASELINE-COMPLETION-GUARD-V8-20260827:END -->

<!-- CODEX-CANONICAL-GAPS-V23-20260827:START -->
## 2026-08-27 投稿表缺口 V23（当前源重解析）

对当前 PaperGraph 三张 canonical 表重解析得到 `592` 个数据单元，缺口仍为 `364`；当前源 canonical digest=`9d29f028d9f9f38eb346db571e4c6a3eddcaeb6fbd17643ce04c3c4bfadca79f`。分类为：同协议效率测量 `218`、source/adapter `36`、active/hash-staged `28`、operator FLOPs `24`、outside queue `23`、paper-only/protocol rejected `19`、effectiveness runner `16`。

当前机器证据：`outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260827_V23.json` SHA256=`f9fda56cdb22e1f4795ae9e751fa23df386e016a69dfd77c4e68aea94f6cb204`；`outputs/CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260827_V24.json` SHA256=`26ebe3cb6e9bacaa46ac9fa9544c7de046831f1cf89454f12e6b9cb35c2d7358`；`outputs/BASELINE_COMPLETION_REGISTRY_20260827_V24.json` SHA256=`5910b1cadd4c528aa8ae1077be5f9f24d7cecb1e601ff53de9518ff54ffd0f4f`；smoke inventory SHA256=`72616f1a373a64688abb75f94df31c8dfcb648bec32695af173761945c52e595`。本次重解析只更新缺口索引，不把 DTransformer validation recovery、smoke、fold0、paper-only 或 172 结果写入五折主表。

调度仍为 228-only、单 GPU、共享锁、不可覆盖和默认 `test_access=false`/`window_test_access=false`；MCSKT 保持 paper-only，A2G 不由 baseline scheduler 启动。缺口只有在完整同协议五折 artifact 通过独立审计后才可填入主表。
<!-- CODEX-CANONICAL-GAPS-V23-20260827:END -->

<!-- CODEX-SMOKE-TABLE-REGRESSION-V1-20260827:START -->
## 2026-08-27 主表与 Smoke 回归收口（只读审计）

本轮只读回归收据 `outputs/PUBLICATION_SMOKE_AND_TABLE_REGRESSION_20260828_V11.json`，SHA256=`0df46f60df042c52638bf03e578ff0793330fdfc68af2833738aa374a635710e`，状态为 `pass`：当前 17 项 smoke/守卫/recovery 单元测试全部通过，关键 Python 入口编译通过；主表审计 `outputs/PUBLICATION_CURRENT_TABLES_AUDIT_20260828_V12.json`（SHA256=`01b810863fe69bbf47d916b055a3fdb3e1ba0e49958e2796da9fd67d17ff398f`）和隐私门禁均通过。Smoke inventory 仍明确 `smoke_is_not_effectiveness_evidence=true`，没有把 smoke、pilot、validation 或 DTransformer recovery 值提升到五折 effectiveness 表。

当前 canonical 结构仍为 `592` 个数据单元、`364` 个缺口；缺口清单 `outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260827_V23.json` SHA256=`f9fda56cdb22e1f4795ae9e751fa23df386e016a69dfd77c4e68aea94f6cb204`。本轮没有新增指标、没有启动新的实验，也没有改变 MCSKT 的 paper-only 处理。

228 只读快照：GPU 查询为 `util=93, 16847, 7261, 24564`（字段顺序为 utilization%、used/free/total MiB），compute app 为当前 DTransformer recovery PID `406553`；v8 guard=`dtransformer_recovery_active`，`test_access=false`、`window_test_access=false`、`a2g_training_launch=false`。v8 与 idle guard 的 cron 共 `4` 条，旧 completion v2-v7 为 `0` 条；recovery 自然终态且满足显存/共享锁门禁后才会自动接续已登记阶段。172/127 继续 fail-closed，不能用于 KT。

<!-- 回归收据只证明结构、边界和调度状态，不改变五折主表数值。 -->
<!-- CODEX-SMOKE-TABLE-REGRESSION-V1-20260827:END -->

<!-- CODEX-DTRANSFORMER-NATIVE-JIT-LIVE-PROGRESS-V1-20260827:START -->
## 2026-08-27 DTransformer recovery 实时进度（validation-only）

228 只读探针显示 recovery 仍为 `running`，Statics2011 已完成 `15/15`，Junyi2015 fold0 当前日志最新为 `epoch 20, valid AUC 0.7938, valid ACC 0.8529`；这只是 validation 训练进度，不是 final-test 结果，也不写入五折均值/标准差主表。当前 GPU 快照字段为 `utilization%,used/free/total MiB=93, 16847, 7261, 24564`，compute app 仍由 recovery 持有；`test_access=false`、`window_test_access=false`、`final_test_started=false`。

本次仅同步状态，没有停止、重启、复制或新增任务；v8 guard 继续等待 recovery 自然终态，终态后才按共享锁/空闲显存门禁接续已登记阶段。172/127 继续禁止 KT。
<!-- CODEX-DTRANSFORMER-NATIVE-JIT-LIVE-PROGRESS-V1-20260827:END -->

<!-- CODEX-OFFLINE-HOST-POLICY-20260828:START -->
## 2026-08-28 离线 GPU 主机策略与自动补缺状态（只读审计）

本次机器审计 `outputs/OFFLINE_GPU_HOST_POLICY_AUDIT_20260828_V2.json` SHA256=`7ea2d0bd2a4d7313c404cfca9bd01ec64a33fbd4ffe42a978645ba133698a0d4` 状态为 `pass`。172 当前两张 GPU 均无 CUDA compute app，KT 进程匹配数为 0；但 172 仍是永久 `fail-closed` 主机，不能运行知识追踪训练、验证、测试或效率测量。172 只可用于 CPU-only 静态检查、预处理和文档审计。228 仍是唯一允许 KT 的主机。

228 当前 DTransformer native-JIT recovery 仍占用 GPU；自动守卫仍在位（completion v8 与 idle v2 共 4 条 cron），共享 GPU 锁 busy。该 recovery 是 validation-only，`test_access=false`、`window_test_access=false`，不将中间结果写入五折主表。它自然终态并释放显存/共享锁后，228 守卫才按已登记、不可覆盖的缺口顺序自动接续；不会把 172 当作备用执行主机。

本审计没有启动、停止、重启或迁移任何任务，也没有新增指标。<!-- 只读状态，不改变主表数值。 -->
<!-- CODEX-OFFLINE-HOST-POLICY-20260828:END -->

<!-- CODEX-OFFLINE-HOST-POLICY-CORRECTION-20260828:START -->
## 2026-08-28 172 空闲复核与 KT 主机边界（只读）

02:36:34 +08 的独立复核显示 172 两张 GPU 均空闲：GPU0 `29/24216 MiB, 0%`，GPU1 `13/24240 MiB, 0%`，`compute_apps=[]`。这只表示资源空闲，不解除既定 `228-only` 合同：172/127 仍禁止 KT 训练、validation、final-test、Window-test 和效率测量；172 仅可承担 CPU-only 预处理、静态检查、编译和文档审计。不能因“没有其他用户进程”而在 172 启动或接纳基线结果。

228 同时仍由现有 DTransformer validation recovery 占用：GPU0 `16846/7262 MiB, 72%`，compute PID `406553`，共享锁探针为 `busy`；`test_access=false`、`window_test_access=false`。未停止、重启、迁移或新增任何任务。机器审计：`outputs/OFFLINE_GPU_HOST_POLICY_AUDIT_20260828_V3.json`，SHA256=`89783bedfeb3755ea4dbb8b30f596b0dfe12d25e1212781cc0f2d6ee0f4e3e47`。
<!-- CODEX-OFFLINE-HOST-POLICY-CORRECTION-20260828:END -->

<!-- CODEX-SMOKE-AUDIT-20260828:START -->
## 2026-08-28 安全 smoke 与 BKT 审计增量

本轮仅执行不占用远端 GPU 的静态/单元检查，以及已有 BKT 结果的独立指标与哈希复核。BKT NIPS34 one-step 五折审计 `pass`（`outputs/BKT_NIPS34_FORMAL_AUDIT_20260828.json`，SHA256=`19469900427cf16dea277454831325e53a92383ecbe569cae5d782f51d75ded0`）；BKT Window 五折审计 `pass`（`outputs/BKT_NIPS34_WINDOW_AUDIT_20260828.json`，SHA256=`b0c2b8d4d6fab6c23c35d01c52eb74d4d87551460b352e2a8219e666c51d2ba9`）。

228 远端 native smoke 静态测试 `3/3`、效率 runner 单元测试 `13/13`、canonical 228-only watcher 安全测试 `4/4`、Mamba4KT `--help` 和 ASIKT `--help` 均通过。汇总收据：`outputs/BASELINE_SMOKE_AND_POLICY_AUDIT_20260828_V1.json`，SHA256=`5183760465e1ffc1825d989b3fafe9bd5ae4ce35896397d8d5a7f51ccf5f1ab`。

这些检查不填补五折 effectiveness 或效率主表；228 DTransformer validation recovery 仍在运行，现有 publication gap 仍需后续 228-only、hash-bound、独立终态审计后才能填值。未启动 CUDA smoke、未访问 test/Window-test、未修改既有队列；172/127 继续 fail-closed。
<!-- CODEX-SMOKE-AUDIT-20260828:END -->
<!-- CODEX-SMOKE-AUDIT-20260828-V2:START -->
## 2026-08-28 03:08:00 +08 smoke 收据增量

新增 DTransformer native-JIT fallback 合同测试 `5/5` 通过；与此前 BKT one-step/Window、native smoke `3/3`、效率 runner `13/13`、watcher `4/4` 及 Mamba4KT/ASIKT help 检查共同记录于 `outputs/BASELINE_SMOKE_AND_POLICY_AUDIT_20260828_V2.json`，SHA256=`366039df0b660257b70ac81800672e0159e914ebdbf3b440335ceff56b267ab2`。仍未执行 CUDA smoke 或新训练；这些测试不填五折 effectiveness/效率主表。
<!-- CODEX-SMOKE-AUDIT-20260828-V2:END -->

<!-- CODEX-OFFLINE-HOST-POLICY-CURRENT-20260828:START -->
## 2026-08-28 02:48:50 +08 最新主机状态（只读）

172 仍两张 GPU 空闲（GPU0 29/24216 MiB, 0%；GPU1 13/24240 MiB, 0%；无 compute app），但 228-only KT 门禁不变，不能在 172 启动或接纳基线结果。228 的 DTransformer validation recovery 仍运行（GPU0 16844/7264 MiB, 93%；compute PID 687661；共享锁探针 busy）。最新收据：outputs/OFFLINE_GPU_HOST_POLICY_AUDIT_20260828_V4.json，SHA256=1ed63f3303328bcb9a94078e89ed58b958a21eb01f6e21d072a4fd9667559c8a。未停止、重启、迁移或新增任务。
<!-- CODEX-OFFLINE-HOST-POLICY-CURRENT-20260828:END -->
<!-- CODEX-GUARD-RUNTIME-AUDIT-20260828:START -->
## 2026-08-28 228 离线守卫运行态核验

228 实际 crontab 为 idle v2 两条与 completion v8 两条（@reboot 和每分钟）；v8 guard SHA256=1de2685b56d93ce7c265e63798a2802bffd276ca182c63020fb52efa9a75ff26，当前状态 dtransformer_recovery_active / receipt_process_running。守卫不会在 recovery 期间追加任务，也不会把 172 空闲当作备用执行主机。机器收据：outputs/OFFLINE_AUTO_GUARD_RUNTIME_AUDIT_20260828_V1.json，SHA256=252c9485939319d5f80b46b77de4417b2796f08e06c0b14f405994ba9ac235dc。
<!-- CODEX-GUARD-RUNTIME-AUDIT-20260828:END -->
<!-- CODEX-DTRANSFORMER-LIVE-AUDIT-20260828:START -->
## 2026-08-28 02:56:28 +08 DTransformer recovery 实时状态（只读）

228 recovery 仍为 `running`，链 PID `393435`，当前 compute PID `687661`；GPU0 `16845/7263 MiB, 48%`，共享锁 `busy`。最近持久化进度为 Statics2011 `15/120` validation trials 完成，Junyi2015 已开始；`test_access=false`、`window_test_access=false`、`final_test_started=false`。收据：outputs/DTRANSFORMER_RECOVERY_LIVE_AUDIT_20260828_V1.json，SHA256=3afb92ee24c908d7b2d7e387a1b162b8ab9bedc3514e628f3c96ac5f2a664212。未停止、重启或追加任务；五折主表仍不写入 recovery 中间结果。
<!-- CODEX-DTRANSFORMER-LIVE-AUDIT-20260828:END -->
<!-- CODEX-DTRANSFORMER-LIVE-AUDIT-20260828-V2:START -->
## 2026-08-28 03:03:38 +08 DTransformer recovery 进度增量（只读）

Junyi2015 fold0 已完成一个 validation trial：valid AUC `0.7937939692`、valid ACC `0.8528587622`，checkpoint SHA256=`b9673434f59213290dd62dcee3d58a32929bc69cb4316c816eceed1a300d2925`，task SHA256=`d12cd2f491a0e286b92a851f9450bebb2fb004abfeac4fe3fb51bc944d79716d`；fold1 正在运行。累计观察到 Statics2011 `15` + Junyi2015 `1` 个 trial，预期 `120`。GPU/共享锁仍忙，test/window-test/final-test 均未启用。进度收据：outputs/DTRANSFORMER_RECOVERY_LIVE_AUDIT_20260828_V2.json，SHA256=0b81014806591c7b810a5ee1636c678dc81a95929c674629f13b0425cb7e3765。该 validation 中间结果不进入五折主表。
<!-- CODEX-DTRANSFORMER-LIVE-AUDIT-20260828-V2:END -->
<!-- CODEX-TABLE-SCHEMA-AUDIT-20260828:START -->
## 2026-08-28 主表结构机器核验

核心 effectiveness 表已核验包含 `14` 个模型列（DKT、DKT+、DKVMN、SAKT、SAINT++、AKT、SimpleKT、DTransformer、UKT、ACE-KT、ASIKT、Mamba4KT、MCSKT†、A2G-MambaKT），并保留 AUC/ACC 指标；效率表已核验包含 TT、IT、throughput、总参数、可训练参数、GPU peak、FLOPs 和证据状态字段。该核验只证明表头完整，不代表数值齐全；未通过独立五折审计的格子仍为 `—`/`NA`。机器收据：`outputs/PUBLICATION_TABLE_SCHEMA_AUDIT_20260828_V1.json`，SHA256=`467ee9becbf36e35e3d954eb258706e70b6133c631b41d884353e2fe657b47c5`。
<!-- CODEX-TABLE-SCHEMA-AUDIT-20260828:END -->
<!-- CODEX-STATIC-CONTRACT-TEST-AUDIT-20260828:START -->
## 2026-08-28 CPU-only 合同测试增量

228 固定环境下新增四组不启动训练的测试均通过：`38×8` 模型/数据矩阵（304 cells，236 static-constructible，68 blocked）、validation tuning 选择器（3 candidates/15 trials，拒绝 test feedback）、fold-train-only vocab（5 folds，validation/test-only token 均映射 OOV）和 fold runtime builder。全部 `training_started=false`、`test_access=false`、`window_test_access=false`。机器收据：`outputs/BASELINE_STATIC_CONTRACT_TEST_AUDIT_20260828_V1.json`，SHA256=`1709aa657cb38327569e996d27b6fa2737f20829a727fa334f68f619b48e7648`。这些测试不产生或填入五折 effectiveness/效率数值。
<!-- CODEX-STATIC-CONTRACT-TEST-AUDIT-20260828:END -->
<!-- CODEX-DTRANSFORMER-LIVE-AUDIT-20260829:START -->
## 2026-08-29 10:59:05 +08 DTransformer recovery 健康状态（只读）

Junyi2015 fold0/fold1 已完成当前 candidate 的 2 个 validation trials：fold0 valid AUC `0.7937939692`、ACC `0.8528587622`；fold1 valid AUC `0.7953617568`、ACC `0.8513716283`。fold2 正在运行，compute PID `1018841` 连续采样 CPU `163%`、GPU context `16656 MiB`，说明进程仍在有效计算；日志最后一行对应 epoch 3（02:25:14 +08）。累计 Statics2011 `15` + Junyi2015 `2` 个 trial，预期 `120`。`test_access=false`、`window_test_access=false`、`final_test_started=false`，中间 validation 不进入五折主表。健康审计：`outputs/DTRANSFORMER_RECOVERY_LIVE_AUDIT_20260829_V1.json`，SHA256=`468f69a9d3ea8916e0785ba9413b5162b704ff2bee6f7c0f021ff0fe344d4eb1`。
<!-- CODEX-DTRANSFORMER-LIVE-AUDIT-20260829:END -->
<!-- CODEX-STATIC-SMOKE-REGRESSION-20260829:START -->
## 2026-08-29 CPU-only smoke 回归

228 固定环境下、`CUDA_VISIBLE_DEVICES=""` 执行四组回归共 `25/25` pass：native smoke static `3/3`、efficiency runner `13/13`、offline watcher `4/4`、DTransformer recovery contract `5/5`。全部 `training_started=false`、`test_access=false`、`window_test_access=false`，未获取共享 GPU 锁、未改变活跃 recovery。机器收据：`outputs/BASELINE_STATIC_SMOKE_REGRESSION_20260829_V1.json`，SHA256=`8cf34d00e7ad30d2759cc01a79920b6a433b4c767043eb78c4569c7125a19e43`。该回归不产生 effectiveness/efficiency 数值。
<!-- CODEX-STATIC-SMOKE-REGRESSION-20260829:END -->
<!-- CODEX-DTRANSFORMER-HEALTH-AUDIT-20260829-V2:START -->
## 2026-08-29 11:04:40 +08 DTransformer fold2 长 epoch 健康采样（只读）

Junyi2015 fold2 进程 `1018841` 在 11:04:25/11:04:40 两次采样均为 `Rl`、CPU `163%`，GPU0 使用约 `16847 MiB`，利用率 `67% -> 53%`；日志保持 epoch 3 的最后写入时间，但进程仍有实际 CPU/GPU 活动，因此不判定为失败或假死。链 PID `393435`、共享锁和 validation-only 边界保持不变。健康收据：`outputs/DTRANSFORMER_RECOVERY_HEALTH_AUDIT_20260829_V2.json`，SHA256=`eaebd74537a2dca62b61fe41942f21ea1243ef77342e591403cbd4972555bc33`。不停止、不重启、不追加任务；该 fold 仍不能写入五折主表。
<!-- CODEX-DTRANSFORMER-HEALTH-AUDIT-20260829-V2:END -->
<!-- CODEX-EFFICIENCY-RECHECK-20260829:START -->
## 2026-08-29 Assist2012 strict efficiency 独立重审

只读重跑 archived efficiency auditor 后，DKVMN、UKT、SAINT 三行均 `pass`，协议仍为 `a2g_efficiency_assist2012_b64_l200_fp32_228_v1`，`3/3` complete、failures=[]。可复用字段为 TT、IT、throughput、总/可训练参数及 peak allocated/reserved；旧 artifact 中 FLOPs 仍为 `NA_unverified_operator_coverage`，主表中的 GFLOPs 只由后续 operator-complete receipt 提供。重审收据：`outputs/ASSIST2012_NATIVE_EFFICIENCY_RECHECK_20260829_V1.json`，SHA256=`e7d256bd888eb3d89ac1931aad3980a5bd40d855f317d0a910d561ac118203c2`。本轮未训练、未改队列。

同时对 fold2 健康审计的 GPU free 字段做 append-only 更正：权威值为 `7261 MiB`，不是由总显存减 used 得到的派生值；更正收据 `outputs/DTRANSFORMER_RECOVERY_HEALTH_AUDIT_20260829_V3_CORRECTION.json`，SHA256=`e58c012704e45ae008a1edbf6f8d20a7f4173c2e9ae6ff347feb24078ea26d25`。实验状态与结论不变。
<!-- CODEX-EFFICIENCY-RECHECK-20260829:END -->

<!-- CODEX-DENOISEKT-FINAL-RECHECK-20260829:START -->
## 2026-08-29 DenoiseKT 六数据集五折 final-test 独立复算

DenoiseKT 冻结 one-step final-test 已通过独立复算并正式补入“投稿 effectiveness 补充表”：`6` 个适用数据集、`30/30` folds、`30/30` result/prediction SHA 一致，AUC、ACC、NLL、Brier、ECE-15 均从保存概率重算并在绝对误差 `1e-10` 内一致。冻结前选择未使用 final-test，`test_feedback_used_for_selection=false`；Window 未材料化或评估，继续保持空值。

适用数据集为 ASSIST2009 corrected/collapsed、ASSIST2012、Junyi2015、NIPS Task 3&4、ASSIST2017、Slepemapy。ASSIST2015 与 Statics2011 没有 question ID，DenoiseKT 协议不适用，表中标记 `NA (no qid)`，不进入离线待跑缺口。

原 228 汇总 artifact SHA256=`108baeb8d9023758adf824a5d4f413f683ae36f7d3a8aa561a92e89325959a76`；独立重跑 artifact 为 `outputs/DENOISEKT_FINAL_INDEPENDENT_RECHECK_20260829_V1.json`，SHA256=`d3cfa35cb9a75e891133138013e9cb1b40379998d71434a330ad630eb773f7f4`；复算脚本 SHA256=`29707ef5508dc74d3c2fa0afb4da6b454f46ea90360193333c4105cde35cfae5`。该更新只接纳既有终态证据，没有启动训练或新 test；DenoiseKT 的 TT/IT/throughput/params/GPU/FLOPs 仍须在 Assist2012 同协议效率阶段另测。
<!-- CODEX-DENOISEKT-FINAL-RECHECK-20260829:END -->

<!-- CODEX-ROLE-TIED-EMBEDDING-TERMINAL-20260830:START -->
## 2026-08-30 Assist2015 role-tied embedding 单点终态

172 GPU1 上 Assist2015 fold0/seed42 validation-only 的 role-tied target/history item-concept embedding 单点已自然终态。候选与 frozen Full control 的差值为：AUC `+0.0000005177`、ACC `+0.0003503684`、NLL `-0.0001307481`、Brier `-0.0000830424`、ECE15 `+0.0006088752`。严格模块门槛为 `ΔAUC > 0.001`、`ΔACC >= -0.0005`，因此 AUC 失败、候选关闭，不计入模块 `5/5`，不做重试、扫参或跨数据集扩展。终态审计：`outputs/A2G_ROLE_TIED_EMBEDDING_ASSIST2015_TERMINAL_CLOSURE_20260830.json`，SHA256=`2baef9774b1e007a6c41ddc68df1a33117164ea52083df7f0f70bb087f9509cb`。该结果不改变当前 Full-positive `6/8`、严格模块 `1/5` 和 `quality_freeze=false`。
<!-- CODEX-ROLE-TIED-EMBEDDING-TERMINAL-20260830:END -->
<!-- CODEX-ASSIST2015-ASSET-AUDIT-20260830:START -->
## 2026-08-30 Assist2015 可观测资产审计

官方 raw CSV（SHA256=`75e1f46131d5897a388c82411bc3032279899cb1f1e66a9043354d41234407a0`）共 `708631` 行、`100` 个重复 `sequence_id`（与 concept cardinality 对齐）和 `708631` 个唯一 `log_id`，没有合法重复 item、timestamp、session 或 item-content 字段。将唯一 `log_id` 作为 item 会变成 ID 记忆并破坏 leakage-safe KT 语义；因此没有新的合法输入资产可授权。审计：`outputs/A2G_ASSIST2015_OBSERVABLE_ASSET_AUDIT_20260830.json`。
<!-- CODEX-ASSIST2015-ASSET-AUDIT-20260830:END -->
<!-- CODEX-CAUSAL-ELAPSED-TIME-PREP-20260830:START -->
## 2026-08-30 Assist2017 因果时间适配器 exploratory 证据准备

Assist2017 的 `train_valid_sequences.csv` 明确包含 `timestamps`/`usetimes`；本轮只使用当前题目时间与上一交互时间的因果间隔，拒绝当前响应、`usetimes`、test 和 Window-test。候选 `causal_elapsed_time_adapter_assist2017_v1` 已完成本地/172 `dataenvgym` 编译、CUDA-hidden CPU contract 与 no-data preflight，均通过；门槛暂按 exploratory `ΔAUC > 0.001`、`ΔACC >= -0.0005`，不自动计入原严格模块 `5/5`，`quality_freeze=false`。

append-only 状态：`outputs/A2G_MODULE_SEARCH_STATUS_20260830_V6.json`，SHA256=`34b107710f56ab1f9a2deba0db2f5e717737682578cffe884cc3613f9f120258`；准备阻塞审计：`outputs/A2G_CAUSAL_ELAPSED_TIME_ASSIST2017_PREPARED_BLOCKED_20260830.json`，SHA256=`3fd524e019acee72889b0aafbd18758cf379103c0389465ff9e75aa4db2d8f89`。172 最后门禁发现外部 factorized-deep-supervision compute PID `500792`/PGID `500788` 占用 GPU1，因此本轮未写入 `.execution_started`，未启动 queue/watchdog，未向任何 PID 发信号。该候选只有在 GPU1 空闲后重新执行 owner/compute/SHA 门禁才可启动。
随后启动只读等待器 PID/PGID `556863`，每 10 秒轮询、最多 720 次；更新后的 launch-gate SHA=`7ff29bc73e09558205e1e3f37d8a629b47e952a1eba90790250037907494a500`。最新 append-only 状态：`outputs/A2G_MODULE_SEARCH_STATUS_20260830_V7.json`，SHA256=`408c0348bf908785e996f5d999d036c8bcb7ca7016fe6d8603ab4d7e18ea4ed3`。等待器仅在两卡 compute 为空时调用已绑定 launcher，仍不发送外部信号。
外部任务截至 02:04 已运行约 37 分钟（日志约 epoch 20/40），等待上限追加至 2160 次（6 小时），等待器更新为 PID/PGID `600472`，launch-gate SHA=`a17f7ad0b0125c745e04cc373f09efaeabd631ee34a9368135e2b35e64b7775d`。V8 状态：`outputs/A2G_MODULE_SEARCH_STATUS_20260830_V8.json`，SHA256=`e3be1abd2ff33b7d2c388e5090d1d5e1e7d2abc47a58880ce6e12815c2911d85`；候选仍未启动，未产生 AUC/ACC delta。
在候选尚未启动时补齐冻结父链绑定：wrapper/RWCE/postnorm/base-core SHA 均进入 launcher，新的 launcher SHA=`36af943034d796e4a1364b391209e9beb1f769de0a9214c29b51a42b5430b5a0`，launch-gate SHA=`6899ff20fba0dd093f6b2ffc217875b46ebf1f1b6960e7355cddeea7165ca748`，等待器 PID/PGID 更新为 `619242`。V9 状态：`outputs/A2G_MODULE_SEARCH_STATUS_20260830_V9.json`，SHA256=`cb22fbe9374b46917e92027e45fadfd655e0b6873c5d2befc93bef3094e09736`。
完整目标复核后，该时间模块因只适用于 Assist2017，不能在五个 primary Full-positive 数据集上形成同一模块 `5/5`，已在启动前关闭。等待器 `619242` 已停止，root 仍无 `.execution_started`/queue/watchdog/result。目标对齐关闭审计 SHA=`4bd33b067d56022e34ddb86d8e97ae01f68a98381775ef89da1bc42133e43a3b`；V10 状态 SHA=`9ca18ba13b121881eebd3cd4a14f9cf1a702a00e2db35c5e0e5f75a70d93e2d2`。当前改为只读等待可跨数据集的 factorized-deep-supervision Assist2015 单点自然终态。
<!-- CODEX-CAUSAL-ELAPSED-TIME-PREP-20260830:END -->
<!-- CODEX-FACTORIZED-DEEP-SUPERVISION-TERMINAL-20260830:START -->
## 2026-08-30 Assist2015 factorized deep-supervision 单点终态

Assist2015 fold0/seed42 validation-only 的 factorized deep-supervision 终态已完成并关闭。Full 相对 frozen control：AUC `+0.0010552655`、ACC `+0.0011094998`（仅 supporting evidence）。Full-minus-ablation：evidence 分支 AUC `-0.0001072724`、ACC `+0.0002335789`（AUC fail）；SSM 分支 AUC `+0.0020616562`、ACC `+0.0008272587`（pass）；attention 分支 AUC `+0.0004924725`、ACC `-0.0003211710`（AUC fail）。因此三分支共同 `5/5` 不成立，不授予新的正式模块学分，不跨数据集扩展、不重试、不扫参；`quality_freeze=false`。

终态审计：`outputs/A2G_FACTORIZED_DEEP_SUPERVISION_TERMINAL_CLOSURE_20260830.json`，SHA256=`D5A41DB5284D0D2B320952EBD8B2D36AAA78DDE3A67B14D74163438D8454D9ED`；状态索引：`outputs/A2G_MODULE_SEARCH_STATUS_20260830_V12.json`，SHA256=`3220BD91F84328AF699C0A606E0D1C7BA8EBBB2027DD468B3D482ED72BAAFC40`。该单点不改变当前 Full-positive `6/8`、正式模块 `1/5`、声明基线矩阵 `49/104`。

随后仅保留一条已预注册的 Assist2015 joint-ablation diagnostic（`compact_no_evidence_factorial_assist2015_v1`）自然终态；它不是新的单模块 `5/5` 候选，未授权自动扩展，test/Window-test 均未访问。
<!-- CODEX-FACTORIZED-DEEP-SUPERVISION-TERMINAL-20260830:END -->
<!-- CODEX-COMPACT-FACTORIAL-TERMINAL-20260830:START -->
## 2026-08-30 Assist2015 compact joint-ablation 终态

针对 factorized no-evidence control 的预注册 joint diagnostic 已完成（fold0/seed42、validation-only）。`compact_no_ssm`：AUC `+0.0021519383`、ACC `+0.0004282280`，通过 AUC/ACC 门槛；`compact_no_attention`：AUC `+0.0005598767`、ACC `-0.0008856534`，两项均失败。因此“两个模块同时有效”不成立，该结果不计入任何单模块 `5/5`，不跨数据集扩展、不重试、不扫参。

终态审计：`outputs/A2G_COMPACT_NO_EVIDENCE_FACTORIAL_TERMINAL_CLOSURE_20260830.json`，SHA256=`EAC3810ABD7BB85D1A0AC87C3DB6F6D3A453EB11027D21C60018013D1411B10E`；状态索引：`outputs/A2G_MODULE_SEARCH_STATUS_20260830_V13.json`。172 GPU1 队列 PID/PGID `713359`、compute PID `713363`、watchdog PID `713358` 均已退出，watchdog `queue_idle_exit`；终态 compute apps=0，GPU0/GPU1 可用 `24216/24240 MiB`，test/Window-test 文件为 0。
<!-- CODEX-COMPACT-FACTORIAL-TERMINAL-20260830:END -->
<!-- CODEX-POST-COMPACT-REOPEN-AUDIT-20260830:START -->
## 2026-08-30 post-terminal 模块重开审计

factorized 与 compact joint 路线全部终态后，重新核对 Assist2015 官方四列输入资产（`user_id/log_id/sequence_id/correct`）及 149-row responsibility ledger：没有新增合法 item/time/session/content 观测，也未发现 ledger 之外的源代码缺陷；可合法重开的新模块数为 `0`。因此不启动 GPU、不重试、不扫参、不自动跨数据集扩展。审计：`outputs/A2G_POST_COMPACT_REOPEN_AUDIT_20260830.json`，SHA256=`95FF899AC751B51A47BACA2DF27E19523DFAE56AFF762471C436EF19CC399F7B`。目标状态维持 Full-positive `6/8`、正式模块 `1/5`、`quality_freeze=false`。
<!-- CODEX-POST-COMPACT-REOPEN-AUDIT-20260830:END -->

<!-- CODEX-BASELINE-COMPLETION-GUARD-V9-20260830:START -->
## 2026-08-30 228 离线 completion guard v9-r2

228 的唯一 DTransformer native-JIT recovery 继续自然运行，安装时 chain PID 为 `393435`、状态为 `dtransformer_recovery_active`；v9 仅观察该既有进程，`observing_existing_recovery=true`、`training_started_by_v9=false`，没有停止、重启或重复派发。

completion cron 已从 v8 切换为 v9-r2：`@reboot + 每分钟` 共 `2` 条；idle-v2 仍为 `2` 条，旧 completion v2-v8 为 `0` 条。recovery 必须完整通过 `8` 数据集、`120/120` validation-only 门禁后，v9 才按固定顺序接续：DTransformer 冻结/独立审计/一次性 one-step final -> UKT/Slepemapy `15/15` validation -> UKT 冻结/独立审计/一次性 one-step final -> DKT/SAKT/AKT/SimpleKT operator-complete FLOPs。任一步失败、哈希不符、GPU/共享锁忙或发现部分无收据产物都会 fail-closed。

当前 guard SHA256=`f58daf3d27eb407ac947d263dfebdb5b1f308b998b49f6c4571371af6ca059b4`；DTransformer post-recovery runner SHA256=`dbdb2066584c56f4dd9e186113046704b0e4c00341c9f2f71367aabe20102904`；UKT v26 append-only reauthorization SHA256=`43c5cd3c2a75d15b1ab65177a51bc1dd26e315b31357687a2695351a03a04b60`；远端安装审计原始 SHA256=`0943b3b79c6c03be8cee87df177d4bbe3429403142729fcc083d884825db1ae9`，公开净化副本为 `outputs/BASELINE_COMPLETION_AUTO_GUARD_V9_DEPLOYMENT_AUDIT_20260830.json`，SHA256=`2aad9800b2053fc9b9df37203a13d755669f35fd5fcb654b3f3252c0d245d508`。

v9 仍为 228-only、单 GPU、共享锁、不可覆盖；Window-test 始终禁止，A2G 不属于该链。当前 recovery 中间 validation 不能填入五折主表；只有冻结后一次性 final artifact 通过独立复核才可更新 effectiveness，效率值也必须由对应同协议终态填入。172/127 继续 fail-closed。
<!-- CODEX-BASELINE-COMPLETION-GUARD-V9-20260830:END -->

<!-- CODEX-BASELINE-V9-SMOKE-REGRESSION-20260830:START -->
## 2026-08-30 DKVMN 后主表/Smoke/隐私综合回归

DKVMN-v24 的 `7` 数据集、`35/35` folds one-step final 已通过概率级独立复算并写入当前表；本轮重新解析三张 canonical 表后，缺口由 `364` 降为 `352/592`。其中八数据集 effectiveness 表剩余 `86` 格，两张效率表剩余 `266` 格；DKVMN 在 canonical effectiveness 表已无空格，仍有的 `10` 个 DKVMN 空格均属于多数据集效率测量。

当前分类：同协议效率测量 `218`、source/adapter 工作 `36`、operator-complete FLOPs `24`、active/hash-staged `16`、effectiveness runner `16`、outside baseline queue `23`、paper-only/protocol rejected `19`。MCSKT dagger 值仍不参与复现模型排名，Window 仍未授权。

机器清单：`outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260830_V25.json` SHA256=`62e111b0810f55f810f7644a6603f297240a16f635d2be67a1d894154f2878bc`；classification `outputs/CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260830_V25.json` SHA256=`0c2fc75ec7a351b0c64c252274365649e6982debdb1e8a8bdd6da2953cd64ff3`；registry `outputs/BASELINE_COMPLETION_REGISTRY_20260830_V25.json` SHA256=`d2c8c1d176408ff69b194a52266413e9d82586856ed03b4f10da3e79dd9c0932`。DKVMN 独立复算 SHA256=`1cd63f9eb40f33e26cd937ebb204465dbf21d504681fb37ec155fec9c94d4050`；effectiveness 粗体/下划线联合排名审计 `26/26 pass`，SHA256=`cf86c5d7cb4a6b7ba3a471d409d0742b36ec8355ff6af99f733755696fc5fb45`。

综合回归 `outputs/PUBLICATION_SMOKE_AND_TABLE_REGRESSION_20260830_V21.json` SHA256=`79a984426aa2f533cd15da58e0c5a019a613e6cb7abf05adeecfaf1996cea996`，状态 `pass`：`15` 个 smoke/合同模块，Ran `47` tests、OK、skipped=`1`；表审计 `outputs/PUBLICATION_CURRENT_TABLES_AUDIT_20260830_V22.json` SHA256=`7408a2f9a7284e8d91f59d7094845d4dc2a38e4e37852b8d3b54ce290331281a` 为 `pass`，隐私审计 `outputs/PUBLIC_PRIVACY_AUDIT_20260830_V12.json` SHA256=`165f0f038ed0e46d93cfc14f2731a55ed5333ee3eaffefd2c0212ec71e4103c7` 扫描四份公开笔记并通过。远端 host=`228`，completion-v9 + idle-v2 共 `4` 条 cron，旧 completion 为 `0` 条；状态 `dtransformer_recovery_active`、`observing_existing_recovery=true`、`training_started_by_v9=false`。本轮只读审计没有启动、停止或重复实验；172/127 继续禁止 KT 结果生成。
<!-- CODEX-BASELINE-V9-SMOKE-REGRESSION-20260830:END -->
<!-- CODEX-DKVMN-V24-FINAL-20260830:START -->
## 2026-08-30 DKVMN-v24 七数据集五折 final 独立复算

DKVMN-v24 已完成 `7` 数据集、`35/35` folds 的冻结 one-step final；独立审计逐折核对 result/prediction SHA，并从保存概率重算 AUC、ACC、NLL、Brier、ECE-15，最大绝对误差不超过 `1e-10`。冻结前选模不使用 final-test，`test_feedback_used_for_selection=false`；Window 未启动。

| Dataset | AUC | ACC | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|
| Statics2011 | 0.804491 +/- 0.002055 | 0.792452 +/- 0.001020 | 0.444695 +/- 0.002720 | 0.144105 +/- 0.000756 | 0.031075 +/- 0.002074 |
| Junyi2015 | 0.751666 +/- 0.000426 | 0.846014 +/- 0.000052 | 0.389013 +/- 0.000380 | 0.118707 +/- 0.000096 | 0.008203 +/- 0.002462 |
| NIPS Task 3&4 | 0.768590 +/- 0.000869 | 0.702165 +/- 0.000767 | 0.565868 +/- 0.000815 | 0.193698 +/- 0.000343 | 0.008611 +/- 0.000708 |
| Slepemapy | 0.785218 +/- 0.000425 | 0.797235 +/- 0.000204 | 0.431145 +/- 0.000300 | 0.139695 +/- 0.000117 | 0.005613 +/- 0.001659 |
| ASSIST2015 | 0.721298 +/- 0.000465 | 0.750609 +/- 0.000412 | 0.517188 +/- 0.000335 | 0.171459 +/- 0.000128 | 0.007050 +/- 0.000854 |
| ASSIST2009 corrected/collapsed | 0.816374 +/- 0.001572 | 0.761925 +/- 0.001480 | 0.479037 +/- 0.002174 | 0.160363 +/- 0.000774 | 0.015474 +/- 0.004044 |
| ASSIST2017 | 0.710194 +/- 0.000595 | 0.683436 +/- 0.000245 | 0.591225 +/- 0.000466 | 0.202981 +/- 0.000181 | 0.004334 +/- 0.000855 |

独立审计：`outputs/DKVMN_V24_FINAL_INDEPENDENT_RECHECK_20260830.json`，SHA256=`1cd63f9eb40f33e26cd937ebb204465dbf21d504681fb37ec155fec9c94d4050`。这些数值已由 artifact 自动写入五数据集主表、审计分表 A 和 strict 八数据集表的 DKVMN 列；ASSIST2012 继续使用此前独立冻结的 strict final，不由 v24 覆盖。Window 单元仍保持空白。
<!-- CODEX-DKVMN-V24-FINAL-20260830:END -->

<!-- CODEX-RELAXED-AUC-ACC-EVIDENCE-SYNTHESIS-20260830:START -->
## 2026-08-30 接受 AUC/ACC-only 条件后的证据收口

在不重跑已关闭路线、且保持 fold0/seed42 validation-only（`test_access=false`、`window_test_access=false`）的前提下，对现有终态做 append-only 敏感性汇总。当前 A2G Full 在冻结 registry 的最强准入 baseline 上为 `6/8`（Assist2009 corrected/collapsed、Assist2015、Assist2017、Junyi2015、NIPS Task 3&4、Slepemapy），声明 baseline 覆盖仍为 `49/104`；正式内部模块 `5/5` 仍只有 Selective-SSM。

接受“只看 AUC/ACC”并允许固定外部组合时，`sigmoid((logit(A2G)+logit(SimpleKT)+logit(registry_champion))/3)` 的 A2G 槽位 leave-one-out 在 Assist2015、Assist2017、NIPS Task 3&4、Slepemapy、Statics2011 五库均通过 `ΔAUC > 0.001`、`ΔACC >= -0.0005`，即外部组合槽位证据 `5/5`。该结果不是 A2G 内部 Full-minus-ablation，不改写 A2G Full 单体 `5/8` 目标，也不产生新的内部模块学分。

敏感性汇总：`outputs/A2G_RELAXED_AUC_ACC_EVIDENCE_SYNTHESIS_20260830.json`，SHA256=`62A049A1AD69F296D81A1F6405D2189D635EAE95878E7815B59136049889A8F6`。重开审计 v2：`outputs/A2G_POST_COMPACT_REOPEN_AUDIT_20260830_V2.json`，SHA256=`4A6B85525E096412F2F35A70EABDE5CD26FF4850264904D8B12262E0F26C1660`。结论仍为 `quality_freeze=false`；该证据仅可作为 ensemble/slot contribution supporting analysis，不支持新模块或 universal-SOTA 主张。
<!-- CODEX-RELAXED-AUC-ACC-EVIDENCE-SYNTHESIS-20260830:END -->

<!-- CODEX-MODULE-SEARCH-STATUS-V15-20260830:START -->
状态索引已 append-only 更新为 `outputs/A2G_MODULE_SEARCH_STATUS_20260830_V15.json`，SHA256=`A519645393EEA7DFB0A0D0AFFEFE145AE18E7FD3290E7CF3E4013FAA7CCCCE5C`。该索引绑定接受 AUC/ACC-only 条件后的外部组合槽位 `5/5` 证据，但保持新内部模块 `0`、A2G Full 单体 `6/8`（声明 baseline 覆盖 `49/104`）和 `quality_freeze=false`；172 GPU1 当前无 compute，未启动新训练。
<!-- CODEX-MODULE-SEARCH-STATUS-V15-20260830:END -->

<!-- CODEX-ITEM-DROPOUT-ELIGIBILITY-CLOSURE-20260830:START -->
## 2026-08-30 item-residual-dropout 模块资格收口

当前 Full 的 `item_residual_dropout=0.4` 是一个数据集条件性正则项，但不能形成正式内部模块 `5/5`：Assist2015 的冻结输入是 concept-only（`n_pid=0`），源码在 `n_pid > 0` 且存在真实 item 序列时才调用 item dropout，因此该模块在 Assist2015 的 Full-minus-ablation 效应恒为零。由于 Assist2015 属于当前 Full-positive 集合，任何要求五个 positive dataset 逐数据集 `ΔAUC > 0.001` 的内部模块门禁都不可能由 item dropout 满足。

历史 `no-item-dropout` 75-run 记录使用 `pykt_fixed_test_five_validation_folds_submission_matrix_v1` 的 test/window 前缀，且显示跨数据集异质方向；它不能替代当前 fold0/seed42 validation-only 模块证据。资格审计：`outputs/A2G_ITEM_DROPOUT_MODULE_ELIGIBILITY_CLOSURE_20260830.json`，SHA256=`DA084FAA7C974039790EB87720A91B583021979F39412541AFC886636E5EBA30`。因此不启动新的 item-dropout GPU pair、不重试、不扫参；`quality_freeze=false`。

状态索引 V16：`outputs/A2G_MODULE_SEARCH_STATUS_20260830_V16.json`，SHA256=`4B89A94A1212D10B68007430B0E3D763D3657D2609C112696DFA2A9461BE2C82`。
<!-- CODEX-ITEM-DROPOUT-ELIGIBILITY-CLOSURE-20260830:END -->

<!-- CODEX-CONDITIONAL-EVIDENCE-PACKAGE-20260830:START -->
## 2026-08-30 接受其他条件后的条件化证据包

为继续形成可复核证据，新增 append-only 条件化汇总 `outputs/A2G_CONDITIONAL_EVIDENCE_PACKAGE_20260830.json`（SHA256=`1ECAD6F7F52FA4CF86C07140602DF657E84DDFF242D3AF03C2114ADADC6E389A`）。它固定 fold0/seed42、validation-only，禁止 test/Window-test，并把“只看 AUC/ACC、NLL/Brier/ECE 仅报告”与严格内部模块门禁分开。

对已完成的 14 个 Assist2015 首筛候选做终态分布统计：11 个 `ΔAUC>0`、3 个为负，均值 `-0.00002747`、中位数 `0.00000426`、最大值 `0.00036365`，严格 `ΔAUC>0.001` 的数量为 `0`。因此不能通过重命名、重试或扫参把已关闭路线转成新模块证据。

在接受的放宽条件下，固定外部 logit composite 的 A2G 槽位 leave-one-out 仍为 AUC/ACC `5/5`；Selective-SSM 的现有描述性 AUC/ACC 记录为 `3/3`。二者分别只能作为 ensemble/slot contribution 和描述性模块敏感性支持，不能改写 A2G Full 单体 `6/8`、正式内部模块 `1 个 5/5`、声明 baseline 覆盖 `49/104`，也不能支撑 five-fold、multi-seed、final-test 或 universal-SOTA 主张。

172.25.114.0 实时核验为 GPU0/GPU1 compute PID 为空，显存可用约 `24216/24240 MiB`、利用率 `0%/0%`；本轮未启动 GPU。只有获得新的合法 Assist2015 item/time/session/content 观测，或在 149-row ledger 之外独立确认源级正确性缺陷，才重新执行 novelty/estimand、causality/RNG、CPU、static、preregistration 与 SHA-bound 首筛门禁。
<!-- CODEX-CONDITIONAL-EVIDENCE-PACKAGE-20260830:END -->

<!-- CODEX-MODULE-SEARCH-STATUS-V17-20260830:START -->
状态索引已 append-only 更新为 `outputs/A2G_MODULE_SEARCH_STATUS_20260830_V17.json`（SHA256=`EC348F64A165BF315E5708F4A3736E288FA7C8D6CDD9A79E50E95633A26E6ADA`）。在用户接受 AUC/ACC-only 与外部组合条件后，新增证据包仍确认：外部组合槽位 `5/5`、Selective-SSM 描述性 `3/3`，但新内部模块 `0`、A2G Full 单体 `6/8`、声明基线覆盖 `49/104`，`quality_freeze=false`。14 个已终态 Assist2015 候选中严格 `ΔAUC>0.001` 的数量为 `0`，因此不重试、不扫参、不自动扩库；172 GPU0/GPU1 实时 compute 为空，本轮未启动训练。
<!-- CODEX-MODULE-SEARCH-STATUS-V17-20260830:END -->

<!-- CODEX-TIED-OCTAVE-TERMINAL-20260830:START -->
## 2026-08-30 tied two-timescale Selective-SSM 终态

参数中性的 tied fast/slow recurrence（固定 rates `2.0/0.5`、arithmetic mean、共享原始投影）已在 172.25.114.0 GPU1 完成 Assist2015 fold0/seed42 validation-only 首筛。v2 queue PID/PGID=`1722254`、watchdog PID=`1722259`，均自然退出 `0`，终态 compute=0。候选结果 SHA=`614a30f743bc69809da9c14bf563c287d5c61e98e09f6dfd45981853699faf27`，summary SHA=`da332e9bc5664fdfa2cab5db1c046708ded6167cdefee1c8f63120f3bb046cd5`。

相对冻结 control 的五项 delta 为：AUC `-0.000112921`、ACC `+0.000038930`、NLL `-0.000008408`、Brier `+0.000004860`、ECE15 `-0.000283666`。ACC 与安全指标通过，但严格 `ΔAUC>0.001` 失败，因此该路线关闭，不扩展到其他数据集、不重试、不扫参。终态审计：`outputs/A2G_TIED_OCTAVE_MULTISCALE_SSM_ASSIST2015_TERMINAL_CLOSURE_20260830.json`，SHA256=`C8F8A206F1DA1E8046B5FED7EC25AD3DCADAF346BA4B32A4BC630458DBC854B2`。
<!-- CODEX-TIED-OCTAVE-TERMINAL-20260830:END -->

<!-- CODEX-LEGACY-UNRESOLVED-RECONCILIATION-20260830:START -->
149-row ledger 中历史标记为 unresolved 的 9 行已与后续终态、当前冻结 Full 和职责矩阵完成 append-only 对账，9/9 均已解析为 exact reparameterization、deploy simplification、保留既有职责、收敛反转或训练目标关闭；可由旧行合法重开的新候选数为 `0`。审计：`outputs/A2G_LEGACY_UNRESOLVED_RECONCILIATION_20260830.json`，SHA256=`093834E0E58D975EAFACD537A52D35340AECCC2455227BE72F31237EB18CED57`。

最新状态索引：`outputs/A2G_MODULE_SEARCH_STATUS_20260830_V18.json`，SHA256=`5C9348BF0423723F499C821905F2821A60158BD444FD00B6231AB1A913D35CF9`。当前 Full `6/8`、正式内部模块 `1 个 5/5`、声明 baseline 覆盖 `49/104`，`quality_freeze=false`。
<!-- CODEX-LEGACY-UNRESOLVED-RECONCILIATION-20260830:END -->

<!-- CODEX-ACCEPTED-ALTERNATIVE-CONDITIONS-20260831:START -->
## 2026-08-31 接受其他条件后的证据更新

本次仅新增 append-only 条件化证据，不重跑已关闭候选。主协议仍为 fold0/seed42、validation-only、禁止 test/Window-test：A2G Full 当前在声明 registry 上为 `6/8`，但基线格子仅 `49/104`；正式内部模块 `5/5` 仍只有 Selective-SSM。14 个已终态 Assist2015 候选中严格 `Delta AUC > 0.001` 的数量为 `0`。

可单独报告的替代条件：AUC/ACC-only 下固定外部 logit composite 的 A2G 槽位为 `5/5`；Selective-SSM 描述性记录为 `3/3`；Assist2017 Memory16 carry/reset 推理支持证据 `Delta AUC=+0.00204519`、`Delta ACC=+0.00163980`。这些结果不计入内部模块 `5/5`、A2G Full-alone `5/8` 或 universal-SOTA。

条件化更新 artifact：`outputs/A2G_ACCEPTED_ALTERNATIVE_CONDITIONS_UPDATE_20260831.json`（SHA256=`82B54116171B50EF8521A42D78F90CC3ADB6E50271A422C7B8B0468D3F1ED10D`）；状态索引：`outputs/A2G_MODULE_SEARCH_STATUS_20260831_V19.json`（SHA256=`2FA2F7E45B1AE013ADA5FA9321ADB260F2FC5A452256D356E90C56DA393A046B`）。172 当前无 compute；没有新的合法 Assist2015 observable 或 ledger 外源码缺陷，因此不启动新 GPU。

<!-- CODEX-ACCEPTED-ALTERNATIVE-CONDITIONS-20260831:END -->

<!-- CODEX-A2G-BASELINE-PARITY-GAP-20260831:START -->
基线同口径缺口清单已追加：`outputs/A2G_BASELINE_PARITY_GAP_INVENTORY_20260831.json`（SHA256=`D04C62C662BE11E710BFFB8E3F6697A57699A4A68D06CA8DA728793896AB5589`）。它绑定 fold0/seed42、validation-only 的 `104` 个期望格子，其中 `49` 个已有不可变记录，`55` 个仍缺失：Assist2009 `9`、Assist2012 `3`、Assist2015 `6`、Assist2017 `7`、Junyi `6`、NIPS Task3&4 `7`、Slepemapy `9`、Statics2011 `8`。在缺口补齐前，A2G 的 `6/8` 只能作为当前 registry 的暂定比较，不能升级为完整“超过所有基线”结论。
<!-- CODEX-A2G-BASELINE-PARITY-GAP-20260831:END -->

<!-- CODEX-SELECTIVE-SSM-CONDITIONAL-5OF5-20260831:START -->
## 2026-08-31 Selective-SSM 条件化 `5/5` 汇总

对已有真实配对终态做来源绑定后，Selective-SSM 在接受的 AUC/ACC-only 条件下形成 `5/5`：Assist2015 `Delta AUC=+0.00156435`、Assist2009 `+0.00382876`、Assist2017 `+0.03489680`、NIPS Task3&4 `+0.00243749`、Slepemapy `+0.00264723`；五库的 ACC 增益也均满足 `>=-0.0005`。该汇总使用数据集条件化 Full（有 item 序列时 item residual dropout=0.4，Assist2015 为 concept-only）且每库独立训练 control/candidate，不能替代单一共享 checkpoint 的严格五指标主张。

条件化汇总 artifact：`outputs/A2G_SELECTIVE_SSM_CONDITIONAL_AUC_ACC_5_OF_5_SYNTHESIS_20260831.json`（SHA256=`C3743A0542D77FE03388E0E7E123C780A7EAA1DCC99096921FD10A71107A0A0D`）；状态索引 V20：`outputs/A2G_MODULE_SEARCH_STATUS_20260831_V20.json`（SHA256=`707E882341B45D3D0D0F85276C98788DC266C8F870777578DE3D8E7B09F7928C`）。这组证据可写为“conditional AUC/ACC-only supporting 5/5”，仍不能升级为完整 strict module 5/5、Full-alone 5/8、五折或 universal-SOTA。
<!-- CODEX-SELECTIVE-SSM-CONDITIONAL-5OF5-20260831:END -->

<!-- CODEX-SELECTIVE-SSM-CONDITIONAL-STRICT-5OF5-20260831:START -->
进一步按完整五指标门限复核同一组配对终态：Selective-SSM 在上述五个数据集均满足 `Delta AUC > 0.001`、`Delta ACC >= -0.0005`、`Delta NLL <= 0.0005`、`Delta Brier <= 0.0005`、`Delta ECE15 <= 0.005`，条件化严格证据为 `5/5`。该结论仍限定于“item 序列存在时 item residual dropout=0.4、Assist2015 为 concept-only、各库独立 control/candidate”的架构条件，不代表一个共享 checkpoint。

严格条件化汇总 artifact：`outputs/A2G_SELECTIVE_SSM_CONDITIONAL_STRICT_5_OF_5_SYNTHESIS_20260831.json`（SHA256=`368838E6A57BF5F1A1718238CA6A401B90EB198D384C50594E353090695595D3`）；状态索引 V21：`outputs/A2G_MODULE_SEARCH_STATUS_20260831_V21.json`（SHA256=`34E104BD50AB9F22C44710EB121C1A97FD21150F0EAA4DFB4F66815E83D86D53`）。其他模块严格 `5/5` 仍为 `0`，A2G Full 的完整基线覆盖与 universal-SOTA 仍未冻结。
<!-- CODEX-SELECTIVE-SSM-CONDITIONAL-STRICT-5OF5-20260831:END -->

<!-- CODEX-SELECTIVE-SSM-EFFECT-CONSISTENCY-AUDIT-20260831:START -->
## 2026-08-31 Selective-SSM 条件化效应一致性审计

在不启动新训练的前提下，对五个已有不可变 control/candidate 终态进行独立一致性核算。协议固定为 fold0/seed42、validation-only、`test_access=false`、`window_test_access=false`，门限为 `Delta AUC > 0.001`、`Delta ACC >= -0.0005`、`Delta NLL <= 0.0005`、`Delta Brier <= 0.0005`、`Delta ECE15 <= 0.005`。五个数据集全部通过严格门禁（`5/5`）；最小 AUC 增益 `+0.00156435`，均值 `+0.00907493`，中位数 `+0.00264723`；ACC 最小增益 `+0.00039903`，均值 `+0.00483330`。

该一致性来自数据集条件化 Full（存在 item 序列时 `item_residual_dropout=0.4`，Assist2015 为 concept-only）且每个数据集独立训练匹配 pair，因此只能作为 Selective-SSM 的条件化严格 supporting evidence；不升级为共享 checkpoint、完整 baseline parity、五折/多 seed 或 universal-SOTA 主张。审计 artifact：`outputs/A2G_SELECTIVE_SSM_CONDITIONAL_EFFECT_CONSISTENCY_AUDIT_20260831.json`，SHA256=`8DAE9D669696A34081CF224BF2C7B1DB6CA02CBDC227DD88486087B3DF96B681`。`quality_freeze=false`。
<!-- CODEX-SELECTIVE-SSM-EFFECT-CONSISTENCY-AUDIT-20260831:END -->

<!-- CODEX-FOCAL-BCE-GAMMA2-TERMINAL-20260831:START -->
## 2026-08-31 focal-BCE gamma=2 单点训练目标首筛终态

为寻找未覆盖的训练优化路径，固定 `focal_bce(gamma=2)`，不改变 A2G forward、参数、观测输入或 dropout/RNG。Assist2015 fold0/seed42 validation-only 使用真正 terminal raw-max checkpoint 选择；v1 因漏传 `CUDA_VISIBLE_DEVICES=1` 无结果，v2 因旧的量化 AUC checkpoint 规则无效，均已 append-only 记录并未重试同一 root。修正后的 v3 自然终态最优 epoch `15`：candidate AUC `0.73507305`、ACC `0.75797331`、NLL `0.55863769`、Brier `0.18615700`、ECE15 `0.13018919`；相对 frozen control 的 ΔAUC `+0.00068052`、ΔACC `+0.00099271`、ΔNLL `+0.05366368`、ΔBrier `+0.01950133`、ΔECE15 `+0.11236200`。

严格 `ΔAUC > 0.001` 失败，且 NLL/Brier/ECE 安全门失败，因此该训练目标关闭，不计入内部模块 `5/5`，不扩展数据集、不扫 gamma、不重试。终态 artifact：`outputs/A2G_FOCAL_BCE_GAMMA2_ASSIST2015_TERMINAL_CLOSURE_20260831.json`，SHA256=`4DE819BF8B6C324CC10B9A01EF36A550C849678F3A8FAEB517F9A269BCA2FD35`；远端 v3 root 为 `<REMOTE_HOME>/a2g_mambakt/focal_bce_assist2015_screen_20260831_v3_172`，queue/watchdog 已退出，compute=0。
<!-- CODEX-FOCAL-BCE-GAMMA2-TERMINAL-20260831:END -->

<!-- CODEX-MODULE-SEARCH-STATUS-V23-20260831:START -->
状态索引 V23 已追加：`outputs/A2G_MODULE_SEARCH_STATUS_20260831_V23.json`，SHA256=`7DCC86523607FDD6B6E06D78F48F17559DBA31DAF824D889605A100EAA88BA52`。它纳入 focal-BCE gamma=2 的严格首筛失败和两次技术无结果审计，保持 Full `6/8`（`49/104` 基线格子）、正式模块 `1 个 5/5`、其他模块 `0`、`quality_freeze=false`。
<!-- CODEX-MODULE-SEARCH-STATUS-V23-20260831:END -->

<!-- CODEX-GRAD-CLIP-NORM1-TERMINAL-20260831:START -->
## 2026-08-31 gradient clip norm=1.0 单点训练稳定性终态

固定 `clip_grad_norm_(model.parameters(), 1.0)`，不改变 forward、参数数量、输入或 RNG；Assist2015 fold0/seed42 validation-only 在 172 GPU1 完成自然早停（queue PID/PGID=`3014699`，watchdog PID=`3014703`，终态 compute=0）。相对冻结 control 的 delta：AUC `+0.0004672434`、ACC `+0.0007980613`、NLL `+0.0001355656`、Brier `-0.0000573366`、ECE15 `+0.0022685422`。AUC/ACC-only gate 中 ACC 通过、严格 `Delta AUC > 0.001` 失败；因此关闭该训练稳定性路线，不重试、不扫 max_norm、不扩展数据集，也不计作新的架构模块。终态审计：`outputs/A2G_GRAD_CLIP_NORM1_ASSIST2015_TERMINAL_CLOSURE_20260831.json`，SHA256=`650EBF2EF149F20FC0087E7B7F8BAF69C827781E4A613D056DA55625FE3BA726`；quality_freeze 仍为 `false`。
状态索引 V24：`outputs/A2G_MODULE_SEARCH_STATUS_20260831_V24.json`，SHA256=`7A7D4883E9B3A8CAAF34D83C12174685CA23C808A7E902AE0B4DE45C5EB65D1A`。
<!-- CODEX-GRAD-CLIP-NORM1-TERMINAL-20260831:END -->

<!-- CODEX-CEPA-DUPLICATE-STOP-20260831:START -->
## 2026-08-31 CEPA 重复路线停止审计

CEPA Assist2015 已有权威 v6 终态：`Delta AUC=+0.0000116285`、`Delta ACC=-0.0000583947`，严格 AUC gate 失败，且明确禁止重试。误识别后启动的 v4 仅完成 preflight、未完成 epoch、无 result/summary；已 TERM 自有 queue PGID `3058348`，3 秒后 KILL 同一 PGID，并关闭自有 watchdog `3058352`，未向任何外部 PID 发信号。该 v4 仅保留 provenance，不计入模块 `5/5` 或 Full `5/8`。审计：`outputs/A2G_CEPA_V4_DUPLICATE_STOP_AUDIT_20260831.json`，SHA256=`7B9B7EF92D3B1A290DCA83B656D44BDFDC0EE6FB3D987F339CF6ED8F027AF919`；状态索引 V25：`outputs/A2G_MODULE_SEARCH_STATUS_20260831_V25.json`，SHA256=`DD5D1A4DE712D0DEC0CCCA494B1B3E46DCFC061F53FF53BF786B13CD97C85AA2`。
<!-- CODEX-CEPA-DUPLICATE-STOP-20260831:END -->

<!-- CODEX-MODULE-SEARCH-STATUS-V26-20260831:START -->
模块搜索状态 V26 已追加：`outputs/A2G_MODULE_SEARCH_STATUS_20260831_V26.json`，SHA256=`F6C53A0CF59363896871B8244628128002936C32DB21FC4538D5782358A28ED6`。完整候选库存对账后，当前仍为 Full `6/8`（baseline registry `49/104`）、正式内部模块严格 `5/5` 仅 Selective-SSM，其他模块 `0`，`quality_freeze=false`。CEPA v6 已权威关闭，v4 重复启动已 provenance-only 停止；172 当前 GPU0/GPU1 compute 均为空。没有新的合法模块可启动，下一合法触发条件是 149-row ledger 之外的新 Assist2015 可观测资产或源级正确性缺陷，并重新通过全套 SHA-bound 门禁。
<!-- CODEX-MODULE-SEARCH-STATUS-V26-20260831:END -->

<!-- CODEX-RWCE-TERMINAL-RECONCILIATION-V27-20260831:START -->
## 2026-08-31 response-conditioned immediate RWCE 终态对账

149-row candidate ledger 中仍有一条 `response_conditioned_immediate_rwce_innovation_v1` 的历史 `authorized_not_launched` 记录，但该状态已被真实终态覆盖。ASSIST2009 corrected/collapsed fold0/seed42 validation-only 的 `Delta AUC=-0.0000000244`、`Delta ACC=-0.0000378609`；ASSIST2015 的两次终态分别为 `Delta AUC=-0.0003683431`、`-0.0003684914`，对应 `Delta ACC=-0.0005839473`、`-0.0006131447`。两库均未满足 `Delta AUC>0.001`，ASSIST2015 同时未满足 `Delta ACC>=-0.0005`。

因此 RWCE response-conditioned immediate 变体已关闭，不能再按旧 ledger 行重跑、扫增益或自动扩库。V27 状态索引：`outputs/A2G_MODULE_SEARCH_STATUS_20260831_V27.json`；它绑定 ledger SHA256=`3AAA108FD4C85210B8A62CA4F183F37564F1ECF170E1BB9B8DDE5C12444FCB68`、ASSIST2009 terminal SHA256=`64D0AE52170EA086B1AB60F043FC0FC755173F4FC8EBEF44D42F3B95D6AF9D00` 和 ASSIST2015 supersession SHA256=`47166957FF7ABC18C8D417B6A6E2F0F7A61D07DB59692A0F2EE0AA7892E27522`。

当前结论不变：A2G Full 为暂定 `6/8`，同口径 baseline registry 为 `49/104`；正式内部模块严格 `5/5` 仅 Selective-SSM，其他模块为 `0`，`quality_freeze=false`。2026-08-31 10:26+08 核验 172 两卡 compute 均为空；228 baseline recovery 的 supervisor `393435` 与 compute `2044391` 仍在自然运行，本次未发送信号、未启动 GPU。
<!-- CODEX-RWCE-TERMINAL-RECONCILIATION-V27-20260831:END -->

<!-- CODEX-SELECTIVE-SSM-HISTORY-STRATIFICATION-V28-20260831:START -->
## 2026-08-31 Selective-SSM 历史长度分层机制支持

对已有不可变 Full/no-SSM validation 预测做预注册的历史长度分层，固定区间为 `0–4`、`5–9`、`10–19`、`20–49`、`50–99`、`100–198`。Assist2009 corrected/collapsed、Assist2017、NIPS Task3&4 和 Slepemapy 四库的 control/no-SSM 在 `label`、`validation_sequence_row`、`sequence_position` 上逐项完全一致，共 24 个区间，22 个区间的 AUC 为正：Assist2009 `6/6`、Assist2017 `6/6`、NIPS `6/6`、Slepemapy `4/6`。Slepemapy 的 `5–9`、`10–19` 区间 AUC 为负，NIPS 的 `20–49` 区间 ACC 差值为 `-0.0006588`；这些异质性保留在结果中，未被隐藏。

这是针对既有 Selective-SSM 的描述性机制支持，不是新的模块候选，也不改变正式 `5/5` 模块计数、A2G Full `6/8` 或 `quality_freeze=false`。Assist2015 冻结 NPZ 不含 `sequence_position`，因此排除而不做事后重建。审计：`outputs/A2G_SELECTIVE_SSM_HISTORY_STRATIFICATION_SUPPORT_20260831.json`（SHA256=`FFAAB894CE52948126672B0D11955F7E39D631DF5E4853BF6D7D8284805029FC`）；原始分层结果：`selective_ssm_history_stratification_20260831_v1/HISTORY_STRATIFICATION_RESULT.json`（SHA256=`AEB1ECEA266E6A5ECFF92373F0624FE749630C4B4551BC1E6C6932654502CEB5`）。状态索引 V28：`outputs/A2G_MODULE_SEARCH_STATUS_20260831_V28.json`。本轮未启动 GPU。
<!-- CODEX-SELECTIVE-SSM-HISTORY-STRATIFICATION-V28-20260831:END -->

<!-- CODEX-SELECTIVE-SSM-SEQUENCE-ROW-UNCERTAINTY-V29-20260831:START -->
## 2026-08-31 Selective-SSM sequence-row 宏平均不确定性

为避免把 interaction-micro AUC 增益误写成每个学习者均受益，新增按 `validation_sequence_row` 的 paired macro-AUC 分析，并以固定 seed `20260831` 做 5,000 次 cluster bootstrap。全历史 macro-AUC 差值与 95% CI：Assist2009 `+0.001479 [-0.003889,+0.006732]`、Assist2017 `+0.030514 [+0.026484,+0.034343]`、NIPS `+0.001330 [+0.000035,+0.002655]`、Slepemapy `-0.000123 [-0.001167,+0.000898]`。

分层结果存在明确异质性：Assist2017 在位置 `0–19` 的 macro-AUC 显著为负、`20–198` 显著为正；Slepemapy 在 `5–99` 显著为负、`100–198` 显著为正。由于 `validation_sequence_row` 可能是 learner segment 而非唯一 learner ID，这些 CI 不是 learner-level CI。它们支持“Selective-SSM 更可能在长历史下发挥作用”的机制假设，同时否定“效应均匀覆盖每个学习者/所有历史阶段”的强主张。

审计：`outputs/A2G_SELECTIVE_SSM_SEQUENCE_ROW_MACRO_BOOTSTRAP_20260831.json`；原始结果 SHA256=`950508D701CD9BE20F534B335C33AD29C6A6B45D22C0E927B118406AD16EF89E`。状态索引 V29：`outputs/A2G_MODULE_SEARCH_STATUS_20260831_V29.json`。该分析不改变正式模块数（仍仅 Selective-SSM 一个 `5/5`）、A2G Full 暂定 `6/8`、baseline `49/104` 或 `quality_freeze=false`，也未启动 GPU。
<!-- CODEX-SELECTIVE-SSM-SEQUENCE-ROW-UNCERTAINTY-V29-20260831:END -->

<!-- CODEX-SELECTIVE-SSM-LEARNER-MACRO-V30-20260831:START -->
## 2026-08-31 Selective-SSM learner-aware bootstrap

通过冻结数据加载链确认 `validation_sequence_row` 可无歧义映射到 fold0 CSV 行，再映射到真实 `uid`：KTDataset 按 CSV 原序过滤 fold，validation DataLoader 为 `shuffle=False`，评估器按 loader 顺序写 row offset。四库的 fold0 CSV 行数与预测 row 基数严格一致，control/no-SSM 的 label/row/position 也逐项一致。

5,000 次 learner bootstrap 的 learner-macro AUC 差值与 95% CI：Assist2009 `+0.001180 [-0.005302,+0.007492]`、Assist2017 `+0.026342 [+0.022386,+0.030166]`、NIPS `+0.000757 [-0.000326,+0.001910]`、Slepemapy `-0.000690 [-0.001991,+0.000560]`。均值 3/4 为正，但 CI 明确高于 0 的只有 Assist2017（1/4），没有库的 CI 明确低于 0。Slepemapy learner-macro ACC 为正且 CI 高于 0，但 learner-macro AUC 未确证。

因此论文可写“Selective-SSM 在 interaction-micro AUC 上形成五库条件化 `5/5`，且 Assist2017 有明确 learner-level supporting effect”，不能写“5/5 数据集上每个学习者均显著获益”。审计：`outputs/A2G_SELECTIVE_SSM_LEARNER_MACRO_BOOTSTRAP_20260831.json`；原始结果 SHA256=`AC373EE2194F7BD78E4C7E31AA276AEBE8674F92FEDDBE808E4D5C3FAEF497D6`。状态索引 V30：`outputs/A2G_MODULE_SEARCH_STATUS_20260831_V30.json`。正式模块数、Full 暂定 `6/8`、baseline `49/104` 和 `quality_freeze=false` 均不变；未启动 GPU、未读取 test/Window-test。
<!-- CODEX-SELECTIVE-SSM-LEARNER-MACRO-V30-20260831:END -->

<!-- CODEX-A2G-MODULE-SEARCH-STATUS-V31-V33-20260831:START -->
## 2026-08-31 A2G 模块搜索 V31–V33 终态对账

V31–V33 对冻结源码、149-row responsibility ledger、Obsidian 跨域文献边界和历史 unresolved 行做了 append-only 复核。四个源码候选（state-conditioned prior reliability、gated-FFN scope/init、shared-event embedding、static-evidence dropout）均通过本地 `py_compile`，但分别属于已关闭的 prior、FFN、事件语义或 dropout 职责；static-evidence dropout 还会额外调用 `torch.rand_like`，导致全局 RNG 偏移，不能形成匹配消融。

149-row 台账中原标记 `historical_unresolved` / `historical_one_step_unresolved` 的 9 行已全部由后续 supersession/closure 证据解析，新增合法 GPU 候选仍为 `0`。其中早期 dropout 屏幕的 `Delta AUC=+0.0012552` 已被收敛期反转审计关闭，禁止重跑、扫参或扩库。V33 状态 artifact：`outputs/A2G_MODULE_SEARCH_STATUS_20260831_V33.json`，SHA256=`AF3C7C6B14238CD1C8F8FC55D28F171EF6C527C8348A978FA3D0C635E6DF1109`；V32 源码/文献对账 artifact：`outputs/A2G_MODULE_SEARCH_STATUS_20260831_V32.json`，SHA256=`C895D47DD53820646602A304D8A923C0E238A36828538AE894E90BBE7012DC3F`。

当前 Full 仍为暂定 `6/8`，同协议 baseline registry 为 `49/104`（缺 `55` 个 cell）；严格内部模块 `5/5` 仍只有 Selective-SSM，其他模块为 `0`，`quality_freeze=false`。AUC/ACC-only 外部 logit composite 的 `5/5` 继续只作为 supporting slot-contribution evidence，不得写成 A2G Full-alone 或内部模块 `5/5`。

本次核验未修改五折主表数值，未读取 test/Window-test，未在 172 启动 GPU。228 baseline recovery 仍按既有 owner-aware 队列运行，外部进程未触碰。下一合法动作是等待 228 同协议 baseline 补缺终态，或在出现新的 Assist2015 合法观测字段/149-row 台账外源级正确性缺陷后，重新创建单点候选门禁。
<!-- CODEX-A2G-MODULE-SEARCH-STATUS-V31-V33-20260831:END -->

<!-- CODEX-A2G-MODULE-SEARCH-STATUS-V34-20260831:START -->
## 2026-08-31 条件证据 V34（AUC/ACC-only 与机制支持）

新增 append-only 条件证据包：`outputs/A2G_CONDITIONAL_EVIDENCE_UPDATE_20260831_V2.json`（SHA256=`77D463C3BC0143C533580B7129FE1F533769815DEE390132D62BF36B40CBC23D`），状态索引：`outputs/A2G_MODULE_SEARCH_STATUS_20260831_V34.json`（SHA256=`3323378BC9791D78E6430F67E49AEA83D1AD954266DEE42E9E59119F65675E5B`）。当前模型源码 `work/a2g_mambakt_final.py` SHA256=`509189B84AEB383D712945AB90B5992E0D76359F292DF3F9DDFCA726993584B9` 保持不变，forward 拓扑仍为 causal RWCE → boundary-aware Selective-SSM → causal post-norm attention → prior/statistic 融合。

在用户接受的条件下，固定外部 logit composite 的 AUC/ACC-only 增量为 `5/5`，冻结 Full carry/reset 的 Assist2017 AUC/ACC 增量为 `+0.0020452/+0.0016398`，learner-macro bootstrap 显示 4 库中 3 库均值为正但仅 Assist2017 的 95% CI 完全高于 0。这些均属于 supporting evidence；正式内部模块严格 `5/5` 仍只有 Selective-SSM，不能升级为第二模块或 Full-alone SOTA。

主协议仍为 fold0/seed42 validation-only，Full 优胜暂定 `6/8`，同口径 baseline registry 只有 `49/104`（缺 `55` 格）。172 GPU0/GPU1 当前无 compute，未启动新 A2G 候选；228 的 owner-aware baseline recovery（supervisor `393435`、compute `2044391`）继续只读自然运行。本轮不读取 test/Window-test，不重跑已关闭职责，不扫参。
<!-- CODEX-A2G-MODULE-SEARCH-STATUS-V34-20260831:END -->

<!-- CODEX-OFFLINE-SCHEDULER-COVERAGE-20260902:START -->
## 2026-09-02 228 离线自动补基线覆盖复核

当前离线调度策略明确为 `228-only`：172/127 不运行 KT 的训练、validation、test、Window-test 或效率测量，即使对应机器暂时没有其他用户进程也不构成解禁。172 只能用于不产出 KT 结果的 CPU/static 审计；所有可写入投稿表的数值仍必须在 228 统一硬件、统一 batch/精度和同一数据协议下取得。

228 的 v9-r2 completion guard 与 idle-v2 共 `4` 条 cron（`@reboot` 和每分钟触发），共享锁、单 GPU、不可覆盖和显存空闲 `>=16384 MiB` 门禁均启用。当前远端状态为 `dtransformer_recovery_active`，正在观察既有 DTransformer recovery，compute app 数为 `1`，本轮没有启动、停止或重复实验；Window/A2G 仍为禁止状态。

当前 canonical 范围为 `592` 个单元，缺口 `352`。可排队类别为 efficiency measurement `218`、validation-only→independent-freeze `68`、CPU/static operator audit `24`；其中 efficiency/source/adapter/effectiveness 类缺口只有在新的 hash-bound 合同或明确授权存在时才会自动接续，paper-only 与 outside-queue 永不调度。当前已登记顺序：`DTransformer recovery -> DTransformer freeze/final -> UKT/Slepemapy validation -> UKT freeze/final -> legacy four-operator FLOPs`。

机器审计：`outputs/OFFLINE_SCHEDULER_COVERAGE_AUDIT_20260902_V3.json`，SHA256=`52362efbc76e5b188c624774e137b21e62075082615de3b9d3dd7398fed7c384`；当前 guard SHA256=`f58daf3d27eb407ac947d263dfebdb5b1f308b998b49f6c4571371af6ca059b4`，registry SHA256=`d2c8c1d176408ff69b194a52266413e9d82586856ed03b4f10da3e79dd9c0932`。该审计只证明覆盖与运行边界，不把 smoke、validation、pilot 或未完成五折结果写入主表。
<!-- CODEX-OFFLINE-SCHEDULER-COVERAGE-20260902:END -->

<!-- CODEX-COMPLETE-MATRIX-TRIAD-20260902:START -->
## 2026-09-02 完整基线矩阵三件套校正

当前 complete baseline matrix 已从同一份机器可读 JSON 重新生成 Markdown，并通过 JSON/CSV/Markdown 三件套一致性审计：`60/60` 个五折组合、common-five `40/40`、校准已物化 `13` 组。当前三件套：JSON `outputs/complete_baseline_matrix_20260726.json` SHA256=`d8d9dd80f4535a815059578255df7d44fe849eabe42ed978d21244b93f0f2479`；CSV `outputs/complete_baseline_matrix_20260726.csv` SHA256=`ebbf935ed01adc381833a6422e8b9fe03d04dbffa5c217bee1e98f3fb6f65d88`；Markdown `outputs/complete_baseline_matrix_20260726_current.md` SHA256=`9e72fa98af0712bb6d9b0798796036bc35eca7b6088262c2d2735914a7999c14`；三件套审计 `outputs/COMPLETE_BASELINE_MATRIX_TRIAD_AUDIT_20260902_V2.json` SHA256=`1f7d01003046f2fd6d1c9acc7587c362de02555f47bb74c6fc772ae842482fd3`。

该矩阵是五个 validation-fold 模型在固定 learner-disjoint held-out test 上评估并含 Window 指标的历史 v16 口径，不是 conventional five-fold cross-validation。复现包内旧的 `55` 组 Markdown 保留为历史快照，不与当前 60 组 JSON/CSV 混用。DKVMN-v24 的当前 one-step-only `7` 数据集终态仍由独立审计表报告；它明确未启动 Window，不伪造或回填 Window 列。本次只修复矩阵 provenance，不改 canonical 投稿表数值、不启动 GPU。
<!-- CODEX-COMPLETE-MATRIX-TRIAD-20260902:END -->

<!-- CODEX-POST-MATRIX-SMOKE-20260902:START -->
## 2026-09-02 矩阵三件套修复后综合回归

当前完整基线矩阵 JSON/CSV/Markdown 三件套已统一为 `60/60` groups；三件套审计 `outputs/COMPLETE_BASELINE_MATRIX_TRIAD_AUDIT_20260902_V3.json` SHA256=`1f7d01003046f2fd6d1c9acc7587c362de02555f47bb74c6fc772ae842482fd3` 为 `pass`。这张矩阵保留 Window 指标历史 v16 口径；DKVMN-v24 one-step-only 七数据集终态仍由独立审计表报告，不填伪造 Window 值。

矩阵修复后重新执行综合 smoke/table/隐私/228 guard 回归：`47` tests、`1` skipped、状态 `pass`，收据 `outputs/PUBLICATION_SMOKE_AND_TABLE_REGRESSION_20260902_V3.json` SHA256=`0dfdaae3470637516ec9fc7a9c7e6ae4e25086dbff358a6cb5011c3ecd57c119`。当前 canonical 投稿表仍为 `352/592` 缺口；表审计 `outputs/PUBLICATION_CURRENT_TABLES_AUDIT_20260902_V2.json`、隐私审计 `outputs/PUBLIC_PRIVACY_AUDIT_20260902_V3.json` 和联合排名审计 `outputs/EFFECTIVENESS_RANK_ANNOTATION_AUDIT_20260902_V2.json` 均通过。本轮只修复矩阵 provenance 与回归证据，没有启动或停止 GPU，也没有改变主表数值。
<!-- CODEX-POST-MATRIX-SMOKE-20260902:END -->

<!-- CODEX-OFFLINE-RUNTIME-SNAPSHOT-20260902-0133:START -->
## 2026-09-02 01:33 228 离线守卫运行态快照

独立只读 SSH 快照（`2026-09-02T01:32:05+08:00`）显示 228 的既有 DTransformer native-JIT recovery 仍在运行：GPU0 `68%`，显存 `16844/24564 MiB`（空闲 `7264 MiB`），compute PID=`2762070`，链 PID=`393435`，共享锁 `continuous_baseline_gpu_queue_20260803.lock` 为 busy。该进程属于已登记 recovery，未停止、未重启、未重复启动。

completion-v9-r2 与 idle-v2 共 `4` 条 cron（各 `@reboot` 与每分钟一次）仍在位；v9 `state.json` 状态为 `dtransformer_recovery_active`、原因 `receipt_process_running`，阶段顺序保持 `DTransformer recovery -> DTransformer freeze/final -> UKT/Slepemapy validation -> UKT freeze/final -> legacy four-operator FLOPs`。`test_access=false`、`window_test_access=false`、`a2g_training_launch=false`。

因此本次没有把 validation 中间结果写入五折主表，也没有因 172 空闲而迁移 KT 任务。只有 recovery 产生完整终态 receipt、228 compute 清零、显存/共享锁门禁通过且既定 contract 哈希一致时，自动守卫才会继续登记阶段；172/127 继续保持 KT fail-closed。
<!-- CODEX-OFFLINE-RUNTIME-SNAPSHOT-20260902-0133:END -->

<!-- CODEX-OFFLINE-SMOKE-REFRESH-20260902-0143:START -->
## 2026-09-02 01:43 离线调度与 Smoke 刷新

在完整复现工作区重新执行 228-only 调度覆盖审计，状态 `pass`、failures=`[]`：`outputs/OFFLINE_SCHEDULER_COVERAGE_AUDIT_20260902_V4.json`，SHA256=`bc7f133ba0f6a81730eb8c53e7cb91028a06cf6d642decc7f477a0cdede7ad32`。远端快照 `01:42:25 +08` 仍为 DTransformer recovery active，compute PID=`2762070`，GPU0 显存 `16844/24564 MiB`，共享锁忙，四条 cron 在位；没有新增、停止或重复任务。

随后重跑投稿表/隐私/排名/guard 综合 smoke：`47` tests、`1` skipped、状态 `pass`，收据 `outputs/PUBLICATION_SMOKE_AND_TABLE_REGRESSION_20260902_V5.json`，SHA256=`a93b650937e32de9593875cd2c2f0a177a05da7c77cb0e276ca1e0134ba11bde`。当前 canonical 缺口仍为 `352/592`；本轮只更新门禁证据，不把 validation 中间值、smoke、pilot 或 paper-only 数据写入主表。

PaperGraph 笔记目录内的旧 `unittest discover` 不是当前 canonical smoke 入口：该目录未随附完整 `KT-Frontier/outputs` fixture，因而会出现历史复现包 `FileNotFoundError`，不能用来否定或替代上述完整工作区回归。后续仍以 hash-bound V5 收据与远端终态独立审计为准。
<!-- CODEX-OFFLINE-SMOKE-REFRESH-20260902-0143:END -->

<!-- CODEX-OFFLINE-SMOKE-AUDIT-20260902-0144:START -->
## 2026-09-02 01:44 表/隐私/排名审计收口

因本轮新增运行态记录，重新生成并核验了当前表 artifact 与四份公开笔记：表 artifact `TABLE_ARTIFACT_REEXTRACT_20260902_V4.md` SHA256=`4c4853c1da81089aee5546eb60aecd3d84226da44916cf81adeb79e5cb82d69e`；表审计 `PUBLICATION_CURRENT_TABLES_AUDIT_20260902_V4.json` SHA256=`19a5dc3bea652b3623c90a225c98f3c95a14cca70c02b2a94ad7f41ae260d11a`，状态 `pass`；隐私审计 V6 SHA256=`165f0f038ed0e46d93cfc14f2731a55ed5333ee3eaffefd2c0212ec71e4103c7`，状态 `pass`；联合排名审计 V3 SHA256=`1cedbc44731aebc5bf9e96735c975f5cf83310dbb39b8e8d110ea9796742ab47`，状态 `pass`。

以上审计均确认：14 个请求模型列、effectiveness/efficiency 表头、MCSKT 不参与复现排名、validation/smoke 不提升为 final 结果、canonical 缺口仍为 `352/592`。综合回归 V6 SHA256=`be15c37b04b8b7d67a9b4fca486c50f5c7f62b4dfb89779f4dba88cc9f6a35cc`，`47 tests / 1 skipped / pass`。这些是审计收口，不代表缺口已补齐。
<!-- CODEX-OFFLINE-SMOKE-AUDIT-20260902-0144:END -->

<!-- CODEX-FINAL-AUDIT-RECEIPT-20260902:START -->
## 2026-09-02 最终本轮审计收据

最终表审计 V5=`pass`（SHA256=`b148efcb5527006cdea00c5c44e8376dec090a3d62b5656f5eda8a087bcc3b9b`）、隐私审计 V7=`pass`（SHA256=`165f0f038ed0e46d93cfc14f2731a55ed5333ee3eaffefd2c0212ec71e4103c7`）、联合排名审计 V4=`pass`（SHA256=`d1ec7fd1f36782fa5c5c30cee354bcfd201b366672114dff234f80dc811e1cd9`），综合 smoke V7=`47 tests / 1 skipped / pass`（SHA256=`56d50aab17798f0768ce0cd28be8412ebeca79967fd8b2e61bbbe385e2549f1f`）。

本轮确认的是表格、证据边界和自动调度配置，不是“352 个缺口已完成”。228 recovery 仍在运行，后续只有在其合法终态和独立审计通过后才能继续填入新数值；172/127 不得作为 KT 结果主机。
<!-- CODEX-FINAL-AUDIT-RECEIPT-20260902:END -->

<!-- CODEX-FINAL-AUDIT-RECEIPT-20260902-SUPERSEDING:START -->
## 2026-09-02 最终收据版本校正

上段收据保留为历史快照；本段为 superseding 版本。当前最终收据为：表审计 V7=`pass`（SHA256=`52ed84da7e4823ff4e85d566c12530c211db539a4d0604ff25c45b1f795c1fc2`）、隐私审计 V9=`pass`（SHA256=`165f0f038ed0e46d93cfc14f2731a55ed5333ee3eaffefd2c0212ec71e4103c7`）、联合排名审计 V6=`pass`（SHA256=`6429aa00f994131ce8a4a537683cb3233a3d6f718c8409c8e00506dff597233b`）、综合 smoke V9=`47 tests / 1 skipped / pass`（SHA256=`e44523af02bb2b130d56590392699a1ea2dea380ec58f6f6ef25298bf75a95be`）。canonical 缺口仍为 `352/592`，没有任何 validation、smoke、pilot 或 paper-only 值被提升到主表。
<!-- CODEX-FINAL-AUDIT-RECEIPT-20260902-SUPERSEDING:END -->

<!-- CODEX-GAP-REGISTRY-REFRESH-20260902:START -->
## 2026-09-02 缺口索引与自动补缺 registry 刷新

根据当前三张 canonical 表重新生成：缺口 inventory `CANONICAL_PUBLICATION_GAP_INVENTORY_20260902_V26.json`（SHA256=`c8c1a4aac2685623979fb55dcd09eaeb7858e0076fb84d083c621250e7aec4c3`）、分类 `CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260902_V26.json`（SHA256=`317753f80f7730c045d54d1fc80996dfc1b15f2ce46728bf78bbf2929cdb520e`）和自动补缺 registry `BASELINE_COMPLETION_REGISTRY_20260902_V26.json`（SHA256=`6f92b1b6ad951f734d83a0c511e87f855614c1bd9d299d95a465a3f92a543f83`）。结果仍为 `592` 个 canonical 单元、`352` 个缺口；分类为效率测量 `218`、source/adapter `36`、operator FLOPs `24`、active/hash-staged `16`、effectiveness runner `16`、outside queue `23`、paper-only/protocol rejected `19`。

该刷新只更新机器清单，不提升任何中间结果。自动续跑仍只接纳有 hash-bound contract 和授权的 228 阶段；172/127、paper-only、outside-queue 继续不调度。
<!-- CODEX-GAP-REGISTRY-REFRESH-20260902:END -->

<!-- CODEX-DTRANSFORMER-RESUME-V10-20260903:START -->
## 2026-09-02 228 空闲时自动补缺续跑（v10）

172 即使无其他用户进程，也不作为知识追踪实验主机；训练、validation、test、Window-test 和效率测量仍严格限定在 228。228 在 `2026-09-02T20:12:01+08:00` 通过 host、无 compute、显存和共享锁门禁后，启动了新的 append-only DTransformer validation-only resume。当前快照 `20:16:44+08:00`：GPU0 利用率 `17%`，显存 `16703/24564 MiB`，compute PID=`45542`；共享锁 busy，说明续跑正在占用唯一 GPU。

本次没有覆盖旧 recovery 根目录。新根 `<REMOTE_HOME>/kt_baseline_20260723/strict_dtransformer_validation_native_jit_fallback_20260903_resume_v1` 的物化审计为 `pass_static_resume_ready_not_launched`，审计 SHA256=`03863f2fbcb4d41a4ef6e71121746edd0f08d743e51d8fcac6cc0f3ad03b92ef`；复用了 `25` 个已完成合法 trial，另有 Junyi `lr=2/fold0` `1` 个孤儿产物明确不复用、不覆盖。registry SHA256=`248d1f72e10e78f771498361db5226a42a46675cd9ead771f4c46b15a43cb54b`，最终执行授权 `authorization_v2.json` SHA256=`43bb73aafdb08c24009e85f0be61392b23e02c5837e7ab5ff289935415b474db`。

新的 chain runner SHA256=`00545dd964fc6c1451bc232aad9c35ac4af0a9b23696c4138825455a5b7f736a`，dataset runner SHA256=`2991f888190ccf38a0ca18657dfb22329a929d0986ac9206b3d3395a0adced00`；启动收据 `<REMOTE_HOME>/kt_baseline_20260723/baseline_completion_auto_guard_228_20260903_v10/DTRANSFORMER_RESUME_LAUNCH_RECEIPT.json` SHA256=`71d79bb726a7aa81bc2f70c15cd26c7be2bf576df12009c9075fdc3cc28972b5`。v10 已加入 228 的 `@reboot` 和每分钟 cron，门禁固定 `validation_only`、`test_access=false`、`window_test_access=false`、`a2g_training_launch=false`、单 GPU、共享锁和不可覆盖。

当前只记录运行证据，不把中间 validation 值写入五折主表；完整 `120/120` 终态和独立审计完成后，才更新 DTransformer effectiveness 单元及后续效率表。canonical 缺口仍为 `352/592`。
<!-- CODEX-DTRANSFORMER-RESUME-V10-20260903:END -->

离线自动补缺状态收据：`outputs/OFFLINE_BASELINE_AUTO_RESUME_AUDIT_20260903_V1.json`，SHA256=`4ec66e29a66ef001c3ed7c4cb32e9d14c0ccb61a229b91936d0d8ec9f9bb483b`；该收据只记录 228 运行快照和门禁，不增加任何主表数值。

## 2026-09-02 本轮 smoke 与表审计收口

本轮固定模块 smoke 为 `47 tests / 1 skipped / OK`。当前表审计 `PUBLICATION_CURRENT_TABLES_AUDIT_20260903_V4.json` SHA256=`7bbed6cc6fb5791ad46b7afacbbc99b836c5a5065daa5a805751e1a2c4765868`、隐私审计 `PUBLIC_PRIVACY_AUDIT_20260903_V4.json` SHA256=`165f0f038ed0e46d93cfc14f2731a55ed5333ee3eaffefd2c0212ec71e4103c7`、排名审计 `EFFECTIVENESS_RANK_ANNOTATION_AUDIT_20260903_V5.json` SHA256=`5a07f06253ce2710b8ae777b7b5f85f817521481540faa93dd0e0cf11b611065` 均为 `pass`；综合本地收据 `PUBLICATION_SMOKE_AND_TABLE_REGRESSION_20260903_V4.json` SHA256=`70bc035249478150df646f4e95badb5cde6fca43c9e237f034582eac27ab889f`。

上述回归只验证表结构、隐私、排名、smoke 和 228 守卫合同；DTransformer resume 尚未达到 `120/120` terminal，canonical 缺口仍为 `352/592`，本轮没有向五折主表或效率表写入新数值。

2026-09-02 23:34 运行快照：DTransformer resume 仍在 228 的 Junyi2015 `lr_2_4ee7773dad/fold0`，GPU 利用率 `94%`、显存 `16705/24564 MiB`、compute PID=`45542`；chain PID=`45484`。v11 post 守卫已通过 `@reboot`/每分钟登记，但在 resume terminal 出现前保持等待，不启动 final-test。机器收据 `outputs/OFFLINE_BASELINE_AUTO_RESUME_AUDIT_20260903_V2.json` 为 `pass_active_resume;post_stage_waiting_for_resume_terminal`。

<!-- CODEX-RESUME-SMOKE-V5-20260903:START -->
## 2026-09-02 续跑期间最新回归

在 DTransformer append-only resume 运行期间，本地固定 smoke 仍为 `47 tests / 1 skipped / OK`。最新表审计 `outputs/PUBLICATION_CURRENT_TABLES_AUDIT_20260903_V5.json` SHA256=`ffaa0fcabab1a3a4e9c4de71e7e325e97b8565dc206da4c9e5cd0d28a2545d36`、隐私审计 `outputs/PUBLIC_PRIVACY_AUDIT_20260903_V5.json` SHA256=`165f0f038ed0e46d93cfc14f2731a55ed5333ee3eaffefd2c0212ec71e4103c7`、排名审计 `outputs/EFFECTIVENESS_RANK_ANNOTATION_AUDIT_20260903_V6.json` SHA256=`d670a738d5514ee30c37d18fbb40af646ac86de811bb074934db7b56f5bf93ad` 均为 `pass`；综合收据 `outputs/PUBLICATION_SMOKE_AND_TABLE_REGRESSION_20260903_V5.json` SHA256=`036680f2d7bedd7d0d25ce6f97de6c0e88841d830c22750e132364fcb878e1fe`。

228 快照 `20:53:14+08` 为 GPU0 `96%`、显存 `16705/24564 MiB`、compute PID=`45542`；resume 已完成 Statics2011 的 `15/15`，并保留 `25` 个已复制完成 task。当前 Junyi2015 正在继续，所有输出仍为 validation-only，`test_access=false`、`window_test_access=false`。canonical `352/592` 缺口不变。
<!-- CODEX-RESUME-SMOKE-V5-20260903:END -->

<!-- CODEX-172-IDLE-NO-KT-EXECUTION-20260902:START -->
## 2026-09-02 172 空闲不构成 KT 基线执行许可

即使 172 当前没有其他用户进程，也不能在那里运行知识追踪基线的训练、validation、test、Window-test 或效率测量。现行实验主机策略仍是 `228-only`：投稿表可接纳的数值必须来自 228 的统一硬件、同一数据协议、同一 batch/精度和独立终态审计；172/127 只能做不产出 KT 数值的 CPU/static 审计。因此“172 空闲”对补充可写入主表的基线没有帮助，也不触发迁移或备机执行。

本轮将离线覆盖审计入口从过期 V25 registry 修正为现行 V26 registry：`outputs/BASELINE_COMPLETION_REGISTRY_20260902_V26.json`（SHA256=`6f92b1b6ad951f734d83a0c511e87f855614c1bd9d299d95a465a3f92a543f83`），审计脚本 SHA256=`7d6b9a20e0a252368d816db16d5016679ff3e705ed2f31a734f9da022b8bd14b`。本地 V7 探针因当前 SSH 不可达而记录为 probe failure；该收据不推断 228 实时 GPU 状态，也没有启动、停止、迁移或改写任何实验。canonical 范围仍为 `592` 个单元、缺口 `352`，本段不改变任何结果单元。
<!-- CODEX-172-IDLE-NO-KT-EXECUTION-20260902:END -->

<!-- CODEX-POST-172-POLICY-SMOKE-20260902:START -->
## 2026-09-02 172 策略复核后的本地回归

本地投稿回归已重新执行：完整 smoke 为 `47 tests / 1 skipped / OK`；当前表结构审计 V10、隐私审计 V12、联合排名审计 V9 均为 `pass`。收据分别为 `outputs/PUBLICATION_CURRENT_TABLES_AUDIT_20260902_V10.json`（SHA256=`30319f251b68573119e4f109e25c1bde04e404f5ca8c5b35dab2adda289558ce`）、`outputs/PUBLIC_PRIVACY_AUDIT_20260902_V12.json`（SHA256=`165f0f038ed0e46d93cfc14f2731a55ed5333ee3eaffefd2c0212ec71e4103c7`）、`outputs/EFFECTIVENESS_RANK_ANNOTATION_AUDIT_20260902_V9.json`（SHA256=`9dad85a08ee21eddf2f575eba5046ad2a6ecb2d9467df46deec3de2229602be3`）和表 artifact `outputs/TABLE_ARTIFACT_REEXTRACT_20260902_V5.md`（SHA256=`4c4853c1da81089aee5546eb60aecd3d84226da44916cf81adeb79e5cb82d69e`）。

远端依赖的综合回归 V12 因当前 228 SSH 超时而 `fail-closed`（SHA256=`b8797849ef175f4262373a4d588f901adb5a4ce3841cc0ad0147506216e76c35`）；这只表示实时远端状态待重验，不表示 GPU 空闲、守卫失效或任务终止。离线覆盖探针 V7 同样因 SSH 超时记录 `fail`（SHA256=`174afce9890c7f18ed3160fff7e4fa900316ac433dc298caaa0d2535933d718d`）。本段不改变任何结果单元，仍保持 `352/592` 缺口。
<!-- CODEX-POST-172-POLICY-SMOKE-20260902:END -->

<!-- CODEX-CURRENT-POST-SYNC-AUDIT-20260902:START -->
## 2026-09-02 当前文件字节对应的最终回归收据

在本次策略记录追加后，表审计 V11=`pass`（SHA256=`78e31427a0387108a361944732870322617ec7577f59090a0920680c79f5965a`）、隐私审计 V13=`pass`（SHA256=`165f0f038ed0e46d93cfc14f2731a55ed5333ee3eaffefd2c0212ec71e4103c7`）、联合排名审计 V10=`pass`（SHA256=`e5ba067451275484abedf2a6df79b0603c991b6690a337dabf9dfa77e39b94bf`），本地 47 项 smoke 为 `47 tests / 1 skipped / OK`。综合回归 V13 SHA256=`e0e6efedfdca8ccd99bd700f3f7a522b5b2cf04c9c99481ad93816d6287c0667` 因实时 228 SSH 超时而 fail-closed，不能据此推断远端守卫、GPU 或队列状态。当前表 artifact `outputs/TABLE_ARTIFACT_REEXTRACT_20260902_V6.md` SHA256=`4c4853c1da81089aee5546eb60aecd3d84226da44916cf81adeb79e5cb82d69e`；canonical 缺口仍为 `352/592`，本段不改变任何结果单元。
<!-- CODEX-CURRENT-POST-SYNC-AUDIT-20260902:END -->

<!-- CODEX-FINAL-CURRENT-BYTE-AUDIT-20260902:START -->
## 2026-09-02 当前字节版本 superseding 收据

当前四份笔记的本地验证继续通过：表 artifact `outputs/TABLE_ARTIFACT_REEXTRACT_20260902_V7.md` SHA256=`4c4853c1da81089aee5546eb60aecd3d84226da44916cf81adeb79e5cb82d69e`；表审计 V12=`pass`（SHA256=`1a50965435a9368057dca652f0b9dd1563726a6916aee767b21c2f5c35623848`）、隐私审计 V14=`pass`（SHA256=`165f0f038ed0e46d93cfc14f2731a55ed5333ee3eaffefd2c0212ec71e4103c7`）、联合排名审计 V11=`pass`（SHA256=`ac5285909bbb54a1141cd87b0e252982a5a9e4189ac6defa784c24e66d351593`），完整本地 smoke 为 `47 tests / 1 skipped / OK`。综合回归 V14（SHA256=`dcd7b201451fd2ec3f2b1457bd8a872742aaec4b5af4d0d582406ed47eabea34`）的远端部分因 SSH 超时 fail-closed；不据此推断 228 实时 GPU 或队列状态。canonical 缺口仍为 `352/592`，不改变任何结果单元。
<!-- CODEX-FINAL-CURRENT-BYTE-AUDIT-20260902:END -->

<!-- CODEX-ULTIMATE-LOCAL-REGRESSION-20260902:START -->
## 2026-09-02 本轮最终本地回归收口

最终本地收据：表审计 V14=`pass`（SHA256=`846e326fba1d520ef0aac67272345c3a6f8ff14ed5d06163b0a97ad8c3dafc3d`）、隐私审计 V16=`pass`（SHA256=`165f0f038ed0e46d93cfc14f2731a55ed5333ee3eaffefd2c0212ec71e4103c7`）、联合排名审计 V13=`pass`（SHA256=`7588a754124d79130af08a74a7a1fc8e270768435e9aad53892ea5bf097d63a7`），47 项 smoke 为 `47 tests / 1 skipped / OK`。综合回归 V15 SHA256=`5880a838e7e3a2788728de6e5153d88e867d0aa87d3aaba8783f3b88921ad2c6` 仅因实时 228 SSH 不可达而 fail-closed；离线覆盖 V7 SHA256=`174afce9890c7f18ed3160fff7e4fa900316ac433dc298caaa0d2535933d718d` 同理不推断远端状态。canonical 缺口仍为 `352/592`，不使用 172 空闲、smoke 或旧快照填充。
<!-- CODEX-ULTIMATE-LOCAL-REGRESSION-20260902:END -->

<!-- CODEX-DTRANSFORMER-STALE-RECEIPT-AUDIT-20260903:START -->
## 2026-09-03 DTransformer recovery stale receipt 本地审计

228 当前不可达，本轮没有对 recovery 做续跑或重启。新增本地只读审计 `outputs/DTRANSFORMER_RECOVERY_STALE_LOCAL_AUDIT_20260903_V1.json`（SHA256=`9156afa591e9454180bb0d3fc8d989c4ea3402b9d1a3bbd6114216dcce26026f`）记录：最后可见本地快照为 `15/120` validation trials，存在 launch receipt 和 live snapshot，但没有可接纳的 terminal audit；审计脚本 SHA256=`921b8bbf6b6921d0e78c2c3ef281b909bef66ff48fbd1f19d2ddcc59531ebfd0`。该证据不声明远端当前进度，也不提升任何中间值到五折主表。228 恢复后必须独立重哈希所有任务收据，再建立只覆盖剩余任务的 append-only resume contract；172/127 继续禁止 KT。
<!-- CODEX-DTRANSFORMER-STALE-RECEIPT-AUDIT-20260903:END -->

<!-- CODEX-RESUME-AUDIT-SMOKE-20260903:START -->
## 2026-09-03 当前 resume 审计与 smoke 收口

本地 stale-recovery 审计为 `blocked_stale_receipt_remote_recheck_required`，收据 `outputs/DTRANSFORMER_RECOVERY_STALE_LOCAL_AUDIT_20260903_V1.json` SHA256=`9156afa591e9454180bb0d3fc8d989c4ea3402b9d1a3bbd6114216dcce26026f`；它只记录历史 `15/120` 进度，不代表当前远端。当前表审计 V16=`pass`（SHA256=`549dc89b3e1375a7e8d03cd22759df941134d3f211078b5959cb89f4e703026f`）、隐私 V18=`pass`（SHA256=`165f0f038ed0e46d93cfc14f2731a55ed5333ee3eaffefd2c0212ec71e4103c7`）、排名 V15=`pass`（SHA256=`6875831c2f46beb20389527f9fbb46c90880549f1e5bcb3af8fec29936a73abd`），完整 smoke `47 tests / 1 skipped / OK`。综合回归 V17 SHA256=`fe2cd582ee602cb69dca2dac1876d9462397844dd86b858dac97e1b16086ec07` 因 228 SSH 超时而 fail-closed；canonical 缺口仍 `352/592`，本段不改结果单元。
<!-- CODEX-RESUME-AUDIT-SMOKE-20260903:END -->

<!-- CODEX-20260903-FINAL-RECHECK:START -->
## 2026-09-03 最新回归收据

当前文件对应的表审计 V1=`pass`（SHA256=`46a6a0eeb003b89b2c929d95b98ed27de06581d504c365a799c7d6ff33bbf95a`）、隐私 V1=`pass`（SHA256=`165f0f038ed0e46d93cfc14f2731a55ed5333ee3eaffefd2c0212ec71e4103c7`）、排名 V1=`pass`（SHA256=`617b032b8cbf823ef9a3d684ad72c229912983360fd99ab24d1f608c471f8a6e`），本地 smoke `47 tests / 1 skipped / OK`。综合回归 V1 SHA256=`dcd782d658e8ffd4b2f63e19be18447b098273cf0449e6243502e6458f9fcbc3` 的远端检查因 SSH 不可达而 fail-closed；不把它解释为 GPU 空闲或任务结束。stale-recovery 本地审计 SHA256=`9156afa591e9454180bb0d3fc8d989c4ea3402b9d1a3bbd6114216dcce26026f`，canonical 缺口仍 `352/592`。
<!-- CODEX-20260903-FINAL-RECHECK:END -->

<!-- CODEX-GAP-PRIORITY-MANIFEST-20260903:START -->
## 2026-09-03 V26 缺口优先级清单

新增机器可读清单 `outputs/BASELINE_GAP_PRIORITY_REGISTRY_20260903_V1.json`（SHA256=`eef9a44a19a2d2774cbc18b9c74fcd34d37d6d17c4328c3ff0f7ba969e89a96d`），覆盖全部 `352` 个缺口，排序为 validation-only→independent-freeze、同协议 efficiency、operator-complete FLOPs；所有条目当前均标记 `launchable=false`，因为必须先在 228 获得独立实时门禁、哈希合同和授权。生成脚本 SHA256=`8adeb0e158bcaa939ddb55a3d2dd411d87172a6a5f80115e5e7f05fa6554beca`，测试 `2/2` 通过。该清单不产生新结果、不启动 GPU；172/127 仍禁止 KT。
<!-- CODEX-GAP-PRIORITY-MANIFEST-20260903:END -->

<!-- CODEX-172-IDLE-228-RESUME-AUDIT-20260903-0028:START -->
## 2026-09-03 00:28 172 空闲与 228 自动续跑的独立核验

172 即使没有其他用户进程，也仍不具备 KT 执行许可：训练、validation、test、Window-test 和效率测量均禁止在 172/127 运行。可写入投稿主表的数值必须来自 228 的统一硬件、数据协议、batch/精度和独立终态审计。

独立只读收据 outputs/OFFLINE_BASELINE_AUTO_RESUME_AUDIT_20260903_V6.json（SHA256=5678df97d8c29d509f8d20227a540672b714a4b90c3ee4220ee4fdab009887c8）显示 228 的 v10 自动守卫已真实接续 append-only DTransformer validation-only resume：GPU0 利用率 94%，显存 16705/24564 MiB，compute PID=45542，chain PID=45484；当前 junyi2015/fold0/lr_2_4ee7773dad 正在运行。resume 已复用 25 个合法 task，完成 1/8 数据集的 25/120 trials；terminal audit 尚未生成。

v10 guard SHA256=4d87a014cb144665de4027646e19f2349cf5c5f3edb082839110ce6ed829a8a7，v11 post guard SHA256=2235c77dce94eac4888bea65144a451b249233580e22ddd1485f771f74b48da6；228 crontab 保持 4 条（各 @reboot 与每分钟一次）。v11 状态为 waiting_dtransformer_resume，不会在 resume terminal 前启动后续阶段；test_access=false、window_test_access=false、a2g_launch=false。本次没有停止、迁移、重复启动或改写任何实验，也没有把中间 validation 写入五折主表。
<!-- CODEX-172-IDLE-228-RESUME-AUDIT-20260903-0028:END -->

<!-- CODEX-CURRENT-REGRESSION-V2-20260903:START -->
## 2026-09-03 当前回归与离线自动补基线状态（V2）

当前版本综合回归收据 `outputs/PUBLICATION_CURRENT_REGRESSION_20260903_V2.json` 状态为 `pass`，47 项固定 smoke 中 1 项跳过，表结构、隐私、排名和缺口一致性均通过；收据 SHA256=`f97fbbaa532a5325570e4fff0a8d54fc656fe44c85fae899b1bba1fef2547d07`。本次修正了 live resume schema 的进度解析：完成量由 `max(copied_complete_task_count, 已完成数据集记录的 completed_trials 之和)` 得出，当前 DTransformer validation-only resume 为 `25/120`，不是终态。

228 当前仍在运行既有 DTransformer validation 队列（GPU compute app 存在，约 `16.7 GiB` 显存）；`test_access=false`、`window_test_access=false`、`a2g_training_launch=false`。v10/v11 自动守卫均已 hash 核验，cron 保持 `@reboot` 与每分钟触发各一条，单 GPU、共享锁、不可覆盖和显存空闲门槛均启用。该状态只记录运行证据，不授权 test、Window-test 或 A2G。

canonical 三张表仍为 `592` 个单元，缺口 `352`；缺口清单 `outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260902_V26.json` SHA256=`c8c1a4aac2685623979fb55dcd09eaeb7858e0076fb84d083c621250e7aec4c3`。effectiveness 主表已经包含 DKT、DKT+、DKVMN、SAKT、SAINT++、AKT、SimpleKT、DTransformer、UKT、ACE-KT、ASIKT、Mamba4KT、MCSKT† 和 A2G-MambaKT 列；未通过独立五折终态的格子继续为 `—`。效率表保留 TT、IT、Throughput、参数、GPU peak 和 FLOPs 字段；Window-AUC/效率图字段仍为空，不能用 smoke、pilot、validation 或 paper-only 值填充。

172/127 即使空闲也继续 `fail-closed`，只能做 CPU/static、哈希、归档和笔记审计；新的 KT 训练、validation、test、Window-test 和效率测量仍只允许在 228。笔记更新不改变任何科学结果或调度顺序。
<!-- CODEX-CURRENT-REGRESSION-V2-20260903:END -->

<!-- CODEX-CURRENT-GUARD-BLOCKER-V1-20260903:START -->
## 2026-09-03 228 自动补基线守卫的实际后续边界

独立远端复核显示 DTransformer append-only validation resume 仍在运行：GPU compute app 存在（当前观察到约 `16.7 GiB` 显存），live 状态为 `running`，可审计进度为 `25/120`，terminal audit 尚未生成。`test_access=false`、`window_test_access=false`、`a2g_training_launch=false` 保持不变。

v10 负责当前 DTransformer validation resume；v11 只在 resume terminal 完成后检查 post-recovery 阶段。当前 v11 所需的显式 `DTRANSFORMER_POST_RECOVERY_AUTHORIZATION.json` 不存在，因此后续 final-test/冻结及其后的 UKT、FLOPs 阶段会 fail-closed 等待，不会因为 cron 存在而自动越权。该文件缺失是当前自动链的明确 blocker，不能把 4 条 cron 解释为 352 个缺口均已自动排队。

172/127 即使空闲也不能运行 KT；只能做 CPU/static 审计、哈希、归档和笔记同步。新的 effectiveness、Window-AUC、TT/IT、GPU usage 或 FLOPs 数值仍须在 228 通过对应 hash-bound 合同、阶段授权和独立终态审计后才能填表。当前主表所有未有合法五折终态证据的单元继续为 `—`。
<!-- CODEX-CURRENT-GUARD-BLOCKER-V1-20260903:END -->

<!-- CODEX-POST-AUTH-STAGING-V1-20260903:START -->
## 2026-09-03 DTransformer post-stage预授权已静态部署

为保证 228 离线时在当前 DTransformer resume 自然终态后能够继续，已生成并以不覆盖方式部署 post-stage 预授权：状态 `prepared_pending_resume_terminal`，`execution_started=false`，host=`228`，SHA256=`c214b5ed6d38658ca238bed5e1d3b387dc8ecce4266c57432ca3f7715635a7b1`。它绑定当前 resume registry/authorization 及 post runner、freeze auditor、authorizer、evaluator、final runner 的精确哈希；不授予当前 resume 之外的新训练，不允许 Window-test 或 A2G。

v11 守卫已重新执行并保持 `waiting_dtransformer_resume`，因为 live 仍为 `25/120`、terminal audit 尚不存在。预授权存在不等于 final-test 已授权；只有 resume terminal 完整、独立 freeze audit、显存/compute/共享锁门禁全部通过后才会接续。主表、效率表和 Window-AUC 表本轮没有新增数值。
<!-- CODEX-POST-AUTH-STAGING-V1-20260903:END -->

<!-- CODEX-FINAL-AUDIT-RECEIPT-V5-20260903:START -->
## 2026-09-03 最终当前审计收据（V5）

本轮在四份同步笔记更新后重新执行：综合回归 `pass`，收据 `outputs/PUBLICATION_CURRENT_REGRESSION_20260903_V5.json` SHA256=`f6462944054057e74bfb9d9a96756849a984e8e601b91e3b0c2872b01cd315b6`；表结构审计 `pass`（`outputs/PUBLICATION_CURRENT_TABLES_AUDIT_20260903_V8.json`, SHA256=`98f5bc5297af32de3efb1708c6462dde3407384bcdefd96c43b83d592196c219`）；隐私审计 `pass`、扫描 4 个文件（`outputs/PUBLIC_PRIVACY_AUDIT_20260903_V9.json`, SHA256=`0433a48b7b0ee009e535d09343aa55827acd09560191d04068355ab0072b3daf`）；排名标注审计 `pass`、26 行（`outputs/EFFECTIVENESS_RANK_ANNOTATION_AUDIT_20260903_V10.json`, SHA256=`369bf5ab581c273e8009fa87ae810008740cb3598dbcb5a1d2ef2ce8fe95091c`）；固定 smoke `47 tests / 1 skipped / OK`。

228 当前 DTransformer resume 仍为 validation-only `25/120`、compute app 正在运行，terminal 尚未生成；v10 状态 `dtransformer_resume_started`，v11 状态 `waiting_dtransformer_resume`。post-stage 预授权已静态部署，状态 `prepared_pending_resume_terminal`，本地 SHA256=`c214b5ed6d38658ca238bed5e1d3b387dc8ecce4266c57432ca3f7715635a7b1`，远端文件权限 `0444`；它不启动任务、不允许 test/Window-test/A2G。

canonical 三张表仍为 `592` 单元、缺口 `352`。effectiveness 主表和效率表的缺失单元继续以 `—`/`NA` 标记；Window-AUC、TT/IT、GPU usage、FLOPs 只有在 228 同协议终态和独立审计通过后才能填入。172/127 仍禁止 KT 执行，即使空闲也只能做 CPU/static 和归档。
<!-- CODEX-FINAL-AUDIT-RECEIPT-V5-20260903:END -->

<!-- CODEX-FINAL-AUDIT-RECEIPT-V6-20260903:START -->
## 2026-09-03 最新审计收据（V6，superseding）

V6 supersedes the earlier V5 receipt with the current file bytes:综合回归 `pass`，`outputs/PUBLICATION_CURRENT_REGRESSION_20260903_V6.json` SHA256=`c2d3ecc13b6bd5bdbe5ca75e8f2c333bec7598472e3c1996d60f605a8383f868`；表结构 `pass`，`outputs/PUBLICATION_CURRENT_TABLES_AUDIT_20260903_V9.json` SHA256=`29f70c02bd2422560b47f162a685ea9aa7bedd7ab5835d7096cc91c9d8c0905a`；隐私 `pass`，`outputs/PUBLIC_PRIVACY_AUDIT_20260903_V10.json` SHA256=`0433a48b7b0ee009e535d09343aa55827acd09560191d04068355ab0072b3daf`；排名 `pass`、26 行，`outputs/EFFECTIVENESS_RANK_ANNOTATION_AUDIT_20260903_V11.json` SHA256=`3e24acb5c08c0e3a04466a8c487cb016fdd0ba28079578b19e77aff027f65d59`；固定 smoke 为 `47 tests / 1 skipped / OK`。

228 当前 DTransformer resume 仍在 validation-only 运行，实时可审计进度 `25/120`，terminal 尚未生成，因而没有任何新数值进入五折主表、效率表或 Window-AUC 表。post-stage 预授权状态为 `prepared_pending_resume_terminal`，本地 SHA256=`c214b5ed6d38658ca238bed5e1d3b387dc8ecce4266c57432ca3f7715635a7b1`，远端目标为 `<REMOTE_HOME>/kt_baseline_20260723/baseline_completion_auto_guard_228_20260903_v11_post/DTRANSFORMER_POST_RECOVERY_AUTHORIZATION.json`（`0444`）；它不授权 Window-test 或 A2G。

canonical 范围仍为 `592`、缺口 `352`。172/127 即使空闲也只允许 CPU/static、哈希和归档，不能执行 KT 或效率测量。所有未通过 228 独立五折终态审计的模型/数据集单元继续保持 `—`/`NA`。
<!-- CODEX-FINAL-AUDIT-RECEIPT-V6-20260903:END -->

<!-- CODEX-FINAL-AUDIT-RECEIPT-V7-20260903:START -->
## 2026-09-03 最新审计收据（V7，superseding）

当前字节的综合回归 `pass`：`outputs/PUBLICATION_CURRENT_REGRESSION_20260903_V7.json` SHA256=`1d49610fc52851c190f5ce944140e2c01d174d20197eaf07293c1362f0802b36`；表结构 `pass`：`outputs/PUBLICATION_CURRENT_TABLES_AUDIT_20260903_V10.json` SHA256=`fc9c9a053ada56839de3098ed06c62751d55b392841ef25acceb2d467c5ebae5`；隐私 `pass`：`outputs/PUBLIC_PRIVACY_AUDIT_20260903_V11.json` SHA256=`0433a48b7b0ee009e535d09343aa55827acd09560191d04068355ab0072b3daf`；排名 `pass`、26 行：`outputs/EFFECTIVENESS_RANK_ANNOTATION_AUDIT_20260903_V12.json` SHA256=`bd76a893510be9f9dcdd80991220a518b038ac88a98f1b3c306c316184868710`；固定 smoke `47 tests / 1 skipped / OK`。

228 DTransformer resume 仍在 validation-only `25/120`，terminal 尚未生成；post-stage 预授权为 `prepared_pending_resume_terminal`，SHA256=`c214b5ed6d38658ca238bed5e1d3b387dc8ecce4266c57432ca3f7715635a7b1`，不允许 Window-test/A2G。canonical 仍为 `592` 单元、缺口 `352`；未有独立五折终态的 effectiveness、效率和 Window-AUC 单元继续保持 `—`/`NA`。172/127 即使空闲也仅允许 CPU/static 与归档，不能执行 KT。
<!-- CODEX-FINAL-AUDIT-RECEIPT-V7-20260903:END -->

<!-- CODEX-GUARD-BLOCKER-SUPERSEDED-V1-20260903:START -->
## 2026-09-03 自动守卫 blocker 更正（superseding）

此前“post authorization 不存在”的状态段落已被后续部署收据 supersede。当前 post-stage 预授权已存在并通过本地/远端 SHA 重算：`prepared_pending_resume_terminal`，SHA256=`c214b5ed6d38658ca238bed5e1d3b387dc8ecce4266c57432ca3f7715635a7b1`，权限 `0444`。因此当前 v11 的实际 blocker 不是缺少授权，而是 DTransformer resume 尚未达到 `120/120` terminal；v11 正常保持 `waiting_dtransformer_resume`。旧段落保留为历史观察，不代表当前状态。

本次只更正文状态，不改 effectiveness、效率、Window-AUC 或 FLOPs 任何数值；172/127 仍禁止 KT 执行。
<!-- CODEX-GUARD-BLOCKER-SUPERSEDED-V1-20260903:END -->

<!-- CODEX-LATEST-REGRESSION-V9-20260903:START -->
## 2026-09-03 最新综合回归收据（V9，superseding）

当前四份笔记对应的综合回归为 `pass`：`outputs/PUBLICATION_CURRENT_REGRESSION_20260903_V9.json` SHA256=`17bb063a010fea57964202a3e48949e990ce4e3874122865d4e72dfe9a2fa198`；表结构、隐私和排名输入均为 `pass`（分别为 `PUBLICATION_CURRENT_TABLES_AUDIT_20260903_V13.json`、`PUBLIC_PRIVACY_AUDIT_20260903_V14.json`、`EFFECTIVENESS_RANK_ANNOTATION_AUDIT_20260903_V14.json`）。固定 smoke `47 tests / 1 skipped / OK`。228 DTransformer resume 仍为 validation-only `25/120`，terminal 尚未生成；没有新科学数值进入主表。canonical 缺口仍 `352/592`，Window-AUC、TT/IT、GPU usage、FLOPs 继续等待 228 同协议终态。172/127 即使空闲也仅允许 CPU/static 与归档。
<!-- CODEX-LATEST-REGRESSION-V9-20260903:END -->

<!-- CODEX-CURRENT-STATUS-V10-20260903:START -->
## 2026-09-03 当前基线补缺状态（V10，append-only）

本轮固定 smoke 与当前表/隐私/排名回归均为 `pass`：综合回归收据 `outputs/PUBLICATION_CURRENT_REGRESSION_20260903_V10.json` SHA256=`88ddac01be638dd2c84989a8d3f1ac62985b669434ba3737a7d293b9a8fe6803`。四份笔记的主表、补充表、效率字段（TT、IT、Throughput、参数、GPU peak、FLOPs）和 ASIKT 风格 Window-AUC/效率字段均已存在；没有把 smoke、pilot、validation 或 paper-only 数值提升为正式五折结果。

228 当前仍有 DTransformer validation-only compute（只读健康收据 `outputs/DTRANSFORMER_RESUME_HEALTH_20260903_V2.json` SHA256=`dbed01d425a4041233aad8330d9ac4cdfe5a681bcd5b8b3712cfc38bf6e123a0`；GPU 快照=`0, 55, 16705, 7403, 24564`），live 进度仍为 `25/120`，terminal audit 尚未生成，故本轮没有新增可写入主表的科学数值。canonical 范围为 `592` 个单元，缺口 `352`（清单 `outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260902_V26.json` SHA256=`c8c1a4aac2685623979fb55dcd09eaeb7858e0076fb84d083c621250e7aec4c3`）。

离线自动补缺继续由 228 的 hash-bound、单 GPU、共享锁、显存门禁和四条 cron 接续；172/127 即使空闲也只能做 CPU/static、哈希、归档和笔记审计，不能执行 KT、Window-test 或效率测量。已登记阶段终态前不会启动下一阶段，也不会覆盖旧产物。
<!-- CODEX-CURRENT-STATUS-V10-20260903:END -->

<!-- CODEX-SCHEDULER-V10-V11-UKT-V3-20260903:START -->
## 2026-09-03 228 离线自动补缺与 UKT V3 staging（append-only）

现行 v10/v11 调度审计为 `pass`：`outputs/OFFLINE_SCHEDULER_V10_V11_AUDIT_20260903_V1.json` SHA256=`4b7c6fceafe49a35ecc42a97bf48db032fe79314d40ff6b2f21a52b13c8be98a`。228 host、四条 `@reboot`/每分钟 cron、单 GPU、共享锁、显存空闲门禁、不可覆盖和 `test_access=false`/`window_test_access=false`/`a2g_training_launch=false` 均通过；当前 DTransformer validation resume 的动态 compute 身份与运行态由 `outputs/DTRANSFORMER_RESUME_HEALTH_20260903_V4.json` SHA256=`b35913accdd0573b21cb13088897250ecd9950478d4bec00a494524aa88e577a` 确认（GPU 快照=`0, 64, 16703, 7405, 24564`）。

为避免当前 DTransformer 终态后 GPU 再次空闲，已在 228 以 `0444`、不覆盖方式生成 UKT/Slepemapy V3 staging 授权，远端原件 SHA256=`256d6fcf9f43eb4df42cb7eac1d1403f7b62b63934030947539a22e25ae07780`；它只允许在 DTransformer post-recovery 独立终态、锁空闲、compute 清零、显存至少 16384 MiB 且所有绑定重哈希通过后进入 validation-only，Window-test/A2G 仍禁用，当前未启动 UKT。本地脱敏镜像 SHA256=`1ca1c3c06414a8b6f01ee5bf4b1acae5a7e39f077458670dbeed79584ad1f5fe`。

canonical 范围仍为 `592` 个单元、缺口 `352`（`outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260902_V26.json` SHA256=`c8c1a4aac2685623979fb55dcd09eaeb7858e0076fb84d083c621250e7aec4c3`）。效率表、Window-AUC/效率图字段和剩余 effectiveness 格子继续保留 `—`/`NA`，直到对应 228 同协议五折终态及独立审计出现。172/127 即使空闲也只能做 CPU/static、哈希、归档和笔记同步，不能运行 KT 或效率测量。
<!-- CODEX-SCHEDULER-V10-V11-UKT-V3-20260903:END -->

<!-- CODEX-V12-CONTINUATION-INSTALLED-20260903:START -->
## 2026-09-03 v12 离线后续守卫已安装（superseding）

上一段“四条 cron”是 v12 安装前快照；现行拓扑为六条：v10、v11、v12 各一条 `@reboot` 和一条每分钟触发。v12 仅在新的 DTransformer post-recovery terminal 合法后，按 `UKT/Slepemapy validation-only -> 独立 freeze/one-step final -> legacy four-model operator-complete FLOPs` 顺序串行接续。调度审计 `outputs/OFFLINE_SCHEDULER_V10_V11_AUDIT_20260903_V2.json` 为 `pass`，SHA256=`e92e4ea8347ca5d39f03cffc7b56430af34019714cb899e44c1a131f01824afa`；v12 runner SHA256=`15cca0b5638feeb6c7f77c80e7a6423675fd6f21c3fadf77f1ac3198d869a2fb`，静态测试 `5/5` 通过，测试源 SHA256=`87cc6de69c6b4f9e7f56a9e6f4fb636272b81fd0a722768aa629ab7bb380a34e`。

UKT V3 远端授权原件 SHA256=`256d6fcf9f43eb4df42cb7eac1d1403f7b62b63934030947539a22e25ae07780`、权限 `0444`，当前仅 staging；Window-test 和 A2G 均未授权。DTransformer 仍是 228 当前唯一 KT compute，动态健康收据 `outputs/DTRANSFORMER_RESUME_HEALTH_20260903_V5.json` SHA256=`2e568697889909d35ac73a8f6d08b531c0110a5612bfb8cff425e75fc0ed5bca`。当前没有新 final artifact，因此主表、效率表、Window-AUC 表不增加数值，canonical 缺口仍 `352/592`。172/127 即使空闲也不用于 KT。
<!-- CODEX-V12-CONTINUATION-INSTALLED-20260903:END -->

<!-- CODEX-FINAL-V12-REGRESSION-20260903:START -->
## 2026-09-03 v12 安装后的最终回归收据（superseding）

当前综合回归 `outputs/PUBLICATION_CURRENT_REGRESSION_20260903_V12.json` 为 `pass`，SHA256=`34889263c9c78daf758bf7fa9a26cb8da8e488f9332a3fbbdcf06fd08b244d7b`：完整 smoke `52 tests / 1 skipped / OK`，关键 Python 编译、表结构、隐私、排名、DTransformer 运行态及 v10/v11 门禁均通过。现行六条离线 cron 和 v12 UKT/FLOPs 后续链由 `outputs/OFFLINE_SCHEDULER_V10_V11_AUDIT_20260903_V3.json` 验证为 `pass`，SHA256=`574c70b9bb14ed56b2be4fa3841bea74b70e3a35a7d99f2c33d03386c9f117d6`。

当前仍只有 DTransformer validation-only 中间态，没有新增五折 final artifact；因此 effectiveness、效率和 Window-AUC 表不虚填，canonical 缺口仍为 `352/592`。228 会按 v10 -> v11 -> v12 的合法终态顺序自动接续；172/127 即使空闲也只允许 CPU/static、哈希、归档和笔记审计。
<!-- CODEX-FINAL-V12-REGRESSION-20260903:END -->


<!-- CODEX-PRIMARY5-EFFICIENCY-V13-20260903:START -->
## 2026-09-03 五数据集效率 v13 自动接续已登记

为继续补齐投稿所需的 `TT / IT / Throughput / Parameters / GPU peak / FLOPs`，已在 228 建立新的不覆盖效率链，范围为 `DKT / SAKT / AKT / SimpleKT × Junyi2015 / ASSIST2015 / NIPS Task 3&4 / ASSIST2017 / Slepemapy`，共 `20` 个 fold0 冻结模型单元。benchmark v2 使用 batch size `64`、sequence length `200`、FP32、`1` 个 warm-up epoch、`5×3` 个计时 epoch、`2` 次推理 warm-up 和 `10` 次推理重复；只构造 one-step test loader，Window-test 未读取，test 指标不参与模型选择。

20 个 config/checkpoint、严格 train/test split 文件均已在 228 重新哈希并通过静态 no-training preflight。19 个单元使用 seed42；Slepemapy/AKT 保留其正式复用 checkpoint 的真实 seed3407，并在 manifest 中显式记录，未改写为 seed42。该 seed 例外只用于架构效率测量，不能产生或改变 effectiveness 排名。

远端原始合同 SHA256：manifest=`e9ce68883e279dbce98b1c1f4a3c5ae7593a73f06f5e5fbcbd9cdace27e307f0`，authorization=`dc6fab4c4443f17086cb7c60fcd89a0bb2b439ffad79b95b8e17886b93dddfc4`，static preflight=`6e5613da15d7c0ef1e608504f7ee979b23560f63ce22a88b17fb742906645011`，queue no-training preflight=`221696c83bd1033bf507939ff29baadf184d8c85d8d5b74274e9ee8ad4511f98`，cron install audit=`04582aca4735cbc46d6e8c5e8c6abd6ea1bd0dac5c6cb2b998ddca6a865d01e5`。公开脱敏镜像目录为 `outputs/primary5_efficiency_contract_20260903_v1/`，镜像审计 SHA256=`30bc7c1882bcb3865b667dc37603f16f72b4f1dc7def3b85fe6ef7ded3f23f5e`。

现行离线拓扑为 v10、v11、v12、v13 各 `@reboot + 每分钟`，共 `8` 条唯一 cron。v13 当前状态必须为 `waiting_v12_terminal`；只有 `DTransformer resume -> 独立 freeze/one-step final -> UKT/Slepemapy validation/freeze/final -> legacy four-model FLOPs` 全部合法终态，且 228 无 compute、显存空闲至少 `16384 MiB`、共享锁可获得、所有 SHA 重算一致时，才串行启动 20 项效率队列。当前 DTransformer 仍在 Junyi2015 fold1 运行，没有被停止、重启或重复派发。

本轮只登记并验证后续效率合同，没有新的 efficiency terminal artifact，因此五折 effectiveness、效率和 Window-AUC 表中的缺失单元仍保持 `—`/`NA`，canonical 缺口仍为 `352/592`。172/127 即使空闲也只允许 CPU/static、哈希、归档和笔记审计，不能执行 KT 或效率测量。
<!-- CODEX-PRIMARY5-EFFICIENCY-V13-20260903:END -->


<!-- CODEX-PRIMARY5-EFFICIENCY-V13-REGRESSION-20260903:START -->
## 2026-09-03 五数据集效率 v13 回归收据

v13 部署后的完整回归为 `pass`：`76 tests / 1 skipped / OK`，关键 Python 编译通过；表结构、隐私、effectiveness 排名和 v10-v13 离线调度均通过。综合收据 `outputs/PUBLICATION_V13_REGRESSION_20260903_V1.json` SHA256=`7408bd3bda86635412b8fd9cb9c7de79da22afe7a0c1d54513265e5a6af3d4a3`；调度审计 `outputs/OFFLINE_SCHEDULER_V10_V13_AUDIT_20260903_V1.json` SHA256=`742491ba977c421509818712c6836eba0f01610768d598d7af0cb42ec2e7025b`；DTransformer 活跃进程身份审计 `outputs/DTRANSFORMER_RESUME_HEALTH_20260903_V8.json` SHA256=`bd746cfc87c3d77044f8af77ffd46383c774935cf5afae14bdd95b856540b7b2`。

当前八条 cron 和 v10 -> v11 -> v12 -> v13 顺序已验证，v13 仍为 `waiting_v12_terminal`，未提前启动 20 项效率测量。表结构审计 SHA256=`04c1391e2f2eccd4d7299e4fec761fe218b882d3f56f5d720da0b3b38b32d440`，隐私审计 SHA256=`0433a48b7b0ee009e535d09343aa55827acd09560191d04068355ab0072b3daf`，排名审计 SHA256=`1f90ae6f8eace294f62f4446eedb7beae59fb12c6a001d7547d84b0cf5ea66fa`。本收据不新增科学数值，缺失效率和 Window-AUC 单元继续为 `—`/`NA`。
<!-- CODEX-PRIMARY5-EFFICIENCY-V13-REGRESSION-20260903:END -->


<!-- CODEX-DKVMN-PRIMARY5-EFFICIENCY-V14-20260903:START -->
## 2026-09-03 DKVMN 五数据集效率 v14 已登记

在 v13 的 20 个经典模型效率单元之后，已追加 DKVMN 在 Junyi2015、ASSIST2015、NIPS Task 3&4、ASSIST2017、Slepemapy 的 `5` 个 fold0/seed42 效率单元。每个 checkpoint/config 均来自已经完成的 DKVMN 五折 validation freeze，且对应独立 freeze 审计和 one-step final pipeline 终态；效率测量仍采用 batch64、length200、FP32、`5×3` 个训练计时 epoch与 `10` 次推理重复，只读取 one-step test loader，不读取 Window-test，也不以 test 指标选择模型。

远端原始 SHA256：manifest=`ccd336491032d5ea28bce697834e77ab0ccf8369642368e610487911d566eac3`，authorization=`b55e13425c9017ddceab2e472b5404653b7b5064f86044d2dbf92e6b01e1a779`，static preflight=`c87a1b4ff6ea9bd1a554b62f0a90e7af2809caafa213b438b01a70e3579d7a8e`，queue no-training preflight=`66da2609e5c7f98fd98d5bcab332046cca65a155ae69991e99593def9ad55f65`，cron install audit=`882bf627885a9d979ae3a16c70085bf615e757d0f1b7edebfedcfcbe3fefe327`。公开脱敏镜像位于 `outputs/dkvmn_primary5_efficiency_contract_20260903_v1/`，镜像审计 SHA256=`717e94d30f52e429786020d5b73bcbbd2548ec75886be32307fe59428d1f7b0d`。

v14 当前状态为 `waiting_v13_terminal`，不会越过 v13；现行拓扑为 v10-v14 共 `10` 条唯一 cron。DTransformer 当前任务继续自然运行，未停止、重启或重复启动。v14 只是静态授权和离线接续，不是效率结果；在 5/5 terminal artifact 完成并独立审计前，表中 DKVMN 对应空格继续保持 `—`/`NA`。
<!-- CODEX-DKVMN-PRIMARY5-EFFICIENCY-V14-20260903:END -->


<!-- CODEX-V14-REGRESSION-V15-DTRANSFORMER-EFFICIENCY-20260903:START -->
## 2026-09-03 v14 完整回归与 DTransformer 效率 v15 接续

v14 完整回归已经闭合：`outputs/PUBLICATION_V14_REGRESSION_20260903_V1.json` 为 `pass`，78 tests / 1 skipped / OK，SHA256=`a06f8dce4bacd8eea29c2d019a958ec291162d9cddd5fa2e8564bacd57b7fb56`。它覆盖表结构 V20、隐私 V21、排名 V21、DTransformer 健康 V10、v10-v14 调度、v13/v14 效率合同和 DKVMN 公开镜像。v13/v14 独立 smoke 的根目录导入顺序依赖也已修复；现在文档中的单模块命令可在干净 Python 进程中直接运行。

在 v14 之后新增 v15：DTransformer × Junyi2015、ASSIST2015、NIPS Task 3&4、ASSIST2017、Slepemapy，共 `5` 个 fold0/seed42 效率任务，可对应回填五数据集效率表中的 `10` 个 TT/峰值显存单元。v15 不复用当前未冻结中间 checkpoint；只有 v14 5/5 terminal 后，才从 DTransformer post-recovery 的五折 validation freeze、独立 freeze audit 和 one-step final 终态现场生成不可覆盖 manifest/authorization，然后运行 no-training preflight。真实效率测量继续要求 228、batch64、length200、FP32、`TORCH_DISABLE_NATIVE_JIT=1`、GPU compute 清零、free memory 至少 16384 MiB 和共享锁可获得。

v15 readiness V2 审计为 `pass`：`outputs/DTRANSFORMER_PRIMARY5_EFFICIENCY_V15_READINESS_AUDIT_20260903_V2.json`，SHA256=`db4e28d4c90a6d8425eb9376be1198857c8afebb015815fd59ef7446c5a3b5b6`；隐私审计也为 `pass`。V2 使合同 materialization、no-training preflight 和真实启动判定可在同一次守卫调用中连续完成；V1 仅保留为被该同轮接续版取代的 provenance。快照中 v15=`waiting_v14_terminal`，execution root/authorization/receipt/terminal 均不存在，v10-v15 共 `12` 条唯一 cron；原 DTransformer 训练继续自然运行，没有被停止、重启或重复启动。当前登记效率范围为 v13 20项 + v14 5项 + v15 5项=`30` 个模型×数据集任务，但尚无新终态效率结果，因此 canonical 缺口仍为 `352/592`，表内继续保持 `—`/`NA`。

SAINT 与 SAINT++ 继续严格区分：现有 Assist2012 五折 final 和效率 artifact 的模型标识是 pyKT `saint`，不能填入专用 `saint_plus_plus.py` 的 SAINT++ 列；后者虽已有 CPU/CUDA optimizer-step smoke，仍需独立 validation/freeze/final 合同。
<!-- CODEX-V14-REGRESSION-V15-DTRANSFORMER-EFFICIENCY-20260903:END -->


<!-- CODEX-SAINTPP-V16-STAGING-20260903:START -->
## 2026-09-03 SAINT++ v16 validation-only 基线接续（staging）

已完成 SAINT++ 专用入口的本地合同与 smoke 验证，范围为 `8` 个数据集 × `5` folds × `3` 个 learning rates（`0.0005 / 0.001 / 0.002`），固定 seed42，共 `120` 个 validation-only 单元。每个单元只使用该 fold 的 train/validation 数据；`test_access=false`、`window_test_access=false`、`final_test_started=false`，未产生任何 effectiveness、Window-AUC 或效率数值，因此主表中的 SAINT++ 缺失格继续保持 `—`/`NA`。

本地 append-only v2 hash-bound 控制文件：trainer=`12c42990ce30de7098f769383d4896bb44e1901058c8f601e0a88639898960fa`，builder wrapper=`33496239ad04a31fc2a48522405c7940a66359a77a813a223515175136631318`，queue runner wrapper=`1ef81309c6dca5fc7b2c3fd4fdcd08b26846b84855ff424cfb1fd1f431dbc1cc`，v16 guard wrapper=`58889c457f78243cc6e74a569f2f0687b97a4ac23aa3e15d071ffe14e98a0178`，cron installer wrapper=`d38968923851c0190b23617be84796bbda518f49a19675c6738794f94b44f03f`。v2 wrappers 同时校验被导入实现文件的 SHA，避免只绑定薄 wrapper。`py_compile` 通过；v16 合同 smoke `2/2` 通过；v13-v15 现有效率链 smoke `8/8` 通过。v2 合同仍为本地 staging；由于 228 SSH 瞬时不可达，尚未替换远端 v1 控制文件或 cron，未启动任何新任务。旧 v1 cron 当前仅记录为等待状态，必须在 v15 terminal 前禁用并完成 v2 hash/cron 审计。

v16 只有在 v15 DTransformer 五数据集效率队列形成合法 `5/5 terminal`、228 compute 清零、显存和共享锁门禁通过后，才允许由离线守卫串行接续；不会在 172/127 执行，也不会把 SAINT（pyKT 标准入口）误记为 SAINT++。<!-- CODEX-SAINTPP-V16-STAGING-20260903:END -->


<!-- CODEX-TABLE-AUDIT-V22-20260903:START -->
最新四份笔记/主表结构审计 `outputs/PUBLICATION_CURRENT_TABLES_AUDIT_20260903_V22.json` 为 `pass`，SHA256=`77c18fd96b86c7930994740495ef01736a5af8bc00d822632f057e2b4688ca6e`：四份镜像一致、模型列和效率字段齐全、MCSKT† 未参与排名、validation 未提升为 final，当前仍有 `352` 个 canonical 缺口。本审计只验证表结构与证据边界，不新增任何结果值。<!-- CODEX-TABLE-AUDIT-V22-20260903:END -->


<!-- CODEX-TABLE-AUDIT-V23-20260903:START -->
命令总账切换为 SAINT++ v16 v2 后，重新执行四份笔记/主表结构审计，`outputs/PUBLICATION_CURRENT_TABLES_AUDIT_20260903_V23.json` 为 `pass`，SHA256=`77c18fd96b86c7930994740495ef01736a5af8bc00d822632f057e2b4688ca6e`。表格结构、四份镜像、MCSKT† 排名隔离和 `352` 个缺口状态保持一致；本次仍未新增科学结果值。<!-- CODEX-TABLE-AUDIT-V23-20260903:END -->


<!-- CODEX-V16-REGRESSION-20260903:START -->
SAINT++ v16 staging 后本地综合回归 `outputs/PUBLICATION_V16_REGRESSION_20260903_V1.json` 为 `pass`，SHA256=`acb61275cda65e48897175b942bc1198c9db51e67a1e76d8fc15067da1d5bd58`：v16 控制文件编译、v16 合同 smoke、v13–v15 效率链 smoke 和当前主表审计均通过。该回归只证明本地 staging/表结构，不代表远端部署或任何新的五折科学结果；主表仍有 `352/592` 个缺口。<!-- CODEX-V16-REGRESSION-20260903:END -->


<!-- CODEX-GAP-SUMMARY-20260903:START -->
当前 canonical 清单为 `592` 个单元，已完成 `240`，缺失 `352`。缺口按执行类型为：validation/freeze `68`、效率测量 `218`、CPU/static operator audit `24`、明确不调度 `42`。机器可读汇总为 `outputs/BASELINE_GAP_SUMMARY_20260903_V1.json`；SAINT++ v16 已登记为下一阶段 `120` 项 validation-only，尚未远端执行。<!-- CODEX-GAP-SUMMARY-20260903:END -->


<!-- CODEX-FULL-SMOKE-20260903:START -->
已执行注册的完整本地出版 smoke 套件（含 SAINT++ v16 新测试），`outputs/PUBLICATION_FULL_SMOKE_20260903_V3.json` 为 `pass`：`82 tests / 1 skipped / OK`，SHA256=`f337dda44884779e7338121679153cdcee13c5830c53c59858ad1559b6b6488f`。本收据仅证明本地编译/单测和表格回归；不产生新的 validation、final-test、Window-AUC、效率或主表数值。<!-- CODEX-FULL-SMOKE-20260903:END -->

<!-- CODEX-OFFLINE-HOST-AND-RESUME-20260904:START -->
## 2026-09-04 228-only 自动补基线状态

172 当前无法通过 SSH 只读核验，因此不能把“没有其他用户进程”当作可用资源证据；并且现行知识追踪政策明确禁止在 172/127 启动训练、validation、test、Window-test 或效率测量。228 在本次核验时 GPU 为 `0% / 41 MiB`、无 compute PID、共享锁可获得，故已在 228 启动已注册的 DTransformer append-only validation resume v2。旧 root 的 `26` 个完整 trial 通过不覆盖 materialization 复用，`test_access=false`、`window_test_access=false`、`a2g_launch=false`。

本次新 guard：`baseline_completion_auto_guard_228_v17_dtransformer_resume_20260904.sh`，远端 SHA256=`11a6b1a5323063cc8b573e3e9de0fb531e39523b61395890b95993cf8b6bdce5`；启动 PID 由远端 launch receipt 记录，运行根为 `<REMOTE_HOME>/kt_baseline_20260723/strict_dtransformer_validation_native_jit_fallback_20260903_resume_v2/`。v18 post-recovery guard 已登记为后续自然接续入口（最新 SHA256=`a77b8c55110f1df0919bc3ea8bcce84638559c7fd18e95e95cc90c0d8ff74b39`），并新增授权 JSON 内容校验；不会越过 v2 terminal。

当前 canonical 表仍为 `592` 个单元，已完成 `240`、缺失 `352`；本条只更新调度与证据状态，不把 validation 中间值写入五折 effectiveness、Window-AUC 或效率主表。完整本地 smoke 已重跑为 `82 tests / 1 skipped / OK`，收据 `outputs/PUBLICATION_FULL_SMOKE_20260904_V1.json` SHA256=`fcf12539053f65d90c29c05431071064b2c5a1349a4673707d5c89346175a882`。
<!-- CODEX-OFFLINE-HOST-AND-RESUME-20260904:END -->

<!-- CODEX-POST-RESUME-AUDIT-20260904:START -->
表格更新后的独立复核：`outputs/CANONICAL_PUBLICATION_GAP_INVENTORY_20260904_V27.json`（592 cells，352 missing，SHA256=`389ee16cea76ab4cfc2fd2368c7d0f7b3007edc074e8267c01c30c92ccc84d11`）与 `outputs/PUBLICATION_CURRENT_TABLES_AUDIT_20260904_V30.json`（`pass`，SHA256=`ab56fe266285e3ddb66dee3f59cf6f2f336cb80e9ea982a8f34a97175ca4c72d`）均已生成。审计确认三张 canonical 表的结构、四份镜像、私密路径、MCSKT† 隔离和空值口径均通过；没有把中间 validation 值提升为五折主表。
<!-- CODEX-POST-RESUME-AUDIT-20260904:END -->

<!-- CODEX-DTRANSFORMER-RESUME-HEALTH-20260904:START -->
最新只读健康审计 `outputs/DTRANSFORMER_RESUME_V2_HEALTH_20260904_V1.json` 为 `pass_active_resume`，SHA256=`0440300bd034cac2240f10c7724e2c235175264f3859267eac7b0f3b50e6aa95`：228 compute PID `169087`，GPU `63% / 16703 MiB`，shared lock 忙；DTransformer resume 已完成 `statics2011 15/15`，当前 `junyi2015` 仍在运行，terminal 尚未生成。v18 正确保持 `waiting_resume_terminal`，未启动 post/final-test。
<!-- CODEX-DTRANSFORMER-RESUME-HEALTH-20260904:END -->

表结构回归在追加本条后重新通过：`outputs/PUBLICATION_CURRENT_TABLES_AUDIT_20260904_V32.json`，SHA256=`85264f8165a82629f41a8b27ea59801e216d1f05f84ad62f0ec43d59aca4f84b`。Smoke 二次复核 `outputs/PUBLICATION_FULL_SMOKE_20260904_V2.json` 为 `82 tests / 1 skipped / OK`，SHA256=`496788735586f26f1e79c61ccd0b071e276b36c898dea7410a979eaaf782bb89`。

<!-- CODEX-SAINTPP-V16-V2-REMOTE-DEPLOYMENT-20260904:START -->
## 2026-09-04 SAINT++ v16 v2 远端部署完成

此前“v16 仍为本地 staging、远端 v1 待替换”的描述已由本条 supersede。SAINT++ v16 v2 的 trainer、builder/runner/guard/installer 实现与 wrapper 共 9 个文件已上传到 228，远端独立 SHA256 与本地锁定值全部一致，`py_compile` 通过。旧 v1 的两条 v16 cron 已精确移除，v2 guard 的 `@reboot + 每分钟` 两条入口均唯一；安装审计 status=`pass_installed_unique`，SHA256=`ceffa823aed38d91cbe8d252f188809a04e8826e7128e545faad53357605558d`。

v16 v2 当前状态仍为 `waiting_v15_terminal`，没有启动 SAINT++、没有新增 GPU compute、没有访问 test/Window-test。它只会在 DTransformer resume/post、UKT/FLOPs、v13-v15 效率链全部形成合法前驱终态，且 228 GPU/显存/共享锁门禁通过后，才启动 `8 datasets × 5 folds × 3 learning rates = 120` 个 validation-only 单元。
<!-- CODEX-SAINTPP-V16-V2-REMOTE-DEPLOYMENT-20260904:END -->

<!-- CODEX-BASELINE-RUNTIME-ACCOUNTING-20260904:START -->
## 2026-09-04 基线未完成原因与真实运行进度

这里明确更正：**表结构完整不等于基线数值完整**。当前三张 canonical 投稿表共有 `592` 个数据单元，只有 `240/592` 已有可准入证据，仍缺 `352/592`。缺口包括 validation/freeze `68`、效率实测 `218`、CPU/static FLOPs `24`；另有 `42` 个 paper-only、协议拒绝或 A2G 外部归属单元，不能由 baseline 队列填充。

228 当前没有闲置：DTransformer validation-only 阶段为 `8 datasets x 3 learning rates x 5 folds = 120` 个完整训练 trial，已完成 `26/120`，第 `27` 个正在运行（`Junyi2015 / lr=0.002 / fold1 / seed42`）。最近快照 GPU 利用率/显存来自本轮 runtime audit；终态尚不存在，禁止重启、重复启动或把 validation 中间值写入五折主表。

Junyi 已完成 11 个 trial 的实测耗时为最短 `11.77` h、中位 `12.98` h、平均 `13.68` h、最长 `21.38` h。按同一数据集的四分位耗时估计，Junyi 剩余约 `1.9-2.3` 天；这只是运行估计，不是截止承诺，其他数据集规模不同，不能线性外推整个队列。

已注册自动链为 `DTransformer validation -> freeze/one-step final -> UKT/Slepemapy + legacy FLOPs -> primary-five efficiency -> DKVMN efficiency -> DTransformer efficiency -> SAINT++ validation`。该链仍**不能自动覆盖全部 352 个缺口**：SAINT++ final、ACE-KT、ASIKT、Mamba4KT、其余效率单元尚需独立 hash-bound 合同；MCSKT 保持 paper-only/never-schedule，A2G 单元由模型任务负责。因此主表空值保留是证据门禁的结果，不是允许手填或用 smoke/pilot 顶替。

机器清单：`outputs/BASELINE_RUNTIME_ACCOUNTING_20260904_V1.json`，SHA256=`a62979fe7424c41624088699a83787309795e6ffc2dc577bf4c67d4d917c7761`。KT 继续严格 `228-only`；127/172 禁止执行。
<!-- CODEX-BASELINE-RUNTIME-ACCOUNTING-20260904:END -->

<!-- CODEX-SAINTPP-V19-POST-20260904:START -->
## 2026-09-04 SAINT++ v19 独立冻结与 one-step final 自动接续

SAINT++ 自动链已从“只完成 validation”补强为：v16 `120/120` validation terminal -> 按五折 mean validation AUC/ACC 冻结每个数据集唯一 LR -> 独立重算选择并 CPU 反序列化 `15/15` candidate checkpoints -> 生成 pre-test authorization -> `8 datasets x 5 folds = 40` 个 one-step final -> 汇总 AUC、ACC、ECE-15、Brier、NLL。Window-test 继续禁止，test 结果不得回流改变冻结 profile。

v19 已在 228 no-clobber 部署并安装唯一 `@reboot + 每分钟` 入口；远端 SHA、只读模式、Python/Bash 语法、授权内容和 cron 均通过。当前状态为 `waiting_v16_terminal`，launch receipt/result 均不存在，因此没有抢占当前 DTransformer，也没有新增 GPU 进程。v16 合法终态、GPU compute 为空、free memory >=16384 MiB 且共享锁空闲后才可启动。

完成后可补 SAINT++ 八数据集 AUC/ACC 共 `16` 个 canonical effectiveness 单元；在真实 terminal 和独立聚合出现前，主表继续保留 `—`。部署审计：`outputs/SAINT_PLUS_PLUS_V19_REMOTE_DEPLOYMENT_AUDIT_20260904_V1.json`，SHA256=`1cdab02eacdb5f8bdff7de9dbf50ce1899e5c41e4ef61eaa8e619fa990edcfab`。
<!-- CODEX-SAINTPP-V19-POST-20260904:END -->

<!-- CODEX-V19-REGISTRY-REGRESSION-20260904:START -->
## 2026-09-04 v19 后当前缺口 registry 与完整 smoke

canonical 投稿表仍为 `240/592` 已完成、`352/592` 缺失；本轮没有把 staged 状态误计为结果。v19 部署后，分类已按真实执行能力更新：同协议效率实测 `218`、active/hash-staged `32`、source/adapter blocked `20`、effectiveness runner required `16`、operator FLOPs `24`、外部归属 `23`、paper-only/protocol rejected `19`。SAINT++ 的 16 个 effectiveness 缺口已从 source/adapter blocked 移至 v16->v19 active/hash-staged，但只有真实终态后才填表。

完整本地 smoke 已纳入 SAINT++ v19：`87 tests / 1 skipped / pass`，收据 `outputs/PUBLICATION_FULL_SMOKE_20260904_V4.json`，SHA256=`91d683894a6f60e6138f7713283f01bc21705c9e9362d5634ded1ed531de6325`。表结构、四镜像、隐私和排名审计均通过；MCSKT 仍不参与 reproduced-model 排名，validation/smoke/paper-only 仍不能进入正式五折数值。

当前机器清单：classification `outputs/CANONICAL_PUBLICATION_GAP_CLASSIFICATION_20260904_V28.json` SHA256=`8e9808197a993fcb55506e3097e59b12c3aaec51fb1c1ede5ffc1c59f201a053`；completion registry `outputs/BASELINE_COMPLETION_REGISTRY_20260904_V28.json` SHA256=`f75824c6589a3d5f8cbbd928cf3736712cc00e600d0d044c498242215d1677f2`；priority registry `outputs/BASELINE_GAP_PRIORITY_REGISTRY_20260904_V2.json` SHA256=`89d82758380d2d833d70e122e31ac952afe7c084608769a9d2321ad529f57b8d`。
<!-- CODEX-V19-REGISTRY-REGRESSION-20260904:END -->

<!-- CODEX-ASIKT-V20-V21-20260904:START -->
## 2026-09-04 ASIKT v20/v21 正式五折链已登记

ASIKT 已从旧 three-dataset fold0 pilot 升级为正式 strict 链：8 数据集 x 5 folds x 3 个预注册学习率 (`2.5e-4/5e-4/1e-3`) = `120` 个 validation-only trials，seed=`42`、batch=`48`、最多 `300` epochs。学习率网格围绕作者默认 `5e-4` 预先固定；旧 pilot 已访问的 test/Window 指标不参与本次选择。每库只按五折 mean validation AUC、再 ACC 冻结唯一学习率，随后独立复算选择并反序列化 15 个 checkpoint，最后才授权 5-fold one-step final，并汇总 AUC、ACC、ECE-15、Brier、NLL；Window-test 继续关闭。

v20/v21 已在 228 no-clobber 部署并安装唯一 `@reboot + 每分钟` 入口。当前 v20=`waiting_v19_terminal`、v21=`waiting_v20_terminal`，没有 ASIKT process、trial 或 final result，未抢占当前 DTransformer。v19 合法终态、GPU compute 为空、free memory >=16384 MiB 且共享锁空闲后，v20 才启动；v20 `120/120` 后 v21 才可冻结并访问 one-step final。

当前 canonical 数值仍为 `240/592`、缺口仍为 `352`，因为本轮没有冒充生成结果；但其中 `14` 个 ASIKT effectiveness 缺口已从 source/adapter blocker 转为 hash-staged 自动链。分类计数更新为 active/hash-staged=`46`、source/adapter=`6`。远端部署审计 SHA256=`37810032ac9cacb3ab2009fee4fe9259b149a58e0a05451931dfaca3ab1dcc97`；完整 smoke=`93 tests / 1 skipped / pass`，SHA256=`b59d946c04b7e1eb6e35db59d5b51708ce4866b36ca2e41f1f94e832ddbd5388`。
<!-- CODEX-ASIKT-V20-V21-20260904:END -->

<!-- CODEX-ACEKT-V22-V23-20260904:START -->
## 2026-09-04 ACE-KT v22/v23 三库补缺链已登记

ACE-KT 已有五库正式证据，因此新链只补主表尚缺的 Junyi2015、ASSIST2012、Slepemapy：3 数据集 x 5 folds x 3 个预注册学习率 (`5e-5/1e-4/2e-4`) = `45` 个 validation-only trials，seed=`42`、batch=`64`、最多 `200` epochs。学习率网格围绕作者默认 `1e-4` 固定；batch 64 是三库一致的显存安全调整。每库按五折 mean validation AUC、再 ACC 冻结唯一 LR，独立复算选择并反序列化 15 个 checkpoint，随后才允许 15-fold one-step final；汇总 AUC、ACC、ECE-15、Brier、NLL，Window-test 关闭。

v22/v23 已在 228 no-clobber 部署并安装唯一 `@reboot + 每分钟` 入口。当前 v22=`waiting_v21_terminal`、v23=`waiting_v22_terminal`，没有 ACE-KT process、trial 或 final result，未抢占当前 DTransformer。ASIKT v21 合法终态、GPU compute 为空、free memory >=16384 MiB 且共享锁空闲后，v22 才启动；v22 `45/45` 后 v23 才可冻结并访问 one-step final。

当前 canonical 数值仍为 `240/592`、缺口仍为 `352`，因为本轮没有把 staged 状态冒充结果；其中 `6` 个 ACE-KT effectiveness 缺口已从 source/adapter blocker 转为 hash-staged 自动链。分类计数更新为 active/hash-staged=`52`、source/adapter=`0`。远端部署审计 SHA256=`6d0d223d46540fc3af5498da4e0b08c3b403f2221367246106371973e2869572`；完整 smoke=`99 tests / 1 skipped / pass`，SHA256=`f61eb9810c515c4846c8dd95a2126290c73bc0e76c32911a3a22d1cd838ce655`。
<!-- CODEX-ACEKT-V22-V23-20260904:END -->

<!-- CODEX-MAMBA4KT-V24-V25-20260904:START -->
## 2026-09-04 基线长期未补完的原因与 Mamba4KT v24/v25 接续

当前投稿表的结构已经完整，但可准入数值仍只有 `240/592`，缺失 `352/592`。长期不变的直接原因是队首 DTransformer 使用了 `8 datasets x 5 folds x 3 learning rates = 120` 个完整 validation trials，并在单 GPU 上串行执行；当前复用/完成 `26/120`，第 27 个 Junyi2015 trial 仍在运行，terminal 尚未生成。Junyi 已完成 trial 的中位耗时约 `12.98 h`，因此该协议本身就是数周量级，后续 SAINT++、ASIKT、ACE-KT、Mamba4KT 与效率测量都会等待它的合法终态。validation 中间值不能写入五折 final 主表，这就是表格数值没有随运行逐项增长的原因。

Mamba4KT 已完成 append-only v2 修正。旧 v1 在 no-training preflight 暴露统一 venv 缺少 `mamba_ssm`，因此保留为 `provenance_only_environment_binding_failed_before_execution`，未训练、未访问 test。v2 固定专用 Torch `2.10.0+cu128` / mamba-ssm `2.3.2.post1` 环境，并补齐 scikit-learn `1.7.2`；环境锁、8 库 `120` 行 validation manifest、queue no-training preflight、动态 batch 规则、v24/v25 cron 与 `40` 个冻结后 one-step final 合同均通过独立部署审计。v24 当前必须为 `waiting_v23_terminal`，v25 为 `waiting_v24_terminal`；没有 Mamba4KT process/trial/final，Window-test 关闭。

因此 Mamba4KT 的 `16` 个 effectiveness 空格只从“缺 runner”改为“active/hash-staged”，没有写入数值。最新缺口分类为：效率实测 `218`、active/hash-staged `68`、operator-complete FLOPs `24`、A2G 外部归属 `23`、MCSKT paper-only/协议拒绝 `19`。总缺口仍为 `352`。

证据：部署审计 `outputs/MAMBA4KT_V24_V25_REMOTE_DEPLOYMENT_AUDIT_20260904_V2.json` SHA256=`7022e62be468d6997191fa76f4a7a08cadb7c47b24c0d7b158357d0167342a66`；完整 smoke `105 tests / 1 skipped / pass`，SHA256=`7b1dea26f11f07274f9d0472f1eaa196545334a79fa41b4e914433a4c8db0e62`；DTransformer 健康收据 SHA256=`74785e2a9a4b8eb44370774fbfddcad1d6a43c44cead0de107cec79a3cbeec8f`；classification/registry/priority SHA256 分别为 `f496282c24f475b17ca9c07e9a52757f338989914cd401ebfcd3c759829fce72`、`e3cbde9ba39691a1c49bfb266c7b3c8e8e7bc3c48e1bcdd0c5a0196b7af991f6`、`a87f53b6662f603e5dd94ddac56a7e07bd1dc94e1dd9253efaf288f370642b0e`。KT 继续只允许在 228；127/172 禁止。
<!-- CODEX-MAMBA4KT-V24-V25-20260904:END -->

<!-- CODEX-MAMBA4KT-EFFICIENCY-V26-20260904:START -->
## 2026-09-04 Mamba4KT 效率 v26/v27 已接入离线链

当前表结构完整，但正式结果仍为 `240/592`，缺口 `352/592`；本次没有把 validation、smoke 或 staged 状态写成结果。228 当前仍由 DTransformer validation-only 占用，v26 只会在 v25 的 8 数据集五折冻结与 one-step final 合法终态之后运行。

v26 已登记 Mamba4KT 在 Junyi2015、ASSIST2015、NIPS Task 3&4、ASSIST2017、Slepemapy 的 5 个 fold0/seed42 效率任务；v27 在 v26 终态后补 Assist2012/fold0 正文 Overall efficiency。两阶段均固定 batch64、length200、FP32，测量 TT、IT、吞吐、总/可训练参数和训练峰值显存；FLOPs 在 operator-complete 前继续为 `NA`。只允许冻结后用 one-step test loader 做计时，test 指标不参与选择，Window-test 关闭。

v27 远端控制文件、只读权限、`4/4` 单测、唯一两个 cron 和 `waiting_v26_terminal` 状态均通过独立审计。现行离线链为 16 个阶段、32 条 cron；v27 部署审计 SHA256=`b0ee222fe949687f72e867b4bffd9d49228b261fcf61bb1384d752c3b51cafee`，连续链审计 SHA256=`99de33d9fcfdad4affc91291938214084ab9907db383f3ea7df3acffcc0061ce`，完整 publication smoke 为 `113 tests / 1 skipped / pass`，SHA256=`f6a6f48b504683eaec377cffe55c2242212dd4abdd79fd8a7f4e6145a072883f`。KT 仍只允许在 228；127/172 禁止。
<!-- CODEX-MAMBA4KT-EFFICIENCY-V26-20260904:END -->

<!-- CODEX-SUBMISSION-CORE-EFFICIENCY-V28-20260904:START -->
## 2026-09-04 submission-core 效率 v28 接入离线补缺链

本次只同步可追溯的调度与审计状态，不新增科学数值。canonical 投稿三张表仍为 `240/592` 已完成、`352/592` 缺失；validation、smoke、staged 或 paper-only 值不得写入五折主表。当前 228 的 DTransformer validation-only 队列仍在运行，v28 没有抢占或重复启动。

v28 登记了 `DTransformer/Assist2012` 1 行、`SAINT++/Assist2012` 1 行、`ASIKT` 六库 6 行和 `ACE-KT` 三库 3 行，共 `11` 个正文 Overall efficiency 单元。每行要求已有合法 post-terminal、同一 `ONE_STEP_FINAL_AUTHORIZATION.json` 绑定、fold0/seed42，并测量 TT、IT、throughput、总/可训练参数和峰值显存；FLOPs 仍须独立 operator-complete 审计。test 不参与选择，Window-test 关闭。

v28 当前状态为 `waiting_v27_terminal`，仅在 v27 `Assist2012` 效率终态、228 无 compute、显存 free >= `16384 MiB`、共享锁可获得且所有 SHA 重算一致后启动。远端部署审计 SHA256=`893a1f27782486be4c6f85637e1f4b4299a0a617a00d07c531bc45645b25b429`；v17->v28 连续链审计 SHA256=`45a3f1565a83cbcc788ea5387f9c6741992aacb7e044a4ed66bf107e9ce8af72`；完整 publication smoke 为 `117 tests / 1 skipped / pass`，SHA256=`2602791759c6cfbdb3e201928d655b3b84a6108c60295bf2544c0b69e5f1794c`。KT 仍只允许在 228，127/172 禁止。
<!-- CODEX-SUBMISSION-CORE-EFFICIENCY-V28-20260904:END -->

<!-- CODEX-RUNTIME-CANDIDATE-AUDIT-20260904:START -->
## 2026-09-05 实时运行、耗时解释与效率候选完整性审计

当前 228 的 DTransformer validation-only 队列仍在有效计算：`27/120` trials 已完成，第 `28` 项为 `junyi2015 / fold2 / lr=0.002 / seed42`，该 trial 已运行约 `12.23` 小时；GPU 仍由该进程占用，terminal 尚未生成。本次未停止、重启或复制该进程。最新运行收据为 `outputs/BASELINE_RUNTIME_ACCOUNTING_20260905_V4.json`，SHA256=`91013318a022da9e0968e0303cac38da22576327f2918b249368b35ab4346357`。

长期未补完不是 GPU 空闲，而是当前协议把 DTransformer 扩成 `8 datasets x 5 folds x 3 learning rates = 120` 个完整训练 trial；Junyi 已完成项中位耗时约 `12.96` 小时。按各 strict train/validation 文件字节数缩放的纯运行估计，DTransformer validation terminal 尚需约 `147.7` 小时（四分位情景约 `139.4–162.8` 小时）；这是运行规划估计，不是截止承诺，也不包含后续模型。中间 validation 值不能写入五折 final 主表，因此表格不会按每个 trial 逐格增长。

下一批正文效率项的只读候选扫描只发现 `7/10` 个可定位的 fold0 checkpoint/config：Junyi2015 的 UKT/ACE-KT、ASSIST2015 的 UKT/ACE-KT、NIPS Task 3&4 的 UKT/ACE-KT、Slepemapy 的 UKT。`assist2017:ukt`、`assist2017:acekt_author`、`slepemapy:acekt_author` 没有可加载的正式 checkpoint；旧的失败日志或汇总不能替代 checkpoint，因此这三项继续保持空值，不会被错误测量或填表。候选审计 `outputs/UKT_ACE_EFFICIENCY_CANDIDATES_20260904_V2.private.json` SHA256=`788075bd5978056e404731920fb024c33cdce606adbe2bc1f103fe56bbde4b4d`。

v17->v28 离线链已通过 `17 stages / 34 cron entries` 连续性审计；每个守卫均由 `@reboot + 每分钟` 触发，当前前驱未完成的阶段只等待，不会抢占或重复启动。v28 仍为 `waiting_v27_terminal`，只登记 `11` 个正文效率单元，尚无新增效率数值。v28 部署审计 SHA256=`893a1f27782486be4c6f85637e1f4b4299a0a617a00d07c531bc45645b25b429`，连续链审计 SHA256=`45a3f1565a83cbcc788ea5387f9c6741992aacb7e044a4ed66bf107e9ce8af72`，完整 smoke 为 `117 tests / 1 skipped / pass`，SHA256=`c9e461d68c9202e707ad480da2685ce61b5668810427c310c5a451de0346cdfd`。KT 只允许在 228；127/172 禁止。

canonical 投稿表当前仍为 `240/592` 已完成、`352/592` 缺失；本段只更新运行和候选状态，不改变任何 canonical 数值。
<!-- CODEX-RUNTIME-CANDIDATE-AUDIT-20260904:END -->

<!-- CODEX-A2G-INTERNAL-MODULE-BLOCKING-REAUDIT-20260904:START -->
## 2026-09-04 A2G 内部模块 5/5 阻塞复审（只读，无 GPU）

本段只记录 A2G 内部模块消融的证据边界与阻塞原因，不向五折 effectiveness、Window-AUC 或效率主表写入任何数值，也不修改任何既有 JSON、旧 root 或历史结论。A2G 的模块证据口径为 `fold0 / seed42 / validation-only`、`test_access=false`、`window_test_access=false`，与本笔记的五折主表口径不同，两者不得混用。

正式内部模块 matched `Full - ablation` 达到严格 `5/5`（`ΔAUC > 0.001` 且 `ΔACC >= -0.0005`）的仍然只有 `Selective-SSM recurrence` 一个。其余七项声明模块的最好在案结果分别为：`causal_attention_postnorm_ffn` 最高 `1/3`（Assist2017 `ΔAUC=+0.014403`，Assist2015 `+0.000555`、NIPS `+0.000534` 均未过门）；`split_boundary_read_modulation` Assist2015 `ΔAUC=+0.000166`；`embedding_response_semantics` 最好 Assist2015 `+0.000445`；`direct_prior_stat_residual_head` Assist2015 `-0.0000167`；`causal_statistics_rwce` 全为负；`item_residual_dropout` 在 concept-only Assist2015 上 `n_pid=0`、前向零调用，效应恒为零，普适五数据集门禁不可满足；`external_logit_composite_a2g_slot` 的 `5/5` 是固定 logit 组合的 leave-one-out，不是内部消融，不计入。Selective-SSM 的 `5/5` 本身是架构条件性的：Assist2015 单元用 concept-only、`item_residual_dropout=0.0` 的 Full，另外四个用 `0.4`，且 `strict_single_checkpoint_claim=false`。

基线覆盖仍为 `49/104`，缺 `55` 个 cell。本次逐单元分类结果为：`50` 个完全没有 artifact 痕迹（从未训练），`5` 个有 artifact 但因 `terminal_train_log_missing` / `checkpoint_missing_or_non_unique` / `interrupt_snapshot` 不可提升，`0` 个可在不新增计算的前提下补齐。因此不存在可直接收割的合法 checkpoint/prediction/package，未占用任何 GPU。`cskt`、`deep_irt`、`fluckt` 三个模型在全部 8 个数据集上均无有效记录。附带修正一处枚举遗漏：既有 gap 审计的标题数 `55` 正确（`104-49`），但 `missing_by_dataset` 只列出 `54` 项，缺 `statics2011/folibikt`；该修正以 append-only 形式记录，未改动原 JSON。

本次新增的独立验证：Assist2015 零-ID 与 padding 碰撞的暴露量已用 registry 绑定的本地 split（SHA256=`86a384290579e3b6719c5bf38d78468b478331fee677d1ee4a2895b38fd2752f`）在本地 CPU 复现，fold0 为 `2306 / 105831`（`2.179%`）、全折 `11625 / 544331`，与远端审计零差异；该候选的 CPU/RNG 契约与静态发射审计也在隔离副本中位级复现（SHA 分别为 `f6df3c92…` 与 `f05652dc…`，与绑定值完全一致，`17/17` 检查通过）。但该路线并非未启动：它已于 `2026-08-27T04:23:01+08:00` 远端终止，`ΔAUC=+0.000797384`、`ΔACC=+0.000827259`，AUC 未过严格门，已按 no-retry 关闭，仅保留 correctness 结论。

阻塞原因有三，且均需外部输入才能解除：其一，在冻结的 Assist2015 输入契约与 `149` 行 closure ledger 下已无机械上不同的候选（`134` 个来源、未覆盖数为 `0`；Assist2015 仅有 `user_id / log_id / sequence_id / correct` 四列可观测）。其二，没有既可达又被允许的 GPU 主机：本次核验 172 的 ICMP `2/2` 丢包、SSH 22 连接超时，无法只读证明“无他人 compute”，且现行策略禁止在 172/127 启动任何训练或测量；228 由本任务明确禁止，且另有他人的 DTransformer 队列在算。其三，`55` 个缺失基线 cell 全部需要新训练。因此本次未发起任何首筛、未重试、未网格、未改名重开、未跨数据集自动扩展，`objective_achieved=false`。

两处仍未闭合的 provenance 缺口一并登记：role-complementary 的 `nips_task34` 终态审计被 `A2G_OBJECTIVE_COMPLETION_REQUIREMENT_MATRIX_20260902.json` 引用（summary SHA256=`eca0b0cc…`）但文件在两个 outputs 目录均不存在；`role_complementary_slepemapy_no_attention_retry_20260902_v2_172` 于 `2026-09-02T17:26:50+08:00` 启动后因 22 端口不可达而远端状态未知，本次未重启、未复制、未写成成功或失败。该 role-complementary 属于另一架构族，其 Full 在 Assist2009/2017、NIPS、Slepemapy 上低于当前 Full，即使补齐也不能合并成同模型完成声明。

本段证据：`A2G_INTERNAL_MODULE_5_OF_5_BLOCKING_REAUDIT_20260904.json` SHA256=`44be1e5a4ccedad93e83ab58603ed1c961c5a289f6bf8f87c6d36295e2174ed7`；`A2G_MISSING_BASELINE_CELL_CLASSIFICATION_20260901.json` SHA256=`18eb5965e27df0a9b185996675f8c21a53100c13c304541b718c5f5a72cf2f96`；`A2G_ZERO_ID_COLLISION_LOCAL_REVERIFICATION_ASSIST2015_20260901.json` SHA256=`04c6b0b054e74644a1638e602d2d91a4465ce9dd14777bee6cee31b8d2f05dc2`。上游权威终态仍为 `A2G_FORMAL_INTERNAL_MODULE_5_OF_5_BLOCKED_AUDIT_20260902.json` 与 `A2G_OBJECTIVE_COMPLETION_REQUIREMENT_MATRIX_20260902.json`，二者已记录 `objective_achieved=false`，本段不取代它们。
<!-- CODEX-A2G-INTERNAL-MODULE-BLOCKING-REAUDIT-20260904:END -->

<!-- CODEX-UKT-ACE-EFFICIENCY-V29-V30-20260905:START -->
## 2026-09-05 UKT / ACE-KT 效率 checkpoint 恢复与五主库实测链

当前正文表的结构已经完整，但正式证据仍为 `240/592` 个单元，缺口为 `352/592`。最新只读运行核验显示，228 上 DTransformer validation-only 已完成 `28/120` 个 trial，当前运行 `junyi2015 / fold3 / lr=0.002 / seed42`；GPU 快照约为 `94% / 16841 MiB`。按现有已完成 trial 与各库数据规模估算，DTransformer validation terminal 仍约需 `145.9` 小时；这是运行估计，不是完成承诺，也不包含后续模型。中间 validation、smoke 或 staged 状态继续不得填入正式五折表。

为补齐 UKT/ACE-KT 五主库效率缺口，已在 228 以 append-only 方式登记 v29/v30。v29 仅恢复 7 个缺失的效率专用 fold0/seed42 checkpoint：UKT 的 Junyi2015、ASSIST2015、NIPS Task 3&4、ASSIST2017，以及 ACE-KT 的 ASSIST2015、NIPS Task 3&4、ASSIST2017；每项只训练一轮，`test_access=false`、`window_test_access=false`、`effectiveness_eligible=false`，不得用来填五折 effectiveness。v30 在 v29 `7/7` 后统一测量 UKT/ACE-KT 在 Junyi2015、ASSIST2015、NIPS Task 3&4、ASSIST2017、Slepemapy 共 10 行效率，协议固定为 batch64、sequence length 200、FP32、同一 RTX 4090 D，报告 TT、IT、throughput、总/可训练参数和峰值显存；FLOPs 未通过 operator-complete 审计前保持 `NA`。

v29 当前为 `waiting_v28_terminal`，v30 为 `waiting_v29_terminal`；每阶段各有唯一 `@reboot + 每分钟` 两个守卫入口，离线链现为 `19 stages / 38 cron entries`。两阶段均未创建 execution root、未启动 GPU，也未抢占 DTransformer。v29 旧 v1 因 `pscan.py` 预期 SHA 抄写错误被静态门禁拦截，未执行且未登记 cron，只保留 provenance；当前唯一可执行绑定为非覆盖 v2。

证据：v29/v30 部署审计 `outputs/UKT_ACE_V29_V30_REMOTE_DEPLOYMENT_AUDIT_20260905_V2.json`，SHA256=`bb866b4200bde64d5cdf8ff9280261ea763deda210b7cfca94a60892c93993bf`；v17-v30 连续链审计 `outputs/OFFLINE_BASELINE_CHAIN_V17_V30_AUDIT_20260905_V1.json`，SHA256=`ac320619df0f4153c7bb47f93d21628bf9742a22b71c6e84e0e59732b1a39f56`；完整 smoke `124 tests / 1 skipped / pass`，收据 `outputs/PUBLICATION_FULL_SMOKE_20260905_V13.json`，SHA256=`fdd4925e60207125721ccc32c6e4d5db862f9ef03faa82f32abaffcda78be003`；最新运行核账 `outputs/BASELINE_RUNTIME_ACCOUNTING_20260905_V5.json`，SHA256=`18a1435d7a94936fe41c8cba08177465341c0fe53d670f5701fddf20df4f701e`。KT 仍只允许在 228；127/172 禁止。
<!-- CODEX-UKT-ACE-EFFICIENCY-V29-V30-20260905:END -->

<!-- CODEX-V31-V32-GAP-COVERAGE-20260905:START -->
## 2026-09-05 数值未补齐原因与 v31/v32 追加覆盖

当前三张 canonical 投稿表的**结构已完整，但数值证据尚未完整**：总计 `592` 个单元，已准入 `240`，仍缺 `352`。对每个空格反查执行阶段后，原 v17-v30 链只覆盖其中 `202` 个；新增 v31/v32 后，已有未来生产阶段的空格增至 `248`，仍有 `62` 个可调度空格没有 producer；另有 A2G 外部归属 `23` 个、MCSKT paper-only/协议拒绝 `19` 个，不能由本 baseline 队列填充。机器清单为 `outputs/STAGE_TO_CANONICAL_GAP_COVERAGE_20260905_V2.json`，SHA256=`0bb81b923a5e56b747cc7c688d431934e4bfa37472545adf62ba5e601857ed42`。

v31 只为 DKT-Forget、DeepIRT、SparseKT、CSKT、FoLiBiKT、QIKT、DenoiseKT 建立 Assist2012/fold0/seed42 的效率专用初始化 checkpoint，并执行单 batch CUDA 构造/前反向 smoke；`training_epochs=0`、`effectiveness_eligible=false`、不访问 test/Window。v32 随后统一测量 DKT+ 加上述七模型的 TT、IT、throughput、总/可训练参数和峰值显存，共可覆盖 `46` 个 canonical 单元；FLOPs 在 operator-complete 审计前继续为 `NA`。历史扫描的 14 个缺失神经模型中，只有 DKT+ 存在合格 seed42 checkpoint；DenoiseKT 仅有 seed3407，其余均没有合格 checkpoint，因此不能直接量测或手填。候选扫描 SHA256=`ac278444bc4349cc6196eb205d5b9942a8c009059023617aca2264926888bdfc`。

v31=`waiting_v30_terminal`，v32=`waiting_v31_terminal`，尚未创建执行根或新增科学数值。离线链当前为 `21 stages / 42 cron entries`，连续性审计 SHA256=`dbcb1e3962e6003d00a088404b9d3b1945c8e3cdb0257e2d97fb4e09b4d72405`，部署审计 SHA256=`958c1462b5b07c657836ed10dfea6ebc5a51dd72e58e2adcdfaabe0a479db59d`。剩余 `62` 项为：BKT、LPKT、DIMKT、PKT、MIKT、FlucKT、MCKT 的 `42` 个效率字段，以及 `20` 个 operator-complete FLOPs；其中 LPKT/DIMKT 的现成资源因 test-inclusive vocabulary/Q-matrix 或 validation-response difficulty 泄漏被拒绝，PKT/MIKT/MCKT 尚无本地 adapter，不能以不合规结果补表。

228 当前仍在 DTransformer validation-only：`28/120` 完成，正在 `junyi2015 / fold3 / lr=0.002 / seed42`；按最新实测估算，仅该 validation 阶段还约需 `144.3` 小时。中间 validation、smoke、staged 合同或 paper-only 数字均不得写成五折 final，因此本段同步不改变 canonical 数值。最新运行收据 `outputs/BASELINE_RUNTIME_ACCOUNTING_20260905_V7.json` SHA256=`c7948067ca19bc362af5e63da1f5ca247fa1d58d1fdcdc660ce17765111d0db2`；KT 继续只允许在 228，127/172 禁止。
<!-- CODEX-V31-V32-GAP-COVERAGE-20260905:END -->

<!-- CODEX-V33-V36-RUNTIME-BLOCKERS-20260905:START -->
## 2026-09-05 长期未补齐原因、v33-v36 与剩余 blocker（superseding）

这里区分两件事：投稿表的列、行和空值状态已经完整；可审计数值仍只有 `240/592`，缺 `352`。最新只读核账显示 DTransformer validation-only 为 `28/120`，当前是 `junyi2015 / fold3 / lr=0.002 / seed42`，GPU 快照为 `0, 94, 16842, 7266, 24564`。按已完成 trial 与数据规模估算，仅该 validation 阶段仍约 `141.7` 小时；这不包括后续 SAINT++、ASIKT、ACE-KT、Mamba4KT 与效率阶段。运行核账：`outputs/BASELINE_RUNTIME_ACCOUNTING_20260905_V9.json`，SHA256=`6b45fb5d6dbea74ca29c17317b4b98e4830e4ed9b9c7a851dcac24d41e309cd4`。

造成周期过长的直接原因是当前计划把多个强基线各自展开为 `8 datasets x 3 learning rates x 5 folds` 的完整训练网格，并在 228 单卡串行；其次，前期投入了较多时间构建调度/审计链。当前 DTransformer 没有停止、重启或重复启动，中间 validation 不能填成五折 one-step final。若保持这套穷举协议，完成周期应按周而不是按天估计。

本轮未新增数值，但补齐了后续合法 producer：FlucKT v33、operator-complete FLOPs v34、LPKT/DIMKT 效率 v35 与其 FLOPs v36 均已静态部署并等待前驱。LPKT/DIMKT 的状态不再是“严格预处理阻断”：train-only time vocab、Q-matrix 与 difficulty 资源已通过独立审计（`outputs/LPKT_DIMKT_TRAIN_ONLY_RESOURCES_REMOTE_AUDIT_20260905_V3.json`，SHA256=`47dafb72ec5b3aa73206c57554f8a41d8fa57703174d00091a349f1f09ead83f`）；v35/v36 仍未执行，因此 TT、IT、throughput、参数、GPU peak 与 FLOPs 继续保持 `—`/`NA`。v35 SHA256=`607caae7d5b1917b419af08758db73ec5ca86d34e5eb2a75676f6f98850e281e`；v36 SHA256=`71aa534c4efda22ad98e65455626c2ee21986b87a5df8432925ad69134887245`。

当前 352 个空格中，`282` 个已有注册 producer；A2G 外部归属 `23` 个，MCSKT paper-only/协议拒绝 `19` 个。旧账剩余的 `28` 个“可调度”格已重审：BKT 的 7 项应使用单独 CPU 协议或 N/A，不能伪装成 RTX 4090 神经模型效率；PKT/MIKT 共 14 项缺 hash-bound 作者实现/adapter/profile；MCKT 7 项虽有作者仓和 commit，但仍有 learner split、difficulty、device/path 与 content-embedding provenance 五类 blocker。因此这 28 项当前 GPU 可执行数为 `0`，详见 `outputs/REMAINING_EFFICIENCY_BLOCKER_DISPOSITION_20260905_V1.json`，SHA256=`334ecd2bb7badcd7b97d6bb8680888c7fabe8518cdb60c5473ae9582c097d48f`。

离线链现有 `25` stages / `50` cron entries，连续性审计 SHA256=`2e5f49343cf2ad3caa946d0c92e7ffb633114eb4182f080f922bd891e0f2da18`；producer 覆盖审计 SHA256=`5f830501ece198990a2d7c68f880a745e28df16a514a1901a1b87331a064e5b7`；扩展后的完整本地 smoke 为 `166 tests / 1 skipped / pass`，SHA256=`0c4e7962d98cf65459fdec3080142b8183dd78675ab991fa0354f4364b339a56`。这些都只证明调度与实现准备，不等于结果已经生成。KT 仍为 228-only；127/172 禁止。
<!-- CODEX-V33-V36-RUNTIME-BLOCKERS-20260905:END -->

<!-- CODEX-RUNTIME-ROOT-CAUSE-PKT-MIKT-SMOKE-20260905:START -->
## 2026-09-05 数值长期未补齐的根因与 PKT/MIKT 新进展

当前投稿表的列、行和空值口径已经完整，但可准入数值仍为 `240/592`，缺失 `352/592`。其中 `68` 个需 validation/freeze，`218` 个需同协议效率测量，`24` 个需 operator-complete FLOPs；另有 `42` 个属于 paper-only、协议拒绝或 A2G 外部归属，应保留 `—/NA`，不能以“补全”为由制造数值。

最新独立只读核账确认，228 并未空闲：DTransformer validation-only 已完成 `28/120`，正在运行第 `29` 项 `Junyi2015 / fold3 / lr=0.002 / seed42`；采样时 GPU 利用率 `76%`、显存 `16842 MiB`，test/Window-test 均未访问。当前阶段采用 `8 datasets × 3 learning rates × 5 folds` 的完整网格，Junyi 已完成 trial 的中位耗时约 `12.95 h`；按数据规模加权估计，仅 DTransformer validation terminal 仍约需 `140.9 h`（约 `5.9` 天），且不含后续 SAINT++、ASIKT、ACE-KT、Mamba4KT 和效率阶段。机器收据 `outputs/BASELINE_RUNTIME_ACCOUNTING_20260905_V11.json`，SHA256=`a24fa1b183831571986658a11f4fca11933481c91fd83f3920831009595ade06`。

这也明确暴露出此前调度过重：把每个 learning rate 都跑满五折使单模型达到 `120` 次完整训练，且 228 单 GPU串行；同时中间 validation 不能提升为五折 one-step final，因此主表不会随每个 trial 逐格增加。若后续改为“固定 fold0 预筛 learning rate，再用唯一配置补齐五折”，每数据集可由 `15` 次降为 `7` 次；该变更会改变预注册协议，必须在当前 trial 或数据集自然边界后以新合同实施，不能静默修改正在运行的队列。

PKT/MIKT 的旧“无作者源码”阻塞已被新证据取代。作者仓已按不可变 commit 私有引用，严格 train-only question-concept 资源已通过审计；228 上又以 `CUDA_VISIBLE_DEVICES=''` 完成 synthetic CPU author-class forward/backward/optimizer-step smoke：PKT 参数 `20,375,484`，MIKT 参数 `10,310,823`，两者均通过，未构造训练/validation/test loader，也未占用 GPU。smoke SHA256=`ed3eeb2f6dc8b13fe24795aefa15af9378c93e085e952f858d75f4967230a4e6`；train-only resource audit SHA256=`381d88d0d1b0ba4647490f5bdb998df063b9f7ce656b7f0ea7b5f585e14e736c`；matrix SHA256=`6facf356bbeb977ef9480da092049dc48f5c6a3714d125c072838feec1e90380`。这只解除 source/class-wiring 阻塞，不能填效率数值；仍需 CUDA real-batch smoke、冻结 profile、效率合同和 operator-complete FLOPs。旧 blocker disposition V1 已由 `outputs/REMAINING_EFFICIENCY_BLOCKER_DISPOSITION_20260905_V2.json` 取代，V2 SHA256=`0212fa32cb89a9656b091c02326241bebdba92290fe9dda6f15b5548936d2c1f`。纳入新 adapter 测试后的完整本地回归为 `170 tests / 1 skipped / pass`，SHA256=`753d577c45507c3688d9b4d65b3e15fae7e5ca6c4af34f06cd0156cd298b6219`。
<!-- CODEX-RUNTIME-ROOT-CAUSE-PKT-MIKT-SMOKE-20260905:END -->

<!-- CODEX-PKTMikt-V37-V38-STAGED-20260905:START -->
## 2026-09-05 PKT/MIKT 效率与 FLOPs 离线接续已部署

PKT/MIKT 旧的“无作者源码”结论已被 source audit 与 adapter smoke 取代。现已追加两阶段 producer：v37 为 Assist2012 fold0/seed42 的 PKT、MIKT 同协议效率测量（TT、IT、吞吐、参数、峰值显存），v38 为同一真实 train batch 的 operator inventory 与 100% 覆盖 analytic FLOPs。两阶段均固定 `228-only`、RTX 设备、batch64、length200、FP32、共享锁、free memory >=16384 MiB、不可覆盖；test 指标不参与选择，Window-test 关闭。

v37/v38 已 no-clobber 部署并安装唯一 `@reboot + 每分钟` cron。v37 安装审计 SHA256=`7cc0841cc0dfc2c50fa56562c32bb3645ae3a0a2005b7de1816097a475ae1d70`，v38 安装审计 SHA256=`b34585c056a945e208566eccd6fb353263cf4a2a9491e1453ce4316b832c1a18`。当前因 v36 前驱仍为 `waiting_v35_terminal`，v37 状态为 `waiting_runtime`，v38 状态为 `waiting_runtime`；没有创建执行 root、没有启动 PKT/MIKT GPU、没有修改正在运行的 DTransformer。前驱终态、空闲 GPU、共享锁和各控制 SHA 全部满足后，离线 cron 才会按 v37 -> v38 串行执行。

本轮本地定向 smoke 为 `9/9` 通过；纳入 v37/v38 测试后的完整 smoke 已更新入口，下一轮收据须重新记录。主表仍保持 `240/592` 正式数值和 `352` 缺口；PKT/MIKT 只有 v37/v38 终态及独立审计通过后才能填入效率/FLOPs，不以当前 staging 状态代填。
<!-- CODEX-PKTMikt-V37-V38-STAGED-20260905:END -->

<!-- CODEX-A2G-ITER0-ALPHA035-ROLE-BLEND-20260905:START -->
## 2026-09-05 A2G iter0/fold0/seed42 五库模块正向复核（探索性 supporting evidence）

本段追加一组真实 GPU1 复核结果，**不改写五折均值/标准差主表**。协议为 fold0、seed42、validation-only、无训练（冻结 checkpoint 参数级双塔推理），`test_access=false`、`window_test_access=false`。统一固定公式为：

`p = sigmoid(0.35 * logit(p_current_strong_full) + 0.65 * logit(p_residual_role_variant))`。

在当前已填充 registry 的五个数据集上，Full AUC 均严格超过该库最高已登记基线；同时 evidence、Selective-SSM、causal cross-attention 三个模块的 Full-minus-ablation AUC 与 ACC 均为正（15/15 AUC、15/15 ACC）。这是用户要求的“先确保所有模块正向”的直接参数级证据，但 alpha=0.35 来自 validation-only 敏感性探索，故标为 exploratory，`quality_freeze=false`，不能写成预注册确认、五折结论或 universal-SOTA。

| 数据集 | Full AUC | 最强已登记基线 | ΔAUC | evidence ΔAUC/ΔACC | Selective-SSM ΔAUC/ΔACC | causal attention ΔAUC/ΔACC |
|---|---:|---|---:|---:|---:|---:|
| ASSIST2009 corrected/collapsed | 0.8590440530 | UKT 0.8566249815 | +0.0024190715 | +0.0016841300 / +0.0004353999 | +0.0013721460 / +0.0024988168 | +0.0015292736 / +0.0013251301 |
| ASSIST2015 | 0.7358063839 | AKT 0.7337756158 | +0.0020307680 | +0.0009260280 / +0.0009148508 | +0.0007469052 / +0.0008369911 | +0.0007613477 / +0.0008369911 |
| ASSIST2017 | 0.7835910032 | ACE-KT 0.7800295288 | +0.0035614744 | +0.0009884098 / +0.0011171578 | +0.0015933105 / +0.0007447719 | +0.0035481894 / +0.0026393671 |
| NIPS Task 3&4 | 0.7984959190 | AKT 0.7961115327 | +0.0023843863 | +0.0010084596 / +0.0004119262 | +0.0008299658 / +0.0006044569 | +0.0011649629 / +0.0012313010 |
| Slepemapy | 0.7980883198 | SimpleKT 0.7968250064 | +0.0012633133 | +0.0012668229 / +0.0011455351 | +0.0000592193 / +0.0000365469 | +0.0005747966 / +0.0001689544 |

真实执行 root：`<REMOTE_HOME>/a2g_mambakt/role_blend_alpha035_all5_gpu_recheck_20260905_v1_172`；服务器 `172.25.114.0`、物理 GPU1。queue PID/PGID=`2262930`，watchdog PID=`2262940`，watchdog SHA256=`ef59a3a32a032a8482f406b39e0f03fbbc396cd0695d523690fc058ad9939c5e`；watchdog 以 1 秒轮询并于 `2026-09-05T16:03:16+08:00` `queue_idle_exit`，未检测外部 compute，终态两卡 compute=0。统一终态审计：`A2G_ROLE_BLEND_ALPHA035_ALL5_EXPLORATORY_TERMINAL_CLOSURE_20260905.json`，SHA256=`e48d072a4eacf6059c49c9a415e530fd765318d1ec54520b19d6dce0bba5a7cf`。

五个真实 per-dataset summary SHA256：ASSIST2009=`522618098216efc620fe55162092f0076d9447b725d9fc04e3efe8058f2fc0a6`；ASSIST2015=`3b20306eb79dbcb03a5babb0f4c3b5d47cbe81debc0e877790615c17d2b2365b`；ASSIST2017=`42b9f301205608fd5b7190b7f702b754b89e1ec0569db3ffdf3f32ec8c0aca1f`；NIPS=`92c5d28e6594307ae40fde128803d6f45011991a0a7f753f4be74dace95a6466`；Slepemapy=`dbcbb069cd35e470b9ce7a6ce12278c9c173267d8e7c0cfc616beb219ced9bf8`。所有结果均声明参数级 inference-only、无 test/Window-test；该段不增加 canonical 五折数值，也不把“5/5”解释为同一单一 checkpoint 或正式主表结论。
<!-- CODEX-A2G-ITER0-ALPHA035-ROLE-BLEND-20260905:END -->

<!-- CODEX-BASELINE-RUNTIME-STATUS-20260905-V12:START -->
## 2026-09-05 17:48 运行状态与主表缺口解释

228 当前确实在使用 GPU：DTransformer validation-only 的 compute PID=`1221468`，GPU 利用率约 `72%`，显存 `16838/7270 MiB (used/free)`；chain runner PID=`168959`，当前数据集 runner PID=`168964`，命令定位为 `Junyi2015 / fold4 / lr=0.002 / seed42`。本次只读核验没有停止、重启或复制该队列，也没有访问 test 或 Window-test。

主表不是“没有补”，而是正式准入门槛尚未允许回填：三张 canonical 表共 `592` 个单元，当前有合法终态证据 `240/592`，缺 `352/592`，其中 validation/freeze `68`、效率实测 `218`、operator-complete FLOPs `24`；另有 `42` 个 paper-only、协议拒绝或 A2G 外部归属单元，不属于该 baseline 队列。当前 DTransformer 阶段采用 `8 datasets × 3 learning rates × 5 folds = 120` 个 validation trial，必须先产生完整 terminal，再冻结唯一配置并进入 one-step final；单个 validation 中间值不能写成五折均值/标准差。

v37/v38 的 PKT/MIKT 效率与 FLOPs producer 已部署，但分别为 `waiting_runtime`，因为 v36 前驱终态尚未出现；它们不是空闲时未启动，而是按共享锁和前驱门禁 fail-closed。当前离线链不会把 cron/staging/smoke 当作结果，也不会在 `172/127` 执行 KT。该状态块只更新运行证据，不改变 canonical 数值。
<!-- CODEX-BASELINE-RUNTIME-STATUS-20260905-V12:END -->

<!-- CODEX-BASELINE-RUNTIME-STATUS-20260905-V13:START -->
## 2026-09-05 18:45 实时状态核验（只读）

独立健康审计 `outputs/DTRANSFORMER_RESUME_V2_HEALTH_20260905_V1.json` 返回 `pass_active_resume`，SHA256=`88bc762052782eb6073e519e73a7908ec51f8bd4e0320e6dc0a3d04f404c1ee9`。228 当前 compute PID=`1221468`，GPU 利用率约 `95%`，显存 `16838/7270 MiB (used/free)`；chain runner=`168959`，dataset runner=`168964`。当前任务为 `Junyi2015 / fold4 / lr=0.002 / seed42`，属于第 `29/120` 个 validation trial；前 `28/120` 已完成，当前 trial 尚未形成终态。共享锁保持占用，DTransformer terminal 尚未生成，因此没有回填新的五折 final-test 数值。

这解释了表格看起来“很久没有变化”：正式准入要求 `8 datasets × 3 learning rates × 5 folds = 120` 个 validation trial 全部合法完成后，才允许冻结配置并进行 one-step final；单个 validation 中间结果不能写入五折均值/标准差。当前三张 canonical 表仍为 `240/592` 个已审计单元、`352` 个缺口，缺口为 validation/freeze `68`、效率 `218`、operator-complete FLOPs `24`；其中 `42` 个属于 paper-only、协议拒绝或 A2G 外部归属，原则上不由该基线队列填充。下游 v18-v38 继续按前驱终态、共享锁和显存门禁串行等待，未抢占当前训练，也未访问 test/Window-test。
当前表结构复核 `outputs/PUBLICATION_CURRENT_TABLES_AUDIT_20260905_V54.json` 为 `pass`，SHA256=`1109179454fca503dcd03412fc005df385a358b50f6201da33e284469e83f486`；该复核确认两份镜像相等、模型列与效率表存在、MCSKT 未参与排名，并确认 `352` 个缺口仍未被不当回填。
<!-- CODEX-BASELINE-RUNTIME-STATUS-20260905-V13:END -->

<!-- CODEX-PUBLICATION-FULL-SMOKE-20260905-V18:START -->
## 2026-09-05 完整 smoke 回归（V18）

本地完整出版 smoke 命令已重新执行，结果为 `175 tests / 1 skipped / pass`，收据 `outputs/PUBLICATION_FULL_SMOKE_20260905_V18.json`，SHA256=`af9786e16cb145898bb777e56e04e2aa7af128dc45abc0e13f7fcd2c5fe0f349`。该 smoke 仅验证 adapter、guard、表结构、隐私路径和审计契约，不访问 SSH/GPU，不训练/验证/测试，也不把 smoke 结果提升为五折主表数值；canonical 缺口仍按独立 gap inventory 处理。
<!-- CODEX-PUBLICATION-FULL-SMOKE-20260905-V18:END -->
<!-- CODEX-BASELINE-RUNTIME-STATUS-20260906-V14:START -->
## 2026-09-06 22:12 runtime progress audit

The 228 DTransformer validation-only chain advanced to 45/120 completed trials. Junyi2015, NIPS Task 3&4, and Statics2011 each have 15/15 completed trials; Slepemapy trial 46 is active. Health audit `outputs/DTRANSFORMER_RESUME_V2_HEALTH_20260906_V1.json` is `pass_active_resume`, SHA256=`ddac68f3df103fe3522596db5b127513f2cf6f233ecb8146cda2239593242447`; compute PID=1923340, GPU=57%, memory=15868/8240 MiB used/free. No terminal exists, so intermediate validation values are not promoted.
<!-- CODEX-BASELINE-RUNTIME-STATUS-20260906-V14:END -->
<!-- CODEX-BASELINE-RUNTIME-STATUS-20260906-V15:START -->
## 2026-09-06 22:42 实时进度复核

228 当前仍由 DTransformer validation-only 链占用 GPU：compute PID=`1923340`，GPU 利用率约 `58%`，显存 `15868/8240 MiB (used/free)`。链状态为 `45/120` 个 trial 完成，`Junyi2015`、`NIPS Task 3&4`、`Statics2011` 各 `15/15`，当前 `Slepemapy/fold0/lr=0.0005/seed42` 正在运行；DTransformer terminal 尚未生成。最新只读健康审计 `outputs/DTRANSFORMER_RESUME_V2_HEALTH_20260906_V2.json` 状态为 `pass_active_resume`，SHA256=`828fe78e481b6fa28018b03546c8e7994336e43b31d211e0980da8a823d28063`。因此没有新的五折 final 数值可合法回填，v18 及后续基线阶段继续等待前驱终态。
<!-- CODEX-BASELINE-RUNTIME-STATUS-20260906-V15:END -->
<!-- CODEX-BASELINE-RUNTIME-STATUS-20260906-V16:START -->
## 2026-09-06 23:36 最新运行核验

228 仍在运行 DTransformer validation-only 链，compute PID=`1923340`，GPU 利用率约 `93%`，显存 `15868/8240 MiB (used/free)`；Slepemapy runner 仍在执行，DTransformer 总 terminal 尚未生成。当前已完成数据集仍为 Junyi2015、NIPS Task 3&4、Statics2011 各 `15/15`，总进度 `45/120`；因此本轮没有合法的新五折 final-test 数值可回填。v18-v38 离线阶段继续保持前驱终态、共享锁和显存门禁，不并发抢占当前任务。

最新健康审计 `outputs/DTRANSFORMER_RESUME_V2_HEALTH_20260906_V3.json` 为 `pass_active_resume`，SHA256=`fb53c14bc3b3fe412e4a084abfda7872fddcff0a70a596c69c1736b7573ae31b`；运行核算 `outputs/BASELINE_RUNTIME_ACCOUNTING_20260906_V3.json` 为 `pass_active_incomplete`，SHA256=`36a594163cc00f16a768a98d6c2844c24357dca0954a8b1e118e483c31924dfe`。
<!-- CODEX-BASELINE-RUNTIME-STATUS-20260906-V16:END -->
<!-- CODEX-SMOKE-20260906-V3:START -->
最新完整 smoke：`175 tests / 1 skipped / pass`，收据 `outputs/PUBLICATION_FULL_SMOKE_20260906_V3.json`，SHA256=`a535d88bccd89907f3fb1162557a8f5f44fdc681f1ce4075816e2fd330a22d7b`。该结果只证明 smoke/契约回归通过，不产生新的五折数值。
<!-- CODEX-SMOKE-20260906-V3:END -->
<!-- CODEX-SMOKE-TABLE-AUDIT-20260906-V2:START -->
2026-09-06 latest regression: full smoke `175 tests / 1 skipped / pass`, receipt `outputs/PUBLICATION_FULL_SMOKE_20260906_V2.json`, SHA256=`7bb981c639b45de4a224c23297a5b7b7a246361621d0c7dd5952cff7ac1a5238`. Publication table audit remains `pass`; audit `outputs/PUBLICATION_CURRENT_TABLES_AUDIT_20260906_V3.json` confirms mirror equality, requested model columns, efficiency-table presence, MCSKT non-ranking, and 352-cell gap consistency.
<!-- CODEX-SMOKE-TABLE-AUDIT-20260906-V2:END -->
<!-- CODEX-PUBLICATION-FULL-SMOKE-20260906-V1:START -->
Full publication smoke rerun on 2026-09-06: `175 tests / 1 skipped / pass`. Receipt: `outputs/PUBLICATION_FULL_SMOKE_20260906_V1.json`, SHA256=`eba8884590e4ca8bda7793a3a7239557869186f4f8d4296aa183100496ec05ff`. This is a contract/static regression only and does not promote validation or smoke values to the five-fold table.
<!-- CODEX-PUBLICATION-FULL-SMOKE-20260906-V1:END -->
Table mirror audit `outputs/PUBLICATION_CURRENT_TABLES_AUDIT_20260906_V1.json` is `pass`, SHA256=`c21da18e3dceea3e7199542bafc062b3803e746f6665dabbc1169c2b85611cf9`; canonical gap remains 352 cells.


<!-- CODEX-RUNTIME-CHECK-20260908-1344:START -->
## 2026-09-08 13:44 228 运行态复核

228 仍由 DTransformer validation-only 链占用 GPU：compute 进程存在，显存约 15.9 GiB、GPU 利用率约 48%。当前可见 checkpoint 覆盖 Junyi2015 15、NIPS Task 3&4 15、Statics2011 15、Slepemapy 6；Slepemapy/后续链仍未形成 DTransformer 总 terminal。由于这是 validation/tuning 中间态，不向五折 final-test 主表或效率表回填。228-only 离线守卫继续保持每分钟检查、共享锁、前驱 terminal、不可覆盖和 test/window 禁止；当前没有合法条件启动并行缺失基线。
<!-- CODEX-RUNTIME-CHECK-20260908-1344:END -->
<!-- CODEX-RUNTIME-CHECK-20260908-1431:START -->
## 2026-09-08 14:31 228 最新运行核验

DTransformer validation-only 链仍在运行，compute PID 存在，GPU 利用率约 16%，显存约 15.9 GiB used / 8.2 GiB free，共享锁 busy。Slepemapy 的 lr=0.001/fold0 已完成，fold1 正在运行；当前任务 JSON 可确认 `6` 个 Slepemapy trial complete，尚未生成 DTransformer 总 terminal。该中间状态不进入五折 final-test 或效率主表。离线守卫继续保持单 GPU 串行与前驱终态门禁，不启动重复基线。
<!-- CODEX-RUNTIME-CHECK-20260908-1431:END -->
<!-- CODEX-RUNTIME-CHECK-20260908-1503:START -->
## 2026-09-08 15:03 228 运行态复核

228 DTransformer validation-only 链仍在 Slepemapy/fold1/lr=0.001 训练（日志已到 epoch 3，当前 GPU compute 仍存在，约 15.9 GiB 显存、94% 利用率）；共享锁继续 busy。该链尚未生成总 terminal，故没有新五折 final-test 或效率值可合法回填；离线守卫继续等待自然终态后串行接续。
<!-- CODEX-RUNTIME-CHECK-20260908-1503:END -->
<!-- CODEX-RUNTIME-CHECK-20260908-1542:START -->
## 2026-09-08 15:42 运行态进展

DTransformer validation-only 链仍在 228 运行，Slepemapy/fold1/lr=0.001 日志已推进至 epoch 20，当前 validation AUC 约 `0.7871`；GPU compute 仍存在且显存约 15.9 GiB。该值是中间 validation 观测，不进入五折 final-test 主表。自动守卫继续等待链终态后串行接续缺失基线。
<!-- CODEX-RUNTIME-CHECK-20260908-1542:END -->
<!-- CODEX-OFFLINE-GUARD-CHECK-20260908-1708:START -->
## 2026-09-08 17:08 离线自动补基线守卫核验

228 当前仍有 DTransformer compute，GPU 利用率约 45%、显存约 15.9 GiB，故守卫按单 GPU 串行规则等待；未启动并行或重复任务。远端 crontab 实际包含 54 条 baseline continuation/completion guard（含 `@reboot` 与每分钟项）；当前链自然终态后才允许继续补缺失五折基线和效率字段。旧 watcher 日志仅作历史证据，不代表当前运行。
<!-- CODEX-OFFLINE-GUARD-CHECK-20260908-1708:END -->
<!-- CODEX-RUNTIME-CHECK-20260908-1716:START -->
## 2026-09-08 17:16 228 运行态复核

DTransformer validation-only 链仍在 Slepemapy/fold1/lr=0.001：chain runner PID=`168959` 等待当前 fold，训练子进程 PID=`3040133` CPU 约 `163%`，GPU SM 约 `60%`、显存约 `15.8 GiB`，因此不是 GPU 空闲或孤儿进程。当前 fold 尚未写入新的完成 task/总 terminal；validation 中间值不进入五折 final-test 主表。离线守卫继续在共享锁内串行等待自然终态。
<!-- CODEX-RUNTIME-CHECK-20260908-1716:END -->
<!-- CODEX-OFFLINE-AUTO-CONTINUED-20260908-1740:START -->
## 2026-09-08 17:40 离线守卫自动接续实证

上一 fold 自然释放 GPU 后，228 离线守卫自动启动 DTransformer validation-only 的 Slepemapy/fold2/lr=0.001；当前 compute PID=`3225013`，显存约 `15.8 GiB`，GPU 利用率约 `94%`，共享锁 busy。该启动由既有 continuation guard 实际完成，未人工重复启动；test/window-test 仍禁用，结果在总 terminal 前不进入正式五折主表。
<!-- CODEX-OFFLINE-AUTO-CONTINUED-20260908-1740:END -->
<!-- CODEX-OFFLINE-AUTO-CONTINUED-20260908-1751:START -->
## 2026-09-08 17:51 自动续跑进展

Slepemapy DTransformer validation-only 的 fold1 已由守卫写入完成任务 `07_lr_1_54ad8f7117_f1.json`；GPU 随后自动接续 fold2，当前 compute PID=`3225013`，运行约 17 分钟，GPU 利用率约 `33%`、显存约 `15.9 GiB`。这仍是 validation/tuning 中间态，不进入正式五折 final-test 主表；共享锁保持 busy，未启动并行任务。
<!-- CODEX-OFFLINE-AUTO-CONTINUED-20260908-1751:END -->