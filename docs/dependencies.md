# 依赖与运行边界

此页说明保留的比赛提取工作流。独立平台的 Docker、MySQL 和模型接入见 [平台配置与评估](platform.md)。

## 本地无模型检查

- Python 3.12/3.13 和 Git。`unittest` 用标准库；`requirements-dev.txt` 仅供 `check_acceptance.py` 的 pytest 夹具使用。建议装到独立 venv，不修改正在运行的实验环境。
- `setup_experiment.py` 默认只指纹化仓库内的六个 Agent 文件并探测机器；只有显式指定 `--tasks` 或 `--official-sources` 才读取对应外部输入。生成的 `evidence/` 和 `data/` 不跟踪。不要在未隔离的工作目录运行模型 Harness。

## 历史实机依赖（不是通用 lockfile）

| 用途 | 外部依赖与边界 |
| --- | --- |
| Agent / Harness | `swegemma==0.2.11`、`adk-submission==0.2.13`、`adk-eval-core==0.1.0`、Google ADK 和 LiteLLM；用获授权的离线 wheelhouse 和锁定清单，仓库不提供官方包。历史上检验过 Python 3.13 的分角色环境；不等于不同版本的接口兼容。 |
| Docker 主流程 | Linux、Docker、外部 sandbox 镜像和只读任务依赖。`pipeline.py` 使用官方接口；已存在的 graphs/embeddings 参数只是适配 Harness 的外部输入，平台的代码检索独立于该 Harness。 |
| P1 分角色校准 | Linux `/proc`、独立 Generator/Validator、控制端批准回执、干净源码、模型 parser 回执。`p1_generate.py` 显式收 `--task-id` / `--source-sha256`；`p1_validate.py` 显式收任务 ID、任务表和快照及其预先核准的 SHA-256。只能向 Generator 提供审查后的输入，参考正控保留在 Validator。不要用本次算出的哈希替代已核准的基线。 |
| Gemma 推理与 tokenizer | 本地指定 `gemma-4-31b-it-qat-w4a16-ct` 权重。历史模型环境安装检查：Python 3.12.3、Torch 2.10.0+cu128、vLLM 0.19.1、Transformers 5.13.1；T4×2 下实机推理失败。`bootstrap_*.py` 依赖外部锁文件和 wheel bundles，不会被本仓库补齐。 |
| SFT/QLoRA | Linux、兼容的 CUDA/BF16 GPU、PyTorch、Transformers、Accelerate、`compressed-tensors`、PEFT、bitsandbytes。`train.py` 保留 CT 解量化后再做 NF4 的路线及 RAM/磁盘门禁；未验证训练组合，因此不提供会被误认为可复现的训练 lockfile。 |

`recording.py` 会在本地写请求、响应、工具参数及差异等原始记录。输出只放在受控目录；`.gitignore` 是防误提交措施，不是敏感数据审查或执行隔离。`compare.py` 的输入是冻结分区和带溯源的评估结果，不是排行榜抓取器。
