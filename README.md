# PatchPilot

从 Gemma 4 31B Developer Agent 提取的单 Agent 代码修复实验基础。保留原有 Google ADK 入口、Harness 编排、轨迹收集、拒绝采样 SFT/QLoRA 与评估代码；不包含 LangGraph、RAG、产品后端或可直接提交的 Kaggle 包。

| 路径 | 内容 |
| --- | --- |
| `workflows/rs_sft/agent/` | 单 Agent YAML、采样与评估配置、系统提示、verify-change 技能 |
| `workflows/rs_sft/` | Harness、分组切分、Generator/Validator 门禁、轨迹记录、SFT 导出、QLoRA、Base/LoRA 比较及本地测试 |
| `workflows/rs_sft/configs/` | 实验超参数和执行限制，不包含真实切分、任务 ID 或运行凭证 |
| `experiments/` | [脱敏实验记录](experiments/README.md)与独立的历史 edit-recovery 实验，不接入主线 Agent |
| `docs/` | [依赖与运行边界](docs/dependencies.md)及提取计划 |

## 本地验证

在仓库根目录执行。以下测试使用合成数据，不读比赛数据或调用模型：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s workflows/rs_sft -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/edit-recovery -v
```

安装 `requirements-dev.txt` 后，可用当前 Python 环境运行真实 Git/pytest 夹具：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 workflows/rs_sft/check_acceptance.py
```

## 真实运行状态

- **本次提取验证：** 37 个合成 `unittest` 用例及 16 个本地 Git/pytest 验收夹具通过；未启动模型或 GPU 作业。
- **历史实测：** V5 单 Agent 配置通过官方编译，ADK 调用过 `inspect`；Gemma tokenizer 的两项 loss mask 检查通过。RS-SFT P0 的独立 Generator/Validator 边界与 P1 的 Validator 空补丁/参考正控有实机记录。
- **实机尝试但未通过：** RS-SFT P1 的原始 31B CT 权重在 T4×2 装载过，但没有一次成功完成推理响应；没有完成该路线的 Agent 修复，也没有可用的 RS 训练轨迹。
- **尚未完成或未运行：** 真实训练 JSONL 端到端导出、CT→BF16→NF4 训练路线、QLoRA 单步及正式训练、Adapter 重载与原始 CT 兼容性、冻结验证集 Base/LoRA 对照。无 Adapter、无涨分结论。历史实测不等于此独立目录复跑实机通过。

## 数据边界

真实输入、按任务校准参数及私有证据继续留在原实验或授权运行环境，不随仓库迁移。即使仓库是私有的，也不要提交真实任务表/快照、测试或参考 Patch、模型/权重/Checkpoint、Wheel/官方源码、密钥、Notebook 和原始日志。P1 需要外部提供固定任务和哈希，仍须完成隔离与模型预检；`configs/experiment.json` 默认禁用批量生成、正式训练和冻结集评测。
