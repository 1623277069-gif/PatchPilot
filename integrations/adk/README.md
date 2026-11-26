# Google ADK 集成

`agent/` 是 PatchPilot 保留的单 Agent 配置包，包含采样配置、评估预算和 `verify-change` 验收工具。运行这个配置包需要已有的 Google ADK 和官方 Harness 环境。历史可用依赖为 `swegemma==0.2.11`、`adk-submission==0.2.13` 和 `adk-eval-core==0.1.0`，由调用方提供。

在仓库根目录执行本地验收夹具。

```sh
uv run python integrations/adk/check_acceptance.py
```

FastAPI 提交的任务由 PatchPilot 的 LangGraph Worker 执行。该 Worker 的模型配置和任务验证见[平台说明](../../docs/platform.md)。
