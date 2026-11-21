# 平台配置与评估

## 执行模型与恢复

任务固定仓库 URL、完整 Commit、需求、测试 profile、模型模式和动作预算。Worker 按 MySQL advisory lock 串行处理一个数据库中的任务，租用标识保护任务状态写入。Worker 崩溃后，新的进程可重领 `running` 任务。Git 拉取、Gemma 服务和 Sandbox 的临时故障最多自动尝试三次，可从失败状态手动重试。

LangGraph 以任务 ID 为 thread ID，使用 MySQL checkpoint 库同步保存各节点结果。源代码工作区在每个读、写、测试步骤重建。恢复时从固定 Git tree 重放规范 diff，并核验基线 Commit、任务入参和已审批的哈希。模型的编辑动作只替换已跟踪 UTF-8 文件中的唯一文字片段。一次任务含最多 28 个动作，默认软时间预算 420 秒；Git、模型请求及测试另有各自的超时。因基础设施故障重放节点时，该次工具调用可再次执行，Git 编辑和测试均通过重建工作区保持可重放性。

初始验收需返回退出码 `1`。候选审批前，同一 profile 必须在最新 diff 上返回 `0`。任何编辑都会使之前的测试失效。加上外部可信回归测试可以提高对需求的覆盖度。一般 profile 的退出码是测试套件结果，无法独立证明模型满足了 Issue 的全部要求。

## 测试隔离

Worker 将已跟踪文件的 tar 通过标准输入送入一次性 Docker 容器。容器使用非 root 用户、只读根文件系统、限量 tmpfs、禁网、无附加 capabilities，并限制 CPU、内存、进程数和执行时间。Docker socket 仅挂载在受信任的 Worker 上。测试容器不接收 DB、GitHub 或模型服务凭据，也不包含宿主文件挂载或 Git 元数据。未安装 Docker 或测试镜像时任务失败关闭。

内置 `demo`、`unittest`、`pytest` 三种测试 profile，适合 Python 公开演示和已预装依赖的项目。需要其他语言或依赖时，由操作员预构建受信任的 Sandbox 镜像，通过 `PATCHPILOT_SANDBOX_IMAGE` 指定。Linux 部署可通过 `PATCHPILOT_SANDBOX_RUNTIME=runsc` 选择已安装的 gVisor。Docker 容器共享其运行主机的内核，建议在专用虚拟机或专用主机上部署处理不可信代码的 Worker。此配置面向单操作员本地部署。

可导入公开仓库，也可在 Worker 上设置仅有目标仓库内容读取权限的 `PATCHPILOT_GITHUB_TOKEN`。拉取只允许 GitHub HTTPS 及指定 Commit，跳过子模块。工作区拒绝符号链接、子模块、冲突路径和过大的树。补丁可查看或下载，审批只更新 PatchPilot 任务状态。

## 本地向量检索

默认仅启用关键词检索。开启语义检索时用 FastEmbed 在 CPU 上运行 BGE 模型，下载一次后可离线使用：

```sh
uv sync --frozen --extra embedding
uv run --extra embedding python -c 'from fastembed import TextEmbedding; m=TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir="data/embeddings"); next(iter(m.embed(["code search"])))'
```

原生 Worker 设置 `PATCHPILOT_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`，默认读取 `data/embeddings`。在 Compose 下，可先将模型下载到 Worker 的持久卷：

```sh
docker compose run --rm worker python -c 'from fastembed import TextEmbedding; m=TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir="/var/lib/patchpilot/embeddings"); next(iter(m.embed(["code search"])))'
```

随后在 `.env` 启用 `PATCHPILOT_EMBEDDING_MODEL`，执行 `docker compose up -d worker`。配置了 embedding 而缓存缺失时，任务会失败并可在缓存就绪后重试。

索引以仓库 ID、Commit 和 embedding 配置版本为键。Python 使用 AST 抽取函数、类及方法的行号。其余支持的文本后缀按行切片。更新时复用相同路径和 blob 的代码块及向量，将新结果写入单独的 FAISS 文件。查找时仅使用当前 Commit 的快照，经倒数排名融合关键词和向量结果。检索内容来自基线，Agent 使用 `read_file` 确认当前候选版本。

## MySQL 表结构

| 表 | 主要字段与约束 |
| --- | --- |
| `repositories` | `id`、唯一 `url`、`created_at` |
| `jobs` | `repository_id`、`commit`、`issue`、模型及 profile、状态、尝试次数、Worker claim、diff、审批哈希及决定、测试摘要 |
| `job_events` | 自增 `id`、`job_id`、事件类型、JSON 摘要、写入时间。按 `(job_id, id)` 回放 SSE |
| `index_snapshots` | `repository_id`、`commit`、`embedding` 联合唯一。快照 ID 对应 FAISS 文件 |
| `code_chunks` | `snapshot_id`、`path`、Git `blob`、`symbol`、起止行、代码、FAISS `vector_id`。通过快照关联 Commit |
| `checkpoints`、`checkpoint_blobs`、`checkpoint_writes`、`checkpoint_migrations` | MySQL Checkpointer 管理 StateGraph 状态与 schema 版本 |

业务 schema 由 `alembic upgrade head` 初始化。Worker 持锁初始化 Checkpointer 自带的表。MySQL 和 FAISS 文件需共同持久化。数据库保存源码片段、Issue、补丁和有限的测试反馈。SSE 只广播状态和指标摘要。数据库和状态卷应留在受控环境。

### 原生运行

使用 MySQL 8.0.19 或更高版本。创建应用数据库及用户，在 `.env` 设置 `PATCHPILOT_DATABASE_URL=mysql+pymysql://...` 和 `PATCHPILOT_API_TOKEN`。构建同一个 Sandbox 镜像后，在不同终端执行：

```sh
uv run alembic upgrade head
uv run uvicorn patchpilot.api:create_app --factory --host 127.0.0.1 --port 8080
uv run python -m patchpilot.worker
```

## Base、LoRA 与后训练

使用已有权重启动本机 OpenAI 兼容服务。原生 Worker 连接 `http://127.0.0.1:8000/v1`，容器 Worker 可连接 `http://host.docker.internal:8000/v1`。Linux Docker Engine 若未提供该 hostname，可为 Worker 增加 `extra_hosts: ["host.docker.internal:host-gateway"]` 并在主机侧让服务对 Docker 网桥可达。服务地址由操作员配置。

Gemma 4 vLLM 使用 `gemma4` tool parser、`gemma4` reasoning parser 和 `--enable-auto-tool-choice`。要评估 LoRA，GPU 服务应将兼容的 Adapter 挂载为独立 served ID，例如 `rs_lora`。分别设置 `PATCHPILOT_BASE_MODEL` 与 `PATCHPILOT_LORA_MODEL`。客户端校验 `/v1/models` 存在所选的精确 ID，并使用 OpenAI 原生 function call 协议。用 `scripts/demo.py --mode base` 或 `--mode lora` 提交同一公开例子，检查实际补丁、模型 ID、基线退出码和候选测试结果。

`workflows/rs_sft/` 保留原 ADK 单 Agent、官方工具、Harness、轨迹采集、Rejection Sampling、SFT/QLoRA 以及冻结集评估。其 `serve.py` 可作为本地 Gemma 服务入口，具体权重与 LoRA 加载门禁见 [原工作流依赖](dependencies.md)。平台复用同一推理协议和 Base/LoRA serving ID。平台任务使用自己的 Git 重建、工具适配和 Sandbox。比赛采集、轨迹校验及评估仍由原官方 Harness 执行。

评估应记录模型 ID、Commit、候选 SHA、基线与候选测试退出码，以及动作数。正式比较使用同一冻结任务集、预算和验收条件，运行原 `workflows/rs_sft/compare.py` 检查配对完整性。Mock 与 HTTP stub 只覆盖执行链路和协议。真实模型推理、Adapter 加载和正式训练需有独立实测记录。

## 测试

```sh
uv run pytest -q -m 'not mysql and not sandbox'
uv run python -m unittest discover -s workflows/rs_sft -p 'test_*.py' -q
uv run python -m unittest discover -s experiments/edit-recovery -q
uv run python workflows/rs_sft/check_acceptance.py
```

安装 `requirements-dev.txt` 中的 pytest 或使用 `uv sync` 的开发依赖可运行 Git/pytest 夹具。使用单独的 MySQL 测试库运行真实迁移、API、Checkpoint 和 FAISS 集成测试。构建 Sandbox 镜像后可同时开启容器集成及公开演示：

```sh
export PATCHPILOT_TEST_DATABASE_URL='mysql+pymysql://test_user:test_password@127.0.0.1/patchpilot_test'
PATCHPILOT_TEST_DOCKER=1 uv run pytest -q
```

设计参考包括 [Open SWE 的持久执行与 Sandbox](https://github.com/langchain-ai/open-swe/blob/main/openwiki/architecture/overview.md)、[Aider 的结构化仓库地图](https://aider.chat/2023/10/22/repomap.html)、[BLAgent 的代码片段组织方式](https://github.com/afifaniks/BLAgent)和 [vLLM 的 Gemma 4 配置说明](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html)。
