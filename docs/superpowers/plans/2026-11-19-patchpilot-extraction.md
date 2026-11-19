# PatchPilot 提取与交付计划

**目标：** 在原实验目录之外建立仅含自有可复用源码及脱敏记录的独立私有仓库。

**结构：** `workflows/rs_sft/` 保留原工作流的相对路径与单 Agent 包；`experiments/` 保留可复用的历史编辑工具和脱敏实验索引。外部运行输入不进入 Git。

1. 从主线 RS-SFT 实验按文件白名单复制代码、Agent 配置和夹具测试；另保留历史的 edit-recovery 技能与单元测试。
2. 将 P1 的单题 ID 和强制输入哈希留给调用方提供，不放宽校验或启用原先关闭的批量生成、正式训练和评测门禁。
3. 添加 `README.md`、`docs/dependencies.md`、`experiments/README.md` 和 `.gitignore`，区分真实校准、夹具测试以及尚未执行的训练和对照评估。
4. 只运行 AST、CLI help、合成夹具测试及 Git 暂存区检查；不加载权重、挂载比赛数据、启动推理服务或接触运行中的 Notebook。
5. 提交前核对全部暂存文件来源、文件名、类型及凭据/数据特征；发布时确认 GitHub 账号和 private 选项，最后核对 `main` 提交及远程可见性。不可用时保留本地提交。
