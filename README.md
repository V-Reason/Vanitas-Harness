# Vanitas-Harness

单机本地推理 CLI：为 RTX 4060 Laptop（8 GB）准备的 llama.cpp 启动器、单实例守卫、交互式 REPL、批量翻译与批处理。模型、端口、上下文与 KV 类型全部由 `config\` 驱动。

- 通用主力：Qwen2.5-7B-Instruct Q6_K（4060，2 分片，只传首片）
- 轻量高质：MiniCPM5-2B **Q8_0**（4061）
- 专业翻译：HY-MT2-7B Q6_K（4062）
- 回归基线：Qwen2.5-3B Q4_K_M（4063）

---

## 先读哪份文档

| 文档 | 你会在里面找到 |
| :--- | :--- |
| [docs/用户使用手册.md](docs/用户使用手册.md) | 安装、命令、REPL、翻译、批处理、排障 |
| [docs/总设计方案.md](docs/总设计方案.md) | 架构、模块边界、配置/进程/接口契约、取舍 |
| [docs/项目现状.md](docs/项目现状.md) | 实测矩阵、已知问题、未验证边界、下一步 |
| [docs/ClaudeCode范式笔记.md](docs/ClaudeCode范式笔记.md) | 交互范式来源与 D1–D6 取舍 |
| [AGENTS.md](AGENTS.md) | 开发纪律、红线、验证门禁、协作契约 |

---

## 三步上手

1. 把 llama.cpp CUDA 预编译包**完整解压**到 `bin\`（需 `llama-server.exe`、`llama.dll`、`ggml-base.dll`、`ggml-cuda.dll`）。
2. 把 GGUF 按 `config\models.json` 的路径放入 `models\`。7B 两片必须同名同目录，启动只传 `-00001-of-00002.gguf`。
3. `python -m pip install prompt_toolkit`（唯一第三方依赖；未安装会自动降级为 `input()`，功能不受阻）。

入口是仓库根的 `vanitas.bat`（等价于 `python -m vanitas`）：

```bat
vanitas.bat              :: 进入 REPL，无服务端时引导启动模型
vanitas.bat models       :: 看模型/端口/档位/别名/路径
vanitas.bat status       :: 看进程/端口/显存
vanitas.bat serve qwen7b :: 后台启动并等待 /health
vanitas.bat agent        :: Agent 规划模式（只输出 JSON，不自动执行）
vanitas.bat translate a.srt :: 翻译并显示 5 段实况看板 + 终态统计
vanitas.bat stop         :: 停止所有 llama-server
```

更多子命令与示例见 [用户使用手册](docs/用户使用手册.md)。

---

## 目录结构

```text
Vanitas-Harness\
├── vanitas.bat                 # 唯一入口（5 行薄壳，CRLF）
├── AGENTS.md                   # 开发与协作规范
├── README.md                   # 本页索引
├── bin\                        # llama.cpp CUDA 构建（不入库）
├── models\                     # GGUF 权重（不入库；7B 为 2 分片）
├── config\
│   ├── models.json             # 模型/端口/别名/分片/档位
│   ├── settings.json           # 超时/重试/启动/REPL/工具/翻译
│   └── prompts\                # system_default + task_*.md
├── vanitas\                    # CLI 实现（标准库 + prompt_toolkit）
├── tests\
│   ├── multi_model_benchmark.py
│   ├── test_*.py               # 144 个单元测试
│   └── reports\                # 评测与冒烟产物（不入库）
└── docs\                       # 总设计方案 / 项目现状 / 使用手册 / 范式笔记
```

---

## 关键约定

- **配置驱动**：新增模型只允许在 `config\models.json` 追加 `models[]` 条目，字段名不可改；`shards>1` 时 `path` 必须指向 `-00001-of-0000N` 首片。
- **单实例**：一次只服务一个模型；`serve` 会先回收旧 `llama-server`（8 GB 显存禁不起双实例）。
- **端口**：4060–4063；`VANITAS_PORT` / `VANITAS_PROFILE` / `VANITAS_CONFIG_DIR` 可临时覆盖。
- **退出码**：`0` 成功 / `1` 中断 / `2` 配置错误 / `3` 守卫失败 / `4` 参数错误 / `5` 引擎或网络失败。
- **依赖红线**：标准库 + `prompt_toolkit`；`bin\`、`models\` 只读；禁止 `git clean -fdx`。
- **行尾**：`*.bat` 为 UTF-8 无 BOM + CRLF，其余文本 LF。

## 验证

```powershell
python -m unittest discover -s tests
python -m vanitas serve qwen7b
python tests\multi_model_benchmark.py --port 4060 --suite agent --runs 1
python -m vanitas stop
```

评测退出码：`0` 通过 / `1` 未就绪 / `2` 协议或性能失败 / `3` 质量失败 / `4` 参数错误；报告落到 `tests\reports\`。

## 版本

`v0.4.1` — 界面现代化 + 后续重构：符号化输出、`ui` 主题、Tab 补全、`/model` 自动切换、Stream 遥测、`vanitas agent` 规划模式、翻译 5 段看板/滑动上下文/截断重译；144 个单测。
`v0.4.0` — 入口迁到根目录 `vanitas.bat`；文档重组为总设计方案 / 项目现状 / 用户使用手册 / 范式笔记 + `AGENTS.md`。
`v0.3.1` — `/exec` 支持 cmd 内置命令，并拒绝 `& | < > ^ %` 串联。
`v0.3.0` — CLI 化重构：配置驱动、Python 守卫、REPL、批量翻译/批处理，端口迁到 4060–4063。
`v0.2.0` — 三模型真实权重实测；7B 分片续载与守卫全分片校验。
