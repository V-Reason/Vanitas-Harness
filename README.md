# Vanitas-Harness

单机本地推理 CLI：为 RTX 4060 Laptop（8 GB）准备的 llama.cpp 启动器、单实例守卫、交互式 REPL 与批量翻译。模型、端口、上下文与 KV 类型全部由 `config\` 驱动，代码里没有硬编码。

- 通用主力：Qwen2.5-7B-Instruct Q6_K（4060，2 分片，只传首片）
- 轻量高质：MiniCPM5-2B **Q8_0**（4061）
- 专业翻译：HY-MT2-7B Q6_K（4062）
- 回归基线：Qwen2.5-3B Q4_K_M（4063）

实测显存/吞吐与未验证边界以 [docs/项目现状.md](docs/项目现状.md) 为准；本轮 CLI 化的交互设计依据见 [docs/ClaudeCode范式笔记.md](docs/ClaudeCode范式笔记.md)。

---

## 目录结构

```text
Vanitas-Harness\
├── bin\                        # llama.cpp CUDA 预编译包（不入库，约 1.1 GB）
├── models\                     # GGUF 权重（不入库；7B 为 2 分片）
├── config\
│   ├── models.json             # 4 个模型：路径/端口/别名/分片/档位
│   ├── settings.json           # 超时、重试、渲染、工具、翻译参数
│   └── prompts\                # system_default + task_chat/translate/code/agent
├── vanitas\                    # CLI 实现（标准库 + prompt_toolkit）
│   ├── cli.py                  # 子命令与退出码
│   ├── config.py / guard.py    # 配置校验 / 守卫（0/1/2/3/4/9）
│   ├── launcher.py / engine.py # 进程生命周期 / OpenAI 兼容客户端
│   ├── repl.py / render.py     # 交互循环 / 流式渲染
│   ├── session.py / prompts.py # 会话与预算 / 模板
│   ├── tools.py / translate.py / batch.py
│   └── errors.py
├── scripts\
│   └── vanitas.bat             # 薄壳：cd 到仓库根后 python -m vanitas %*
├── tests\
│   ├── multi_model_benchmark.py
│   ├── test_*.py               # 配置/守卫/翻译/模板/工具/CLI 冒烟
│   └── reports\                # 评测与冒烟产物（不入库）
└── docs\
    ├── 任务书.md                # 交付契约与修订记录
    ├── 项目现状.md              # 状态事实源
    ├── ClaudeCode范式笔记.md    # 交互范式提炼与取舍
    └── 前期方案.md              # 历史设计（冻结）
```

## 依赖与三步上手

1. 把 llama.cpp CUDA 预编译包**完整解压**到 `bin\`（需 `llama-server.exe`、`llama.dll`、`ggml-base.dll`、`ggml-cuda.dll`）。
2. 把 GGUF 按 `config\models.json` 的路径放入 `models\`。7B 两片必须同名同目录，启动只传 `-00001-of-00002.gguf`。
3. `python -m pip install prompt_toolkit`（唯一第三方依赖，仅用于行编辑/历史；未安装时自动降级为 `input()`）。

入口二选一：`scripts\vanitas.bat <子命令>`，或在仓库根 `python -m vanitas <子命令>`（bat 只负责切换目录，无业务逻辑）。

## 常用命令

```text
vanitas                        # 交互式 REPL（无服务端时引导选择并启动模型）
vanitas models                 # 列出模型/端口/档位/别名
vanitas status                 # 进程、端口、显存、运行记录
vanitas serve qwen7b           # 后台启动并等待 /health（默认 4K + Q8_0）
vanitas serve qwen7b --profile 2 --foreground
vanitas stop                   # 结束所有 llama-server
vanitas chat --model minicpm   # 直接进入 REPL（未启动时自动拉起）
vanitas translate a.srt b.md --out-dir out\
vanitas batch tasks.jsonl --model qwen7b
```

退出码：`0` 成功 / `1` 用户取消或中断 / `2` 配置错误 / `3` 守卫失败 / `4` 参数错误 / `5` 引擎或网络失败。守卫失败时 stderr 首行形如 `[GUARD-CODE=1] [GUARD-FATAL] 端口 4060 被 PID 1234 (x.exe) 占用。`

## 端口与档位

| 模型 id | 端口 | 档位 1（默认） | 档位 2 | 档位 3 |
| :--- | :-: | :--- | :--- | :--- |
| `qwen7b` | 4060 | 4K + Q8_0 KV | 8K + Q4_0 KV | 4K + F16 KV（需空闲 ≥7.2 GB） |
| `minicpm` | 4061 | 8K + F16 KV | 32K + F16 KV | — |
| `hymt` | 4062 | 4K + Q8_0 KV | 8K + Q4_0 KV | 4K + F16 KV（需空闲 ≥7.2 GB） |
| `qwen3b` | 4063 | 8K + Q4_0 KV | 4K + F16 KV | — |

- 环境变量：`VANITAS_PORT` 覆盖端口，`VANITAS_PROFILE` 覆盖档位，`VANITAS_CONFIG_DIR` 覆盖配置目录。
- **单实例策略**：一次只服务一个模型；`serve` 会先回收存量 `llama-server`（8 GB 显存禁不起双实例），再启动目标模型。
- 服务只监听 `127.0.0.1`，提供 OpenAI 兼容 API（`/v1/chat/completions`、`/v1/models`、`/health`）；本 CLI 不再使用浏览器 Web UI。
- 新增模型只允许在 `config\models.json` 追加 `models[]` 条目，字段名不可改；`shards>1` 时 `path` 必须指向 `-00001-of-0000N` 首片。

## REPL 内置命令

`/help`、`/model [id]`、`/profile [N]`、`/system [text|file]`、`/task [name]`、`/clear`、`/save [path]`、`/load <path>`、`/exec <cmd>`、`/stop`、`/status`、`/quit`（别名 `/exit`）。

- 仅以 `/` 开头且第 2 个字符不是空格的行当命令；其余整行都是用户输入。
- `/model` 只切换连接，不会自动启停服务端；目标未就绪时提示先 `vanitas serve`。
- `/profile` 只影响下次 `serve`。
- `/task translate` 会把每轮输入填入模板 `{input}`；`/system` 参数是存在的文件路径时读文件，否则按文本处理。
- `/exec` 默认关闭（`settings.tools.exec_enabled`）；启用后仍受白名单与逐次确认约束，`shell=False`，`& | >` 只能当字面参数。
- 上下文预算 = `min(档位 ctx, repl.max_context_tokens) - 512`，超预算时丢弃最早轮次并提示。

## 批处理与翻译

```text
vanitas translate <file...> [--model hymt] [--out-dir DIR] [--jobs 1]
vanitas batch <tasks.jsonl> [--model qwen7b]
```

- `.txt` / `.md`：按空行分段，段内超 `translate.chunk_chars` 再按句末标点切；空行结构原样保留。
- `.srt`：只翻译字幕文本，序号、时间轴、空行、CRLF/LF 与 BOM 原样保留；多行字幕整体翻译并保留行结构。
- 输出：同目录（或 `--out-dir`），文件名插入 `.zh`，如 `a.srt → a.zh.srt`。
- 任一段失败保留原文并继续，最后汇总 `[TRANSLATE-FAIL]` 并返回 `5`；`--jobs >1` 时提示实际串行（服务端 `-np 1`）。
- `batch` 的 JSONL 行格式：`{"id","prompt","system"?,"task"?,"max_tokens"?}`，结果写 `<input>.out.jsonl`。

## 验收评测

```bat
python tests\multi_model_benchmark.py --port 4060 --suite agent --runs 3
python tests\multi_model_benchmark.py --port 4062 --suite translation --runs 3
```

退出码：`0` 通过 / `1` 未就绪 / `2` 协议或性能失败 / `3` 质量失败 / `4` 参数错误。报告落到 `tests\reports\`。

指标口径：性能取 llama-server 返回的 `timings` 与 `usage`；流式只校验 SSE 帧、TTFT 与 `[DONE]`，不用分片数冒充 token。请求统一带 `chat_template_kwargs`（由 `config\models.json` 的 `template_kwargs` 决定）。

## 常见故障

| 现象 | 处置 |
| :--- | :--- |
| `[GUARD-CODE=1] ... 端口 4060 被 PID ... 占用` | 结束占用进程，或 `set VANITAS_PORT=<空闲端口>` |
| `[GUARD-CODE=2] ... 模型文件缺失` | 核对 `config\models.json` 的 `path` 与 `models\` 实际文件名 |
| `[GUARD-CODE=2] ... 模型分片缺失 / 分片序号错误` | 7B 两片同名同目录；只传首片 |
| `[GUARD-CODE=2] ... 文件头不是 GGUF` | 下载得到的是网页/压缩包，重新下载 |
| `[GUARD-CODE=3] ... 运行时缺失` | 缺 `bin\` 下的 DLL，补全 llama.cpp 构建 |
| 启动即退出、无日志 | CLI 会打印日志尾部，日志在 `~/.vanitas_logs/` |
| `[ERROR] ... 返回 HTTP 4xx` | 请求本身不合法，不会重试；看响应体提示 |
| 显存不足 / 加载中断 | 关闭 Steam/浏览器/壁纸引擎，或换低档位 |

## 仓库约定

- 分支：`main` 保持可用；特性分支短生命周期，`--no-ff` 合并后打标签。
- `bin\`、`models\`、`tests\reports\` 永不入库；**不要执行 `git clean -fdx`**，它会连带删除权重。
- `scripts\*.bat` 固定 UTF-8（无 BOM）+ CRLF，且只允许薄壳，不得承载启动逻辑。
- 运行产物：`~/.vanitas_logs/`（服务端日志）、`~/.vanitas_run/`（pid 记录）、`~/.vanitas_sessions/`（`/save`）、`~/.vanitas_history`（输入历史）。

## 版本

`v0.3.0` — CLI 化重构：配置驱动、Python 守卫、REPL 与 12 个内置命令、批量翻译/批处理，端口迁到 4060–4063。
`v0.2.0` — 三模型真实权重实测；7B 分片自动续载与守卫全分片校验。
