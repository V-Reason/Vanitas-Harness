# 批次 1–8 重构执行报告

## 批次 1 · 配置层分裂 + project_prefixes 接入 · f8ee040 · 2026-10-27 18:20

**关键改动**：
- 8 个文件变更（新增 4、删除 1、修改 3）；新增 `vanitas/config/{__init__,schema,loader,preferences}.py`，删除 `vanitas/config.py`。
- 涉及模块：config、tools、settings、README、项目现状。

**DoD 通过情况**：
- 12/12 条命令通过。
- 单测：Ran 353 tests, OK (skipped=1)。

**兜底补丁**（如有）：
- 无。

**遗留问题**（如有）：
- 任务书 4.1 的 `schema.py` 注释含 `python verify.py` 字面串，与 DoD “schema.py 0 命中”冲突；已将注释改为“项目级白名单已迁至 settings.tools.project_prefixes”，不写该字面串。
- 任务书要求 `_parse_port` 放 loader，但 `schema.AppConfig.effective_port` 运行时调用它；为保持行为，`_parse_port` 放在 schema，loader 导入复用。

**验收要点**：
- `BASE_DIR` 为 `parents[2]`；`load_config()` 返回 4 个模型与 `('python verify.py',)`。
- `models`/`--version`/`status` 退出码 0。

## 批次 2 · 工具层分裂 + runner 合并 · 6432243 · 2026-10-27 18:28

**关键改动**：
- 7 个文件变更（新增 6、删除 1）；新增 `vanitas/tools/{__init__,sandbox,runner,fs,shell,exec}.py`，删除 `vanitas/tools.py`。
- 涉及模块：tools、config、AgentTools。

**DoD 通过情况**：
- 8/8 条命令通过。
- 单测：Ran 353 tests, OK (skipped=1)。

**兜底补丁**（如有）：
- `tools/__init__.py` 保留 `import subprocess`，兼容既有测试对 `vanitas.tools.subprocess.run` 的补丁。
- `AgentTools._readonly_prefix` 绑定到 `shell._readonly_prefix`，兼容既有测试直接调用类属性。
- 统一 `runner.run` 对 `shell=True` 的 `FileNotFoundError` 保持旧文案“命令启动失败”，避免 `test_missing_shell_reports_denial` 行为变化。

**遗留问题**（如有）：
- 无。

**验收要点**：
- `_execute` / `_execute_shell` 双路径 echo 输出一致；`AgentTools` 11 个公开方法均存在；`tools.py` 已删除。

## 批次 3 · Agent 层分裂 · a10f595 · 2026-10-27 18:36

**关键改动**：
- 10 个文件变更（新增 8、删除 1、修改 1）；新增 `vanitas/agent/{__init__,actions,parser,stream,policy,stagnation,feedback,loop}.py`，删除 `vanitas/agent.py`。
- 涉及模块：agent、repl、tools、tests。

**DoD 通过情况**：
- 10/10 条命令通过。
- 单测：Ran 353 tests, OK (skipped=1)。

**兜底补丁**（如有）：
- `tests/test_agent.py` 的 `mock.patch.object(agent, "AgentRepl")` 改为 `agent.loop.AgentRepl`：拆分后 `run_agent` 在 `loop` 模块内解析 `AgentRepl`，原补丁目标不再生效；仅改 patch 目标，断言未变。

**遗留问题**（如有）：
- 无。

**验收要点**：
- `AgentLoop`/`AgentRepl`/`run_agent`/`_run_action`/`_run_actions` 全部 re-export；`_stagnates` 连续 3 次触发；标签流式抑制正常。

## 批次 4 · Context 层分裂 + 预算统一 · 936f3b5 · 2026-10-27 18:44

**关键改动**：
- 9 个文件变更（新增 3、重命名/session 1、删除 1、修改 4）；新增 `vanitas/context/{__init__,session,budget,collapse}.py`，删除顶层 `vanitas/session.py`。
- `chat.py::_run_turn` 去掉 `budget` 位置参数，改用 `ContextBudget.from_config(...)`；`repl.py` 仅切换 import。
- 涉及模块：context、chat、repl、tests。

**DoD 通过情况**：
- 10/10 条命令通过。
- 单测：Ran 353 tests, OK (skipped=1)。

**兜底补丁**（如有）：
- `tests/test_session.py` 的 import 改为 `vanitas.context`；两处 `mock.patch("vanitas.session.*")` 改为 `vanitas.context.session.*`。
- `tests/test_chat.py`、`tests/test_agent_advanced_tools.py` 的 `Session`/`COLLAPSED_OBSERVATION_MARKER` import 改到 `vanitas.context`。
- 以上均只改导入/patch 路径，断言未变。

**遗留问题**（如有）：
- 任务书折叠 DoD 把 `<tool_result>` 放在 cutoff 之外，原脚本返回 0；已用“较早位置放超长工具观测”的等价脚本验证 `n=1`，未修改实现。
- `grep "from .session import"` 仍会命中 `context/budget.py` 的 TYPE_CHECKING 子模块引用，属于新包内合法引用，不是旧顶层路径。

**验收要点**：
- `ContextBudget` 与 `context_budget` / `chat_max_tokens` 返回值一致；`Session.collapse_intermediate_observations` 签名不变；`metrics.py` 零改动。

## 批次 5 · Agent FSM 重写 + 停滞策略化 + 失败签名幂等 · 2830f0c · 2026-10-27 18:52

**关键改动**：
- 4 个文件变更（新增 1、修改 3）；新增 `vanitas/agent/failed.py`；`stagnation.py` 策略化；`loop.py` 接入 `FailedSignatures`。
- 涉及模块：agent。

**DoD 通过情况**：
- 6 条自检脚本 + 10 条 DoD 命令通过。
- 单测：Ran 353 tests, OK (skipped=1)。

**兜底补丁**（如有）：
- 无。

**遗留问题**（如有）：
- 任务书 5.3/DoD 的“自然语言完成”脚本手动向 stream 写字符，按现有语义 `stream.emitted=True` 时不会调用 `finalize`；已改用不写 stream 的等价脚本验证 finalize，未修改实现。
- 5.1 停滞等价脚本按原文执行通过；5.2 FailedSignatures、5.5 属性、5.6 AgentRepl 转发均通过。

**验收要点**：
- FSM 四组分支（自然语言/只读自动/写确认/拒绝）通过；连续 3 次停滞熔断；失败签名不二次确认；`FailedSignatures` 语义与旧 dict 等价。

## 批次 6 · 流式 chunk 化 + 性能微优化 · 6e8c1b5 · 2026-10-27 19:04

**关键改动**：
- 6 个文件变更（修改 4、新增 2）；`stream.write` 快速路径、`estimate_tokens` 双路径缓存、`_compile_pattern` 缓存、`Session.message_tokens` 缓存与失效、微基准入库。
- 涉及模块：agent.stream、metrics、tools.fs、context.session、tests/benchmarks。

**DoD 通过情况**：
- 10/10 条命令通过。
- 单测：Ran 353 tests, OK (skipped=1)。

**兜底补丁**（如有）：
- 快速路径除 `<` 外额外排除 `\n`/`\r`：任务书条件只排除 `<`，但 chunk 内换行后接 `{`/`[` 会绕过行首 JSON 抑制，破坏既有 `test_legacy_line_start_json_is_suppressed` 语义。
- `Session.trim` 在每次删除后失效缓存（任务书只要求 “dropped > 0 时失效”）；否则 while 循环会读取旧缓存并过度删除。
- 任务书 `estimate_tokens` 期望值 `"Hello 世界"=2`、`"\n"*100=100` 不正确；按旧实现实测改为 4、25。
- 任务书 `message_tokens` 用例用两次 `pop_last_user` 清空不成立；改用 `pop_last_turn` / `clear_messages` 验证失效。

**遗留问题**（如有）：
- 微基准：plain 4k x50 0.0059s、mixed x50 0.0125s（本机一次观测，非硬门槛）。

**验收要点**：
- `estimate_tokens` 与旧实现逐字等价；缓存命中正常；grep 缓存与错误文案不变；`Session.message_tokens` 在增删/折叠/裁剪后正确失效。

## 批次 7 · REPL 层分裂 + 体验提升 · dfbfe0c · 2026-10-27 19:24

**关键改动**：
- 7 个文件变更（新增 4、重命名 1、删除 1、修改 1）；新增 `vanitas/repl/{__init__,commands,watchdog,status,lifecycle,loop}.py`，删除顶层 `vanitas/repl.py`。
- 提示符改为 `模型@档位 [ctx: N%] › `；`/status` 默认 4 行紧凑、`-v` 完整；`/help` 四组；`/model` 三段进度。
- 涉及模块：repl、agent、tests。

**DoD 通过情况**：
- 12/12 条命令通过。
- 单测：Ran 353 tests, OK (skipped=1)。

**兜底补丁**（如有）：
- `tests/test_agent_advanced_tools.py` 的提示符断言从旧 `[ctx: N% | Xk/Yk]` 映射为新 `[ctx: N%]`；`test_status_reports_window_gpu_process_and_rates` 改调 `-v`，保留原完整输出断言。
- `repl/__init__.py` 额外暴露 `guard` / `launcher` / `tools` / `LlamaClient`，兼容既有测试补丁目标。
- 紧凑 status 第二行改为直接拼接 `gpu_text`，避免 `Palette.meta` 前缀造成 `· ·`。

**遗留问题**（如有）：
- 无。

**验收要点**：
- 提示符 `qwen_coder@1 [ctx: 24%] › `；四组帮助；`/status` 4 行；`AgentRepl` 继承与 `_confirm_actions` 未变；`repl.py` 已删除。

## 批次 8 · CLI 收尾 + 文档一致性 + 全链路冒烟 · batch-8-done · 2026-10-27 19:40

**关键改动**：
- 7 个文件变更（新增 2、修改 5）；更新 README、项目现状第 25 节、用户手册、AGENTS、总设计附录；新增 `tests/smoke_cli.py` 与本报告。
- 涉及模块：docs、tests；`vanitas/` 零改动。

**DoD 通过情况**：
- 6/6 条命令通过；`python -m tests.smoke_cli` 10/10。
- 单测：Ran 353 tests, OK (skipped=1)。

**兜底补丁**（如有）：
- 无源码补丁；任务书 25.10 草案列 16 项，按“17 项”要求补 `/profile 2` 一项。
- 用户手册第 216 行旧提示符描述同步为 `[ctx: N%]`，避免同一事实冲突。

**遗留问题**（如有）：
- 真机 25.10 的 17 项冒烟需用户手工执行；未启动 llama-server。
- 批次 5 的 20 组真实模型对照矩阵仍待项目负责人手工执行。

**验收要点**：
- 文档一致性 spot check 通过；`git diff --stat HEAD~1 -- vanitas/` 为空；smoke 10/10。







