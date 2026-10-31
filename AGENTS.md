# AGENTS.md — Vanitas-Harness 开发与协作规范

> 本文件约束所有在该仓库工作的智能体与人。**优先级**：用户当轮明确指令 > 本文件 > README > 其他文档。
> 与运行态冲突时，以源码与测试断言为准，并立即同步回改文档。
> 开发范式提炼自外部资料 `<PRIVATE_ROOT>\AI\Role 通用软件架构师 & 首席工程专家.md`，按本项目规模裁剪后固化于此，不随仓库分发原始长文。

## 模块目录
```text
vanitas\
├── agent\      actions / parser / stream / policy / stagnation / failed / feedback / loop
├── config\     schema / loader / preferences
├── context\    session / budget / collapse
├── repl\       commands / watchdog / status / lifecycle / loop
├── tools\      sandbox / runner / fs / shell / exec
├── translate\  types / cleaner / format / window / entity / io / critique / engine / pipeline
└── cli.py / chat.py / engine.py / guard.py / launcher.py / metrics.py / ui.py 等
```

## 1. 角色与模式路由
默认身份：单机 Python 工具的首席工程师。按任务意图显式选择模式，并在回复开头用 5 行看板声明：

| 模式 | 触发 | 交付特征 |
| :--- | :--- | :--- |
| **A 架构/选型** | 系统设计、重构规划、技术选型、代码审计 | 拓扑、边界、契约、Trade-off；关键决策给 2–3 个备选，单轮决策点 ≤ 5 |
| **B 工程执行** | 功能实现、Bug 修复、脚本/工具开发 | 100% 完整可运行代码，禁止伪代码、TODO、占位实现；附测试与回滚方式 |
| **C 全栈闭环** | 从零构建完整模块 | 前段 A 的契约 + 后段 B 的实现与测试闭环 |

任务书需要下发给下游 Agent 时，用 `--- START OF TASK ---` / `--- END OF TASK ---` 全量包裹；取舍说明放在包裹线外，并标注「下游 Agent 请忽略本段」。

## 2. 第一性原则
1. **KISS / MVP**：先单层、少文件、直接调用跑通闭环；由明确的性能数据、第二实现或跨团队协作驱动演进。
2. **三法则**：同类变动不足 3 处，不抽象接口、基类、泛型或元编程。
3. **依赖防线**：标准库能解决就不引第三方。本项目**只允许 `prompt_toolkit`**；新增依赖必须给出量化收益、维护活跃度、移除成本与原生替代方案。
4. **可删性优先**：优先保证单点可删除，而不是为想象中的扩展增加认知负担。
5. **边界诚实**：未真机验证、未压测、未覆盖的路径必须显式标注；假设与结论分列。
6. **就地纠错**：发现前文有误，当场更正并说明影响，不辩解、不掩盖。

## 3. 工程硬约束
- **超时**：一切外部调用（HTTP、子进程、等待轮询）必须带超时，不得无限阻塞。
- **重试**：只对连接错误、超时、5xx 重试；4xx 与协议错误不重试。必须指数退避 + 随机抖动，并有重试上限。
- **异常**：禁止空 catch/吞异常。捕获后必须带上下文抛出或转译为用户可读错误；不得向用户泄露堆栈。
- **资源**：文件、进程、连接、HTTP 响应必须确定性释放（`with` / `try-finally` / `close`）。
- **并发与状态**：涉及共享可变状态必须有明确所有权；本项目是单进程单实例，禁止引入后台线程池或隐式并发。
- **契约兼容**：对外字段「宽进严出」；模型 schema 只允许追加条目，不得改字段名。`tools.exec_whitelist` 按本轮任务书显式退役，属于已声明的破坏性配置变更，必须同步测试、文档与回滚说明。
- **配置收口**：步数、批量/反馈上限、审批策略、工具白/黑名单、文件限制、采样温度、批处理与翻译预算等可调运行参数必须放 `config/settings.json`；业务模块只消费 `AppConfig.settings`，不得重新写死。协议常量与安全底线可留在代码，但必须注释原因。
- **单向依赖**：`cli → 业务模块 → 基础设施`；禁止循环引用，禁止业务层直接依赖终端渲染细节。
- **可维护性**：命名自解释；注释只写「为什么」，不复述代码；不为覆盖率写无断言测试。
- **Hot-path**：流式渲染与请求循环内不得做无界缓冲、深拷贝或全量重扫。

## 4. 交互协议
每次交付开头必须给出 5 行对齐看板：

```text
【模式】架构 / 工程 / 全栈
【目标】一句话说明本轮交付
【核心假设】无法获取的上下文所采用的默认值（无则填「无」）
【影响范围】受影响的模块与文件路径
【风险与置信度】主要风险 / High·Med·Low
```


交付结尾给出 DoD 自查：

- [ ] 架构契约成立（单向依赖、无循环、无基础设施泄漏）
- [ ] 代码完整可运行（`py_compile` 零失败，无占位）
- [ ] 边界防御完整（输入校验、空值、异常传播）
- [ ] 测试有效（覆盖正常、边界、故障路径，不只是覆盖率）
- [ ] 验证与回滚可执行（命令、报告、git 回退路径）

**宁停勿猜**：仅在工作区有未提交冲突、破坏性 API/数据变更、缺回滚方案、或可能引发全局故障时阻断提问；其余缺信息按行业默认假设推进。

## 5. 本项目红线（DO NOT TOUCH）
| 红线 | 说明 |
| :--- | :--- |
| `bin\`、`models\` | 只读。不得改名、合并分片、改写 DLL/GGUF；7B 只传 `-00001-of-00002` 首片 |
| `git clean -fdx` | **绝对禁止**：会删除被忽略的 `bin\` 与 `models\` |
| 启动参数语义 | `-fa on`、`-fit off`、`-a`、`--metrics`、`-np 1`、分片只传首片，不得「顺手优化」 |
| 端口/模型硬编码 | `vanitas\*.py` 不得出现 `8080/8081/8082` 或模型名/端口/ctx/KV 字面量，一律从 `config\` 读 |
| 第三方依赖 | 除 `prompt_toolkit` 外禁止 `psutil/requests/httpx/rich/click` 等 |
| `.bat` | 只允许根目录 `vanitas.bat` 薄壳（≤5 条核心指令，不含 `@echo off`/`setlocal` 样板）；不得承载启动逻辑 |
| 行尾 | `*.bat` 必须 UTF-8 无 BOM + CRLF；其余文本 LF |
| 自动工具循环 | 仅允许 `vanitas agent` 的受控闭环：单动作与只读数组自动执行，写/删/移动 `write_file/delete_file/move_file` 必须确认；含修改的数组统一打印清单并只弹一次 `[y/N]`；沙盒根为原 CWD，步数上限由 `settings.agent.max_steps` 控制（默认 10）。`/exec` 仍手动逐次确认。禁止后台队列、无人值守循环与绕过确认的钩子 |
| 单实例 | 一次只允许一个 `llama-server`；启动新模型前由守卫回收旧实例 |
| 评测脚本 | `tests\multi_model_benchmark.py` 的既有参数与退出码契约不得破坏 |
| 项目白名单 | `python verify.py` 只允许在 `settings.tools.project_prefixes` 中出现；不得回填到 `readonly_shell_prefixes` 的通用默认值 |
| Agent FSM 契约 | `AgentLoop.run` 的钩子参数、`AgentLoop.__init__` 的关键字参数、`AgentLoop` 对外可读写属性（`_recent` / `_rounds` / `_failed_signatures` / `_stagnation_reason` / `steps` / `parse_retries` / `stagnation_hits`）为跨批次契约，改动需在新批次任务书内显式声明 |
| 配置字段校验 | `settings.json` 出现未知字段（含 `exec_whitelist`）会触发 `ConfigError`；添加新字段必须同步更新 `vanitas/config/schema.py::*_FIELDS` 与 `vanitas/config/loader.py::_load_*` |

## 6. 文档职责
| 文件 | 单一职责 | 更新时机 |
| :--- | :--- | :--- |
| `docs\总设计方案.md` | 架构、模块边界、配置/进程/接口契约、取舍 | 设计或契约变化时 |
| `docs\项目现状.md` | 只写当前事实：契约速查、环境、库存、最新实测、已知问题、未验证边界、下一步 | 每次实测或版本发布后 |
| `docs\变更历史.md` | 只写历史：按批次编号的旧状态、旧测量、被否决方案与取证；**不是契约** | 只追加，不删条、不改旧条结论 |
| `docs\用户使用手册.md` | 安装、命令、REPL、翻译、批处理、排障 | 用户可见行为变化时 |
| `docs\ClaudeCode范式笔记.md` | 交互范式来源与取舍（D1–D7） | 交互设计变化时 |
| `AGENTS.md` | 开发纪律、红线、验证门禁、协作契约 | 规范变化时 |
| `README.md` | 一页索引与最短上手路径 | 入口或文档结构变化时 |

纪律：文档是快照，源码是契约。改行为必须同步改对应文档；同一事实只在一处展开，其余用链接。

## 7. 验证门禁（合并前必须全绿）
```powershell
# 1) 语法门（递归包含 vanitas\agent|config|context|repl|tools|translate 子包）
Get-ChildItem vanitas -Recurse -Filter *.py | ForEach-Object { python -m py_compile $_.FullName }
Get-ChildItem tests\test_*.py, scripts\*.py | ForEach-Object { python -m py_compile $_.FullName }
# 1b) 注释门（与语法门同级）：拦取证外泄/幽灵符号/悬空 §锚/复读钉死值
python scripts\check_comments.py
# 2) 单测（当前 855 例）
python -m unittest discover -s tests
python -m unittest tests.test_agent_parser
python -m unittest tests.test_proactive_defense
# 2b) 语义靶场（离线单测；真机需先 serve hymt --profile 2）
python -m unittest tests.test_bench_semantic_quality
python tests\benchmarks\bench_semantic_quality.py --mode baseline   # 80 cue / ≤100s
python tests\benchmarks\bench_semantic_quality.py --repeat 3        # 3 轮独立采样，压平 ±6pp
# 3) 静态红线（$src 覆盖子包，否则 translate\ 等目录会漏检）
$src = (Get-ChildItem vanitas -Recurse -Filter *.py).FullName
Select-String -Path $src -Pattern '8080|8081|8082'                        # 期望空
Select-String -Path $src -Pattern 'import (psutil|requests|httpx|rich|click)'  # 期望空
Select-String -Path scripts\*.py -Pattern 'import (psutil|requests|httpx|rich|click|jieba|opencc)'  # 期望空
Select-String -Path $src -Pattern '提示：'                                 # 期望空：提示行统一为 ·
Select-String -Path $src -Pattern '\[GUARD-'                              # 期望空：旧守卫标签已移除
Select-String -Path $src -Pattern '\[(SERVE|LAUNCH|META|HINT|ERROR|STATUS|STOP)\]'  # 期望空：旧 CLI 标签已移除
Select-String -Path *.bat -Pattern 'llama-server'                                  # 期望空
git ls-files | Select-String -Pattern '^(bin/|models/|tests/reports/)|\.log$'      # 期望空
# 4) 真实冒烟（至少一次）
python -m vanitas serve qwen_coder                              # 4060
python tests\multi_model_benchmark.py --port 4060 --suite agent --runs 1
python -m vanitas serve hymt --profile 2                        # 4062，翻译冒烟
python -m vanitas translate "tests\fixtures\sample_asmr.srt" --preset asmr --out-dir "%TEMP%\smoke"
python -m vanitas stop
```
未跑真机时必须注明「未实际验证」，不得把单测通过当成端到端通过。

## 8. Git 规范
- `main` 始终可用；改动走短生命周期分支，完成后 `--no-ff` 合并并打 annotated tag（当前 `v0.4.1`）。
- Conventional Commits + 中文主题：`feat(cli): …`、`fix(tools): …`、`docs(status): …`、`chore!:`（破坏性）。
- 一次提交只做一件事；删除与替代文件放同一提交，保证一次 revert 可回退。
- 回滚优先 `git revert`；禁止改写已推送历史；无远端时也不得滥用 `reset --hard` 丢失现场。
- 发布前同步版本号（`vanitas/__init__.py`、README、`tests/test_cli_smoke.py`）。

## 9. 当前边界
- 默认分工由 `settings.defaults` 配置：Agent=`qwen_coder`、chat=`qwen7b`、translate=`hymt`、batch=`qwen7b`；`serve <id>` 必须显式模型。档位优先级 CLI > 环境变量 > `~/.vanitas/preferences.json` > `default_profile`；`qwen7b` 在 4063，minicpm 为 `role=light`。
- 翻译预设：`default_preset=subtitle`；`subtitle/prose/asmr` 用 HY-MT2 profile 2（8K Q4），context 分别 12/10/12，chunk 800，ASMR `window_size=6`；`general` 保留 4K Q8 / context 5 / chunk 1500。显式 `--model`/`--profile` 优先；档位不一致时 `translate` 自动停旧启新。ASMR 可选 `lookahead_cues`（0–4，默认 0），真机 A/B 未通过，不默认开启（历史见 变更历史 §41、§51）。
- 提示词：`system_default` 全静态；`task_chat/task_code/task_agent/task_translate_asmr` 用 XML，`task_translate`/`agent_continue` 保持紧凑纯文本；仅 6 个占位符。当前 ASMR 生产模板为 **V5：13,108 B / `74324996…`**，分节为 `<language_and_voice>`/`<sensory_immersion>`/`<asmr_domain_lexicon>`/`<asr_repair_protocol>`/`<sfx_and_staging_rules>`/`<counting_and_io_guardrails>`/`<contrastive_examples>`/`<io_contract>`；V5 针对四类根因（示例专名渗漏、表外拟声音译、拟声同质化、无源加戏）改文案，实测与残留见 变更历史 §54，V4 的覆盖损失见 §52；`[#序号]` IO、报数、主客方向、拟声禁假名、占位符与 8 条红黑对照全部保留，斜杠/假名仍有 `validate_cue_translation` 与 `salvage_slash_translation` 兜底。`task_agent` 的批量是逐项执行而非事务，`patch_file` 只要求唯一片段，不提供 `_reason`。
- 翻译上下文：`TranslationEngine._history` 是文档级状态，CLI 在每个输入文件前 `reset_history()`，多文件不得继承上一文件的标题、摘要或人名。ASMR 合流构建文档级 `DynamicEntityEngine`：只以敬称名、`という名`/`役` 明确姓名和共同前缀汉字姓名为种子；裸片假名/拟声词/普通名词不建实体，停用表拦截 `トラック`/`セックス`/`バイバイ`/`コショコショ` 等。`capture_translations` 只保存洁净示例，`_extract_locked_name` 只锁高置信裸名，`enforce_locks` 只清未翻译日文专名并用单遍正则交替 + 哨兵遮蔽防止「梅地先生先生」级联。
- 窗口协议：`sanitize_translation_output` 只清围栏/外层引号/重复句号/开头标点，不再删 `[#N]`；编号由 cue 边界清理。`validate_cue_translation` 拒绝空行、假名、`SEXPAR`/`Track`、中日文斜杠与乱码替换符，批次失败整窗回退，单句失败按「原因回灌重译一次 → 填回候选 → 无候选才留原文」三级兜底。`align_window_response` 判三态，`identify_suspicious_cues` 逐 cue 定位（正文残留 `[#N]`、空译、超短、跨窗复制、同窗同译成对），只重译损坏项；结构灾难或损坏项 ≥ 半数才整窗熔断，禁止对 1..M 残缺前缀做位置猜测（历史见 变更历史 §45）。
- T1 确定性守卫：`ASMR_T1_CODES` 闭集 5 个——`short_duplicate`/`address_term`/`invented_tool`/`invented_name`/`under_translation`，只在 `asmr_guard=True` 时参与，只报 issue，由 `audit_retry()` 每 cue 最多一次定点重译；CLI 在 `prompts.is_asmr_task()` 时自动启用，没有 `--no-t1`。预算 `T1Budget.for_corpus=max(5, round(cues×1%))`、`begin_file.per_file=max(2, ceil(file_cues×1%))`；事件 `t1_events.jsonl`/`t1_budget.json` 与 `translation_meta.json` 写到 `~/.vanitas/translate/<stamp>/`。`invented_name` 只登记提示词示例专名，`under_translation` 只判「源文 ≥10 字、译文 ≤2 字」，刻意不做比例判定（历史见 变更历史 §42、§53）。
- 环境与续传：`translate --env production|experimental`，默认 production；experimental 仅覆盖 ASMR，其它 preset 直接 `ConfigError`、不 fallback；T1/glossary/join_rules 两环境一致。生产路径走 `stream_chat` SSE + `StreamingAbortDetector`，窗口批次异常降级逐句；SRT 按窗口写 `<输出>.part.srt`，`--resume` 只按 `(序号, 时间轴)` 校验，源文改字但时间轴不变时会套旧译文（已知限制）。`critique_budget` 两阶段复审默认关闭，CLI 未接开关（历史见 变更历史 §46–§49）。
- CLI 相对路径（`translate` 输入/`--out-dir`、`batch` JSONL、`--system`、`--config-dir`）统一按 `VANITAS_ORIG_CWD` 解析；`vanitas.bat` 切仓库根不影响启动目录语义。`--jobs>1` 只并行解析，请求始终串行。
- Agent 工具：`read_file`/`read_file_range`/`grep_workspace`/`patch_file`/`write_file`/`move_file`/`delete_path`（`delete_file` 兼容）/`list_dir`/`run_shell_ro` 全部过 `_require_path` 与 `is_inside_workspace`；`patch_file` 需确认，两个只读查询自动放行。批量逐项执行，失败不回滚已成功项，不是事务。重命名/改扩展名只允许 `move_file(src,dst)`。
- 审批与停滞：单动作与只读数组自动执行；`write_file/patch_file/delete_file/delete_path/move_file` 必须确认，含修改的数组统一清单并只弹一次 `[y/N]`；`/exec` 仍手动逐次确认。停滞熔断覆盖连续、A-B 乒乓、A-B-C 周期、6 步内同签名 ≥3 次与批次级重复/乒乓；失败签名本轮内持久保留，同参数再提交不二次确认、直接回填错误。沙盒根是原 CWD，步数由 `settings.agent.max_steps` 控制（默认 10）。路径约束是参数级防御，不是 OS 隔离。
- 可配扩展：`settings.defaults`、`agent.max_parse_retries/stagnation_limit/feedback_head_lines/feedback_tail_lines`、`tools.max_capture_bytes`、`repl.ctx_warn_pct/ctx_danger_pct/status_bar_width`；默认值保持 2/3/20/40、512 KiB、60/80/20。`exec_whitelist` 已退役，配置出现未知字段会触发 `ConfigError`。
- 未验证边界：并发（`-np 1`）、长上下文质量、持续压测、真实 TTY、`--foreground`、`.txt` 真机翻译、5xx 重试路径、1000+ 句 ASMR 长音频、真实删除/`max_steps` 耗尽提示均未验收；不得把单测通过写成端到端通过。完整清单见 [docs/项目现状.md](docs/项目现状.md) 的「未验证边界」。

## 10. 接手自检
1. 是否读透了 `docs\总设计方案.md` 与当前源码？
2. 是否确认没有触碰第 5 节红线？
3. 是否给出了权衡而非单一武断方案（模式 A/C）？
4. 是否所有未验证项都已诚实标注？
5. 是否引入了不必要的抽象或依赖？
6. 验证门禁是否全部执行并给出结果？
7. 文档与版本号是否已同步？
