# AGENTS.md — Vanitas-Harness 开发与协作规范

> 本文件约束所有在该仓库工作的智能体与人。**优先级**：用户当轮明确指令 > 本文件 > README > 其他文档。
> 与运行态冲突时，以源码与测试断言为准，并立即同步回改文档。
> 开发范式提炼自外部资料 `<PRIVATE_ROOT>\AI\Role 通用软件架构师 & 首席工程专家.md`，按本项目规模裁剪后固化于此，不随仓库分发原始长文。

---

## 模块目录

```text
vanitas\
├── agent\      actions / parser / stream / policy / stagnation / failed / feedback / loop
├── config\     schema / loader / preferences
├── context\    session / budget / collapse
├── repl\       commands / watchdog / status / lifecycle / loop
├── tools\      sandbox / runner / fs / shell / exec
├── translate\  types / cleaner / format / window / entity / io / engine / pipeline
└── cli.py / chat.py / engine.py / guard.py / launcher.py / metrics.py / ui.py 等
```

---

## 1. 角色与模式路由

默认身份：单机 Python 工具的首席工程师。按任务意图显式选择模式，并在回复开头用 5 行看板声明：

| 模式 | 触发 | 交付特征 |
| :--- | :--- | :--- |
| **A 架构/选型** | 系统设计、重构规划、技术选型、代码审计 | 拓扑、边界、契约、Trade-off；关键决策给 2–3 个备选，单轮决策点 ≤ 5 |
| **B 工程执行** | 功能实现、Bug 修复、脚本/工具开发 | 100% 完整可运行代码，禁止伪代码、TODO、占位实现；附测试与回滚方式 |
| **C 全栈闭环** | 从零构建完整模块 | 前段 A 的契约 + 后段 B 的实现与测试闭环 |

任务书需要下发给下游 Agent 时，用 `--- START OF TASK ---` / `--- END OF TASK ---` 全量包裹；取舍说明放在包裹线外，并标注「下游 Agent 请忽略本段」。

---

## 2. 第一性原则

1. **KISS / MVP**：先单层、少文件、直接调用跑通闭环；由明确的性能数据、第二实现或跨团队协作驱动演进。
2. **三法则**：同类变动不足 3 处，不抽象接口、基类、泛型或元编程。
3. **依赖防线**：标准库能解决就不引第三方。本项目**只允许 `prompt_toolkit`**；新增依赖必须给出量化收益、维护活跃度、移除成本与原生替代方案。
4. **可删性优先**：优先保证单点可删除，而不是为想象中的扩展增加认知负担。
5. **边界诚实**：未真机验证、未压测、未覆盖的路径必须显式标注；假设与结论分列。
6. **就地纠错**：发现前文有误，当场更正并说明影响，不辩解、不掩盖。

---

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

---

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

---

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

---

## 6. 文档职责

| 文件 | 单一职责 | 更新时机 |
| :--- | :--- | :--- |
| `docs\总设计方案.md` | 架构、模块边界、配置/进程/接口契约、取舍 | 设计或契约变化时 |
| `docs\项目现状.md` | 环境、库存、实测矩阵、已知问题、未验证边界、下一步 | 每次实测或版本发布后 |
| `docs\用户使用手册.md` | 安装、命令、REPL、翻译、批处理、排障 | 用户可见行为变化时 |
| `docs\ClaudeCode范式笔记.md` | 交互范式来源与取舍（D1–D7） | 交互设计变化时 |
| `AGENTS.md` | 开发纪律、红线、验证门禁、协作契约 | 规范变化时 |
| `README.md` | 一页索引与最短上手路径 | 入口或文档结构变化时 |

纪律：文档是快照，源码是契约。改行为必须同步改对应文档；同一事实只在一处展开，其余用链接。

---

## 7. 验证门禁（合并前必须全绿）

```powershell
# 1) 语法（递归包含 vanitas\agent|config|context|repl|tools|translate 子包）
Get-ChildItem vanitas -Recurse -Filter *.py | ForEach-Object { python -m py_compile $_.FullName }
Get-ChildItem tests\test_*.py, scripts\*.py | ForEach-Object { python -m py_compile $_.FullName }

# 2) 单测（当前 686 例）
python -m unittest discover -s tests
python -m unittest tests.test_agent_parser

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
python -m vanitas serve qwen7b
python tests\multi_model_benchmark.py --port 4060 --suite agent --runs 1
python -m vanitas stop
```

未跑真机时必须注明「未实际验证」，不得把单测通过当成端到端通过。

---

## 8. Git 规范

- `main` 始终可用；改动走短生命周期分支，完成后 `--no-ff` 合并并打 annotated tag（当前 `v0.4.1`）。
- Conventional Commits + 中文主题：`feat(cli): …`、`fix(tools): …`、`docs(status): …`、`chore!:`（破坏性）。
- 一次提交只做一件事；删除与替代文件放同一提交，保证一次 revert 可回退。
- 回滚优先 `git revert`；禁止改写已推送历史；无远端时也不得滥用 `reset --hard` 丢失现场。
- 发布前同步版本号（`vanitas/__init__.py`、README、`tests/test_cli_smoke.py`）。

---

## 9. 当前边界

- 默认分工由 `settings.defaults` 配置：Agent=`qwen_coder`、chat=`qwen7b`（非 Coder）、translate=`hymt`、batch=`qwen7b`；`serve <id>` 必须显式模型。`qwen7b` 在 4063，minicpm 仍为 `role=light`。`~/.vanitas/preferences.json` 记忆档位，优先级 CLI > 环境变量 > 偏好 > `default_profile`。
- 翻译预设：`translate.default_preset=subtitle`；`subtitle/prose/asmr` 使用 HY-MT2 profile 2（8K Q4）、context 12/10/12、chunk 800；`general` 保留 4K Q8 / context 5 / chunk 1500。ASMR 预设可选 `lookahead_cues`（0–4，默认缺省 0）；本轮真机 A/B 未通过，不默认开启。显式 `--model`/`--profile` 优先；档位不一致时 `translate` 自动停旧启新。
- 提示词：`system_default` 全静态；`task_chat/task_code/task_agent/task_translate_asmr` 使用 XML，`task_translate`/`agent_continue` 保持紧凑纯文本；仅 6 个占位符。`task_translate_asmr` 另含 `<asmr_domain_lexicon>`（采耳道具/解剖方位/互动动作/情境用语/声效词），保留 `梵天`/`膝枕`/`添寝`/`特典` 等圈内词并约束自然搭配，附 `お騒がせ` 自谦方向；新增 `<sensory_immersion>` 约束贴耳体感、体感具象、动作本土化与跨行语流，并明确普通片假名不是专名、拟声词不得保留假名、禁止 `SEXPAR`/`Track` 与斜杠并列，不删除既有领域词典/ASR/计数/IO 契约。`task_agent` 的批量是逐项执行而非事务，`patch_file` 只要求唯一片段、不强制固定上下行数；不提供 `_reason` 参数。
- 翻译上下文：`TranslationEngine._history` 是文档级状态；CLI 在每个输入文件前调用 `reset_history()`，多文件不得继承上一文件的标题、摘要或人名。`sanitize_translation_output` 只清代码围栏、外层引号、重复句号与开头 `，、；：`，**不再删除 `[#N]`**；编号由 cue 边界的 `strip_window_id_prefix`/`postprocess_cue_text` 清理。ASMR 合流（`window_size>1`）构建文档级 `DynamicEntityEngine`：只有敬称名、`という名`/`役` 明确姓名和共同前缀汉字姓名为种子；裸片假名/拟声词/普通名词不建实体，停用表拦截 `トラック`/`セックス`/`バイバイ`/`コショコショ` 等。`capture_translations` 只保存洁净示例；`_extract_locked_name` 只锁高置信裸名；`enforce_locks` 只清未翻译日文专名；平假名写法仅通过明确姓名做别名。合流 `context_window=12` 按 cue 记录（每窗 `restore_history(批前快照 + 已接受 cue)`，不再调用 `record_window_history`），不再把 6 条窗口放大成 72 条；批次接受前用 `_copies_history` 查跨窗复制，命中整窗回退。`postprocess_cue_text` 之外新增 `validate_cue_translation`：空行、假名、`SEXPAR`/`Track`、中日文斜杠不写回；批次失败整窗回退，单句失败保留原文并记 failure。合流编号重置恢复、结构守卫、单句回退规则不变。ASMR 合流若模型每窗从 `[#1]` 重编号，`recover_window_response()` 仅在编号为 `1..K` 且数量一致时按位置恢复；**P2 局部回退**：`align_window_response()` 先判三态（完全对齐 / 缺项可用 / 灾难），`identify_suspicious_cues()` 定位到具体 cue（正文残留 `[#N]`、非空原文对应空译文、超短漏译、跨窗复制、同窗不同原文相同长译文——最后一类**成对**标记），只对损坏项发单编号 `[#N] 原文` 重译，其余批次译文直接采纳。整窗熔断条件：结构灾难（重复/未知编号、乱序、无编号、1..M 残缺前缀——该形状与整窗重编号文本上不可区分，禁止位置猜测）或**损坏项 ≥ 半数**（半窗已坏说明整批退化，未标记项也不可信）。单句回退多编号/多行仍记失败并保留原文。
- 质量检查：`scripts\qc_srt.py` 三层分流——繁体/行首标点静默计数，cue 数/序号/时间轴/`[#N]`/重复译文为 error，ASR 疑词/语境“您”/未授权动作/长度/空译/假名/可疑拉丁词/斜杠为 actionable；`duplicate_translation` 对拟声词与近义源文先做 `ッ`/`ー` 归一化，阈值 0.60；`--json` 含 `summary` 与 `flagged_cues`，默认仅 error 返回 `1`，`--strict` 时 actionable 也返回 `1`。QC 的“您”检查改为文档语域优先：service 不报、intimate 必报、neutral 仍走原单句启发式。
- 确定性质量守卫（P1）：`vanitas\quality.py`（纯标准库，不 import `vanitas.translate`）提供 ASMR 术语表（`config\glossary\asmr.tsv`，`source<TAB>preferred<TAB>forbidden`）、文档语域检测、`您→你` 人称策略与文档级重复索引（精确哈希 + 3-gram 倒排，源文相似 ≥0.75 视为合法重复，译文 ≥12 字才判定）。`translate_srt` 的 `window_size<=1` 与合流两条路径统一走 `postprocess_cue_text → enforce_locks → glossary.apply → apply_pronoun_policy → repair_first_person_name → repair_quote_private → validate_cue_translation`；审计命中每个 cue 最多一次 `_translate_single_cue` 定点重译，前后用 `history_snapshot`/`restore_history` 隔离，失败不写 `failures`、保留原译；fallback 逐句路径硬校验通过后同样做 duplicate/honorific 审计与一次重译，最终接受项才 `index.add`。`SINGLE_RETRY_HINT` 要求人名/称谓/助词/片假名名词全部译成中文、禁止整句日文。合流每窗无条件 `restore_history(批前快照 + 已接受 cue)`，`translate_srt` 不再调用 `record_window_history`（方法保留供兼容）。术语与 Prompt 只新增 `アマアマ→甜腻`、授受/被动方向、引用句 3 条规则，`_render` 不注入术语表。跨 cue 语义、主客方向与引用归属已由 P2a 窄规则覆盖一部分；未覆盖的 residual 见 `docs\项目现状.md` 第 39、42 节。
- P2a 高置信语义审计与一次定点重译（2026-10-29）：`DuplicateIndex.find_match()` 返回 `DuplicateMatch(cue, source, translation)`，`find()` 只委托 `.cue`，阈值/顺序/并列规则不变；`audit_duplicate` 提示同时带重复条与本条原文片段（各 24 字）。重复索引另加窄规则：源文规范化相似度 ≤0.5 且译文共同前缀 ≥6 字时命中，可检测 `自由談 #80` 这类译文整体相似度不足的长共同开头；但模型一次重译仍可能复用，属“检测覆盖、修复不保证”，QC/人工仍须复核，#80 留作 residual。不改 0.75/0.80 全局阈值。五类窄规则 `customer_direction` / `quote_private` / `negative_question` / `first_person` / `tag_question` 全部 AND 触发、空源/空译直接放行；`quote_private` 的坏模式收窄为 `让我叫|我的名字|让你叫`，单独“叫我”不再自动判坏（“他说，叫我雫就好”放行）。`translate_srt` 的 `finalize` 在人称策略之后、`validate` 之前增加窄修复：仅源含 `下の名前で呼んであげて` 且源无第一人称时，把 `我的名字/让我/叫我` 收敛为 `她的名字/让她/叫她`（`学生 #8`）。`audit_translation` 固定按 duplicate → customer_direction → quote_private → negative_question → first_person → tag_question → honorific 汇总（T1 开启时在 duplicate 后插 `short_duplicate`、在 tag_question 后插 `address_term`/`invented_tool`），`combine_hints` 去重取前 2 条。单句、合流与 fallback 三处候选统一走 `candidate_ok()`（硬校验、空译、换行、长度、触发码与 duplicate 残余），每 cue 仍最多一次重译，异常保留原译且不写 failure；正确候选不产生额外调用。术语表追加 `お耳をマッサージ→耳朵按摩`（专用行在前）与 `マッサージ→按摩`，`耳かき` 不受影响。
- P2b ASR 断词修复（2026-10-29）：新增纯函数 `vanitas\asr_repair.py` 与 CLI `scripts\repair_asr_srt.py`，表 `config\asr_joins.tsv`（3 列、`#` 注释、坏行/重复键带 `文件:行号`）。`parse_srt` 后正文多行无空格拼接并压成单行；相邻 cue 按 `left.endswith(tail) and right.startswith(head)` 贪心合并，守卫为左前缀非空、timing 可解析、`right.start >= left.end`、gap ≤1000ms、合并后 ≤8s；合并后 start=左.start、end=右.end，右 cue 删除，序号连续重排 1..N，链式跑到不动点且二次运行 0 变更。CLI 默认 dry-run，`--write` 才写；`--out-dir` 必填且不得等于源目录，拒绝覆盖 `.zh.srt`/已存在目标/输出冲突；`--check` 有待合并返回 `1`，坏表/IO/覆盖返回 `2`；不原地、不产出 `.join.srt`。
- V5 跨 cue 右吸附（2026-10-29）：新增 `vanitas\cue_boundary.py`，共享规则器从 `asr_repair.py` 抽出；`asr_repair.py` 的 merge CLI 语义不变，但不进入默认 `translate_srt`。`TranslationEngine` 仅 ASMR 且 `config\asr_joins.tsv` 存在时暴露只读 `join_rules`；`translate_srt(..., join_rules=())`/`translate_file(..., join_rules=())` 为向后兼容可选参数，CLI 仅在 `window_size>1` 传入。`translate_srt` 在幻觉置空、`translatable` 构建后、`source_snapshot`/`detect_register` 前把左 cue 末尾 `tail` 右吸附到右 cue 开头（右 = `replacement + 右去 head`），cue 数/序号/时间轴不变；守卫为 `replacement == tail+head`、`tail≤2`/`head≤4`、tail 非助词、左前缀可收尾、`gap≤500ms`、timing 不重叠，多行源文保留换行；不满足只 skip。批量窗口另有 `_translation_is_too_short()`：源文规范化 ≥30 字且译文 ≤6 字整窗回退逐句。`QUOTE_PRIVATE_BAD_RE` 增加 `叫我叫`，`repair_quote_private()` 仅在 `私のことは…って` 场景收敛为“叫我”，在 `finalize` 的人称修复之后调用。另加 `audit_negative_question()`：源文 `〜ていませんか/ではありませんか` 的否定疑问若译文缺否定标记则报 `negative_question`，只做一次定点重译；不覆盖 `〜ませんか` 邀请句。硬校验因中日文斜杠失败时，`salvage_slash_translation()` 优先唯一中文项、否则取第一个非空项，避免把整句日文源写回；正常后处理的保守斜杠规则不变。首批只放行 1ms 治疗例；1479/5240ms 断词保留 residual。
- V6 后文参考（2026-10-29，未默认启用）：`TranslationEngine.set_lookahead()` 把下一窗最多 N 条原文、≤120 字追加到 `<reference_context>` 的 `【后文参考（仅用于理解指代/省略/语气，禁止翻译或写入当前编号）】`，`clear_lookahead()` 在批次返回的 `finally` 立即清空，不进 `_history`、不写回。后文不带 `[#N]` 标签，避免诱导模型把参考项当输入。`PRESET_FIELDS`/loader 校验 0–4，默认缺省 0；CLI 只读 preset 并透传。真机 dev 单文件 A/B 出现明显慢于基线/回退增多，未达到发布门槛，故默认关闭、保留为显式实验开关；不得把单元测试通过当作收益。
- E0 + T1-only（2026-10-30，主树 `c31e160`）：E0 `d2e801d` 在 `postprocess_cue_text`/`validate_cue_translation`/单句裸文本出口三处折叠连续空行，防止 1 条 cue 写成 2 个 SRT block；T1 `36471a2` 合入 `c31e160`（6 files / +2117−89，生产模板 12,784 B 未变，674 例 / 1 skip；环境隔离后 686）。T1 三 code `short_duplicate`/`address_term`/`invented_tool` 只报 issue，在 `finalize`/`validate` 后由 `audit_retry()` 做一次定点重译；`T1Budget.for_corpus` = `max(5, round(total_cues*1%))`、`begin_file` 的 `per_file` = `max(2, ceil(file_cues*1%))`（514→5、1404→14、712→8）；`total` 跨文件共享、`file_total` 每文件重置；CLI 在 `prompts.is_asmr_task(task_template)` 时自动 `asmr_guard=True`（production/experimental 均算），当前没有 `--no-t1`；旁路产物为 `t1_events.jsonl` 与 `t1_budget.json`。T2 c1 `2a1c65e` 仍是实验分支、不默认，实验模板经 `--env experimental` 显式选择；P4 fresh-blind 514 cue / 375 diff / 139 identical、三票外推 107<360 → `INVALID_MEASUREMENT_UNREACHABLE`，不是 PASS/FAIL、不是胜率；P5 same-run v2 QC 5/5（base 8/4、v1 6/5），相对 base 新增 error 0，残差 5 条均为 base 已有 duplicate；v2 的 5 条 actionable 是 `asr_term` 2 + `latin` 3（不是 5 条“您”；“您”7 条属 P4 candidate）；不得写成“语义收益已提升”。
- ASMR 生产/实验环境隔离（2026-10-30，merge `fc2b8c7`）：`translate --env production|experimental`，`translate.environment` 默认 production；`EnvironmentProfile`/`prompts.ASMR_ENVIRONMENT_TEMPLATES` 负责解析，production 原样、experimental 仅 ASMR 用 `translate_asmr_experimental`，其它 preset 直接 `ConfigError`、不 fallback；T1/glossary/join_rules 两环境均启用（`prompts.is_asmr_task`）。`--out-dir` 写 `translation_meta.json`（environment/task_template/prompt_sha256/prompt_bytes/preset/temperature/model/started_at）。主树 686 例/1 skip；生产模板 12,784 B/`CC2B4C82…`，实验模板 5,399 B/`7E589E42…`；T2 仍实验、不默认；P4 `INVALID_MEASUREMENT_UNREACHABLE` 与“语义收益未证实”不变。残留：环境名三处来源、batch 未路由、meta 仅 out-dir。
- 翻译目录：`translate` 的 `inputs` 接受文件、目录或混用；目录用 `os.walk(followlinks=False)` 递归，只收 `settings.translate.source_suffixes`，跳过目录内已有 `<原名><output_suffix>.<扩展名>` 产物，按真实路径去重。`--out-dir` 下目录输入保留相对层级，直接文件平铺；写出前拒绝目标冲突与覆盖输入。空目录/不可遍历目录在启动模型前返回 `5`；显式文件的存在性仍由 `translate_file()` 判定。批处理 UI 按文件显示 `[i/N] 文件 · done/total 段 · 百分比`，结束后由 `TranslationReport` 输出一个汇总块；默认 ≤12 文件列全部，超过只列异常，`translate -v` 显示全部；目标已存在只标 `覆盖` 不阻止。
- 文件名翻译：`--translate-filename`/`--translate-name` 同时翻译正文与文件名，输出 suffix 覆盖为空串（不加 `.zh`），正文仍只接受 `source_suffixes`；`--translate-filename-only`/`--translate-name-only` 只原地改名、不读内容、支持任意扩展名、拒绝 `--out-dir`。两者都用短正向 `FILENAME_SYSTEM_PROMPT`；`_mask_name_literals()` 把高置信编号、全大写/内部大写、版本号、`_临时` 白名单和装饰符号替换为 `[[K0]]` 占位符，模型翻译普通词后 `_restore_literals()` 逐字恢复。`sanitize_filename_stem()` 保留引号/括号/下划线、去 `文件名：`/`翻译：` 前缀；纯中文/纯技术名（`wav`）和去掉占位符后无可译文本的名称不调用模型。占位符丢失/重复、截断、空结果、恢复后超 80 字符，或含假名的名称完全未翻译时最多重试一次并回退原名；单文件回退不中止批处理。CLI 用 `Spinner` 显示 `文件名/目录名 i/N` 进度；正文模式回退且无 `--out-dir` 时跳过该文件避免覆盖源文件。
- 人类 `/exec`/`!` 使用原生 `shell=True`，不再受 `manual_exec_prefixes`/目录限制；`AgentTools.run_shell_ro` 的只读白名单与 `is_inside_workspace` 边界完全不变。
- AgentTools 新增 `patch_file`、`read_file_range`、`grep_workspace`；全部走 `_require_path`。`patch_file` 需确认，另两项只读自动放行；大文件优先 `read_file_range`/`grep_workspace`，小改优先 `patch_file`，多文件优先单轮动作数组；批量逐项执行，失败不回滚已成功项，不是事务。
- `delete_path` 递归删除文件/目录，`delete_file` 为兼容别名；工作区根目录、父级与越界路径硬拒绝，符号链接只删链接本身。`/undo`/`/pop`/`/rollback` 回滚最近一轮并释放 tokens。
- 提示符常驻 `[ctx: N%]`，`/status` 默认 4 行紧凑、`-v` 展示视窗进度条、GPU/进程、TTFT/tps 与 Agent 步数；会话超 65% 时自动折叠旧 `<tool_result>` 观测，动作签名与最近两轮保留。
- 停滞熔断覆盖连续、A-B 乒乓、A-B-C 周期、6 步内同签名 ≥3 次，以及批次级重复/乒乓；失败签名在本轮内持久保留，即使中间穿插只读读取，原参数再提交也不二次确认、直接回填错误。
- 重命名/改扩展名只允许 `move_file(src,dst)`；`patch_file` 只改内容。Prompt 明确禁止用 `patch_file` 绕文件名变更。
- 可配扩展：`settings.defaults`（四模式默认模型）、`agent.max_parse_retries/stagnation_limit/feedback_head_lines/feedback_tail_lines`、`tools.max_capture_bytes`、`repl.ctx_warn_pct/ctx_danger_pct/status_bar_width`。默认值保持 2/3/20/40、512 KiB、60/80/20。
- CLI 相对路径（`translate` 输入/`--out-dir`、`batch` 的 JSONL、`--system`、`--config-dir`）统一按 `VANITAS_ORIG_CWD` 解析；`vanitas.bat` 切仓库根不影响用户在启动目录使用相对路径。
- 并发（`-np 1`）、长上下文质量、持续压测、TTY 交互、`--foreground`、`.txt` 真机翻译、5xx 重试路径尚未验证；`.md` 已用 HY-MT2 做 2 文件目录小样本冒烟，`.mp4/.jpg` 已做仅改名冒烟。
- v0.4.1 的 16 色配色与三处覆盖式进度未做真人 TTY 验收（仅伪 TTY 单测 + 强制 `always` 抓字节）。
- `vanitas agent` 的默认入口、`{cwd}` 注入、只读列目录不确认、批量 `move_file` 清单与一次 `[y/N]`、三连输入不触发上下文裁剪、最终答复流式（实测 111 次写入）、解释文字后的尾部动作 JSON、`run_shell_ro` 拒绝写命令、工作区外读取拦截均已用 qwen7b 真机冒烟；工具调用已改为 `<tool_call>` 标签优先，`raw_decode`/栈平衡兜底与 `<tool_result>` 反馈由单测覆盖；qwen7b 已真机输出 `<tool_call>` 数组并一次确认创建 `a.txt`/`b.txt`。真实终端 `[y/N]` 观感、真实删除、模型在 shell 被拒后自动改用 `move_file`，以及 `max_steps` 耗尽提示仍未验收。
- `exec_whitelist` 已退役；Agent 的步数/批量/反馈/审批在 `settings.agent`，文件限制、只读 Shell、`/exec` 前缀与写参数拦截在 `settings.tools`。配置可调整策略，但破坏性命令与写参数黑名单的安全底线只能追加、不能删除。
- Agent 路径约束是参数级沙盒，不是 OS 级隔离；程序内部自行写文件仍需人工判断。
- 混合换行的 `.srt` 已修复：`parse_srt` 使用 `splitlines()`，`rebuild_srt` 统一目标 EOL（见现状第 17 节）。
- HY-MT2 显存余量约 500 MiB，是本机最紧的模型。
- 完整清单与取证见 [docs/项目现状.md](docs/项目现状.md) 第 5、6 节；不得把未验证项写成已结论。

---

## 10. 接手自检

1. 是否读透了 `docs\总设计方案.md` 与当前源码？
2. 是否确认没有触碰第 5 节红线？
3. 是否给出了权衡而非单一武断方案（模式 A/C）？
4. 是否所有未验证项都已诚实标注？
5. 是否引入了不必要的抽象或依赖？
6. 验证门禁是否全部执行并给出结果？
7. 文档与版本号是否已同步？
