# Claude Code 交互范式笔记

> 目的：为 `vanitas` CLI 提炼可移植的交互设计。**只读勘察、提炼机制、不复制源码**。
> 状态：D1–D6 已落地于 `vanitas\`（v0.3+），入口为仓库根 `vanitas.bat`。架构契约见 [总设计方案.md](总设计方案.md)。
> 勘察对象（`<REFERENCE_PROJECTS>\`）：`Open-ClaudeCode`、`claude-code-haha`、`NanmiCoder-claude-code-haha`、`claude-code无补丁.zip`。
> 阅读入口：`src/commands.ts`、`src/commands/*/index.ts`、`src/history.ts`、`src/types/permissions.ts`、`src/screens/REPL.tsx`、`src/replLauncher.tsx`、`claude-code-haha/README.md`、`claude-code-haha/package.json`。

---

## 一、可移植设计点（D1–D6）

### D1 命令是「注册表数据」，不是散落的 if 分支

- 观察：`src/commands.ts` 集中 import 约 80 个命令模块；每个模块默认导出一个元数据对象，字段固定为 `type`（`local` / `local-jsx` / `prompt`）、`name`、`description`、`aliases`、`argumentHint`、`immediate`、`load`。
- 细节证据：`clear` 声明 `aliases: ['reset','new']`；`exit` 声明 `aliases: ['quit']` 且 `immediate: true`；`model` 用 getter 让 `description` 动态显示「当前模型」，并声明 `argumentHint: '[model]'`；实现体通过 `load: () => import(...)` 懒加载，帮助文本不需要拉起全部实现。
- 可移植性：这是**纯数据结构**，与渲染框架无关。收益是 `/help`、别名解析、参数提示、子命令补全只有一处事实源。
- 本项目映射：`vanitas/repl.py` 的 `COMMANDS` 表；`/help` 由表生成，不手写文案。

### D2 输入历史独立于会话存档

- 观察：`src/history.ts` 用 `MAX_HISTORY_ITEMS = 100` 截断窗口；写盘调用 `appendFile(historyPath, jsonLines.join(''), { mode: 0o600 })` 收紧权限；并发写用 lockfile；另有「粘贴内容引用占位符」（`[Pasted text #N +M lines]`）避免把大段粘贴塞进历史。
- 可移植性：历史是**编辑器状态**，会话是**业务状态**，两者生命周期不同。claude 把历史条数、权限、锁分开处理，正是踩过坑的痕迹。
- 本项目映射：`prompt_toolkit.FileHistory(~/.vanitas/history)`，保留条数上限；非 TTY 分支不写历史；粘贴占位符不引入（单机短输入，KISS）。

### D3 流式输出以「首 token 上屏」为第一体验指标

- 观察：`src/screens/REPL.tsx` 组合 `SpinnerWithVerb`（等待态动词 + 计时）、`VirtualMessageList`（长对话虚拟化）、`useStdin`/`useInput`（按键直通），消息体在流式期间持续重绘。
- 可移植性：**「等待要有可见状态、首字要立刻出现、结束后要有耗时」** 是终端用户的核心感知，跟 Ink 无关。
- 本项目映射：`vanitas/render.py` 增量写 stdout 并 flush；等待期用 `⋯` 覆盖式进度；结束打印 `· token 数 · 耗时 · t/s`，`-v` 追加首字；t/s 优先服务端 `timings.predicted_per_second`，其次 `usage`，绝不用 SSE 分片数冒充 token。

### D4 工具调用必须「按副作用分级」

- 观察：`src/types/permissions.ts` 定义 `PermissionBehavior = 'allow' | 'deny' | 'ask'` 与 `default / acceptEdits / plan / dontAsk / bypassPermissions` 模式，规则来源区分 `userSettings / projectSettings / localSettings / policySettings / cliArg / session`。
- 可移植性：**默认拒绝、显式放行、副作用可见** 是安全模型，不是产品特性；只读操作可以自动，有副作用的操作必须留下确认点。
- 本项目映射：`vanitas/tools.py` 的手动 `/exec` 保留固定兼容前缀与逐次确认；`vanitas/agent.py` 对单动作与只读数组的 `read_file/list_dir/run_shell_ro` 自动执行，对 `write_file/delete_file/move_file` 确认，含修改的数组只弹一次 `[y/N]`。`AgentTools` 以原 CWD 为沙盒根、最多 5 轮；`shell=False` + `shlex.split`，`& | >` 不能串联。最终自然语言增量上屏，动作 JSON 不落屏。

### D5 会话边界清晰：清空、存档、恢复、预算

- 观察：`clear` 的职责是「清对话、释放上下文」，`supportsNonInteractive: false`（非交互场景直接新建会话）；`/resume`、`/session`、`/export` 独立成命令；上下文超限时有 compact 体系兜底。
- 可移植性：用户需要知道「我在哪个会话、能不能带走、上下文还剩多少」。这些是**状态契约**，与存储格式无关。
- 本项目映射：`vanitas/session.py` 维护 messages/system/task；`/clear` 只清对话保留系统提示；`/save`、`/load` 落 JSON；按 `min(档位 ctx, max_context_tokens) - 输出预留` 估算并裁剪最早轮次。

### D6 当前端点必须常驻可见，切换前先验就绪

- 观察：`model` 命令的 `description` 动态呈现当前模型；REPL 状态行（`StatusLine`）常驻显示模型/上下文等；切模型不是静默生效，而是先经过模型配置与就绪检查。
- 可移植性：**「我现在连的是哪个服务、哪个档位」** 一旦不可见，排障就只能靠猜。本机有 3 个模型、3 个端口，这条尤其值钱。
- 本项目映射：提示符 `<id>@<档位> › `（如 `qwen7b@1 › `）；`/model` 无参列全表并标记当前；带 id 时先用 pidfile + `/health` 验运行态，已运行直连，没有运行则在单实例守卫下停旧启新；`/status` 汇总 pidfile + `/health` + 显存 + 待生效档位。

---

## 二、不可移植点及原因

| 机制 | 为什么不移植 |
| :--- | :--- |
| Ink + React 虚拟 DOM 全屏重绘（`replLauncher.tsx` 注入 React `Root`；`REPL.tsx` 用 `useStdin`、`useInput`、React compiler runtime） | 需要 React 渲染器与终端布局引擎；`claude-code-haha/package.json` 有 62 个运行时依赖（`@anthropic-ai/sdk`、`@opentelemetry/*`、`react` 生态等）。本项目红线只允许 `prompt_toolkit`，引入即违约。 |
| bun 打包 + `feature()` 编译期开关（`feature('KAIROS')`、`feature('DAEMON')` 等） | 属于「同一份源码编出不同产品形态」的构建体系，不是运行期可复制的交互设计。 |
| OAuth/登录、遥测、GrowthBook、MCP、插件、Skills、远程会话（`/bridge`、`/teleport`、`/remote-setup`） | 对应云端产品面与多端协同；本机是单机离线推理套件，没有账号、没有远端。 |
| 无人值守自动工具循环、auto-compact、prompt caching、多 Agent 协作 | 依赖大上下文、token 预算与更强的权限模型；本项目只实现“按副作用分级、CWD 沙盒、最多 5 轮”的受控闭环，不做后台队列或无人值守执行。 |
| Vim 模式、键位自定义、剪贴板图片粘贴、语音、主题 | 交互增强层，对「能稳定跑通一轮对话 + 批量翻译」不构成瓶颈，属于过度设计。 |

---

## 三、本项目的取舍

1. **纯文本增量渲染，不做全屏重绘**：流式 token 直接写 stdout 并 flush；代码围栏内缩进 4 空格表示等宽块；着色只在 `vanitas/ui.py` 做（16 色 ANSI，`color: auto|always|never`），重定向即自动降级为纯文本，保证 `| tee`、CI 抓取可读（D3）。
2. **12 个 `/` 命令，一张表**：`/help /model /profile /system /task /clear /save /load /exec /stop /status /quit`（`/exit` 为 `/quit` 别名），全部在 `repl.py` 的 `COMMANDS` 中声明 `name/description/aliases/arg_hint`，`/help` 自动生成（D1）。提示符固定为 `<id>@<档位> › `，与 `/model`、`/status` 显示的模型/档位始终同一来源。TTY 下 Tab 补全命令/别名、`/model ` 的模型 id 与 `/profile ` 的档位。
3. **单引擎、单模型、单档位**：一次只服务一个 llama-server；`/model` 切换前用 pidfile + `/health` 验运行态，已运行直连，没有运行则在单实例守卫下停旧启新（D6）。8 GB 显存仍禁不起双实例；失败时保留可读错误，不假装切换成功。
4. **`/exec` 手动、逐次确认；Agent 闭环按副作用分级**：手动 `/exec` 继续「固定兼容前缀 + 逐次确认」的最小安全面；`vanitas agent` 自动执行单动作与只读数组，写/删/移动必须确认，含修改的数组一次确认，`AgentTools` 以 CWD 为沙盒根、最多 5 轮，不提供后台队列或绕过修改确认的钩子（D4）。最终回答流式上屏，动作 JSON 不上屏（D3）。
5. **会话与历史分离**：`/save` 存业务消息（可跨进程恢复），输入历史交给 `prompt_toolkit`；`/clear` 保留系统提示（D2、D5）。
6. **上下文预算按档位算，不猜**：历史预算 `min(profile.ctx, repl.max_context_tokens) - context_reserve_tokens`；单轮输出预算 `max(min_output_tokens, profile.ctx - 历史 tokens - reserve)`。超预算丢最早轮次并提示裁剪数量；不实现自动摘要（那属于 auto-compact，明确不做）。
7. **翻译保留“可见的上下文”**：`TranslationEngine` 携带最近 N 段原文/译文摘要；`monitor.py` 用 ANSI 局部重绘最近 5 段（按终端宽高裁剪），让用户能肉眼核对跨段人称与术语；每个文件 reset/finish 独立结算。业务层只发 `SegmentStat`，不直接画终端。`finish_reason=length` 时按句末/中点拆分重译，避免半句静默输出。

### 代码回溯索引（DoD：≥3 个设计点可回溯）

| 设计点 | 落地位置 |
| :--- | :--- |
| D1 命令注册表 | `vanitas/repl.py` → `COMMANDS`、`_cmd_help` |
| D2 输入历史 | `vanitas/repl.py` → `_get_prompt_session` |
| D3 流式渲染 | `vanitas/render.py` → `StreamRenderer` |
| D4 分级审批 / CWD 沙盒 | `vanitas/tools.py` → `run_exec`、`AgentTools`、`is_inside_workspace`、`run_shell_ro`；`vanitas/agent.py` → `AgentRepl`、`_parse_actions` |
| D5 会话边界 | `vanitas/session.py` + `/save` `/load` `/clear` |
| D6 端点可见性 | `vanitas/launcher.py` → `running()`；`vanitas/repl.py` → `/model` `/status` |
| D7 翻译上下文与看板 | `vanitas/translate.py` → `TranslationEngine`；`vanitas/monitor.py` → `TranslationMonitor` |
