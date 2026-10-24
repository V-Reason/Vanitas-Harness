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
- 本项目映射：`prompt_toolkit.FileHistory(~/.vanitas_history)`，保留条数上限；非 TTY 分支不写历史；粘贴占位符不引入（单机短输入，KISS）。

### D3 流式输出以「首 token 上屏」为第一体验指标

- 观察：`src/screens/REPL.tsx` 组合 `SpinnerWithVerb`（等待态动词 + 计时）、`VirtualMessageList`（长对话虚拟化）、`useStdin`/`useInput`（按键直通），消息体在流式期间持续重绘。
- 可移植性：**「等待要有可见状态、首字要立刻出现、结束后要有耗时」** 是终端用户的核心感知，跟 Ink 无关。
- 本项目映射：`vanitas/render.py` 增量写 stdout 并 flush；等待期用 `⋯` 覆盖式进度（批 3）；结束打印 `· token 数 · 耗时 · 首字 · t/s`（来源 `usage`，绝不用 SSE 分片数冒充 token）。

### D4 工具调用必须「白名单 + 逐次确认」

- 观察：`src/types/permissions.ts` 定义 `PermissionBehavior = 'allow' | 'deny' | 'ask'` 与 `default / acceptEdits / plan / dontAsk / bypassPermissions` 模式，规则来源区分 `userSettings / projectSettings / localSettings / policySettings / cliArg / session`。
- 可移植性：**默认拒绝、显式放行、每次可见** 是安全模型，不是产品特性。
- 本项目映射：`vanitas/tools.py` 只做 `/exec`（D1=B 不做自动循环）：`exec_enabled=false` 直接拒绝；命令按规范化字符串匹配白名单；每次执行前 `y/N`；`shell=False` + `shlex.split`，`& | >` 只能当字面参数。

### D5 会话边界清晰：清空、存档、恢复、预算

- 观察：`clear` 的职责是「清对话、释放上下文」，`supportsNonInteractive: false`（非交互场景直接新建会话）；`/resume`、`/session`、`/export` 独立成命令；上下文超限时有 compact 体系兜底。
- 可移植性：用户需要知道「我在哪个会话、能不能带走、上下文还剩多少」。这些是**状态契约**，与存储格式无关。
- 本项目映射：`vanitas/session.py` 维护 messages/system/task；`/clear` 只清对话保留系统提示；`/save`、`/load` 落 JSON；按 `min(档位 ctx, max_context_tokens) - 输出预留` 估算并裁剪最早轮次。

### D6 当前端点必须常驻可见，切换前先验就绪

- 观察：`model` 命令的 `description` 动态呈现当前模型；REPL 状态行（`StatusLine`）常驻显示模型/上下文等；切模型不是静默生效，而是先经过模型配置与就绪检查。
- 可移植性：**「我现在连的是哪个服务、哪个档位」** 一旦不可见，排障就只能靠猜。本机有 4 个模型、4 个端口，这条尤其值钱。
- 本项目映射：提示符 `<id>@<档位> › `（如 `qwen7b@1 › `）；`/model` 无参列全表并标记当前；目标未就绪时只提示 `vanitas serve <id>`，不偷偷切换；`/status` 汇总 pidfile + `/health` + 显存。

---

## 二、不可移植点及原因

| 机制 | 为什么不移植 |
| :--- | :--- |
| Ink + React 虚拟 DOM 全屏重绘（`replLauncher.tsx` 注入 React `Root`；`REPL.tsx` 用 `useStdin`、`useInput`、React compiler runtime） | 需要 React 渲染器与终端布局引擎；`claude-code-haha/package.json` 有 62 个运行时依赖（`@anthropic-ai/sdk`、`@opentelemetry/*`、`react` 生态等）。本项目红线只允许 `prompt_toolkit`，引入即违约。 |
| bun 打包 + `feature()` 编译期开关（`feature('KAIROS')`、`feature('DAEMON')` 等） | 属于「同一份源码编出不同产品形态」的构建体系，不是运行期可复制的交互设计。 |
| OAuth/登录、遥测、GrowthBook、MCP、插件、Skills、远程会话（`/bridge`、`/teleport`、`/remote-setup`） | 对应云端产品面与多端协同；本机是单机离线推理套件，没有账号、没有远端。 |
| 自动工具调用循环、auto-compact、prompt caching、多 Agent 协作 | 依赖大上下文与 token 预算模型，且项目边界（D1=B）明确排除自动循环；本机 `-np 1`、8 GB 显存，先保证单轮稳定。 |
| Vim 模式、键位自定义、剪贴板图片粘贴、语音、主题 | 交互增强层，对「能稳定跑通一轮对话 + 批量翻译」不构成瓶颈，属于过度设计。 |

---

## 三、本项目的取舍

1. **纯文本增量渲染，不做全屏重绘**：流式 token 直接写 stdout 并 flush；代码围栏内缩进 4 空格表示等宽块；着色只在 `vanitas/ui.py` 做（16 色 ANSI，`color: auto|always|never`），重定向即自动降级为纯文本，保证 `| tee`、CI 抓取可读（D3）。
2. **12 个 `/` 命令，一张表**：`/help /model /profile /system /task /clear /save /load /exec /stop /status /quit`（`/exit` 为 `/quit` 别名），全部在 `repl.py` 的 `COMMANDS` 中声明 `name/description/aliases/arg_hint`，`/help` 自动生成（D1）。提示符固定为 `<id>@<档位> › `，与 `/model`、`/status` 显示的模型/档位始终同一来源。
3. **单引擎、单模型、单档位**：一次只服务一个 llama-server；`/model` 切换前用 pidfile + `/health` 验就绪，不自动起停（D6）。理由：8 GB 显存禁不起双实例，静默启动比报错更危险。
4. **`/exec` 手动、默认关闭、逐次确认**：只做「白名单内、可见、可拒绝」的最小安全面，不实现自动循环，也不预留空钩子（D4）。
5. **会话与历史分离**：`/save` 存业务消息（可跨进程恢复），输入历史交给 `prompt_toolkit`；`/clear` 保留系统提示（D2、D5）。
6. **上下文预算按档位算，不猜**：`budget = min(profile.ctx, repl.max_context_tokens) - 512`，超预算丢最早轮次并提示裁剪数量；不实现自动摘要（那属于 auto-compact，明确不做）。

### 代码回溯索引（DoD：≥3 个设计点可回溯）

| 设计点 | 落地位置 |
| :--- | :--- |
| D1 命令注册表 | `vanitas/repl.py` → `COMMANDS`、`_cmd_help` |
| D2 输入历史 | `vanitas/repl.py` → `_get_prompt_session` |
| D3 流式渲染 | `vanitas/render.py` → `StreamRenderer` |
| D4 白名单 + 确认 | `vanitas/tools.py` → `run`/`_confirm` |
| D5 会话边界 | `vanitas/session.py` + `/save` `/load` `/clear` |
| D6 端点可见性 | `vanitas/launcher.py` → `running()`；`vanitas/repl.py` → `/model` `/status` |
