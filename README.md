# Vanitas-Harness

单机本地推理 CLI：为 RTX 4060 Laptop（8 GB）准备 llama.cpp 启动器、单实例守卫、交互式 REPL、受控 Agent 闭环、批量翻译与批处理。模型、端口、上下文与 KV 类型全部由 `config/` 驱动。

- 代码主力：Qwen2.5-Coder-7B-Instruct Q6_K（4060，Agent 默认）
- 通用备胎：Qwen2.5-7B-Instruct Q6_K（4063，chat/batch 默认，2 分片只传首片）
- 轻量高质：MiniCPM5-2B Q8_0（4061）
- 专业翻译：HY-MT2-7B Q6_K（4062，translate 默认）

---

## 最短上手

1. 把 llama.cpp CUDA 预编译包**完整解压**到 `bin/`（需 `llama-server.exe`、`llama.dll`、`ggml-base.dll`、`ggml-cuda.dll`）。
2. 把 GGUF 按 `config/models.json` 的路径放入 `models/`。7B 两片必须同名同目录，启动只传 `-00001-of-00002.gguf`。
3. `python -m pip install prompt_toolkit`（唯一第三方依赖；未安装会自动降级为 `input()`，功能不受阻）。
4. 从仓库根运行 `vanitas.bat`；默认进入 Agent 闭环，无服务端时引导启动模型。

```bat
vanitas.bat
vanitas.bat models
vanitas.bat status
vanitas.bat serve qwen7b
vanitas.bat translate a.srt --preset subtitle
vanitas.bat stop
```

`vanitas.bat` 只做四件事：记录原 CWD、切到仓库根、调用 `python -m vanitas`、透传退出码；除 Agent 的 `{cwd}`/沙盒根外，`python -m vanitas ...` 与其等价。

---

## 文档地图

| 文档 | 一句话职责 |
| :--- | :--- |
| [README.md](README.md) | 本页：项目是什么、最短上手、文档索引与常用入口。 |
| [docs/项目现状.md](docs/项目现状.md) | **当前事实的唯一入口**：契约速查、环境/库存、最新实测、已知问题、未验证边界、下一步。 |
| [docs/变更历史.md](docs/变更历史.md) | **历史，不是当前契约**：原 `§1–§53` 批次原文（编号连续）+ 旧 AGENTS §9 + 旧总设计全文；旧值只在这里有效。 |
| [docs/总设计方案.md](docs/总设计方案.md) | 架构、模块边界、配置/进程/接口契约、明确取舍与已否决方案。 |
| [docs/用户使用手册.md](docs/用户使用手册.md) | 安装、命令、REPL、翻译、批处理、排障；命令以 `--help` 实测为准。 |
| [AGENTS.md](AGENTS.md) | 开发纪律、红线、验证门禁、当前契约边界与协作规范。 |
| [docs/ClaudeCode范式笔记.md](docs/ClaudeCode范式笔记.md) | 交互范式来源与 D1–D7 取舍；独立职责，不参与批次历史。 |
| [docs/ASMR语义质量冲刺_开发流程与实验复盘.md](docs/ASMR语义质量冲刺_开发流程与实验复盘.md) | 2026-10-30 语义冲刺的历史复盘；当前结论以项目现状/变更历史 §52–§53 为准。 |
| [docs/ASMR字幕生成与测试.md](docs/ASMR字幕生成与测试.md) | ASR/字幕生成的参数与样本记录（历史材料）；不是翻译质量契约。 |

> 读法：先看 `项目现状.md` 顶部契约速查，再看 `总设计方案.md`；遇到“为什么变成这样”才进 `变更历史.md`。不要把历史章节里的旧测试数、旧指纹或旧结论当成当前值。

---

## 常见入口

| 我想… | 入口 |
| :--- | :--- |
| 看当前模型/端口/档位/别名 | `vanitas.bat models` |
| 看进程、端口、显存 | `vanitas.bat status` |
| 启动/停止服务端 | `vanitas.bat serve <id>` / `vanitas.bat stop` |
| 进入 Agent 闭环 | `vanitas.bat` 或 `vanitas.bat agent` |
| 进入纯聊天 REPL | `vanitas.bat chat` |
| 普通字幕翻译 | `vanitas.bat translate a.srt --preset subtitle` |
| ASMR 翻译 | `vanitas.bat translate asmr.srt --preset asmr` |
| 目录翻译并保留层级 | `vanitas.bat translate novels --preset prose --out-dir translated` |
| 只改文件名/目录名 | `vanitas.bat translate media --translate-filename-only` |
| 跑全量单测 | `python -m unittest discover -s tests` |
| 查文档为何这样设计 | [docs/变更历史.md](docs/变更历史.md) 对应 §n |

配置驱动与红线：新增模型只允许在 `config/models.json` 追加条目；单实例、端口 4060–4063、退出码 `0/1/2/3/4/5`；`bin/`、`models/` 只读；禁止 `git clean -fdx`。完整约束见 [AGENTS.md](AGENTS.md) 与 [总设计方案](docs/总设计方案.md)。
