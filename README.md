# Vanitas-Harness

单机三模型推理套件：为 RTX 4060 Laptop（8 GB）准备的 llama.cpp 启动器、单实例互斥守卫与自动化验收基准。

- 通用主力：Qwen2.5-7B-Instruct Q6_K（8080）
- 轻量高质：MiniCPM5-2B Q6_K（8081）
- 专业翻译：HY-MT2-7B Q6_K（8082）

当前只到位 3B 临时验证权重；三个目标权重与其显存/质量数据仍在等待。**状态、实测数据与未验证边界以 [docs/项目现状.md](docs/项目现状.md) 为准。**

---

## 目录结构

```text
Vanitas-Harness\
├── bin\                        # llama.cpp CUDA 预编译包（不入库，约 1.1 GB）
├── models\                     # GGUF 权重（不入库）
├── scripts\
│   ├── _env_guard.bat          # 单实例互斥 + 运行时/模型/端口校验
│   ├── run_qwen2.5_7b_q6.bat   # 8080
│   ├── run_minicpm5_2b_q6.bat  # 8081
│   ├── run_hymt2_7b_q6.bat     # 8082
│   └── engine_control.bat      # 交互总控
├── tests\
│   ├── multi_model_benchmark.py
│   └── reports\                # 评测报告（不入库）
└── docs\
    ├── 任务书.md                # 交付契约与修订记录
    ├── 项目现状.md              # 状态事实源
    └── 前期方案.md              # 历史设计（冻结，路径已作废）
```

## 三步上手

1. 把 llama.cpp CUDA 预编译包**完整解压**到 `bin\`（需 `llama-server.exe`、`llama.dll`、`ggml-base.dll`、`ggml-cuda.dll`）。
2. 把 GGUF 按约定文件名放入 `models\`（见 [docs/项目现状.md](docs/项目现状.md) 第 2 节）。
3. 双击 `scripts\engine_control.bat`，按菜单选择。

启动器会先跑守卫：清理旧实例 → 校验模型 → 检查端口；任何一步失败都会给出可执行的处置提示，不会被静默忽略。

## 端口与档位

| 启动器 | 端口 | 档位 1（默认） | 档位 2 | 档位 3 |
| :--- | :-: | :--- | :--- | :--- |
| Qwen2.5-7B | 8080 | 4K + Q8_0 KV | 8K + Q4_0 KV | 4K + F16 KV（需空闲 ≥7.2 GB） |
| MiniCPM5-2B | 8081 | 8K + F16 KV | 32K + F16 KV | — |
| HY-MT2-7B | 8082 | 4K + Q8_0 KV | 8K + Q4_0 KV | 4K + F16 KV（需空闲 ≥7.2 GB） |

- 服务只监听 `127.0.0.1`，浏览器打开 `http://127.0.0.1:<端口>` 即为 Web UI，同时提供 OpenAI 兼容 API。
- 8080 若被 Steam 占用，退出 Steam；或临时 `set VANITAS_PORT=8090` 后再启动（默认值不变）。
- 手动启动示例：`scripts\run_qwen2.5_7b_q6.bat "<REPOSITORY_ROOT>\models\qwen2.5-3b-instruct-q4_k_m.gguf" 2`

## 验收评测

```bat
python tests\multi_model_benchmark.py --port 8080 --suite agent
python tests\multi_model_benchmark.py --port 8082 --suite translation
```

退出码：`0` 通过 / `1` 未就绪 / `2` 协议或性能失败 / `3` 质量失败 / `4` 参数错误。报告自动落到 `tests\reports\`。

指标口径：性能取 llama-server 返回的 `timings` 与 `usage`；流式只校验 SSE 帧、TTFT 与 `[DONE]`，不用分片数冒充 token。

## 常见故障

| 现象 | 处置 |
| :--- | :--- |
| `[GUARD-FATAL] 端口 8080 被 PID ... 占用` | 按提示结束占用进程，或 `set VANITAS_PORT=<空闲端口>` |
| `[GUARD-FATAL] 模型文件缺失` | 核对 `models\` 文件名是否与约定完全一致 |
| `[GUARD-FATAL] 文件头不是 GGUF` | 下载得到的是网页/压缩包，重新下载 |
| 启动即退出、无日志 | 缺 `bin\` 下的 DLL；守卫会指明缺哪一个 |
| 显存不足 / 加载中断 | 关闭 Steam/浏览器/壁纸引擎，或降档位（见上表） |

## 仓库约定

- 分支：`main` 保持可用；特性分支短生命周期，`--no-ff` 合并后打标签。
- `bin\`、`models\`、`tests\reports\` 永不入库；**不要执行 `git clean -fdx`**，它会连带删除权重。
- `.bat` 固定 UTF-8（无 BOM）+ CRLF，改完必须实跑一次守卫与 3B 冒烟。

## 版本

`v0.1.0` — 运行时脚手架、总控、评测套件与现状文档。
