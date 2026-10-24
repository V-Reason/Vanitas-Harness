"""Vanitas-Harness 本地推理 CLI。

包内只允许标准库 + prompt_toolkit（后者仅用于 REPL 行编辑，缺失时降级为 input）。
模型、端口、上下文与 KV 类型一律来自 config\\，代码不得硬编码。
"""

__version__ = "0.3.0"

__all__ = ["__version__"]
