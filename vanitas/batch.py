"""JSONL 批处理：一行一个任务，结果写回 <input>.out.jsonl。

行格式（未在任务书中固定，这里取最小集）：
  {"id": "t1", "prompt": "……", "system": "可选覆盖", "task": "可选模板名", "max_tokens": 512}
坏行计入失败但不中断，最终由 CLI 汇总并返回非零。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import prompts
from .errors import ConfigError, EngineError, VanitasError

ChatFn = Callable[[str, str, int], str]


@dataclass
class BatchItem:
    id: str
    ok: bool
    output: str = ""
    error: str = ""


@dataclass
class BatchResult:
    items: list[BatchItem] = field(default_factory=list)
    output_path: Path | None = None

    @property
    def failures(self) -> list[BatchItem]:
        return [item for item in self.items if not item.ok]


def output_path_for(tasks: Path) -> Path:
    return tasks.with_suffix(".out.jsonl")


def run_batch(
    tasks: Path,
    chat: ChatFn,
    *,
    prompt_dir: Path,
    system_default: str,
    default_max_tokens: int = 512,
    echo: Callable[[str], None] = print,
) -> BatchResult:
    tasks = Path(tasks)
    try:
        raw = tasks.read_text(encoding="utf-8")
    except OSError as exc:
        raise VanitasError(f"批处理文件读取失败：{tasks}（{exc.strerror or exc}）") from exc

    lines = raw.splitlines()
    result = BatchResult(output_path=output_path_for(tasks))
    total = sum(1 for line in lines if line.strip())
    position = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        position += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            result.items.append(
                BatchItem(
                    id=f"line{line_number}",
                    ok=False,
                    error=f"第 {line_number} 行不是合法 JSON：{exc.msg}",
                )
            )
            echo(f"[BATCH] {position}/{total} line{line_number} 失败：JSON 解析")
            continue
        if not isinstance(record, dict):
            result.items.append(
                BatchItem(id=f"line{line_number}", ok=False, error="行内容必须是 JSON 对象")
            )
            echo(f"[BATCH] {position}/{total} line{line_number} 失败：不是对象")
            continue
        item_id = str(record.get("id") or f"line{line_number}")
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            result.items.append(
                BatchItem(id=item_id, ok=False, error="缺少非空字符串字段 prompt")
            )
            echo(f"[BATCH] {position}/{total} {item_id} 失败：缺少 prompt")
            continue
        system = record.get("system")
        system_text = system.strip() if isinstance(system, str) and system.strip() else system_default
        task = record.get("task")
        if isinstance(task, str) and task.strip():
            try:
                template = prompts.load_task(prompt_dir, task.strip())
                prompt = prompts.render(template, {"input": prompt})
            except ConfigError as exc:
                result.items.append(BatchItem(id=item_id, ok=False, error=str(exc)))
                echo(f"[BATCH] {position}/{total} {item_id} 失败：模板")
                continue
        max_tokens = record.get("max_tokens", default_max_tokens)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 8192:
            result.items.append(
                BatchItem(id=item_id, ok=False, error=f"max_tokens 必须是 1–8192 的整数，实际 {max_tokens!r}")
            )
            echo(f"[BATCH] {position}/{total} {item_id} 失败：max_tokens")
            continue
        try:
            output = chat(system_text, prompt, max_tokens)
        except (EngineError, ConfigError) as exc:
            result.items.append(BatchItem(id=item_id, ok=False, error=str(exc)))
            echo(f"[BATCH] {position}/{total} {item_id} 失败：{exc}")
            continue
        result.items.append(BatchItem(id=item_id, ok=True, output=output))
        echo(f"[BATCH] {position}/{total} {item_id} 完成")

    try:
        result.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = "\n".join(
            json.dumps(
                {
                    "id": item.id,
                    "ok": item.ok,
                    "output": item.output,
                    "error": item.error,
                },
                ensure_ascii=False,
            )
            for item in result.items
        )
        result.output_path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    except OSError as exc:
        raise VanitasError(
            f"批处理结果写入失败：{result.output_path}（{exc.strerror or exc}）"
        ) from exc
    return result
