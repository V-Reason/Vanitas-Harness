"""批次 6 微基准：4k 回答的流式 CPU 时间对比。

不参与单测，仅作观测指标。
"""
from __future__ import annotations

import io
import time

from vanitas.agent.stream import _AgentAnswerStream


def _feed_in_chunks(stream, text: str, chunk_size: int = 4) -> None:
    for index in range(0, len(text), chunk_size):
        stream.write(text[index:index + chunk_size])


def bench(text: str, repeats: int = 50) -> float:
    stream = _AgentAnswerStream()
    start = time.perf_counter()
    for _ in range(repeats):
        stream.__init__()  # 重置状态
        stream._target = io.StringIO()  # 基准只计时，不写终端
        _feed_in_chunks(stream, text)
    return time.perf_counter() - start


def main() -> None:
    plain = "这是一段普通的中文文本，不包含任何标签或 JSON。" * 20
    long_text = plain * 4
    mixed = (
        long_text
        + "\n<tool_call>[{\"action\":\"read_file\",\"parameters\":"
        "{\"path\":\"x\"}}]</tool_call>\n"
        + long_text
    )

    print(f"plain 4k x50: {bench(long_text):.4f}s")
    print(f"mixed x50: {bench(mixed):.4f}s")


if __name__ == "__main__":
    main()
