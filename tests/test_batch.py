"""批处理：默认 token 与上限来自 settings，坏行不中断。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import batch  # noqa: E402


class BatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.tasks = self.root / "tasks.jsonl"

    def test_default_and_limit_come_from_call(self) -> None:
        self.tasks.write_text(
            "\n".join(
                [
                    json.dumps({"id": "default", "prompt": "hi"}),
                    json.dumps(
                        {"id": "too-big", "prompt": "hi", "max_tokens": 11}
                    ),
                    json.dumps(
                        {"id": "explicit", "prompt": "hi", "max_tokens": 7}
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        seen: list[int] = []

        def chat(_system: str, _prompt: str, max_tokens: int) -> str:
            seen.append(max_tokens)
            return "ok"

        result = batch.run_batch(
            self.tasks,
            chat,
            prompt_dir=ROOT / "config" / "prompts",
            system_default="sys",
            default_max_tokens=5,
            max_tokens_limit=10,
            echo=lambda _text: None,
        )
        self.assertEqual(seen, [5, 7])
        self.assertEqual(sum(1 for item in result.items if item.ok), 2)
        failed = [item for item in result.items if not item.ok]
        self.assertEqual(len(failed), 1)
        self.assertIn("1–10", failed[0].error)
        self.assertTrue(result.output_path.is_file())


if __name__ == "__main__":
    unittest.main()
