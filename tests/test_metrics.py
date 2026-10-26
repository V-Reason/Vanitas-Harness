"""metrics：token 估算、聊天/翻译预算、汉字统计与耗时格式。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.metrics import (  # noqa: E402
    chat_max_tokens,
    compute_rate,
    context_budget,
    count_cjk,
    estimate_tokens,
    format_duration,
    translation_budget,
)


class MetricsTests(unittest.TestCase):
    def test_estimate_tokens_cjk_and_ascii(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)
        self.assertGreaterEqual(estimate_tokens("你好世界"), 4)
        self.assertLessEqual(estimate_tokens("hello world"), 4)

    def test_context_budget(self) -> None:
        self.assertEqual(context_budget(4096, 8192), 3584)
        self.assertEqual(context_budget(1024, 8192), 512)
        self.assertEqual(context_budget(4096, 8192, reserve=1000), 3096)

    def test_chat_max_tokens_matches_task_formula(self) -> None:
        self.assertEqual(chat_max_tokens(4096, 1000), 2584)
        self.assertEqual(chat_max_tokens(1024, 900), 256)
        self.assertEqual(chat_max_tokens(4096, 1000, min_output=300), 2584)

    def test_translation_budget_prefers_desired_then_caps(self) -> None:
        budget = translation_budget(8192, prompt_tokens=1000, input_tokens=1000)
        self.assertEqual(budget.desired_tokens, 2500)
        self.assertEqual(budget.max_tokens, 2500)
        self.assertGreater(budget.available_tokens, 2500)

        capped = translation_budget(4096, prompt_tokens=1800, input_tokens=1500)
        self.assertEqual(capped.desired_tokens, 3750)
        self.assertLess(capped.max_tokens, capped.desired_tokens)
        self.assertGreaterEqual(capped.max_tokens, 512)

    def test_translation_budget_raises_when_too_small(self) -> None:
        with self.assertRaises(ValueError):
            translation_budget(1024, prompt_tokens=1000, input_tokens=100)
        with self.assertRaises(ValueError):
            translation_budget(0, prompt_tokens=0, input_tokens=1)

    def test_count_cjk_only_han(self) -> None:
        self.assertEqual(count_cjk("你好，世界！"), 4)
        self.assertEqual(count_cjk("hello 你好"), 2)
        self.assertEqual(count_cjk("こんにちは"), 0)

    def test_format_duration(self) -> None:
        self.assertEqual(format_duration(135), "02分15秒")
        self.assertEqual(format_duration(0), "00分00秒")
        self.assertEqual(format_duration(3661), "01时01分01秒")

    def test_compute_rate(self) -> None:
        self.assertAlmostEqual(compute_rate(10, 2), 5.0)
        self.assertEqual(compute_rate(10, 0), 0.0)
        self.assertEqual(compute_rate(10, -1), 0.0)


if __name__ == "__main__":
    unittest.main()
