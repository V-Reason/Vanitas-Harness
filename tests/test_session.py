"""session：默认 chat 的秒级时间戳与预算转发。"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.config import load_config  # noqa: E402
from vanitas.session import Session, context_budget, estimate_tokens  # noqa: E402

STAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


class SessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.prompt_dir = self.config.config_dir / "prompts"

    def test_default_chat_injects_second_precision_time(self) -> None:
        session = Session(system="sys")
        message = session.build_user_message("hello", self.prompt_dir, "alias")
        self.assertTrue(message.startswith("当前时间："), message)
        stamp = message.splitlines()[0].split("：", 1)[1]
        self.assertRegex(stamp, STAMP)
        self.assertTrue(message.endswith("hello"))

    def test_task_template_uses_date_placeholder(self) -> None:
        session = Session(system="sys", task="chat")
        message = session.build_user_message("hello", self.prompt_dir, "alias")
        self.assertIn("hello", message)
        self.assertNotIn("{date}", message)
        self.assertIn("当前日期：", message)
        stamp = message.splitlines()[0].split("：", 1)[1]
        self.assertRegex(stamp, STAMP)

    def test_budget_helpers_are_reexported(self) -> None:
        self.assertEqual(context_budget(4096, 8192), 3584)
        self.assertGreaterEqual(estimate_tokens("你好"), 2)


if __name__ == "__main__":
    unittest.main()
