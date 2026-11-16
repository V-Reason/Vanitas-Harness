"""session：默认 chat 的秒级时间戳与预算转发。"""

from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.config import load_config  # noqa: E402
from vanitas.context import (  # noqa: E402
    Session,
    context_budget,
    default_save_path,
    estimate_tokens,
)

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
        stamp_line = next(
            line for line in message.splitlines() if "当前日期：" in line
        )
        stamp = stamp_line.split("：", 1)[1]
        self.assertRegex(stamp, STAMP)

    def test_agent_template_uses_explicit_cwd(self) -> None:
        session = Session(system="sys", task="agent")
        with mock.patch.dict(os.environ, {"VANITAS_ORIG_CWD": ""}):
            message = session.build_user_message(
                "hello",
                self.prompt_dir,
                "alias",
                cwd=r"<WORKSPACE>\demo\workspace",
            )
        self.assertIn("当前工作目录：<WORKSPACE>\\demo", message)
        self.assertNotIn("{cwd}", message)

    def test_agent_later_user_message_uses_compact_template(self) -> None:
        session = Session(system="sys", task="agent")
        session.add_user("first request")
        session.add_assistant("first answer")
        message = session.build_user_message(
            "second request", self.prompt_dir, "alias"
        )
        self.assertNotIn("【输出格式强约束】", message)
        self.assertNotIn("【唯一允许的动作】", message)
        self.assertIn("【Agent 继续】", message)
        self.assertIn("<tool_call>[...]</tool_call>", message)
        self.assertIn("second request", message)

    def test_agent_template_prefers_original_cwd_env(self) -> None:
        session = Session(system="sys", task="agent")
        with mock.patch.dict(os.environ, {"VANITAS_ORIG_CWD": r"<WORKSPACE>"}):
            message = session.build_user_message(
                "hello",
                self.prompt_dir,
                "alias",
                cwd=r"<WORKSPACE>\repo",
            )
        self.assertIn("当前工作目录：<WORKSPACE>", message)

    def test_agent_template_defaults_to_process_cwd(self) -> None:
        session = Session(system="sys", task="agent")
        with (
            mock.patch.dict(os.environ, {"VANITAS_ORIG_CWD": ""}),
            mock.patch(
                "vanitas.context.session.os.getcwd", return_value=r"<WORKSPACE>\fallback"
            ),
        ):
            message = session.build_user_message("hello", self.prompt_dir, "alias")
        self.assertIn("当前工作目录：<WORKSPACE>\\fallback", message)

    def test_budget_helpers_are_reexported(self) -> None:
        self.assertEqual(context_budget(4096, 8192), 3584)
        self.assertGreaterEqual(estimate_tokens("你好"), 2)

    def test_default_save_path_is_under_vanitas_sessions(self) -> None:
        home = Path(r"<WORKSPACE>\home")
        with mock.patch(
            "vanitas.context.session.Path.home", return_value=home
        ):
            path = default_save_path()
        self.assertEqual(path.parent, home / ".vanitas" / "sessions")
        self.assertEqual(path.suffix, ".json")


if __name__ == "__main__":
    unittest.main()
