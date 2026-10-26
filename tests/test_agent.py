"""agent：规划模式只包装 REPL，不执行任何工具。"""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import agent  # noqa: E402
from vanitas.config import load_config  # noqa: E402
from vanitas.ui import Palette  # noqa: E402


class AgentTests(unittest.TestCase):
    def test_run_agent_uses_task_agent_without_execution(self) -> None:
        config = load_config()
        model = config.model("qwen7b")
        profile = config.resolve_profile(model)
        fake_repl = mock.Mock()
        fake_repl.run.return_value = 0
        out = io.StringIO()
        with (
            mock.patch.object(agent, "Repl", return_value=fake_repl) as repl_cls,
            contextlib.redirect_stdout(out),
        ):
            code = agent.run_agent(
                config,
                model,
                profile.key,
                palette=Palette({}, enabled=False),
            )
        self.assertEqual(code, 0)
        repl_cls.assert_called_once()
        fake_repl.configure.assert_called_once_with(system=None, task="agent")
        fake_repl.run.assert_called_once()
        self.assertIn("不自动执行", out.getvalue())


if __name__ == "__main__":
    unittest.main()
