"""运行产物路径收口：统一到 ~/.vanitas/{run,logs} 与 ~/.vanitas/history。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import launcher  # noqa: E402
from vanitas.config import load_config  # noqa: E402


class LauncherPathTests(unittest.TestCase):
    def test_run_and_log_dirs_share_vanitas_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with mock.patch("vanitas.launcher.Path.home", return_value=home):
                self.assertEqual(launcher.run_dir(), home / ".vanitas" / "run")
                self.assertEqual(launcher.log_dir(), home / ".vanitas" / "logs")
                self.assertEqual(
                    launcher.record_path("minicpm"),
                    home / ".vanitas" / "run" / "minicpm.json",
                )

    def test_ensure_dirs_creates_new_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with mock.patch("vanitas.launcher.Path.home", return_value=home):
                launcher._ensure_dirs()
            self.assertTrue((home / ".vanitas" / "run").is_dir())
            self.assertTrue((home / ".vanitas" / "logs").is_dir())

    def test_history_file_uses_new_config_layout(self) -> None:
        config = load_config()
        self.assertEqual(
            config.settings.repl.history_file,
            Path.home() / ".vanitas" / "history",
        )


if __name__ == "__main__":
    unittest.main()
