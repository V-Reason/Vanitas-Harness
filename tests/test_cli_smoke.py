"""CLI 冒烟：只验证参数解析、配置加载与退出码映射，不启动真实模型。"""

from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import cli  # noqa: E402
from vanitas.config import DEFAULT_THEME, load_config  # noqa: E402
from vanitas.errors import GuardError  # noqa: E402
from vanitas.ui import Palette  # noqa: E402

ENV_KEYS = ("VANITAS_PORT", "VANITAS_PROFILE", "VANITAS_CONFIG_DIR")


@contextmanager
def clean_env(**values):
    saved = {key: os.environ.get(key) for key in ENV_KEYS}
    for key in ENV_KEYS:
        os.environ.pop(key, None)
    for key, value in values.items():
        os.environ[key] = value
    try:
        yield
    finally:
        for key in ENV_KEYS:
            if saved[key] is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = saved[key]


class FakeClient:
    def chat(self, messages, *, max_tokens, temperature=0.2, response_format=None, timeout=None):
        return {"choices": [{"message": {"content": "译文"}}]}

    def health(self, timeout=None) -> bool:
        return True


def run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class CliSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_version(self) -> None:
        with clean_env():
            code, out, _err = run(["--version"])
        self.assertEqual(code, 0)
        self.assertIn("vanitas 0.4.1", out)

    def test_models_lists_config(self) -> None:
        with clean_env():
            code, out, _err = run(["models"])
        self.assertEqual(code, 0)
        for model_id in ("qwen7b", "minicpm", "hymt", "qwen3b"):
            self.assertIn(model_id, out)
        self.assertIn("4060", out)
        self.assertIn("qwen2.5-7b-instruct-q6_k", out)

    def test_status_without_server(self) -> None:
        with clean_env():
            code, out, _err = run(["status"])
        self.assertEqual(code, 0)
        self.assertIn("qwen7b", out)
        self.assertIn("停止", out)

    def test_unknown_subcommand_exit_4(self) -> None:
        with clean_env():
            code, _out, err = run(["nope"])
        self.assertEqual(code, 4)
        self.assertIn("✗", err)

    def test_unknown_model_exit_2(self) -> None:
        with clean_env():
            code, _out, err = run(["serve", "nope"])
        self.assertEqual(code, 2)
        self.assertIn("未知模型", err)

    def test_bad_config_dir_exit_2(self) -> None:
        with clean_env():
            code, _out, err = run(["--config-dir", str(self.dir / "missing"), "models"])
        self.assertEqual(code, 2)
        self.assertIn("配置目录不存在", err)

    def test_guard_failure_maps_to_3(self) -> None:
        with clean_env():
            with mock.patch.object(
                cli.launcher,
                "start",
                side_effect=GuardError(1, ["✗ 端口 4060 被 PID 1 (x.exe) 占用"]),
            ):
                code, _out, err = run(["serve", "qwen7b"])
        self.assertEqual(code, 3)
        self.assertIn("✗ 端口 4060 被 PID 1", err)
        self.assertNotIn("GUARD-CODE", err)

    def test_translate_with_fake_engine(self) -> None:
        source = self.dir / "sample.srt"
        source.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n2\n00:00:03,000 --> 00:00:04,000\nBye\n",
            encoding="utf-8",
        )
        out_dir = self.dir / "out"
        with clean_env():
            with mock.patch.object(
                cli, "_ensure_engine", return_value=(FakeClient(), 4060)
            ):
                code, out, err = run(
                    ["translate", str(source), "--out-dir", str(out_dir)]
                )
        self.assertEqual(code, 0, err)
        target = out_dir / "sample.zh.srt"
        self.assertTrue(target.is_file())
        # 完成行与进度同走 stderr，避免与 stdout 混排
        self.assertIn("2 段", err)
        self.assertNotIn("\x1b[", err)

    def test_translate_missing_file_exit_5(self) -> None:
        with clean_env():
            with mock.patch.object(
                cli, "_ensure_engine", return_value=(FakeClient(), 4060)
            ):
                code, _out, err = run(["translate", str(self.dir / "nope.srt")])
        self.assertEqual(code, 5)
        self.assertIn("✗", err)
        self.assertNotIn("TRANSLATE-FAIL", err)

    def test_translate_rejects_bad_jobs(self) -> None:
        with clean_env():
            code, _out, err = run(["translate", "a.srt", "--jobs", "0"])
        self.assertEqual(code, 4)
        self.assertIn("--jobs", err)

    def test_palette_disabled_in_pipe(self) -> None:
        # redirect_stdout 之后 sys.stdout 不是 TTY，auto 必须降级为纯文本
        with clean_env():
            code, out, err = run(["models"])
        self.assertEqual(code, 0)
        self.assertNotIn("\x1b[", out)
        self.assertNotIn("\x1b[", err)

    def test_agent_subcommand_dispatches(self) -> None:
        with clean_env():
            with mock.patch.object(cli.agent_mod, "run_agent", return_value=0) as called:
                code, _out, _err = run(["agent"])
        self.assertEqual(code, 0)
        called.assert_called_once()

    def test_translate_two_files_get_independent_compact_stats(self) -> None:
        first = self.dir / "a.srt"
        first.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
            encoding="utf-8",
        )
        second = self.dir / "b.srt"
        second.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nBye\n",
            encoding="utf-8",
        )
        out_dir = self.dir / "out"
        with clean_env():
            with mock.patch.object(
                cli, "_ensure_engine", return_value=(FakeClient(), 4060)
            ):
                code, _out, err = run(
                    ["translate", str(first), str(second), "--out-dir", str(out_dir)]
                )
        self.assertEqual(code, 0, err)
        summaries = [
            line
            for line in err.splitlines()
            if re.match(r"^\d+ 段 · \d+ 字 · ", line)
        ]
        self.assertEqual(len(summaries), 2, err)
        self.assertTrue(summaries[0].startswith("1 段"))
        self.assertTrue(summaries[1].startswith("2 段"))
        self.assertNotIn("翻译任务已完成", err)
        self.assertNotIn("统计段数", err)
        self.assertNotIn("累计汉字", err)

    def test_status_state_uses_error_color(self) -> None:
        config = load_config()
        palette = Palette(DEFAULT_THEME, enabled=True)
        fake_client = mock.Mock()
        fake_client.health.return_value = False
        out = io.StringIO()
        with (
            mock.patch.object(cli.LlamaClient, "for_model", return_value=fake_client),
            mock.patch.object(cli.launcher, "read_records", return_value=[]),
            mock.patch.object(cli.guard, "gpu_status", return_value=None),
            contextlib.redirect_stdout(out),
        ):
            code = cli._cmd_status(
                config, mock.Mock(verbose=False), palette
            )
        self.assertEqual(code, 0)
        self.assertIn("\x1b[31m✗ 停止\x1b[0m", out.getvalue())


if __name__ == "__main__":
    unittest.main()
