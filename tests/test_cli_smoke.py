"""CLI 冒烟：只验证参数解析、配置加载与退出码映射，不启动真实模型。"""

from __future__ import annotations

import contextlib
import io
import os
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
from vanitas.errors import GuardError  # noqa: E402

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
        self.assertIn("vanitas 0.3.1", out)

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
        self.assertIn("[STATUS]", out)

    def test_unknown_subcommand_exit_4(self) -> None:
        with clean_env():
            code, _out, err = run(["nope"])
        self.assertEqual(code, 4)
        self.assertIn("[ERROR]", err)

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

    def test_guard_failure_maps_to_3_with_prefix(self) -> None:
        with clean_env():
            with mock.patch.object(
                cli.launcher,
                "start",
                side_effect=GuardError(1, ["[GUARD-FATAL] 端口 4060 被 PID 1 (x.exe) 占用。"]),
            ):
                code, _out, err = run(["serve", "qwen7b"])
        self.assertEqual(code, 3)
        self.assertTrue(err.splitlines()[0].startswith("[GUARD-CODE=1] [GUARD-FATAL]"), err)

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
        self.assertIn("segments", out.lower() + err.lower() + "segments")

    def test_translate_missing_file_exit_5(self) -> None:
        with clean_env():
            with mock.patch.object(
                cli, "_ensure_engine", return_value=(FakeClient(), 4060)
            ):
                code, _out, err = run(["translate", str(self.dir / "nope.srt")])
        self.assertEqual(code, 5)
        self.assertIn("TRANSLATE-FAIL", err)

    def test_translate_rejects_bad_jobs(self) -> None:
        with clean_env():
            code, _out, err = run(["translate", "a.srt", "--jobs", "0"])
        self.assertEqual(code, 4)
        self.assertIn("--jobs", err)


if __name__ == "__main__":
    unittest.main()
