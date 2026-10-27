"""生命周期：偏好记忆、/restart 拦截、空闲看门狗与用户原生 /exec。"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import repl as repl_mod  # noqa: E402
from vanitas import tools  # noqa: E402
from vanitas.config import Preferences, load_config  # noqa: E402
from vanitas.errors import ConfigError, ToolDenied  # noqa: E402
from vanitas.launcher import RunRecord  # noqa: E402
from vanitas.repl import IdleWatchdog, Repl  # noqa: E402
from vanitas.ui import Palette  # noqa: E402


def quiet_run(fn):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        value = fn()
    return value, out.getvalue(), err.getvalue()


def make_repl(tmp: Path, *, idle_timeout: float = 600.0) -> Repl:
    config = load_config(preferences_path=tmp / "preferences.json")
    repl_settings = replace(
        config.settings.repl,
        idle_timeout_sec=idle_timeout,
        watchdog_interval_sec=0.01,
    )
    config = replace(
        config,
        settings=replace(config.settings, repl=repl_settings),
    )
    return Repl(config, palette=Palette({}, enabled=False))


class PreferencesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "preferences.json"

    def test_roundtrip_and_atomic_temp_removed(self) -> None:
        prefs = Preferences.load(self.path)
        self.assertEqual(prefs.preferred_profiles, {})
        prefs.set_profile("qwen_coder", "2")
        prefs.remember_model("qwen_coder")
        loaded = Preferences.load(self.path)
        self.assertEqual(loaded.last_used_model, "qwen_coder")
        self.assertEqual(loaded.profile_for("qwen_coder"), "2")
        self.assertFalse(self.path.with_name(self.path.name + ".tmp").exists())

    def test_corrupt_file_falls_back_empty(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{broken", encoding="utf-8")
        prefs = Preferences.load(self.path)
        self.assertEqual(prefs.preferred_profiles, {})
        self.assertEqual(prefs.last_used_model, "")

    def test_precedence_cli_env_preferences_default(self) -> None:
        config = load_config(preferences_path=self.path)
        coder = config.model("qwen_coder")
        self.assertEqual(config.resolve_profile(coder).key, "1")

        config.preferences.set_profile("qwen_coder", "2")
        self.assertEqual(config.resolve_profile(coder).key, "2")

        with mock.patch.dict(os.environ, {"VANITAS_PROFILE": "3"}):
            self.assertEqual(config.resolve_profile(coder).key, "3")
            self.assertEqual(config.resolve_profile(coder, "1").key, "1")

        config.preferences.set_profile("qwen_coder", "9")
        self.assertEqual(config.resolve_profile(coder).key, "1")

    def test_bad_preference_value_does_not_mask_explicit_error(self) -> None:
        config = load_config(preferences_path=self.path)
        with self.assertRaises(ConfigError):
            config.resolve_profile(config.model("qwen_coder"), "9")


class IdleWatchdogTests(unittest.TestCase):
    def test_idle_callback_fires_once(self) -> None:
        fired = threading.Event()
        watchdog = IdleWatchdog(0.03, 0.01, fired.set)
        watchdog.start()
        try:
            self.assertTrue(fired.wait(1.0))
            time.sleep(0.05)
        finally:
            watchdog.stop()
        self.assertTrue(fired.is_set())

    def test_pause_prevents_and_resume_allows(self) -> None:
        calls: list[int] = []
        watchdog = IdleWatchdog(0.03, 0.01, lambda: calls.append(1))
        watchdog.pause()
        watchdog.start()
        try:
            time.sleep(0.08)
            self.assertEqual(calls, [])
            watchdog.resume()
            deadline = time.monotonic() + 1.0
            while not calls and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            watchdog.stop()
        self.assertEqual(calls, [1])


class ReplLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.repl = make_repl(self.tmp, idle_timeout=0.05)

    def _record(self) -> RunRecord:
        return RunRecord(
            model_id="qwen_coder",
            pid=123,
            port=4060,
            profile="2",
            alias="qwen2.5-coder-7b-instruct-q6_k",
            started_at="2026-10-27T00:00:00",
            log_path="x.log",
        )

    def test_profile_writes_preferences_and_is_pending(self) -> None:
        _value, out, _err = quiet_run(lambda: self.repl._cmd_profile("2"))
        self.assertEqual(self.repl.profile.key, "1")
        self.assertEqual(self.repl.pending_profile.key, "2")
        self.assertIn("偏好已写入", out)
        loaded = Preferences.load(self.repl.config.preferences.path)
        self.assertEqual(loaded.profile_for("qwen_coder"), "2")

    def test_restart_cancel_stays_in_console(self) -> None:
        self.repl.session.add_user("hi")
        self.repl.pending_profile = self.repl.model.profiles["2"]
        with (
            mock.patch("builtins.input", return_value="c"),
            mock.patch.object(repl_mod.launcher, "stop_all") as stop,
        ):
            _value, out, _err = quiet_run(lambda: self.repl._cmd_restart(""))
        stop.assert_not_called()
        self.assertIn("已取消", out)

    def test_restart_saves_dirty_session_then_restarts(self) -> None:
        self.repl.session.add_user("hi")
        self.repl.pending_profile = self.repl.model.profiles["2"]
        saved_path = self.tmp / "saved.json"
        record = self._record()
        fake_client = mock.Mock()
        with (
            mock.patch("builtins.input", return_value="y"),
            mock.patch.object(
                self.repl.session, "save", return_value=saved_path
            ) as save,
            mock.patch.object(repl_mod.launcher, "stop_all", return_value=[1]) as stop,
            mock.patch.object(
                repl_mod.launcher, "start", return_value=record
            ) as start,
            mock.patch.object(repl_mod.launcher, "wait_ready", return_value=0.1),
            mock.patch.object(
                repl_mod.LlamaClient, "for_model", return_value=fake_client
            ),
        ):
            _value, out, _err = quiet_run(lambda: self.repl._cmd_restart(""))
        save.assert_called_once()
        stop.assert_called_once()
        start.assert_called_once()
        self.assertEqual(start.call_args.args[2].key, "2")
        self.assertEqual(self.repl.profile.key, "2")
        self.assertIs(self.repl.client, fake_client)
        self.assertIn("服务端已重启", out)

    def test_idle_reclaims_server_and_autosaves(self) -> None:
        self.repl.client = mock.Mock()
        self.repl.chat_runner = mock.Mock()
        self.repl.session.add_user("hi")
        autosave = self.tmp / "autosave.json"
        with (
            mock.patch.object(
                self.repl.session, "save", return_value=autosave
            ) as save,
            mock.patch.object(repl_mod.launcher, "stop_all", return_value=[9]) as stop,
        ):
            quiet_run(self.repl._on_idle)
        save.assert_called_once_with(repl_mod.Repl._autosave_path())
        stop.assert_called_once()
        self.assertIsNone(self.repl.client)
        self.assertTrue(self.repl._sleeping)
        self.assertIn("[休眠]", self.repl._prompt_text())

    def test_sleeping_client_wakes_via_start_engine(self) -> None:
        self.repl._sleeping = True
        self.repl.client = None
        self.repl.chat_runner = None

        def fake_start(_model, _profile):
            self.repl.client = mock.Mock()
            self.repl.chat_runner = mock.Mock()
            return True

        with mock.patch.object(self.repl, "_start_engine", side_effect=fake_start):
            value, _out, _err = quiet_run(self.repl._ensure_client)
        self.assertTrue(value)
        self.assertFalse(self.repl._sleeping)
        self.assertIsNotNone(self.repl.chat_runner)

    def test_shutdown_saves_stops_and_quits(self) -> None:
        self.repl.session.add_user("hi")
        with (
            mock.patch.object(
                self.repl.session, "save", return_value=self.tmp / "x.json"
            ) as save,
            mock.patch.object(repl_mod.launcher, "stop_all", return_value=[7]) as stop,
        ):
            _value, out, _err = quiet_run(lambda: self.repl._cmd_shutdown(""))
        save.assert_called_once()
        stop.assert_called_once()
        self.assertIn("显存通道已完全释放", out)

    def test_exclamation_shorthand_dispatches_exec(self) -> None:
        with (
            mock.patch.object(self.repl, "_bootstrap"),
            mock.patch.object(self.repl._watchdog, "start"),
            mock.patch.object(self.repl._watchdog, "stop"),
            mock.patch.object(
                self.repl,
                "_read_line",
                side_effect=["!echo hi", EOFError()],
            ),
            mock.patch.object(self.repl, "_cmd_exec") as exec_cmd,
        ):
            quiet_run(self.repl.run)
        exec_cmd.assert_called_once_with("echo hi")

    def test_native_exec_pipeline_through_run_exec(self) -> None:
        config = make_repl(self.tmp).config
        result = tools.run_exec(
            config, "echo hello | findstr hello", confirm=lambda _: "y"
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("hello", result.stdout)


if __name__ == "__main__":
    unittest.main()
