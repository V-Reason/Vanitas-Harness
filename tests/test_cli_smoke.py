"""CLI 冒烟：只验证参数解析、配置加载与退出码映射，不启动真实模型。"""

from __future__ import annotations

import contextlib
import io
import json
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
from vanitas.translate import T1Budget, T1Event  # noqa: E402
from vanitas.ui import Palette  # noqa: E402

ENV_KEYS = (
    "VANITAS_PORT",
    "VANITAS_PROFILE",
    "VANITAS_CONFIG_DIR",
    "VANITAS_ORIG_CWD",
)


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


class NamingClient(FakeClient):
    def chat(self, messages, *, max_tokens, temperature=0.2, response_format=None, timeout=None):
        prompt = str(messages[-1]["content"])
        if "文件名：" in prompt:
            name = prompt.split("文件名：", 1)[1]
            if "Clock" in name:
                content = "时钟"
            elif "Station" in name:
                content = "车站"
            elif "English Dir" in name:
                content = "中文目录"
            elif "Clip" in name:
                content = "片段"
            elif "cover" in name:
                content = "封面"
            elif "Hello" in name:
                content = "你好"
            else:
                content = "译名"
            return {"choices": [{"message": {"content": content}}]}
        return super().chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            timeout=timeout,
        )


class PartialNamingClient(NamingClient):
    def chat(self, messages, *, max_tokens, temperature=0.2, response_format=None, timeout=None):
        prompt = str(messages[-1]["content"])
        if "文件名：" in prompt and "NYC_notes" in prompt:
            return {"choices": [{"message": {"content": "文件名：纽约市_笔记"}}]}
        return super().chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            timeout=timeout,
        )


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
        for model_id in ("qwen_coder", "qwen7b", "minicpm", "hymt"):
            self.assertIn(model_id, out)
        self.assertIn("4060", out)
        self.assertIn("qwen2.5-coder-7b-instruct-q6_k", out)

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

    def test_translate_filename_single_file(self) -> None:
        source = self.dir / "The Clock.md"
        source.write_text("Hello\n", encoding="utf-8")
        out_dir = self.dir / "out"
        with clean_env():
            with mock.patch.object(
                cli, "_ensure_engine", return_value=(NamingClient(), 4060)
            ):
                code, _out, err = run(
                    [
                        "translate",
                        str(source),
                        "--out-dir",
                        str(out_dir),
                        "--translate-filename",
                    ]
                )
        self.assertEqual(code, 0, err)
        self.assertTrue((out_dir / "时钟.md").is_file())
        self.assertFalse((out_dir / "时钟.zh.md").exists())
        self.assertIn("文件名 1/1", err)
        self.assertIn("The Clock.md → 时钟.md", err)

    def test_translate_filename_directory_names(self) -> None:
        source_dir = self.dir / "src"
        source_dir.mkdir()
        (source_dir / "The Clock.md").write_text("A\n", encoding="utf-8")
        (source_dir / "Station.md").write_text("B\n", encoding="utf-8")
        out_dir = self.dir / "out"
        with clean_env():
            with mock.patch.object(
                cli, "_ensure_engine", return_value=(NamingClient(), 4060)
            ):
                code, _out, err = run(
                    [
                        "translate",
                        str(source_dir),
                        "--out-dir",
                        str(out_dir),
                        "--translate-name",
                    ]
                )
        self.assertEqual(code, 0, err)
        self.assertTrue((out_dir / "时钟.md").is_file())
        self.assertTrue((out_dir / "车站.md").is_file())

    def test_translate_filename_collision_is_rejected(self) -> None:
        source_dir = self.dir / "src"
        source_dir.mkdir()
        (source_dir / "one.md").write_text("A\n", encoding="utf-8")
        (source_dir / "two.md").write_text("B\n", encoding="utf-8")
        out_dir = self.dir / "out"
        with clean_env():
            with mock.patch.object(
                cli, "_ensure_engine", return_value=(NamingClient(), 4060)
            ):
                code, _out, err = run(
                    [
                        "translate",
                        str(source_dir),
                        "--out-dir",
                        str(out_dir),
                        "--translate-filename",
                    ]
                )
        self.assertEqual(code, 5)
        self.assertIn("输出路径冲突", err)
        self.assertFalse((out_dir / "译名.md").exists())

    def test_translate_filename_only_renames_media_without_content(self) -> None:
        root = self.dir / "media"
        folder = root / "English Dir"
        folder.mkdir(parents=True)
        clip = folder / "Clip.mp4"
        clip.write_bytes(b"\x00\x01\xff")
        cover = folder / "cover.jpg"
        cover.write_bytes(b"\xff\xd8\xff")
        with clean_env():
            with mock.patch.object(
                cli, "_ensure_engine", return_value=(NamingClient(), 4060)
            ):
                code, out, err = run(
                    ["translate", str(root), "--translate-name-only"]
                )
        self.assertEqual(code, 0, err)
        target_dir = root / "中文目录"
        self.assertTrue((target_dir / "片段.mp4").is_file())
        self.assertTrue((target_dir / "封面.jpg").is_file())
        self.assertEqual((target_dir / "片段.mp4").read_bytes(), b"\x00\x01\xff")
        self.assertFalse(folder.exists())
        self.assertIn("目录名 1/3", err)
        self.assertIn("文件名 2/3", err)
        self.assertIn("文件名 3/3", err)
        self.assertIn("名称翻译完成", out)
        self.assertIn("重命名 3", out)

    def test_translate_filename_only_continues_after_fallback(self) -> None:
        root = self.dir / "media"
        root.mkdir()
        (root / "NYC_notes.mp4").write_bytes(b"\x00")
        (root / "Hello.mp4").write_bytes(b"\x01")
        with clean_env():
            with mock.patch.object(
                cli, "_ensure_engine", return_value=(PartialNamingClient(), 4060)
            ):
                code, _out, err = run(
                    ["translate", str(root), "--translate-name-only"]
                )
        self.assertEqual(code, 0, err)
        self.assertTrue((root / "你好.mp4").is_file())
        self.assertTrue((root / "NYC_notes.mp4").is_file())
        self.assertIn("保留原名", err)

    def test_translate_filename_content_fallback_without_out_dir_skips(self) -> None:
        source = self.dir / "NYC_notes.md"
        source.write_text("Body\n", encoding="utf-8")
        with clean_env():
            with mock.patch.object(
                cli, "_ensure_engine", return_value=(PartialNamingClient(), 4060)
            ):
                code, _out, err = run(
                    ["translate", str(source), "--translate-filename"]
                )
        self.assertEqual(code, 5)
        self.assertIn("回退原名会覆盖源文件", err)
        self.assertTrue(source.is_file())
        self.assertFalse((self.dir / "NYC_notes.zh.md").exists())

    def test_translate_filename_content_fallback_with_out_dir_keeps_name(self) -> None:
        source = self.dir / "NYC_notes.md"
        source.write_text("Body\n", encoding="utf-8")
        out_dir = self.dir / "out"
        with clean_env():
            with mock.patch.object(
                cli, "_ensure_engine", return_value=(PartialNamingClient(), 4060)
            ):
                code, _out, err = run(
                    [
                        "translate",
                        str(source),
                        "--translate-filename",
                        "--out-dir",
                        str(out_dir),
                    ]
                )
        self.assertEqual(code, 0, err)
        self.assertTrue((out_dir / "NYC_notes.md").is_file())
        self.assertIn("保留原名", err)

    def test_translate_filename_only_rejects_out_dir(self) -> None:
        source = self.dir / "clip.mp3"
        source.write_bytes(b"\x00")
        out_dir = self.dir / "out"
        with clean_env():
            with mock.patch.object(cli, "_ensure_engine") as engine:
                code, _out, err = run(
                    [
                        "translate",
                        str(source),
                        "--translate-filename-only",
                        "--out-dir",
                        str(out_dir),
                    ]
                )
        self.assertEqual(code, 4)
        self.assertIn("不支持 --out-dir", err)
        engine.assert_not_called()

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

    def test_translate_help_lists_preset(self) -> None:
        with clean_env():
            code, out, _err = run(["translate", "--help"])
        self.assertEqual(code, 0)
        self.assertIn("--preset", out)
        self.assertIn("--translate-name", out)
        self.assertIn("--translate-name-only", out)
        self.assertIn("--translate-filename-only", out)

    def test_translate_unknown_preset_exit_4(self) -> None:
        with clean_env():
            code, _out, err = run(
                ["translate", "a.srt", "--preset", "nope"]
            )
        self.assertEqual(code, 4)
        self.assertIn("未知翻译预设", err)
        self.assertIn("general", err)
        self.assertIn("prose", err)
        self.assertIn("subtitle", err)
        self.assertIn("asmr", err)

    def test_translate_scene_presets_choose_effective_settings(self) -> None:
        cases = (
            ([], "subtitle", "2", 12, 800, 1),
            (["--preset", "prose"], "prose", "2", 10, 800, 1),
            (["--preset", "asmr"], "asmr", "2", 12, 800, 6),
            (["--preset", "general"], "general", "1", 5, 1500, 1),
        )
        for extra, name, profile_key, context, chunk, window in cases:
            with self.subTest(preset=name):
                outcome = mock.Mock()
                outcome.source = Path("a.srt")
                outcome.output = Path("a.zh.srt")
                outcome.segments = 1
                outcome.failures = []
                with clean_env():
                    with (
                        mock.patch.object(
                            cli,
                            "_ensure_engine",
                            return_value=(FakeClient(), 4060),
                        ) as engine,
                        mock.patch.object(
                            cli.translate,
                            "translate_file",
                            return_value=outcome,
                        ) as translate_file,
                    ):
                        code, _out, err = run(
                            ["translate", "a.srt", *extra]
                        )
                self.assertEqual(code, 0, err)
                self.assertEqual(engine.call_args.args[2].key, profile_key)
                passed = translate_file.call_args.args[2]
                self.assertEqual(passed.context_window, context)
                self.assertEqual(passed.chunk_chars, chunk)
                self.assertEqual(
                    translate_file.call_args.kwargs["window_size"], window
                )
                self.assertEqual(
                    translate_file.call_args.kwargs["max_window_chars"], 250
                )

    def test_count_asmr_cues_ignores_non_srt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            srt = root / "a.srt"
            body = (
                "1\n00:00:01,000 --> 00:00:02,000\n你好\n\n"
                "2\n00:00:03,000 --> 00:00:04,000\n再见\n"
            )
            srt.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
            markdown = root / "b.md"
            markdown.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n不是字幕\n",
                encoding="utf-8",
            )
            total, counts = cli._count_asmr_cues([srt, markdown])
        self.assertEqual(total, 2)
        self.assertEqual(counts, {str(srt): 2})
        self.assertNotIn(str(markdown), counts)

    def test_translate_asmr_precount_budget(self) -> None:
        source = self.dir / "long.srt"
        blocks = [
            f"{index}\n00:00:01,000 --> 00:00:02,000\n第{index}句"
            for index in range(1, 1405)
        ]
        source.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
        out_dir = self.dir / "out"
        outcome = mock.Mock()
        outcome.source = source
        outcome.output = out_dir / "long.zh.srt"
        outcome.segments = 1404
        outcome.failures = []
        outcome.t1_events = ()
        with clean_env():
            with (
                mock.patch.object(
                    cli, "_ensure_engine", return_value=(FakeClient(), 4060)
                ),
                mock.patch.object(
                    cli.translate,
                    "translate_file",
                    return_value=outcome,
                ) as translate_file,
            ):
                code, _out, err = run(
                    [
                        "translate",
                        str(source),
                        "--preset",
                        "asmr",
                        "--out-dir",
                        str(out_dir),
                    ]
                )
        self.assertEqual(code, 0, err)
        budget = translate_file.call_args.kwargs["t1_budget"]
        self.assertEqual(budget.max_total, 14)
        self.assertEqual(
            translate_file.call_args.kwargs["file_cues"], 1404
        )
        budget_file = out_dir / "t1_budget.json"
        data = json.loads(budget_file.read_text(encoding="utf-8"))
        self.assertEqual(data["max_total"], 14)
        self.assertEqual(data["total_cues"], 1404)

    def test_translate_asmr_guard_passthrough(self) -> None:
        cases = (
            ([], False),
            (["--preset", "subtitle"], False),
            (["--preset", "prose"], False),
            (["--preset", "general"], False),
            (["--preset", "asmr"], True),
        )
        for extra, expected in cases:
            with self.subTest(extra=extra):
                outcome = mock.Mock()
                outcome.source = Path("a.srt")
                outcome.output = Path("a.zh.srt")
                outcome.segments = 1
                outcome.failures = []
                with clean_env():
                    with (
                        mock.patch.object(
                            cli,
                            "_ensure_engine",
                            return_value=(FakeClient(), 4060),
                        ),
                        mock.patch.object(
                            cli.translate,
                            "translate_file",
                            return_value=outcome,
                        ) as translate_file,
                    ):
                        code, _out, err = run(
                            ["translate", "a.srt", *extra]
                        )
                self.assertEqual(code, 0, err)
                self.assertIs(
                    translate_file.call_args.kwargs["asmr_guard"], expected
                )
                self.assertIsInstance(
                    translate_file.call_args.kwargs["t1_budget"], T1Budget
                )

    def test_translate_reports_t1_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outcome = mock.Mock()
            outcome.source = Path("a.srt")
            outcome.output = Path(tmp) / "a.zh.srt"
            outcome.segments = 1
            outcome.failures = []
            outcome.t1_events = (
                T1Event(2, "short_duplicate", "accepted"),
                T1Event(4, "address_term", "budget"),
            )
            with clean_env():
                with (
                    mock.patch.object(
                        cli,
                        "_ensure_engine",
                        return_value=(FakeClient(), 4060),
                    ),
                    mock.patch.object(
                        cli.translate, "translate_file", return_value=outcome
                    ),
                ):
                    code, out, err = run(
                        [
                            "translate",
                            "a.srt",
                            "--preset",
                            "asmr",
                            "--out-dir",
                            tmp,
                        ]
                    )
        self.assertEqual(code, 0, err)
        self.assertIn("T1 守卫", out)
        self.assertIn("接受 1", out)
        self.assertIn("超预算 1", out)
        self.assertIn("cue 2 short_duplicate accepted", out)

    def test_translate_t1_events_file_scoped_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)

            def fake_translate(chat, source, settings, **kwargs):
                outcome = mock.Mock()
                outcome.source = Path(source)
                outcome.output = out_dir / (
                    Path(source).stem + ".zh.srt"
                )
                outcome.segments = 1
                outcome.failures = []
                outcome.t1_events = (
                    T1Event(
                        2,
                        "short_duplicate",
                        "accepted",
                        source=str(source),
                    ),
                )
                return outcome

            with clean_env():
                with (
                    mock.patch.object(
                        cli,
                        "_ensure_engine",
                        return_value=(FakeClient(), 4060),
                    ),
                    mock.patch.object(
                        cli.translate,
                        "translate_file",
                        side_effect=fake_translate,
                    ),
                ):
                    code, out, err = run(
                        [
                            "translate",
                            "a.srt",
                            "b.srt",
                            "--preset",
                            "asmr",
                            "--out-dir",
                            str(out_dir),
                        ]
                    )
            self.assertEqual(code, 0, err)
            self.assertIn("a.srt cue 2 short_duplicate accepted", out)
            self.assertIn("b.srt cue 2 short_duplicate accepted", out)
            self.assertIn("事件 2", out)
            self.assertIn("唯一 cue 2", out)

            events_file = out_dir / "t1_events.jsonl"
            self.assertTrue(events_file.is_file())
            records = [
                json.loads(line)
                for line in events_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(
                {Path(record["source"]).name for record in records},
                {"a.srt", "b.srt"},
            )
            self.assertTrue(
                all(
                    record["cue"] == 2
                    and record["code"] == "short_duplicate"
                    and record["outcome"] == "accepted"
                    for record in records
                )
            )
            budget_file = out_dir / "t1_budget.json"
            self.assertTrue(budget_file.is_file())
            budget = json.loads(
                budget_file.read_text(encoding="utf-8")
            )
            self.assertEqual(budget["total"], 0)
            self.assertEqual(budget["max_total"], 5)
            self.assertIn("total_cues", budget)

    def test_translate_t1_summary_counts_unique_cues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            outcome = mock.Mock()
            outcome.source = Path("a.srt")
            outcome.output = out_dir / "a.zh.srt"
            outcome.segments = 1
            outcome.failures = []
            outcome.t1_events = (
                T1Event(
                    2,
                    "short_duplicate",
                    "accepted",
                    source="a.srt",
                ),
                T1Event(
                    2,
                    "invented_tool",
                    "accepted",
                    source="a.srt",
                ),
            )
            with clean_env():
                with (
                    mock.patch.object(
                        cli,
                        "_ensure_engine",
                        return_value=(FakeClient(), 4060),
                    ),
                    mock.patch.object(
                        cli.translate,
                        "translate_file",
                        return_value=outcome,
                    ),
                ):
                    code, out, err = run(
                        [
                            "translate",
                            "a.srt",
                            "--preset",
                            "asmr",
                            "--out-dir",
                            str(out_dir),
                        ]
                    )
            self.assertEqual(code, 0, err)
            self.assertIn("事件 2", out)
            self.assertIn("唯一 cue 1", out)
            self.assertIn("a.srt cue 2 short_duplicate accepted", out)
            self.assertIn("a.srt cue 2 invented_tool accepted", out)

    def test_translate_explicit_profile_overrides_preset(self) -> None:
        outcome = mock.Mock()
        outcome.source = Path("a.srt")
        outcome.output = Path("a.zh.srt")
        outcome.segments = 1
        outcome.failures = []
        with clean_env():
            with (
                mock.patch.object(
                    cli, "_ensure_engine", return_value=(FakeClient(), 4060)
                ) as engine,
                mock.patch.object(
                    cli.translate, "translate_file", return_value=outcome
                ),
            ):
                code, _out, err = run(
                    [
                        "translate",
                        "a.srt",
                        "--preset",
                        "subtitle",
                        "--profile",
                        "1",
                    ]
                )
        self.assertEqual(code, 0, err)
        self.assertEqual(engine.call_args.args[2].key, "1")

    def test_ensure_engine_restarts_when_profile_differs(self) -> None:
        config = load_config()
        model = config.model("hymt")
        profile = config.resolve_profile(model, "2")
        client = mock.Mock()
        client.health.return_value = True
        record = mock.Mock(model_id="hymt", pid=123, profile="1")
        started = mock.Mock(port=4062)
        with (
            mock.patch.object(cli.LlamaClient, "for_model", return_value=client),
            mock.patch.object(
                cli.launcher, "read_records", return_value=[record]
            ),
            mock.patch.object(cli.launcher, "pid_running", return_value=True),
            mock.patch.object(cli.launcher, "stop_all") as stop,
            mock.patch.object(
                cli.launcher, "start", return_value=started
            ) as start,
            mock.patch.object(cli.launcher, "wait_ready"),
        ):
            _client, port = cli._ensure_engine(
                config, model, profile, Palette({}, enabled=False)
            )
        stop.assert_called_once()
        start.assert_called_once()
        self.assertEqual(port, 4062)

    def test_ensure_engine_reuses_matching_profile(self) -> None:
        config = load_config()
        model = config.model("hymt")
        profile = config.resolve_profile(model, "2")
        client = mock.Mock()
        client.health.return_value = True
        record = mock.Mock(model_id="hymt", pid=123, profile="2")
        with (
            mock.patch.object(cli.LlamaClient, "for_model", return_value=client),
            mock.patch.object(
                cli.launcher, "read_records", return_value=[record]
            ),
            mock.patch.object(cli.launcher, "pid_running", return_value=True),
            mock.patch.object(cli.launcher, "stop_all") as stop,
            mock.patch.object(cli.launcher, "start") as start,
        ):
            _client, port = cli._ensure_engine(
                config, model, profile, Palette({}, enabled=False)
            )
        stop.assert_not_called()
        start.assert_not_called()
        self.assertEqual(port, 4062)

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

    def test_default_command_dispatches_agent_without_subcommand_args(self) -> None:
        with clean_env():
            with mock.patch.object(cli.agent_mod, "run_agent", return_value=0) as called:
                code, _out, err = run([])
        self.assertEqual(code, 0, err)
        called.assert_called_once()
        args, kwargs = called.call_args
        self.assertEqual(args[1].id, "qwen_coder")
        self.assertIsNone(kwargs["system"])

    def test_chat_uses_configured_non_coder_default(self) -> None:
        with clean_env():
            with mock.patch.object(cli, "Repl") as repl_cls:
                repl_cls.return_value.run.return_value = 0
                code, _out, err = run(["chat"])
        self.assertEqual(code, 0, err)
        repl_cls.assert_called_once()
        self.assertEqual(repl_cls.call_args.args[1].id, "qwen7b")

    def test_models_command_shows_default_routes(self) -> None:
        with clean_env():
            code, out, err = run(["models"])
        self.assertEqual(code, 0, err)
        self.assertIn("默认 agent=qwen_coder", out)
        self.assertIn("chat=qwen7b", out)
        self.assertIn("translate=hymt", out)
        self.assertIn("batch=qwen7b", out)

    def test_translate_resolves_relative_paths_against_invocation_cwd(self) -> None:
        source = self.dir / "a.srt"
        source.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
            encoding="utf-8",
        )
        out_dir = self.dir / "out"
        outcome = mock.Mock()
        outcome.source = source
        outcome.output = out_dir / "a.zh.srt"
        outcome.segments = 1
        outcome.failures = []
        with clean_env(VANITAS_ORIG_CWD=str(self.dir)):
            with mock.patch.object(
                cli, "_translate_chat", return_value=(FakeClient(), 0)
            ):
                with mock.patch.object(
                    cli.translate,
                    "translate_file",
                    return_value=outcome,
                ) as translate_file:
                    code, _out, err = run(
                        ["translate", "a.srt", "--out-dir", "out"]
                    )
        self.assertEqual(code, 0, err)
        call = translate_file.call_args
        self.assertEqual(call.args[1].resolve(), source.resolve())
        self.assertEqual(call.kwargs["out_dir"].resolve(), out_dir.resolve())

    def test_translate_resets_history_between_files(self) -> None:
        outcome = mock.Mock()
        outcome.source = Path("a.md")
        outcome.output = Path("a.zh.md")
        outcome.segments = 1
        outcome.failures = []
        chat = mock.Mock()
        chat.reset_history = mock.Mock()
        with clean_env():
            with (
                mock.patch.object(
                    cli, "_translate_chat", return_value=(chat, 0)
                ),
                mock.patch.object(
                    cli.translate, "translate_file", return_value=outcome
                ),
            ):
                code, _out, err = run(["translate", "a.md", "b.md"])
        self.assertEqual(code, 0, err)
        self.assertEqual(chat.reset_history.call_count, 2)

    def test_translate_two_files_get_index_and_single_report(self) -> None:
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
                code, out, err = run(
                    ["translate", str(first), str(second), "--out-dir", str(out_dir)]
                )
        self.assertEqual(code, 0, err)
        self.assertIn("正在翻译 a.srt", err)
        self.assertIn("[1/2]", err)
        self.assertIn("[2/2]", err)
        self.assertIn("翻译完成", out)
        self.assertEqual(out.count("翻译完成"), 1)
        self.assertIn("[1/2]", out)
        self.assertIn("[2/2]", out)
        self.assertNotRegex(err, r"^\d+ 段 · \d+ 字 · ")
        self.assertNotIn("翻译任务已完成", err)
        self.assertNotIn("统计段数", err)
        self.assertNotIn("累计汉字", err)

    def test_translate_partial_failure_reports_warn_and_exit_5(self) -> None:
        source = self.dir / "a.srt"
        source.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
            encoding="utf-8",
        )
        outcome = mock.Mock()
        outcome.source = source
        outcome.output = self.dir / "a.zh.srt"
        outcome.segments = 2
        outcome.failures = [
            mock.Mock(segment="line 2", reason="模型返回空内容")
        ]
        with clean_env():
            with (
                mock.patch.object(
                    cli, "_ensure_engine", return_value=(FakeClient(), 4060)
                ),
                mock.patch.object(
                    cli.translate, "translate_file", return_value=outcome
                ),
            ):
                code, out, err = run(["translate", str(source)])
        self.assertEqual(code, 5, err)
        self.assertIn("⚠", err)
        self.assertIn("1 段失败", err)
        self.assertIn("部分失败 1", out)
        self.assertIn("失败明细", out)
        self.assertIn("line 2：模型返回空内容", out)

    def test_translate_file_error_continues_and_reports(self) -> None:
        first = self.dir / "a.srt"
        first.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
            encoding="utf-8",
        )
        second = self.dir / "b.srt"
        second.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nBye\n",
            encoding="utf-8",
        )
        outcome = mock.Mock()
        outcome.source = second
        outcome.output = self.dir / "b.zh.srt"
        outcome.segments = 1
        outcome.failures = []

        def fake_translate(_chat, path, _settings, **_kwargs):
            if Path(path).name == "a.srt":
                raise cli.TranslateError("读取失败")
            return outcome

        with clean_env():
            with (
                mock.patch.object(
                    cli, "_ensure_engine", return_value=(FakeClient(), 4060)
                ),
                mock.patch.object(
                    cli.translate, "translate_file", side_effect=fake_translate
                ),
            ):
                code, out, err = run(
                    ["translate", str(first), str(second)]
                )
        self.assertEqual(code, 5, err)
        self.assertIn("✗ [1/2]", err)
        self.assertIn("读取失败", err)
        self.assertIn("✓ [2/2]", err)
        self.assertIn("失败 1", out)

    def test_translate_marks_existing_output_as_overwrite(self) -> None:
        source = self.dir / "a.srt"
        source.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
            encoding="utf-8",
        )
        (self.dir / "a.zh.srt").write_text("old\n", encoding="utf-8")
        outcome = mock.Mock()
        outcome.source = source
        outcome.output = self.dir / "a.zh.srt"
        outcome.segments = 1
        outcome.failures = []
        with clean_env():
            with (
                mock.patch.object(
                    cli, "_ensure_engine", return_value=(FakeClient(), 4060)
                ),
                mock.patch.object(
                    cli.translate, "translate_file", return_value=outcome
                ),
            ):
                code, out, err = run(["translate", str(source)])
        self.assertEqual(code, 0, err)
        self.assertIn("覆盖", err)
        self.assertIn("覆盖 1", out)

    def test_translate_verbose_shows_all_report_rows(self) -> None:
        names = [f"{index}.srt" for index in range(1, 14)]
        for name in names:
            (self.dir / name).write_text(
                f"1\n00:00:01,000 --> 00:00:02,000\n{name}\n",
                encoding="utf-8",
            )
        outcome = mock.Mock()
        outcome.source = self.dir / names[0]
        outcome.output = self.dir / "1.zh.srt"
        outcome.segments = 1
        outcome.failures = []
        with clean_env():
            with (
                mock.patch.object(
                    cli, "_ensure_engine", return_value=(FakeClient(), 4060)
                ),
                mock.patch.object(
                    cli.translate, "translate_file", return_value=outcome
                ),
            ):
                code, out, err = run(
                    ["translate", *[str(self.dir / name) for name in names], "-v"]
                )
        self.assertEqual(code, 0, err)
        self.assertIn("[1/13]", out)
        self.assertIn("[13/13]", out)

    def test_translate_keyboard_interrupt_reports_partial_batch(self) -> None:
        first = self.dir / "a.srt"
        first.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
            encoding="utf-8",
        )
        second = self.dir / "b.srt"
        second.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nBye\n",
            encoding="utf-8",
        )

        def fake_translate(_chat, _path, _settings, **_kwargs):
            raise KeyboardInterrupt

        with clean_env():
            with (
                mock.patch.object(
                    cli, "_ensure_engine", return_value=(FakeClient(), 4060)
                ),
                mock.patch.object(
                    cli.translate, "translate_file", side_effect=fake_translate
                ),
            ):
                code, out, err = run(
                    ["translate", str(first), str(second)]
                )
        self.assertEqual(code, 1, err)
        self.assertIn("翻译中断", out)
        self.assertIn("中断 1", out)
        self.assertIn("已中断", err)

    def test_translate_wires_on_segment_callback(self) -> None:
        source = self.dir / "a.srt"
        source.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
            encoding="utf-8",
        )
        outcome = mock.Mock()
        outcome.source = source
        outcome.output = self.dir / "a.zh.srt"
        outcome.segments = 1
        outcome.failures = []
        seen: list[tuple[int, int]] = []

        def fake_translate(_chat, _path, _settings, **kwargs):
            callback = kwargs["on_segment"]
            callback(1, 4)
            seen.append((1, 4))
            return outcome

        with clean_env():
            with (
                mock.patch.object(
                    cli, "_ensure_engine", return_value=(FakeClient(), 4060)
                ),
                mock.patch.object(
                    cli.translate, "translate_file", side_effect=fake_translate
                ),
            ):
                code, _out, err = run(["translate", str(source)])
        self.assertEqual(code, 0, err)
        self.assertEqual(seen, [(1, 4)])

    def test_translate_directory_recursively_preserves_tree(self) -> None:
        source_dir = self.dir / "src"
        (source_dir / "sub").mkdir(parents=True)
        (source_dir / "a.md").write_text("A\n\nB\n", encoding="utf-8")
        (source_dir / "sub" / "a.md").write_text("C\n", encoding="utf-8")
        (source_dir / "sub" / "b.srt").write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
            encoding="utf-8",
        )
        (source_dir / "skip.json").write_text("{}", encoding="utf-8")
        (source_dir / "old.zh.md").write_text("旧译文\n", encoding="utf-8")
        out_dir = self.dir / "out"
        with clean_env():
            with mock.patch.object(
                cli, "_ensure_engine", return_value=(FakeClient(), 4060)
            ):
                code, out, err = run(
                    ["translate", str(source_dir), "--out-dir", str(out_dir)]
                )
        self.assertEqual(code, 0, err)
        self.assertTrue((out_dir / "a.zh.md").is_file())
        self.assertTrue((out_dir / "sub" / "a.zh.md").is_file())
        self.assertTrue((out_dir / "sub" / "b.zh.srt").is_file())
        self.assertFalse((out_dir / "skip.zh.json").exists())
        self.assertFalse((out_dir / "old.zh.zh.md").exists())
        self.assertIn("3 个文件", out)
        self.assertIn("sub/a.md", err)

    def test_translate_empty_directory_fails_before_engine_start(self) -> None:
        source_dir = self.dir / "empty"
        source_dir.mkdir()
        with clean_env():
            with mock.patch.object(cli, "_ensure_engine") as engine:
                code, _out, err = run(["translate", str(source_dir)])
        self.assertEqual(code, 5)
        self.assertIn("没有可翻译文件", err)
        engine.assert_not_called()

    def test_translate_output_path_that_is_file_exits_4(self) -> None:
        source = self.dir / "a.md"
        source.write_text("A\n", encoding="utf-8")
        target = self.dir / "out"
        target.write_text("not a directory\n", encoding="utf-8")
        with clean_env():
            code, _out, err = run(
                ["translate", str(source), "--out-dir", str(target)]
            )
        self.assertEqual(code, 4)
        self.assertIn("输出路径不是目录", err)

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
