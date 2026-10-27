"""翻译：SRT 时间轴/多行/BOM/换行、文本分块、失败汇总。"""

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.config import TranslateSettings  # noqa: E402
from vanitas.errors import EngineError, TranslateError  # noqa: E402
from vanitas.translate import (  # noqa: E402
    NameTarget,
    TranslationEngine,
    TranslationOptions,
    apply_name_renames,
    chunk_text,
    collect_name_targets,
    collect_translate_inputs,
    parse_srt,
    parse_window_response,
    plan_name_renames,
    plan_translation_outputs,
    rebuild_srt,
    sanitize_filename_stem,
    sanitize_translation_output,
    translate_file,
)

SETTINGS = TranslateSettings(
    source_suffixes=(".txt", ".md", ".srt"),
    chunk_chars=1500,
    output_suffix=".zh",
)
SMALL_CHUNKS = TranslateSettings(
    source_suffixes=(".txt", ".md", ".srt"),
    chunk_chars=8,
    output_suffix=".zh",
)
OPTION_DEFAULTS = {
    "n_ctx": 8192,
    "context_window": 5,
    "context_summary_chars": 200,
    "min_output_tokens": 512,
    "output_ratio": 2.5,
    "prompt_safety_tokens": 64,
    "temperature": 0.2,
    "max_split_depth": 3,
}


def translation_options(**overrides) -> TranslationOptions:
    return TranslationOptions(**{**OPTION_DEFAULTS, **overrides})

SRT_CRLF = (
    "1\r\n"
    "00:00:01,000 --> 00:00:02,500\r\n"
    "Hello world\r\n"
    "Second line\r\n"
    "\r\n"
    "2\r\n"
    "00:00:03,000 --> 00:00:04,000\r\n"
    "Goodbye\r\n"
)


def echo_chat(text: str) -> str:
    return "译" + text


class TranslateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write(self, name: str, text: str, *, bom: bool = False) -> Path:
        path = self.dir / name
        data = text.encode("utf-8")
        path.write_bytes(b"\xef\xbb\xbf" + data if bom else data)
        return path

    def test_srt_preserves_index_timing_crlf_bom(self) -> None:
        source = self.write("sample.srt", SRT_CRLF, bom=True)
        outcome = translate_file(echo_chat, source, SETTINGS)
        self.assertEqual(outcome.output, self.dir / "sample.zh.srt")
        self.assertEqual(outcome.segments, 2)
        self.assertEqual(outcome.failures, [])

        raw = outcome.output.read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"), "BOM 必须保留")
        text = raw[3:].decode("utf-8")
        self.assertIn("\r\n", text)
        self.assertNotIn("\n\n\n", text)

        blocks = parse_srt(text, "\r\n")
        self.assertEqual([block.index for block, _ in blocks], ["1", "2"])
        self.assertEqual(
            [block.timing for block, _ in blocks],
            [
                "00:00:01,000 --> 00:00:02,500",
                "00:00:03,000 --> 00:00:04,000",
            ],
        )
        first_text = blocks[0][0].text
        # 多行字幕整体翻译（一次请求），行结构保留
        self.assertEqual(first_text.split("\r\n"), ["译Hello world", "Second line"])

    def test_srt_without_bom_keeps_no_bom(self) -> None:
        source = self.write("plain.srt", SRT_CRLF.replace("\r\n", "\n"))
        outcome = translate_file(echo_chat, source, SETTINGS)
        raw = outcome.output.read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\r\n", raw)
        self.assertEqual(parse_srt(raw.decode("utf-8"), "\n")[0][0].index, "1")

    def test_parse_srt_mixed_eol_recovers_all_blocks(self) -> None:
        text = (
            "1\r\n00:00:01,000 --> 00:00:02,000\r\nHello\r\n\r\n"
            "2\n00:00:03,000 --> 00:00:04,000\nWorld\n\n"
            "3\r\n00:00:05,000 --> 00:00:06,000\n"
            "Third one\r\nThird two\r\n"
        )
        blocks = parse_srt(text, "\r\n")
        self.assertEqual([block.index for block, _ in blocks], ["1", "2", "3"])
        self.assertEqual(
            [block.timing for block, _ in blocks],
            [
                "00:00:01,000 --> 00:00:02,000",
                "00:00:03,000 --> 00:00:04,000",
                "00:00:05,000 --> 00:00:06,000",
            ],
        )
        self.assertEqual(blocks[2][0].text, "Third one\r\nThird two")

        rebuilt = rebuild_srt(blocks, "\n")
        self.assertNotIn("\r", rebuilt)
        self.assertTrue(rebuilt.startswith("1\n00:00:01,000"))

        source = self.write("mixed.srt", text)
        outcome = translate_file(echo_chat, source, SETTINGS)
        self.assertEqual(outcome.segments, 3)
        self.assertEqual(outcome.failures, [])
        translated = outcome.output.read_text(encoding="utf-8")
        for expected in ("译Hello", "译World", "译Third one"):
            self.assertIn(expected, translated)

    def test_parse_window_response_success_and_missing(self) -> None:
        ok, aligned, missing = parse_window_response(
            "[#1] 第一句\n[#2] 第二句", ["1", "2"]
        )
        self.assertTrue(ok)
        self.assertEqual(aligned, {"1": "第一句", "2": "第二句"})
        self.assertEqual(missing, [])

        ok, aligned, _missing = parse_window_response(
            "[#1] \n[#2] 第二句", ["1", "2"]
        )
        self.assertTrue(ok)
        self.assertEqual(aligned["1"], "")
        self.assertEqual(aligned["2"], "第二句")

        ok, aligned, missing = parse_window_response(
            "[#1] 第一句", ["1", "2"]
        )
        self.assertFalse(ok)
        self.assertEqual(aligned, {})
        self.assertEqual(missing, ["2"])

        ok, aligned, _missing = parse_window_response(
            "[#2] 第二句\n[#1] 第一句", ["1", "2"]
        )
        self.assertFalse(ok)
        self.assertEqual(aligned, {})

        ok, aligned, _missing = parse_window_response(
            "[#1] 第一句\n[#2] 第二句\n[#3] 多翻", ["1", "2"]
        )
        self.assertFalse(ok)
        self.assertEqual(aligned, {})

    def _windowed_srt(self) -> str:
        return (
            "1\n00:00:01,000 --> 00:00:02,000\n一\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n二\n\n"
            "3\n00:00:05,000 --> 00:00:06,000\n三\n\n"
            "4\n00:00:07,000 --> 00:00:08,000\n四\n"
        )

    def test_translate_srt_windowed_flow(self) -> None:
        source = self.write("window.srt", self._windowed_srt())
        calls: list[str] = []

        def chat(prompt: str) -> str:
            calls.append(prompt)
            if "\n" not in prompt:
                return f"译{prompt}"
            lines: list[str] = []
            for line in prompt.splitlines():
                match = re.match(r"^\[#(\d+)\]\s*(.*)$", line)
                if match:
                    lines.append(f"[#{match.group(1)}] 译{match.group(2)}")
            return "\n".join(lines)

        progress: list[tuple[int, int]] = []
        outcome = translate_file(
            chat,
            source,
            SETTINGS,
            window_size=2,
            on_segment=lambda done, total: progress.append((done, total)),
        )
        self.assertEqual(outcome.segments, 4)
        self.assertEqual(outcome.failures, [])
        self.assertEqual(len(calls), 2)
        self.assertTrue(all("\n" in call for call in calls))
        self.assertEqual(progress, [(2, 4), (4, 4)])
        translated = outcome.output.read_text(encoding="utf-8")
        blocks = parse_srt(translated, "\n")
        self.assertEqual([block.index for block, _ in blocks], ["1", "2", "3", "4"])
        self.assertEqual(
            [block.timing for block, _ in blocks],
            [
                "00:00:01,000 --> 00:00:02,000",
                "00:00:03,000 --> 00:00:04,000",
                "00:00:05,000 --> 00:00:06,000",
                "00:00:07,000 --> 00:00:08,000",
            ],
        )
        self.assertEqual(
            [block.text for block, _ in blocks],
            ["译一", "译二", "译三", "译四"],
        )

    def test_translate_srt_windowed_fallback_on_guardrail_trip(self) -> None:
        source = self.write("fallback.srt", self._windowed_srt())
        batch_calls = 0
        single_calls = 0

        def chat(prompt: str) -> str:
            nonlocal batch_calls, single_calls
            if "\n" in prompt:
                batch_calls += 1
                # 只回第一行，故意漏掉同窗第二个序号
                return prompt.splitlines()[0]
            single_calls += 1
            return f"译{prompt}"

        progress: list[tuple[int, int]] = []
        outcome = translate_file(
            chat,
            source,
            SETTINGS,
            window_size=2,
            on_segment=lambda done, total: progress.append((done, total)),
        )
        self.assertEqual(batch_calls, 2)
        self.assertEqual(single_calls, 4)
        self.assertEqual(outcome.segments, 4)
        self.assertEqual(outcome.failures, [])
        self.assertEqual(progress, [(2, 4), (4, 4)])
        translated = outcome.output.read_text(encoding="utf-8")
        blocks = parse_srt(translated, "\n")
        self.assertEqual([block.index for block, _ in blocks], ["1", "2", "3", "4"])
        self.assertEqual(
            [block.text for block, _ in blocks],
            ["译一", "译二", "译三", "译四"],
        )

    def test_translate_file_with_asmr_preset_end_to_end(self) -> None:
        source = ROOT / "tests" / "fixtures" / "sample_asmr.srt"
        raw = source.read_bytes()
        self.assertIn(b"\r\n", raw)
        self.assertIn(b"\n", raw.replace(b"\r\n", b""))
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))

        out_dir = self.dir / "asmr"
        calls: list[str] = []

        def mock_chat(prompt: str) -> str:
            calls.append(prompt)
            if "[#1]" in prompt:
                return (
                    "[#1] 那、那个……是离耳朵太近了吗？\n"
                    "[#2] 啾……呵呵，连耳垂都红透了呢。\n"
                    "[#3] （在耳边低语）哈啊……已经忍不住了吗？"
                )
            return f"译{prompt}"

        outcome = translate_file(
            mock_chat,
            source,
            SETTINGS,
            out_dir=out_dir,
            window_size=4,
        )
        self.assertEqual(outcome.output, out_dir / "sample_asmr.zh.srt")
        self.assertTrue(outcome.output.is_file())
        self.assertEqual(outcome.segments, 4)
        self.assertEqual(outcome.failures, [])
        self.assertEqual(len(calls), 1)

        translated = outcome.output.read_text(encoding="utf-8")
        blocks = parse_srt(translated, "\r\n")
        self.assertEqual([block.index for block, _ in blocks], ["1", "2", "3", "4"])
        self.assertEqual(
            [block.timing for block, _ in blocks],
            [
                "00:00:01,000 --> 00:00:03,500",
                "00:00:04,000 --> 00:00:06,000",
                "00:00:07,000 --> 00:00:09,000",
                "00:00:10,000 --> 00:00:12,000",
            ],
        )
        self.assertEqual(
            blocks[0][0].text, "那、那个……是离耳朵太近了吗？"
        )
        self.assertIn("啾", blocks[1][0].text)
        self.assertIn("（在耳边低语）", blocks[2][0].text)
        self.assertIn("哈啊……", blocks[2][0].text)
        self.assertEqual(blocks[3][0].text.strip(), "")
        self.assertNotIn("ご視聴", translated)

    def test_asr_hallucination_hard_filter(self) -> None:
        text = (
            "1\n00:00:01,000 --> 00:00:02,000\n正常台词\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nご視聴ありがとうございました\n\n"
            "3\n00:00:05,000 --> 00:00:06,000\nPlease subscribe\n"
        )
        source = self.write("hallucination.srt", text)
        calls: list[str] = []

        def chat(prompt: str) -> str:
            calls.append(prompt)
            return f"译{prompt}"

        outcome = translate_file(chat, source, SETTINGS)
        self.assertEqual(outcome.segments, 3)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("ご視聴", calls[0])
        self.assertNotIn("Please subscribe", calls[0])
        translated = outcome.output.read_text(encoding="utf-8")
        blocks = parse_srt(translated, "\n")
        self.assertEqual([block.index for block, _ in blocks], ["1", "2", "3"])
        self.assertEqual(blocks[0][0].text, "译正常台词")
        self.assertEqual(blocks[1][0].timing, "00:00:03,000 --> 00:00:04,000")
        self.assertEqual(blocks[1][0].text, "")
        self.assertEqual(blocks[2][0].timing, "00:00:05,000 --> 00:00:06,000")
        self.assertEqual(blocks[2][0].text, "")

    def test_srt_segment_failure_keeps_original_and_reports(self) -> None:
        def failing_chat(text: str) -> str:
            if "Goodbye" in text:
                raise EngineError("模拟引擎失败")
            return "译" + text

        source = self.write("mixed.srt", SRT_CRLF)
        outcome = translate_file(failing_chat, source, SETTINGS)
        self.assertEqual(len(outcome.failures), 1)
        self.assertIn("模拟引擎失败", outcome.failures[0].reason)
        self.assertEqual(outcome.failures[0].segment, "line 2#1")
        text = outcome.output.read_text(encoding="utf-8")
        self.assertIn("Goodbye", text)
        self.assertIn("译Hello world", text)

    def test_prose_preserves_blank_lines(self) -> None:
        text = "## Title\n\nParagraph one.\n\n\nParagraph two.\n"
        source = self.write("doc.md", text)
        outcome = translate_file(echo_chat, source, SETTINGS)
        translated = outcome.output.read_text(encoding="utf-8")
        self.assertEqual(
            translated, "译## Title\n\n译Paragraph one.\n\n\n译Paragraph two.\n"
        )

    def test_txt_chunking_long_paragraph(self) -> None:
        text = "句子一。句子二。句子三。"
        source = self.write("long.txt", text)
        outcome = translate_file(echo_chat, source, SMALL_CHUNKS)
        translated = outcome.output.read_text(encoding="utf-8")
        # chunk_chars=8：前两句落在同一块（8 字符），第三句单独成块
        self.assertEqual(translated, "译句子一。句子二。译句子三。")

    def test_chunk_text_keeps_all_characters(self) -> None:
        text = "句子一。句子二。句子三。"
        chunks = chunk_text(text, 5)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(chunk) <= 5 for chunk in chunks))

    def test_out_dir_and_suffix_naming(self) -> None:
        source = self.write("a.txt", "hello")
        out_dir = self.dir / "out"
        outcome = translate_file(echo_chat, source, SETTINGS, out_dir=out_dir)
        self.assertEqual(outcome.output, out_dir / "a.zh.txt")
        self.assertTrue(outcome.output.is_file())

    def test_unsupported_suffix_rejected(self) -> None:
        source = self.write("a.json", "{}")
        with self.assertRaises(TranslateError):
            translate_file(echo_chat, source, SETTINGS)

    def test_on_segment_called_per_block(self) -> None:
        text = "Para one.\n\nPara two.\n\nPara three.\n"
        source = self.write("three.md", text)
        calls: list[tuple[int, int]] = []
        outcome = translate_file(
            echo_chat, source, SETTINGS, on_segment=lambda done, total: calls.append((done, total))
        )
        self.assertEqual(calls, [(1, 3), (2, 3), (3, 3)])
        self.assertEqual(outcome.segments, 3)

    def test_on_segment_optional(self) -> None:
        # 不传回调时行为与旧版完全一致（向后兼容）
        source = self.write("three.md", "A.\n\nB.\n\nC.\n")
        outcome = translate_file(echo_chat, source, SETTINGS)
        self.assertEqual(outcome.segments, 3)
        self.assertEqual(outcome.failures, [])

    def test_missing_file_rejected(self) -> None:
        with self.assertRaises(TranslateError):
            translate_file(echo_chat, self.dir / "nope.srt", SETTINGS)


class InputDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write(self, name: str, text: str = "x") -> Path:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def relative(self, sources) -> list[str]:
        return [source.path.relative_to(self.dir).as_posix() for source in sources]

    def test_directory_recursively_collects_supported_files(self) -> None:
        self.write("b.md")
        self.write("A.SRT")
        self.write("sub/c.txt")
        self.write("skip.json")
        self.write("old.zh.md")
        self.write("sub/old.zh.srt")

        sources = collect_translate_inputs([self.dir], SETTINGS)

        self.assertEqual(self.relative(sources), ["A.SRT", "b.md", "sub/c.txt"])
        self.assertEqual(sources[0].relative_parent, Path())
        self.assertEqual(sources[1].relative_parent, Path())
        self.assertEqual(sources[2].relative_parent, Path("sub"))

    def test_explicit_generated_name_is_still_translated(self) -> None:
        source = self.write("old.zh.md")
        sources = collect_translate_inputs([source], SETTINGS)
        self.assertEqual([item.path for item in sources], [source])

    def test_overlapping_inputs_are_deduplicated_in_first_seen_order(self) -> None:
        root = self.write("a.md")
        nested = self.write("sub/b.md")
        sources = collect_translate_inputs([root, self.dir, self.dir / "sub"], SETTINGS)
        self.assertEqual([item.path for item in sources], [root, nested])

    def test_empty_or_only_generated_directory_is_rejected(self) -> None:
        self.write("sub/old.zh.md")
        with self.assertRaisesRegex(TranslateError, "没有可翻译文件"):
            collect_translate_inputs([self.dir], SETTINGS)

    def test_exclude_dir_prunes_previous_output_subtree(self) -> None:
        self.write("a.md")
        out_dir = self.dir / "out"
        self.write("out/leftover.md")
        sources = collect_translate_inputs(
            [self.dir], SETTINGS, exclude_dir=out_dir
        )
        self.assertEqual(self.relative(sources), ["a.md"])

    def test_exclude_nested_dir_does_not_drop_sibling_sources(self) -> None:
        self.write("a.md")
        self.write("sub/chapter.md")
        out_dir = self.dir / "sub" / "out"
        self.write("sub/out/leftover.md")
        sources = collect_translate_inputs(
            [self.dir], SETTINGS, exclude_dir=out_dir
        )
        self.assertEqual(self.relative(sources), ["a.md", "sub/chapter.md"])

    def test_plan_preserves_directory_tree_under_out_dir(self) -> None:
        self.write("a.md")
        self.write("sub/b.md")
        out_dir = self.dir / "out"
        plans = plan_translation_outputs(
            collect_translate_inputs([self.dir], SETTINGS), SETTINGS, out_dir
        )
        self.assertEqual(
            [plan.target.relative_to(out_dir).as_posix() for plan in plans],
            ["a.zh.md", "sub/b.zh.md"],
        )
        self.assertEqual(plans[1].out_dir, out_dir / "sub")

    def test_plan_uses_translated_stem(self) -> None:
        self.write("The Clock.md")
        source = collect_translate_inputs([self.dir / "The Clock.md"], SETTINGS)
        source = [replace(source[0], translated_stem="时钟")]
        out_dir = self.dir / "out"
        plans = plan_translation_outputs(
            source, SETTINGS, out_dir, output_suffix=""
        )
        self.assertEqual(plans[0].target, out_dir / "时钟.md")

    def test_plan_rejects_colliding_outputs(self) -> None:
        first = self.dir / "first"
        second = self.dir / "second"
        self.write("first/a.md")
        self.write("second/a.md")
        sources = collect_translate_inputs([first, second], SETTINGS)
        with self.assertRaisesRegex(TranslateError, "输出路径冲突"):
            plan_translation_outputs(sources, SETTINGS, self.dir / "out")

    def test_plan_rejects_output_overwriting_another_input(self) -> None:
        self.write("a.md")
        self.write("a.zh.md")
        sources = collect_translate_inputs(
            [self.dir / "a.md", self.dir / "a.zh.md"], SETTINGS
        )
        with self.assertRaisesRegex(TranslateError, "输出会覆盖输入"):
            plan_translation_outputs(sources, SETTINGS, None)

    def test_explicit_missing_path_is_left_for_translate_file(self) -> None:
        missing = self.dir / "missing"
        sources = collect_translate_inputs([missing], SETTINGS)
        self.assertEqual([item.path for item in sources], [missing])

    def test_directory_order_is_stable_case_insensitive(self) -> None:
        self.write("b.md")
        self.write("A.md")
        self.write("sub/z.md")
        self.write("sub/C.md")
        sources = collect_translate_inputs([self.dir], SETTINGS)
        self.assertEqual(
            self.relative(sources), ["A.md", "b.md", "sub/C.md", "sub/z.md"]
        )


class NameOnlyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write(self, name: str, data: bytes = b"x") -> Path:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_collect_accepts_arbitrary_suffixes_but_skips_hidden(self) -> None:
        self.write("Clip.mp4", b"\x00\x01")
        self.write("cover.JPG", b"\xff\xd8")
        self.write("sub/track.wav", b"RIFF")
        self.write(".hidden.mp3")
        self.write(".git/config")

        targets = collect_name_targets([self.dir])
        relative = {
            (target.path.relative_to(self.dir).as_posix(), target.is_dir)
            for target in targets
        }
        self.assertEqual(
            relative,
            {
                ("Clip.mp4", False),
                ("cover.JPG", False),
                ("sub", True),
                ("sub/track.wav", False),
            },
        )

    def test_explicit_arbitrary_file_is_accepted(self) -> None:
        source = self.write("sound.flac", b"fLaC")
        targets = collect_name_targets([source])
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].path, source)
        self.assertFalse(targets[0].is_dir)

    def test_plan_keeps_file_suffix_and_replaces_directory_name(self) -> None:
        file_path = self.write("Clip.mp4")
        dir_path = self.write("sub/track.wav").parent
        renames = plan_name_renames(
            [
                NameTarget(file_path, False, "片段"),
                NameTarget(dir_path, True, "子目录"),
            ]
        )
        by_old = {rename.old: rename.new for rename in renames}
        self.assertEqual(by_old[file_path], file_path.with_name("片段.mp4"))
        self.assertEqual(by_old[dir_path], dir_path.with_name("子目录"))

    def test_plan_rejects_translated_name_collision(self) -> None:
        first = self.write("one.mp3")
        second = self.write("two.mp3")
        with self.assertRaisesRegex(TranslateError, "译名冲突"):
            plan_name_renames(
                [
                    NameTarget(first, False, "同名"),
                    NameTarget(second, False, "同名"),
                ]
            )

    def test_plan_rejects_existing_unmoved_target(self) -> None:
        source = self.write("one.mp3")
        self.write("目标.mp3")
        with self.assertRaisesRegex(TranslateError, "已存在"):
            plan_name_renames([NameTarget(source, False, "目标")])

    def test_apply_renames_nested_file_and_directory(self) -> None:
        source = self.write("English Dir/Clip.mp4", b"\x00\x01\x02")
        directory = source.parent
        renames = plan_name_renames(
            [
                NameTarget(directory, True, "中文目录"),
                NameTarget(source, False, "片段"),
            ]
        )
        apply_name_renames(renames)

        target = self.dir / "中文目录" / "片段.mp4"
        self.assertTrue(target.is_file())
        self.assertEqual(target.read_bytes(), b"\x00\x01\x02")
        self.assertFalse((self.dir / "English Dir").exists())


class FakeTranslationClient:
    def __init__(self, *, truncate_over: int | None = None) -> None:
        self.calls: list[dict] = []
        self.truncate_over = truncate_over

    def chat(self, messages, *, max_tokens, temperature=0.2):
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        prompt = str(messages[-1]["content"])
        marker = "【当前待翻译内容】\n"
        source = prompt.split(marker, 1)[1] if marker in prompt else prompt
        if self.truncate_over is not None and len(source) > self.truncate_over:
            return {
                "choices": [
                    {
                        "message": {"content": "半句"},
                        "finish_reason": "length",
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {"content": f"译{len(self.calls)}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 3},
            "timings": {"predicted_per_second": 10.0, "predicted_n": 3},
        }


class FilenameTranslationClient:
    def __init__(
        self,
        content: str = "文件名：“时钟.md”",
        finish: str = "stop",
        responses: list[tuple[str, str]] | None = None,
    ) -> None:
        self.content = content
        self.finish = finish
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def chat(self, messages, *, max_tokens, temperature=0.2):
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if self.responses:
            content, finish = (
                self.responses.pop(0)
                if len(self.responses) > 1
                else self.responses[0]
            )
        else:
            content, finish = self.content, self.finish
        return {
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": finish,
                }
            ]
        }


class TranslationEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompt_dir = ROOT / "config" / "prompts"

    def make_engine(self, client, **options) -> TranslationEngine:
        return TranslationEngine(
            client,
            self.prompt_dir,
            "系统提示",
            options=translation_options(**options),
        )

    def test_engine_uses_configured_task_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = Path(tmp)
            (prompt_dir / "task_translate_custom.md").write_text(
                "CUSTOM {input}", encoding="utf-8"
            )
            engine = TranslationEngine(
                FakeTranslationClient(),
                prompt_dir,
                "系统提示",
                options=translation_options(),
                task_template="translate_custom",
            )
        self.assertEqual(engine._template, "CUSTOM {input}")

    def test_translate_prompt_keeps_context_before_source(self) -> None:
        client = FakeTranslationClient()
        engine = self.make_engine(client)
        engine.translate("alpha")

        prompt = client.calls[0]["messages"][-1]["content"]
        context_marker = "【参考前文背景"
        source_marker = "【当前待翻译内容】"
        self.assertIn(context_marker, prompt)
        self.assertIn(source_marker, prompt)
        self.assertLess(
            prompt.index(context_marker), prompt.index(source_marker)
        )
        self.assertTrue(prompt.rstrip().endswith("alpha"))

    def test_translation_engine_reset_history(self) -> None:
        client = FakeTranslationClient()
        engine = self.make_engine(client, context_window=2)
        engine.translate("alpha")
        engine.reset_history()
        engine.translate("beta")

        latest = client.calls[-1]["messages"][-1]["content"]
        self.assertIn("无前文，故事开端", latest)
        self.assertNotIn("alpha", latest)

    def test_translate_filename_uses_model_and_sanitizes(self) -> None:
        client = FilenameTranslationClient(content="文件名：时钟.md")
        engine = self.make_engine(client)

        self.assertEqual(engine.translate_filename("The Clock.md"), "时钟")
        prompt = client.calls[-1]["messages"][-1]["content"]
        self.assertIn("The Clock", prompt)
        self.assertGreater(client.calls[-1]["max_tokens"], 0)
        system = client.calls[-1]["messages"][0]["content"]
        self.assertIn("Hello_World", system)
        self.assertIn("Chapter_01", system)
        self.assertIn("_临时", system)
        self.assertNotIn("宁可保留原文", system)

    def test_translate_filename_translates_underscored_words(self) -> None:
        engine = self.make_engine(
            FilenameTranslationClient(content="文件名：你好_世界")
        )
        result = engine.translate_filename_result("Hello_World.md")
        self.assertEqual(result.name, "你好_世界")
        self.assertIsNone(result.fallback_reason)

    def test_translate_filename_translates_chapter_and_year(self) -> None:
        cases = (
            ("Chapter_01.md", "文件名：第01章", "第01章"),
            ("photos_2024.zip", "文件名：照片_2024", "照片_2024"),
        )
        for filename, response, expected in cases:
            with self.subTest(filename=filename):
                engine = self.make_engine(
                    FilenameTranslationClient(content=response)
                )
                result = engine.translate_filename_result(filename)
                self.assertEqual(result.name, expected)
                self.assertIsNone(result.fallback_reason)

    def test_translate_filename_preserves_marker_without_stalling(self) -> None:
        engine = self.make_engine(
            FilenameTranslationClient(content="文件名：报告[[K0]]")
        )
        result = engine.translate_filename_result("report_临时.md")
        self.assertEqual(result.name, "报告_临时")
        self.assertIsNone(result.fallback_reason)

    def test_translate_filename_all_protected_skips_model(self) -> None:
        for filename, expected in (
            ("NYC_2024.mp4", "NYC_2024"),
            ("RJ401457_临时.mp4", "RJ401457_临时"),
            ("AI简报_v2.md", "AI简报_v2"),
        ):
            with self.subTest(filename=filename):
                client = FilenameTranslationClient(content="不应调用")
                engine = self.make_engine(client)
                result = engine.translate_filename_result(filename)
                self.assertEqual(result.name, expected)
                self.assertIsNone(result.fallback_reason)
                self.assertEqual(client.calls, [])

    def test_translate_filename_restores_decorative_symbols(self) -> None:
        client = FilenameTranslationClient(
            content="文件名：[[K0]] [[K1]]序幕[[K2]][[K3]][[K4]]"
        )
        engine = self.make_engine(client)
        result = engine.translate_filename_result("① (プロローグ)…!.wav")
        self.assertEqual(result.name, "① (序幕)…!")
        self.assertIsNone(result.fallback_reason)

    def test_translate_filename_restores_trailing_symbols(self) -> None:
        client = FilenameTranslationClient(
            content="文件名：03_用游泳锻炼的私处榨出精液[[K0]][[K1]]"
        )
        engine = self.make_engine(client)
        result = engine.translate_filename_result(
            "03_水泳で鍛えたおまんこでザーメン搾り♪..wav"
        )
        self.assertEqual(result.name, "03_用游泳锻炼的私处榨出精液♪.")
        self.assertIsNone(result.fallback_reason)

    def test_translate_filename_pure_chinese_and_technical_skip_model(self) -> None:
        for name in ("情侣日常", "私密掏耳", "大購入特典", "wav"):
            with self.subTest(name=name):
                client = FilenameTranslationClient(content="不应调用")
                engine = self.make_engine(client)
                result = engine.translate_filename_result(name + ".wav")
                self.assertEqual(result.name, name)
                self.assertIsNone(result.fallback_reason)
                self.assertEqual(client.calls, [])

    def test_translate_filename_retries_when_japanese_is_untranslated(self) -> None:
        masked = "[[K0]]最初は優しく感度上げ[[K1]]はむはむふぅふぅ"
        client = FilenameTranslationClient(
            responses=[
                (masked, "stop"),
                ("[[K0]]最初是温柔提升敏感度[[K1]]是吧唧吧唧呼呼声", "stop"),
            ]
        )
        engine = self.make_engine(client)
        result = engine.translate_filename_result(
            "①最初は優しく感度上げ♪はむはむふぅふぅ.wav"
        )
        self.assertEqual(result.name, "①最初是温柔提升敏感度♪吧唧吧唧呼呼声")
        self.assertIsNone(result.fallback_reason)
        self.assertEqual(len(client.calls), 2)

    def test_translate_filename_overlong_result_falls_back(self) -> None:
        client = FilenameTranslationClient(content="长" * 120)
        engine = self.make_engine(client)
        result = engine.translate_filename_result("The Clock.md")
        self.assertEqual(result.name, "The Clock")
        self.assertIn("过长", result.fallback_reason or "")

    def test_translate_filename_retries_when_code_is_lost(self) -> None:
        client = FilenameTranslationClient(
            responses=[
                ("文件名：纽约市_笔记", "stop"),
                ("文件名：[[K0]]_笔记", "stop"),
            ]
        )
        engine = self.make_engine(client)
        result = engine.translate_filename_result("NYC_notes.md")
        self.assertEqual(result.name, "NYC_笔记")
        self.assertIsNone(result.fallback_reason)
        self.assertEqual(len(client.calls), 2)

    def test_translate_filename_falls_back_after_retry(self) -> None:
        client = FilenameTranslationClient(
            responses=[
                ("文件名：纽约市_笔记", "stop"),
                ("文件名：纽约市_笔记", "stop"),
            ]
        )
        engine = self.make_engine(client)
        result = engine.translate_filename_result("NYC_notes.md")
        self.assertEqual(result.name, "NYC_notes")
        self.assertIn("占位符", result.fallback_reason or "")
        self.assertEqual(len(client.calls), 2)

    def test_translate_filename_falls_back_on_truncation(self) -> None:
        client = FilenameTranslationClient(finish="length")
        engine = self.make_engine(client)
        result = engine.translate_filename_result("The Clock.md")
        self.assertEqual(result.name, "The Clock")
        self.assertIn("截断", result.fallback_reason or "")
        self.assertEqual(len(client.calls), 2)

    def test_translate_filename_falls_back_on_empty_result(self) -> None:
        client = FilenameTranslationClient(content="...")
        engine = self.make_engine(client)
        result = engine.translate_filename_result("The Clock.md")
        self.assertEqual(result.name, "The Clock")
        self.assertIn("结果为空", result.fallback_reason or "")
        self.assertEqual(len(client.calls), 2)

    def test_context_sentinel_and_fifo(self) -> None:
        client = FakeTranslationClient()
        stats = []
        engine = TranslationEngine(
            client,
            self.prompt_dir,
            "系统提示",
            options=translation_options(context_window=2),
            on_stat=stats.append,
        )
        engine.translate("alpha")
        engine.translate("beta")
        engine.translate("gamma")
        engine.translate("delta")

        first_prompt = client.calls[0]["messages"][-1]["content"]
        self.assertIn("无前文，故事开端", first_prompt)
        second_prompt = client.calls[1]["messages"][-1]["content"]
        self.assertIn("alpha", second_prompt)
        third_prompt = client.calls[2]["messages"][-1]["content"]
        self.assertIn("alpha", third_prompt)
        fourth_prompt = client.calls[3]["messages"][-1]["content"]
        self.assertNotIn("alpha", fourth_prompt)
        self.assertIn("beta", fourth_prompt)
        self.assertEqual(len(stats), 4)
        self.assertTrue(all(item.tokens == 3 for item in stats))
        self.assertTrue(all(item.tps == 10.0 for item in stats))

    def test_truncated_output_is_split_and_retried(self) -> None:
        client = FakeTranslationClient(truncate_over=5)
        engine = self.make_engine(client, max_split_depth=3)
        translated = engine.translate("abcdefghij")
        self.assertEqual(translated, "译2译3")
        self.assertGreaterEqual(len(client.calls), 3)

    def test_budget_drops_old_context(self) -> None:
        client = FakeTranslationClient()
        engine = self.make_engine(
            client,
            n_ctx=2000,
            context_window=5,
            min_output_tokens=128,
            prompt_safety_tokens=32,
        )
        # 先塞入足够长的历史，再请求一次，必须从最老端丢弃而不是直接报错。
        for index in range(4):
            engine.translate(f"MARK{index}-" + "历史" * 200)
        try:
            engine.translate("当前" * 60)
        except TranslateError as exc:
            self.fail(f"预算门禁不应在这里失败：{exc}")
        last_prompt = client.calls[-1]["messages"][-1]["content"]
        self.assertNotIn("MARK0-", last_prompt)

    def test_sanitize_quotes_and_fences(self) -> None:
        self.assertEqual(sanitize_translation_output('"你好"', "hello"), "你好")
        self.assertEqual(sanitize_translation_output("“你好”", "hello"), "你好")
        self.assertEqual(
            sanitize_translation_output('```text\n“你好”\n```', "hello"),
            "你好",
        )
        self.assertEqual(
            sanitize_translation_output("“早上好”", "「おはよう」"),
            "“早上好”",
        )
        self.assertEqual(
            sanitize_translation_output("“未闭合", "hello"),
            "“未闭合",
        )

    def test_sanitize_removes_period_after_ellipsis(self) -> None:
        self.assertEqual(sanitize_translation_output("你好……。", "hello"), "你好……")
        self.assertEqual(sanitize_translation_output("你好——。", "hello"), "你好——")

    def test_sanitize_filename_stem_preserves_quotes_and_strips_prefix(self) -> None:
        self.assertEqual(sanitize_filename_stem("文件名：时钟.md", ".md"), "时钟")
        self.assertEqual(sanitize_filename_stem("“时钟”", ".md"), "“时钟”")
        self.assertEqual(sanitize_filename_stem("《雨の駅》.srt", ".srt"), "《雨の駅》")

    def test_sanitize_filename_stem_replaces_invalid_characters(self) -> None:
        self.assertEqual(
            sanitize_filename_stem("雨/车站？", ".md"),
            "雨_车站？",
        )
        self.assertEqual(sanitize_filename_stem("...", ".md"), "")

    def test_sanitize_filename_stem_preserves_markers(self) -> None:
        self.assertEqual(sanitize_filename_stem("_临时", ".md"), "_临时")
        self.assertEqual(sanitize_filename_stem("【临时】", ".md"), "【临时】")

    def test_sanitize_filename_stem_takes_explanation_payload(self) -> None:
        self.assertEqual(
            sanitize_filename_stem("Hello_World翻译：你好_世界", ".md"),
            "你好_世界",
        )

    def test_sanitize_filename_stem_leaves_length_to_restore_step(self) -> None:
        cleaned = sanitize_filename_stem("长" * 120, ".md")
        self.assertEqual(len(cleaned), 120)

    def test_synthetic_test_ja_quotes_are_preserved(self) -> None:
        source_blocks = [
            block.strip()
            for block in (ROOT / "tests" / "fixtures" / "test_ja.md")
            .read_text(encoding="utf-8")
            .split("\n\n")
            if block.strip()
        ]
        translated_blocks = [
            block.strip()
            for block in (ROOT / "tests" / "fixtures" / "test_ja.zh.md")
            .read_text(encoding="utf-8")
            .split("\n\n")
            if block.strip()
        ]
        self.assertEqual(len(source_blocks), len(translated_blocks))
        for source, translated in zip(source_blocks, translated_blocks):
            self.assertEqual(
                sanitize_translation_output(translated, source), translated
            )
            self.assertEqual(translated.count("“"), translated.count("”"))


if __name__ == "__main__":
    unittest.main()
