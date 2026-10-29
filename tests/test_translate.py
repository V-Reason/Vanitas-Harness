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
from vanitas.cue_boundary import load_joins  # noqa: E402
from vanitas.errors import EngineError, TranslateError  # noqa: E402
from vanitas.translate import (  # noqa: E402
    DynamicEntityEngine,
    NameTarget,
    TranslationEngine,
    TranslationOptions,
    apply_name_renames,
    chunk_text,
    collect_name_targets,
    collect_translate_inputs,
    parse_srt,
    parse_window_response,
    recover_window_response,
    plan_name_renames,
    plan_translation_outputs,
    postprocess_cue_text,
    rebuild_srt,
    sanitize_filename_stem,
    sanitize_translation_output,
    strip_window_id_prefix,
    translate_file,
    translate_srt,
    validate_cue_translation,
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

    def test_recover_window_response_for_reset_ids(self) -> None:
        ok, aligned, missing = recover_window_response(
            "[#1] 第七句\n[#2] 第八句", ["7", "8"]
        )
        self.assertTrue(ok)
        self.assertEqual(aligned, {"7": "第七句", "8": "第八句"})
        self.assertEqual(missing, [])

    def test_recover_window_response_rejects_reorder_and_count(self) -> None:
        ok, aligned, _missing = recover_window_response(
            "[#2] 第八句\n[#1] 第七句", ["7", "8"]
        )
        self.assertFalse(ok)
        self.assertEqual(aligned, {})
        ok, aligned, _missing = recover_window_response(
            "[#1] 第七句", ["7", "8"]
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

    def test_short_window_translation_falls_back_to_single_cues(self) -> None:
        text = (
            "1\n00:00:01,000 --> 00:00:02,000\n"
            "体におかけしているタオルがちょっとでも動いたら、本当に許しませんからね\n"
            "\n"
            "2\n00:00:03,000 --> 00:00:04,000\n"
            "許しませんからね\n"
        )
        source = self.write("short-window.srt", text)
        batch_calls = 0
        single_calls = 0

        def chat(prompt: str) -> str:
            nonlocal batch_calls, single_calls
            if "\n" in prompt:
                batch_calls += 1
                # 批量把长句压成拟声词；必须整窗回退，不能静默接受。
                return "[#1] 哒——哒——\n[#2] 好"
            single_calls += 1
            if "タオル" in prompt:
                return "要是搭在身上的毛巾有半点动的话"
            return "我可不会放过你的哦"

        outcome = translate_file(chat, source, SETTINGS, window_size=2)
        self.assertEqual(batch_calls, 1)
        self.assertEqual(single_calls, 2)
        self.assertEqual(outcome.failures, [])
        blocks = parse_srt(outcome.output.read_text(encoding="utf-8"), "\n")
        self.assertEqual(len(blocks), 2)
        self.assertIn("毛巾", blocks[0][0].text)
        self.assertIn("放过", blocks[1][0].text)

    def test_translate_srt_recovers_reset_window_ids(self) -> None:
        source = self.write("reset-ids.srt", self._windowed_srt())
        calls: list[str] = []

        def chat(prompt: str) -> str:
            calls.append(prompt)
            output: list[str] = []
            for index, line in enumerate(prompt.splitlines(), start=1):
                _tag, text = line.split(" ", 1)
                output.append(f"[#{index}] 译{text}")
            return "\n".join(output)

        outcome = translate_file(
            chat,
            source,
            SETTINGS,
            window_size=2,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(outcome.failures, [])
        translated = outcome.output.read_text(encoding="utf-8")
        blocks = parse_srt(translated, "\n")
        self.assertEqual(
            [block.text for block, _ in blocks],
            ["译一", "译二", "译三", "译四"],
        )
        self.assertNotIn("[#", translated)

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

    def test_translate_srt_rejects_multiline_single_fallback(self) -> None:
        source = self.write("multiline-fallback.srt", self._windowed_srt())

        def chat(prompt: str) -> str:
            if "\n" in prompt:
                return "[#1] 不完整"
            return "第一行\n第二行\n第三行"

        outcome = translate_file(chat, source, SETTINGS, window_size=2)
        self.assertEqual(len(outcome.failures), 4)
        translated = outcome.output.read_text(encoding="utf-8")
        self.assertEqual(
            [block.text for block, _ in parse_srt(translated, "\n")],
            ["一", "二", "三", "四"],
        )

    def test_translate_srt_fallback_strips_window_id_prefix(self) -> None:
        source = self.write("fallback-ids.srt", self._windowed_srt())

        def chat(prompt: str) -> str:
            if "\n" in prompt:
                return "[#1] 不完整"
            return f"[#1] 译{prompt}"

        outcome = translate_file(
            chat,
            source,
            SETTINGS,
            window_size=2,
        )
        self.assertEqual(outcome.failures, [])
        translated = outcome.output.read_text(encoding="utf-8")
        blocks = parse_srt(translated, "\n")
        self.assertEqual(
            [block.text for block, _ in blocks],
            ["译一", "译二", "译三", "译四"],
        )
        self.assertNotIn("[#", translated)

    def test_strip_window_id_prefix_removes_inline_ids(self) -> None:
        self.assertEqual(
            strip_window_id_prefix("译文 [#12] 内容 [#13]"),
            "译文内容",
        )
        self.assertEqual(strip_window_id_prefix("[#1] 开头"), "开头")

    def test_translate_srt_rejects_inline_window_id_and_falls_back(self) -> None:
        source = self.write("inline-id.srt", self._windowed_srt())
        batch_calls = 0
        single_calls = 0

        def chat(prompt: str) -> str:
            nonlocal batch_calls, single_calls
            if "\n" in prompt:
                batch_calls += 1
                return "[#1] 译一\n[#2] 译二 [#1]"
            single_calls += 1
            return f"译{prompt}"

        outcome = translate_file(chat, source, SETTINGS, window_size=2)
        self.assertEqual(outcome.failures, [])
        self.assertEqual(batch_calls, 2)
        self.assertEqual(single_calls, 4)
        translated = outcome.output.read_text(encoding="utf-8")
        self.assertNotIn("[#", translated)
        self.assertEqual(
            [block.text for block, _ in parse_srt(translated, "\n")],
            ["译一", "译二", "译三", "译四"],
        )

    def test_translate_srt_rejects_duplicate_translation_for_distinct_sources(self) -> None:
        source = self.write(
            "duplicate.srt",
            "1\n00:00:01,000 --> 00:00:02,000\n"
            "今日はとても楽しかったです\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n"
            "明日の予定を確認します\n",
        )
        batch_calls = 0
        single_calls = 0

        def chat(prompt: str) -> str:
            nonlocal batch_calls, single_calls
            if "\n" in prompt:
                batch_calls += 1
                return "[#1] 这是完全相同的一条长译文\n[#2] 这是完全相同的一条长译文"
            single_calls += 1
            return "译今日" if "今日" in prompt else "译明日"

        outcome = translate_file(chat, source, SETTINGS, window_size=2)
        self.assertEqual(outcome.failures, [])
        self.assertEqual(batch_calls, 1)
        self.assertEqual(single_calls, 2)
        translated = outcome.output.read_text(encoding="utf-8")
        self.assertIn("译今日", translated)
        self.assertIn("译明日", translated)

    def test_translate_srt_rejects_partial_translation_overlap(self) -> None:
        source = self.write(
            "overlap.srt",
            "1\n00:00:01,000 --> 00:00:02,000\n今回初めて割と音だけ\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nを楽しむ部分が\n",
        )
        batch_calls = 0
        single_calls = 0

        def chat(prompt: str) -> str:
            nonlocal batch_calls, single_calls
            if "\n" in prompt:
                batch_calls += 1
                return (
                    "[#1] 这次是第一次，单纯享受声音的部分可真多呢。\n"
                    "[#2] 享受声音的部分可真多呢。"
                )
            single_calls += 1
            return "译单句"

        outcome = translate_file(chat, source, SETTINGS, window_size=2)
        self.assertEqual(outcome.failures, [])
        self.assertEqual(batch_calls, 1)
        self.assertEqual(single_calls, 2)

    def test_batch_history_is_recorded_per_cue(self) -> None:
        source = "\n\n".join(
            f"{index}\n00:00:01,000 --> 00:00:02,000\n一"
            for index in range(1, 13)
        )
        first = "\n".join(f"[#{index}] 译{index}" for index in range(1, 7))
        second = "\n".join(f"[#{index}] 译{index}" for index in range(7, 13))
        client = ScriptedTranslationClient([first, second])
        engine = TranslationEngine(
            client,
            ROOT / "config" / "prompts",
            "",
            options=translation_options(context_window=12),
            task_template="translate_asmr",
        )
        failures: list = []
        translate_srt(
            engine,
            source,
            chunk_chars=800,
            source="history.srt",
            eol="\n",
            failures=failures,
            window_size=6,
            max_window_chars=1000,
        )
        self.assertEqual(failures, [])
        self.assertEqual(len(engine.history_snapshot()), 12)
        self.assertIn("译6", client.calls[1])

    def test_cross_window_copy_falls_back(self) -> None:
        source = (
            "1\n00:00:01,000 --> 00:00:02,000\n"
            "今日はとても楽しかったです\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n"
            "明日の予定を確認します\n\n"
            "3\n00:00:05,000 --> 00:00:06,000\n"
            "これは別の長い台詞です\n"
        )
        client = ScriptedTranslationClient(
            [
                "[#1] 今天真的非常开心，发生了很多事。\n[#2] 明天确认计划。",
                "[#3] 今天真的非常开心，发生了很多事。",
                "[#3] 这是另一段很长的台词。",
            ]
        )
        engine = TranslationEngine(
            client,
            ROOT / "config" / "prompts",
            "",
            options=translation_options(context_window=12),
            task_template="translate_asmr",
        )
        failures: list = []
        output = translate_srt(
            engine,
            source,
            chunk_chars=800,
            source="copy.srt",
            eol="\n",
            failures=failures,
            window_size=2,
            max_window_chars=1000,
        )
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(failures, [])
        self.assertIn("这是另一段很长的台词。", output)
        self.assertEqual(output.count("今天真的非常开心，发生了很多事。"), 1)

    def test_batch_validation_rejects_sexpar_and_falls_back(self) -> None:
        source = (
            "1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nさようなら\n"
        )
        client = ScriptedTranslationClient(
            [
                "[#1] 做SEXPAR啦\n[#2] 正常",
                "[#1] 做爱啦",
                "[#2] 正常",
            ]
        )
        engine = TranslationEngine(
            client,
            ROOT / "config" / "prompts",
            "",
            options=translation_options(),
            task_template="translate_asmr",
        )
        failures: list = []
        output = translate_srt(
            engine,
            source,
            chunk_chars=800,
            source="sexpar.srt",
            eol="\n",
            failures=failures,
            window_size=2,
            max_window_chars=1000,
        )
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(failures, [])
        self.assertIn("做爱啦", output)
        self.assertNotIn("SEXPAR", output)

    def test_fallback_retries_after_validation(self) -> None:
        source = "1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n"
        client = ScriptedTranslationClient(
            ["[#1] 做SEXPAR啦", "[#1] 做SEXPAR啦", "[#1] 做爱啦"]
        )
        engine = TranslationEngine(
            client,
            ROOT / "config" / "prompts",
            "",
            options=translation_options(),
            task_template="translate_asmr",
        )
        failures: list = []
        output = translate_srt(
            engine,
            source,
            chunk_chars=800,
            source="retry.srt",
            eol="\n",
            failures=failures,
            window_size=2,
            max_window_chars=1000,
        )
        self.assertEqual(failures, [])
        self.assertEqual(len(client.calls), 3)
        self.assertIn("做爱啦", output)

    def test_fallback_validation_keeps_source_and_records_failure(self) -> None:
        source = "1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n"
        client = ScriptedTranslationClient(
            ["[#1] 做SEXPAR啦", "[#1] 做SEXPAR啦"]
        )
        engine = TranslationEngine(
            client,
            ROOT / "config" / "prompts",
            "",
            options=translation_options(),
            task_template="translate_asmr",
        )
        failures: list = []
        output = translate_srt(
            engine,
            source,
            chunk_chars=800,
            source="bad.srt",
            eol="\n",
            failures=failures,
            window_size=1,
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("臆造", failures[0].reason)
        self.assertIn("こんにちは", output)
        self.assertNotIn("SEXPAR", output)

    def test_validate_cue_translation_rejects_kana_sexpar_and_slash(self) -> None:
        self.assertIn("假名", validate_cue_translation("まだ仮名") or "")
        self.assertIn("臆造", validate_cue_translation("做SEXPAR啦") or "")
        self.assertIn("臆造", validate_cue_translation("用Track的话") or "")
        self.assertIn("斜杠", validate_cue_translation("先生/女士") or "")
        self.assertIsNone(validate_cue_translation("ASMR和NG，1/2 也可以"))

    def test_translate_srt_accepts_repeated_short_source_lines(self) -> None:
        source = self.write(
            "repeat.srt",
            "1\n00:00:01,000 --> 00:00:02,000\n好き\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n好き\n",
        )
        batch_calls = 0

        def chat(prompt: str) -> str:
            nonlocal batch_calls
            if "\n" in prompt:
                batch_calls += 1
                return "[#1] 喜欢你\n[#2] 喜欢你"
            return "喜欢你"

        outcome = translate_file(chat, source, SETTINGS, window_size=2)
        self.assertEqual(outcome.failures, [])
        self.assertEqual(batch_calls, 1)
        translated = outcome.output.read_text(encoding="utf-8")
        self.assertEqual(
            [block.text for block, _ in parse_srt(translated, "\n")],
            ["喜欢你", "喜欢你"],
        )

    def test_translate_srt_accepts_similar_source_lines_with_same_translation(self) -> None:
        source = self.write(
            "similar.srt",
            "1\n00:00:01,000 --> 00:00:02,000\nもうやれやれみたいな感じで\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nやれやれみたいな感じが\n",
        )
        batch_calls = 0

        def chat(prompt: str) -> str:
            nonlocal batch_calls
            if "\n" in prompt:
                batch_calls += 1
                return "[#1] 好像有种“真是的”的感觉。\n[#2] 好像有种“真是的”的感觉。"
            return "好像有种“真是的”的感觉。"

        outcome = translate_file(chat, source, SETTINGS, window_size=2)
        self.assertEqual(outcome.failures, [])
        self.assertEqual(batch_calls, 1)

    def test_translate_srt_rejects_empty_translation_for_nonempty_source(self) -> None:
        source = self.write("empty-window.srt", self._windowed_srt())
        batch_calls = 0
        single_calls = 0

        def chat(prompt: str) -> str:
            nonlocal batch_calls, single_calls
            if "\n" in prompt:
                batch_calls += 1
                return "[#1] \n[#2] 译二"
            single_calls += 1
            return f"译{prompt}"

        outcome = translate_file(chat, source, SETTINGS, window_size=2)
        self.assertEqual(outcome.failures, [])
        self.assertEqual(batch_calls, 2)
        self.assertEqual(single_calls, 4)
        translated = outcome.output.read_text(encoding="utf-8")
        self.assertEqual(
            [block.text for block, _ in parse_srt(translated, "\n")],
            ["译一", "译二", "译三", "译四"],
        )

    def test_batch_window_preserves_ids_through_engine(self) -> None:
        client = ScriptedTranslationClient(["[#1] 译一\n[#2] 译二"])
        engine = TranslationEngine(
            client,
            ROOT / "config" / "prompts",
            "",
            options=translation_options(),
            task_template="translate_asmr",
        )
        failures = []
        output = translate_srt(
            engine,
            "1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nさようなら\n",
            chunk_chars=800,
            source="batch.srt",
            eol="\n",
            failures=failures,
            window_size=2,
        )
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(
            [block.text for block, _ in parse_srt(output, "\n")],
            ["译一", "译二"],
        )
        self.assertEqual(failures, [])

    def test_single_fallback_rejects_multi_id_after_engine(self) -> None:
        client = ScriptedTranslationClient(
            [
                "[#1] 译一",
                "[#1] 译一\n[#2] 译二",
                "[#1] 译一\n[#2] 译二",
            ]
        )
        engine = TranslationEngine(
            client,
            ROOT / "config" / "prompts",
            "",
            options=translation_options(),
            task_template="translate_asmr",
        )
        failures = []
        output = translate_srt(
            engine,
            "1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nさようなら\n",
            chunk_chars=800,
            source="fallback.srt",
            eol="\n",
            failures=failures,
            window_size=2,
        )
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(len(failures), 2)
        self.assertTrue(all("续写" in item.reason for item in failures))
        self.assertEqual(
            [block.text for block, _ in parse_srt(output, "\n")],
            ["こんにちは", "さようなら"],
        )

    def test_batch_postprocess_cleans_traditional_and_staging(self) -> None:
        client = ScriptedTranslationClient(
            ["[#1] ，妳的話我會聽\n[#2] （在耳后）喜欢你哦"]
        )
        engine = TranslationEngine(
            client,
            ROOT / "config" / "prompts",
            "",
            options=translation_options(),
            task_template="translate_asmr",
        )
        failures = []
        output = translate_srt(
            engine,
            "1\n00:00:01,000 --> 00:00:02,000\n好きだよ\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n大好きだよ\n",
            chunk_chars=800,
            source="postprocess.srt",
            eol="\n",
            failures=failures,
            window_size=2,
        )
        self.assertEqual(
            [block.text for block, _ in parse_srt(output, "\n")],
            ["你的话我会听", "喜欢你哦"],
        )
        self.assertEqual(failures, [])

    def test_single_cue_postprocess_cleans_traditional(self) -> None:
        client = ScriptedTranslationClient(["，妳的話我會聽"])
        engine = TranslationEngine(
            client,
            ROOT / "config" / "prompts",
            "",
            options=translation_options(),
            task_template="translate_asmr",
        )
        failures = []
        output = translate_srt(
            engine,
            "1\n00:00:01,000 --> 00:00:02,000\n大好きだよ\n",
            chunk_chars=800,
            source="single.srt",
            eol="\n",
            failures=failures,
            window_size=1,
        )
        self.assertEqual(
            [block.text for block, _ in parse_srt(output, "\n")],
            ["你的话我会听"],
        )
        self.assertEqual(failures, [])

    def test_entity_context_injected_into_second_window(self) -> None:
        client = ScriptedTranslationClient(
            [
                "[#1] 诺亚，早上好\n[#2] 今天天气真好",
                "[#3] 诺亚，再见\n[#4] 晚安",
            ]
        )
        engine = TranslationEngine(
            client,
            ROOT / "config" / "prompts",
            "",
            options=translation_options(),
            task_template="translate_asmr",
        )
        failures = []
        output = translate_srt(
            engine,
            "1\n00:00:01,000 --> 00:00:02,000\nノアちゃん、おはよう\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n今日はいい天気だね\n\n"
            "3\n00:00:05,000 --> 00:00:06,000\nノアチップ、またね\n\n"
            "4\n00:00:07,000 --> 00:00:08,000\nおやすみ\n",
            chunk_chars=800,
            source="entity.srt",
            eol="\n",
            failures=failures,
            window_size=2,
        )
        self.assertEqual(len(client.calls), 2)
        self.assertIn("【本文实体", client.calls[1])
        self.assertIn("已锁定译名：诺亚", client.calls[1])
        self.assertEqual(
            [block.text for block, _ in parse_srt(output, "\n")],
            ["诺亚，早上好", "今天天气真好", "诺亚，再见", "晚安"],
        )
        self.assertEqual(failures, [])

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

    def test_translate_file_moves_split_word_across_cue_boundary(self) -> None:
        joins = load_joins(ROOT / "config" / "asr_joins.tsv")
        self.assertTrue(joins)
        text = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "いらっしゃいましたらカ\n"
            "\n"
            "2\n"
            "00:00:02,001 --> 00:00:03,500\n"
            "ウンターへお越しください。\n"
        )
        source = self.write("counter.srt", text)
        calls: list[str] = []

        def mock_chat(prompt: str) -> str:
            calls.append(prompt)
            if "ウンター" in prompt:
                return "请到柜台来。"
            return "译…"

        baseline_calls: list[str] = []

        def baseline_chat(prompt: str) -> str:
            baseline_calls.append(prompt)
            if "ウンター" in prompt:
                return "请到柜台来。"
            return "译…"

        # 不开合并规则时同一 fixture 的请求次数：窗口批调用 1 次 + 逐句回退 2 次。
        translate_file(baseline_chat, source, SETTINGS, window_size=2)
        self.assertEqual(len(baseline_calls), 3)

        outcome = translate_file(
            mock_chat,
            source,
            SETTINGS,
            join_rules=joins,
            window_size=2,
            out_dir=self.dir / "joined",
        )
        self.assertEqual(outcome.failures, [])
        # 右吸附只改正文，不增加模型请求。
        self.assertEqual(len(calls), len(baseline_calls))

        translated = outcome.output.read_text(encoding="utf-8")
        blocks = parse_srt(translated, "\n")
        source_blocks = parse_srt(text, "\n")
        self.assertEqual(len(blocks), len(source_blocks))
        self.assertEqual(
            [block.index for block, _ in blocks],
            [block.index for block, _ in source_blocks],
        )
        self.assertEqual(
            [block.timing for block, _ in blocks],
            [block.timing for block, _ in source_blocks],
        )
        self.assertIn("柜台", blocks[1][0].text)
        # 合并后的右 cue 应把“カ”带回来，提示词里不再是半截词。
        self.assertTrue(
            any("カウンターへお越しください。" in prompt for prompt in calls)
        )
        self.assertFalse(any("[#2] ウンター" in prompt for prompt in calls))
        self.assertFalse(any("ましたらカ" in prompt for prompt in calls))

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


class ScriptedTranslationClient:
    """按顺序返回脚本内容，用于验证批次编号与回退边界。"""

    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls: list[str] = []

    def chat(self, messages, *, max_tokens, temperature=0.2):
        self.calls.append(str(messages[-1]["content"]))
        content = self.contents.pop(0) if self.contents else ""
        return {
            "choices": [
                {"message": {"content": content}, "finish_reason": "stop"}
            ],
            "usage": {"completion_tokens": 1},
            "timings": {"predicted_n": 1},
        }


class FailingTranslationClient:
    """按序返回脚本内容，并在指定调用序抛 EngineError（模拟审计重译失败）。"""

    def __init__(self, contents: list[str], *, fail_at: set[int]) -> None:
        self.contents = list(contents)
        self.fail_at = set(fail_at)
        self.calls: list[str] = []

    def chat(self, messages, *, max_tokens, temperature=0.2):
        self.calls.append(str(messages[-1]["content"]))
        if len(self.calls) in self.fail_at:
            raise EngineError("模拟审计重译失败")
        content = self.contents.pop(0) if self.contents else ""
        return {
            "choices": [
                {"message": {"content": content}, "finish_reason": "stop"}
            ],
            "usage": {"completion_tokens": 1},
            "timings": {"predicted_n": 1},
        }


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

    def test_sanitize_strips_leading_punctuation_but_keeps_window_ids(self) -> None:
        self.assertEqual(
            sanitize_translation_output("，[#12] 你好", "hello"),
            "[#12] 你好",
        )
        self.assertEqual(
            sanitize_translation_output("[#1] 你好\n[#2] 再见", "hello"),
            "[#1] 你好\n[#2] 再见",
        )
        self.assertEqual(
            sanitize_translation_output("，我们要不要一起看看呢？", "hello"),
            "我们要不要一起看看呢？",
        )

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


class QualityGuardTests(unittest.TestCase):
    """术语、语域/人称、文档级重复审计与定点重译。"""

    SRC_A = "今日はとても楽しかったです、本当にいい一日でした"
    SRC_B = "また何かあったら教えてくださいね"
    SRC_C = "今日はとても退屈でした、本当に長い一日でした"
    ZH_A = "今天真的非常开心呀，真是美好的一天。"
    ZH_B = "明天也请继续听下去哦。"
    ZH_COPY = "今天真的非常开心呢，真是美好的一天。"
    ZH_RETRY = "今天真的非常无聊呢，真是漫长的一天。"

    def srt_text(self, *texts: str) -> str:
        blocks = []
        for index, text in enumerate(texts, start=1):
            start = f"00:00:{index * 2 - 1:02d},000"
            end = f"00:00:{index * 2:02d},000"
            blocks.append(f"{index}\n{start} --> {end}\n{text}")
        return "\n\n".join(blocks) + "\n"

    def engine(self, client, *, context_window: int = 5) -> TranslationEngine:
        return TranslationEngine(
            client,
            ROOT / "config" / "prompts",
            "",
            options=translation_options(context_window=context_window),
            task_template="translate_asmr",
        )

    def run_translate(self, client, *texts: str, window_size: int = 2):
        engine = self.engine(client)
        failures: list = []
        output = translate_srt(
            engine,
            self.srt_text(*texts),
            chunk_chars=800,
            source="quality.srt",
            eol="\n",
            failures=failures,
            window_size=window_size,
            max_window_chars=1000,
        )
        return engine, output, failures

    def run_translate_text(self, client, text: str, *, window_size: int = 2):
        engine = self.engine(client)
        failures: list = []
        output = translate_srt(
            engine,
            text,
            chunk_chars=800,
            source="quality.srt",
            eol="\n",
            failures=failures,
            window_size=window_size,
            max_window_chars=1000,
        )
        return engine, output, failures

    def srt_blocks(self, *items: tuple[int, str]) -> str:
        blocks = []
        for index, text in items:
            start = f"00:00:{index * 2 - 1:02d},000"
            end = f"00:00:{index * 2:02d},000"
            blocks.append(f"{index}\n{start} --> {end}\n{text}")
        return "\n\n".join(blocks) + "\n"

    def block_texts(self, output: str) -> list[str]:
        return [block.text for block, _separator in parse_srt(output, "\n")]

    def test_asmr_engine_loads_glossary_and_translate_does_not(self) -> None:
        self.assertEqual(
            len(self.engine(ScriptedTranslationClient([])).glossary.entries), 3
        )
        plain = TranslationEngine(
            ScriptedTranslationClient([]),
            ROOT / "config" / "prompts",
            "",
            options=translation_options(),
        )
        self.assertEqual(plain.glossary.entries, ())

    def test_cross_window_near_duplicate_triggers_one_retranslation(self) -> None:
        client = ScriptedTranslationClient(
            [
                f"[#1] {self.ZH_A}\n[#2] {self.ZH_B}",
                f"[#3] {self.ZH_COPY}",
                f"[#3] {self.ZH_RETRY}",
            ]
        )
        _engine, output, failures = self.run_translate(
            client, self.SRC_A, self.SRC_B, self.SRC_C
        )
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(failures, [])
        self.assertIn("禁止复用", client.calls[2])
        self.assertEqual(
            self.block_texts(output), [self.ZH_A, self.ZH_B, self.ZH_RETRY]
        )
        self.assertEqual(output.count("今天真的非常开心"), 1)

    def test_clean_window_has_no_extra_calls(self) -> None:
        client = ScriptedTranslationClient(
            ["[#1] 译一\n[#2] 译二", "[#3] 译三\n[#4] 译四"]
        )
        _engine, output, failures = self.run_translate(
            client, "一", "二", "三", "四"
        )
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(failures, [])
        self.assertEqual(self.block_texts(output), ["译一", "译二", "译三", "译四"])

    def test_glossary_replaces_forbidden_translation(self) -> None:
        client = ScriptedTranslationClient(["[#1] 业余水平的耳边絮语"])
        _engine, output, failures = self.run_translate(
            client, "アマアマな声でささやく", window_size=2
        )
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(failures, [])
        self.assertEqual(self.block_texts(output), ["甜腻的耳边絮语"])

    def test_glossary_preferred_synonym_does_not_retranslate(self) -> None:
        client = ScriptedTranslationClient(["[#1] 甜腻的耳边絮语"])
        _engine, output, failures = self.run_translate(
            client, "アマアマな声でささやく", window_size=2
        )
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(failures, [])
        self.assertEqual(self.block_texts(output), ["甜腻的耳边絮语"])

    def test_intimate_register_normalizes_nin(self) -> None:
        client = ScriptedTranslationClient(
            ["[#1] 您真好呢\n[#2] 给您膝枕哦"]
        )
        _engine, output, failures = self.run_translate(
            client, "先輩、好きです", "膝枕してあげるね"
        )
        self.assertEqual(failures, [])
        self.assertEqual(self.block_texts(output), ["你真好呢", "给你膝枕哦"])

    def test_service_and_neutral_registers_keep_nin(self) -> None:
        client = ScriptedTranslationClient(
            ["[#1] 您这边请\n[#2] 您预约好了"]
        )
        _engine, output, failures = self.run_translate(
            client,
            "いらっしゃいませ",
            "お客様、ご予約ありがとうございます",
        )
        self.assertEqual(failures, [])
        self.assertEqual(self.block_texts(output), ["您这边请", "您预约好了"])

        neutral = ScriptedTranslationClient(["[#1] 您好"])
        _engine, output, failures = self.run_translate(
            neutral, "こんにちは", window_size=2
        )
        self.assertEqual(failures, [])
        self.assertEqual(self.block_texts(output), ["您好"])

    def test_asr_hallucination_does_not_shift_register(self) -> None:
        # 片尾套话里塞满服务词也不得把亲密文档判成 service；它本身也不该进模型。
        # 提示词模板自带“チャンネル登録/お客様”示例，故用独有标记验证未被发送。
        client = ScriptedTranslationClient(["[#1] 您好"])
        _engine, output, failures = self.run_translate(
            client,
            "先輩、好きです",
            "チャンネル登録 ASRHALLUCINATIONMARKER お客様 予約 施術 いらっしゃいませ ございます",
            window_size=2,
        )
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(failures, [])
        self.assertEqual(self.block_texts(output), ["你好", ""])
        self.assertNotIn("ASRHALLUCINATIONMARKER", client.calls[0])

    def test_restore_history_keeps_maxlen(self) -> None:
        engine = self.engine(
            ScriptedTranslationClient([]), context_window=2
        )
        engine.restore_history((("a", "1"), ("b", "2"), ("c", "3")))
        self.assertEqual(engine.history_snapshot(), (("b", "2"), ("c", "3")))
        self.assertEqual(engine._history.maxlen, 2)

        narrow = self.engine(
            ScriptedTranslationClient([]), context_window=0
        )
        narrow.restore_history((("a", "1"),))
        self.assertEqual(narrow.history_snapshot(), ())

    def test_audit_retry_exception_keeps_original_without_failure(self) -> None:
        client = FailingTranslationClient(
            [
                f"[#1] {self.ZH_A}\n[#2] {self.ZH_B}",
                f"[#3] {self.ZH_COPY}",
            ],
            fail_at={3},
        )
        _engine, output, failures = self.run_translate(
            client, self.SRC_A, self.SRC_B, self.SRC_C
        )
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(failures, [])
        self.assertEqual(
            self.block_texts(output), [self.ZH_A, self.ZH_B, self.ZH_COPY]
        )

    def test_fallback_success_enters_duplicate_index(self) -> None:
        client = ScriptedTranslationClient(
            [
                "[#1] 只回第一行",
                f"[#1] {self.ZH_A}",
                f"[#2] {self.ZH_B}",
                f"[#3] {self.ZH_COPY}",
                f"[#3] {self.ZH_RETRY}",
            ]
        )
        _engine, output, failures = self.run_translate(
            client, self.SRC_A, self.SRC_B, self.SRC_C
        )
        self.assertEqual(len(client.calls), 5)
        self.assertEqual(failures, [])
        self.assertEqual(
            self.block_texts(output), [self.ZH_A, self.ZH_B, self.ZH_RETRY]
        )
        self.assertEqual(output.count("今天真的非常开心"), 1)

    def test_window_size_one_applies_glossary_and_duplicate_audit(self) -> None:
        client = ScriptedTranslationClient(
            [
                "业余水平的耳边絮语",
                self.ZH_A,
                self.ZH_COPY,
                f"[#3] {self.ZH_RETRY}",
            ]
        )
        _engine, output, failures = self.run_translate(
            client,
            "アマアマな声でささやく",
            self.SRC_A,
            self.SRC_C,
            window_size=1,
        )
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(failures, [])
        self.assertEqual(
            self.block_texts(output),
            ["甜腻的耳边絮语", self.ZH_A, self.ZH_RETRY],
        )

    def test_single_retry_hint_fixes_residual_kana(self) -> None:
        client = ScriptedTranslationClient(
            [
                "不完整",          # 批次解析失败 -> 逐句 fallback
                "[#1] 雫ちゃん…",  # 首轮单句仍保留假名
                "[#1] 雫酱…",      # 带 SINGLE_RETRY_HINT 的重试
            ]
        )
        _engine, output, failures = self.run_translate(
            client, "雫ちゃん、こんにちは", window_size=2
        )
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(failures, [])
        hint = client.calls[2]
        self.assertIn("所有日文假名", hint)
        self.assertIn("人名中的", hint)
        self.assertIn("片假名名词", hint)
        self.assertEqual(self.block_texts(output), ["雫酱…"])

    def test_fallback_copy_is_audited_and_retried(self) -> None:
        client = ScriptedTranslationClient(
            [
                "[#1] 只回第一行",   # 批次缺 #2 -> 逐句 fallback
                f"[#1] {self.ZH_A}",
                f"[#2] {self.ZH_A}",  # 第二句复制第一句
                f"[#2] {self.ZH_B}",  # 带审计 hint 的重试
            ]
        )
        _engine, output, failures = self.run_translate(
            client, self.SRC_A, self.SRC_C, window_size=2
        )
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(failures, [])
        self.assertIn("禁止复用", client.calls[3])
        self.assertEqual(self.block_texts(output), [self.ZH_A, self.ZH_B])
        self.assertEqual(output.count("今天真的非常开心"), 1)


    def test_batch_retranslates_tag_question(self) -> None:
        sources = [
            f"これは学生の台詞その{number}です" for number in range(1, 17)
        ] + ["予算くださいってこと"]
        client = ScriptedTranslationClient(
            [
                "\n".join(
                    f"[#{position}] 第{position}句" for position in range(1, 7)
                ),
                "\n".join(
                    f"[#{position}] 第{position}句" for position in range(7, 13)
                ),
                "\n".join(
                    f"[#{position}] 第{position}句" for position in range(13, 17)
                )
                + "\n[#17] 对吧？",
                "[#17] 也就是说要预算",
            ]
        )
        _engine, output, failures = self.run_translate(
            client, *sources, window_size=6
        )
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(failures, [])
        self.assertIn("标签问", client.calls[3])
        self.assertNotIn("对吧", output)
        self.assertEqual(self.block_texts(output)[16], "也就是说要预算")

    def test_window_size_one_retranslates_same_tag_question(self) -> None:
        sources = [
            f"これは学生の台詞その{number}です" for number in range(1, 17)
        ] + ["予算くださいってこと"]
        contents = [f"第{number}句" for number in range(1, 17)]
        contents += ["对吧？", "[#17] 也就是说要预算"]
        client = ScriptedTranslationClient(contents)
        _engine, output, failures = self.run_translate(
            client, *sources, window_size=1
        )
        self.assertEqual(len(client.calls), 18)
        self.assertEqual(failures, [])
        self.assertNotIn("对吧", output)
        self.assertEqual(self.block_texts(output)[16], "也就是说要预算")

    def customer_direction_srt(self) -> str:
        return self.srt_blocks(
            (14, "いらっしゃいませ。"),
            (15, "ご予約ありがとうございます。"),
            (16, "お客様から私たち触れることは禁止となっております。"),
        )

    def test_customer_direction_retry_accepts_short_candidate(self) -> None:
        client = ScriptedTranslationClient(
            [
                "[#14] 欢迎光临\n"
                "[#15] 感谢您的预约\n"
                "[#16] 我们不可以触碰客人。",
                "[#16] 禁止客人触碰我们",
            ]
        )
        _engine, output, failures = self.run_translate_text(
            client, self.customer_direction_srt(), window_size=6
        )
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(failures, [])
        self.assertEqual(
            self.block_texts(output),
            ["欢迎光临", "感谢您的预约", "禁止客人触碰我们"],
        )

    def test_correct_customer_direction_has_no_extra_call(self) -> None:
        client = ScriptedTranslationClient(
            [
                "[#14] 欢迎光临\n"
                "[#15] 感谢您的预约\n"
                "[#16] 禁止客人触碰我们",
            ]
        )
        _engine, output, failures = self.run_translate_text(
            client, self.customer_direction_srt(), window_size=6
        )
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(failures, [])
        self.assertEqual(self.block_texts(output)[2], "禁止客人触碰我们")

    def test_candidate_fixing_tag_but_duplicating_is_rejected(self) -> None:
        client = ScriptedTranslationClient(
            [
                f"[#1] {self.ZH_A}\n[#2] {self.ZH_B}",
                "[#3] 对吧？",
                f"[#3] {self.ZH_A}",
            ]
        )
        _engine, output, failures = self.run_translate(
            client, self.SRC_A, self.SRC_B, "予算くださいってこと"
        )
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(failures, [])
        self.assertIn("标签问", client.calls[2])
        self.assertEqual(
            self.block_texts(output), [self.ZH_A, self.ZH_B, "对吧？"]
        )

    def test_customer_direction_retry_failure_keeps_original(self) -> None:
        client = FailingTranslationClient(
            [
                "[#14] 欢迎光临\n"
                "[#15] 感谢您的预约\n"
                "[#16] 我们不可以触碰客人。",
            ],
            fail_at={2},
        )
        _engine, output, failures = self.run_translate_text(
            client, self.customer_direction_srt(), window_size=6
        )
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(failures, [])
        self.assertEqual(self.block_texts(output)[2], "我们不可以触碰客人。")

    def test_ear_massage_glossary_specific_row_wins(self) -> None:
        cases = [
            ("お耳をマッサージしてあげる", "[#1] 采耳", "耳朵按摩"),
            ("マッサージ", "[#1] 采耳", "按摩"),
            ("耳かき", "[#1] 采耳", "采耳"),
        ]
        for source, content, expected in cases:
            with self.subTest(source=source):
                client = ScriptedTranslationClient([content])
                _engine, output, failures = self.run_translate(
                    client, source
                )
                self.assertEqual(len(client.calls), 1)
                self.assertEqual(failures, [])
                self.assertEqual(self.block_texts(output), [expected])

    def test_first_person_name_is_repaired_before_validate(self) -> None:
        source = "ちゃんと下の名前で呼んであげてよ"
        bad = ScriptedTranslationClient(["[#1] 请用我的名字好好叫我哦"])
        _engine, output, failures = self.run_translate(bad, source)
        self.assertEqual(len(bad.calls), 1)
        self.assertEqual(failures, [])
        self.assertEqual(
            self.block_texts(output), ["请用她的名字好好叫她哦"]
        )

        good = ScriptedTranslationClient(["[#1] 请用她的名字好好叫她哦"])
        _engine, output, failures = self.run_translate(good, source)
        self.assertEqual(len(good.calls), 1)
        self.assertEqual(failures, [])
        self.assertEqual(
            self.block_texts(output), ["请用她的名字好好叫她哦"]
        )

    def test_quote_private_repair_runs_after_first_person_rename(self) -> None:
        """e2e：引用句“叫我叫”在 finalize 末段收敛，cue/时间轴不变。"""
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "quote.srt"
            source.write_text(
                "1\n00:00:01,000 --> 00:00:02,500\n私のことは雫でいいって\n",
                encoding="utf-8",
            )

            def mock_chat(_prompt: str) -> str:
                return "说叫我叫雫就好"

            outcome = translate_file(mock_chat, source, SETTINGS, window_size=1)

            self.assertEqual(outcome.failures, [])
            translated = outcome.output.read_text(encoding="utf-8")
            blocks = parse_srt(translated, "\n")
            self.assertEqual(len(blocks), 1)
            block, _separator = blocks[0]
            self.assertEqual(block.index, "1")
            self.assertEqual(block.timing, "00:00:01,000 --> 00:00:02,500")
            self.assertEqual(block.text, "说叫我雫就好")


class DynamicEntityEngineTests(unittest.TestCase):
    def test_dynamic_entity_clustering_and_propagation(self) -> None:
        engine = DynamicEntityEngine(
            ["ノアちゃん、おはよう", "ノアチップ、またね"]
        )
        self.assertEqual(engine.canonical_map.get("ノア"), "ノア")
        self.assertEqual(engine.canonical_map.get("ノアチップ"), "ノア")
        engine.capture_translations(
            ["ノアちゃん、おはよう"], ["诺亚，早上好"]
        )
        self.assertEqual(engine.locked_translations.get("ノア"), "诺亚")
        context = engine.get_query_context("ノアチップ、またね")
        self.assertIn("已锁定译名：诺亚", context)
        self.assertIn("ノアチップ", context)
        self.assertIn("诺亚，早上好", context)

    def test_kanji_variant_prefix_clustering(self) -> None:
        engine = DynamicEntityEngine(
            ["美咲日和はどこ？", "美咲希织、大好き"]
        )
        self.assertEqual(engine.canonical_map.get("美咲日和"), "美咲")
        self.assertEqual(engine.canonical_map.get("美咲希织"), "美咲")

    def test_common_words_are_not_entities(self) -> None:
        engine = DynamicEntityEngine(
            [
                "今日はいい天気",
                "大丈夫",
                "お客さん、またね",
                "赤ちゃん、かわいい",
            ]
        )
        for term in ("今日", "大丈夫", "お客", "赤"):
            self.assertNotIn(term, engine.canonical_map)

    def test_no_entity_context_returns_empty(self) -> None:
        engine = DynamicEntityEngine(["今日はいい天気", "大丈夫"])
        self.assertEqual(engine.get_query_context("今日はいい天気"), "")

    def test_common_katakana_and_onomatopoeia_are_not_entities(self) -> None:
        engine = DynamicEntityEngine(
            [
                "トラックさん、こんにちは",
                "トラックさん、またね",
                "セックスしたい",
                "セックス?",
                "コショコショコショコショ",
                "バイバーイ",
                "バイバーイ",
            ]
        )
        for term in ("トラック", "セックス", "コショコショ", "バイバイ"):
            self.assertNotIn(term, engine.canonical_map)
            self.assertNotIn(term, engine.locked_translations)
        for text in ("トラックさん", "セックス", "コショコショ", "バイバーイ"):
            self.assertEqual(engine.get_query_context(text), "")
        engine.capture_translations(
            ["トラックさん、こんにちは", "セックスしたい"],
            ["托拉克，你好", "做SEXPAR"],
        )
        self.assertNotIn("トラック", engine.locked_translations)
        self.assertNotIn("セックス", engine.examples)

    def test_noa_is_not_merged_into_chin_and_pachinko_is_not_variant(self) -> None:
        engine = DynamicEntityEngine(
            [
                "ノアちゃん、おはよう",
                "ノアチン、またね",
                "パチンコの話",
                "チン、聞こえる",
            ]
        )
        self.assertEqual(engine.canonical_map.get("ノア"), "ノア")
        self.assertNotIn("チン", engine.canonical_map)
        variants = engine._variants.get("ノア", set())
        self.assertNotIn("パチンコ", variants)
        self.assertNotIn("チン", variants)

    def test_hiragana_alias_links_to_explicit_name(self) -> None:
        engine = DynamicEntityEngine(
            [
                "ミサキヒヨリという名のサクラです",
                "みさきひよりと申します",
                "さくら役のみさきひよりと",
            ]
        )
        self.assertEqual(engine.canonical_map.get("みさきひより"), "ミサキヒヨリ")
        engine.capture_translations(["みさきひよりと"], ["美咲日和"])
        context = engine.get_query_context("みさきひよりと")
        self.assertIn("みさきひより", context)
        self.assertIn("ミサキヒヨリ", context)

    def test_capture_skips_kana_and_artifact_examples(self) -> None:
        engine = DynamicEntityEngine(["ノアちゃん、おはよう"])
        engine.capture_translations(
            ["ノアちゃん、おはよう"],
            ["ノアちゃん、SEXPAR"],
        )
        self.assertNotIn("ノア", engine.examples)
        self.assertNotIn("ノア", engine.locked_translations)

    def test_capture_does_not_lock_non_leading_name(self) -> None:
        engine = DynamicEntityEngine(
            ["ノアちゃん、おはよう", "ノアチップ、またね"]
        )
        engine.capture_translations(
            ["今日はノアちゃんと"], ["今天和诺亚一起"]
        )
        self.assertNotIn("ノア", engine.locked_translations)
        self.assertTrue(engine.examples.get("ノア"))

    def test_capture_does_not_take_whole_line_prefix(self) -> None:
        engine = DynamicEntityEngine(
            ["ノアちゃん、おはよう", "ノアチップ、またね"]
        )
        engine.capture_translations(
            ["ノアちゃん、おはよう"], ["诺亚早上好"]
        )
        self.assertNotIn("诺亚", engine.locked_translations.values())

    def test_capture_skips_ambiguous_multi_entity_line(self) -> None:
        engine = DynamicEntityEngine(
            ["ノアちゃん、おはよう", "美咲さん、こんばんは"]
        )
        engine.capture_translations(
            ["ノアちゃんと美咲さん"], ["诺亚和美咲"]
        )
        self.assertNotIn("ノア", engine.locked_translations)
        self.assertNotIn("美咲", engine.locked_translations)

    def test_kanji_long_variant_does_not_shorten_lock(self) -> None:
        engine = DynamicEntityEngine(
            ["美咲日和はどこ？", "美咲希织、大好き"]
        )
        engine.capture_translations(
            ["美咲日和、おはよう"], ["美咲日和，早上好"]
        )
        self.assertNotIn("美咲", engine.locked_translations)

    def test_enforce_locks_cleans_untranslated_japanese(self) -> None:
        engine = DynamicEntityEngine(
            ["ノアちゃん、おはよう", "ノアチップ、またね"]
        )
        engine.capture_translations(
            ["ノアちゃん、おはよう"], ["诺亚，早上好"]
        )
        self.assertEqual(engine.enforce_locks("ノア、再见"), "诺亚、再见")
        self.assertEqual(
            engine.enforce_locks("ノアチップ、再见"), "诺亚、再见"
        )
        self.assertEqual(engine.enforce_locks("诺亚，再见"), "诺亚，再见")


class PostprocessTests(unittest.TestCase):
    def test_traditional_to_simplified_cleaner(self) -> None:
        self.assertEqual(
            postprocess_cue_text("妳的話我會聽", "好きだよ"),
            "你的话我会听",
        )

    def test_leading_punctuation_cleanup_keeps_ellipsis(self) -> None:
        self.assertEqual(postprocess_cue_text("，你好", "hello"), "你好")
        self.assertEqual(postprocess_cue_text("……喜欢你", "hello"), "……喜欢你")

    def test_repeated_punctuation_cleanup(self) -> None:
        self.assertEqual(
            postprocess_cue_text("好舒服……。", "hello"),
            "好舒服……",
        )
        self.assertEqual(postprocess_cue_text("呼~。", "hello"), "呼~")

    def test_unauthorized_staging_brackets_stripped(self) -> None:
        self.assertEqual(
            postprocess_cue_text("（在耳后）喜欢你哦", "好きだよ"),
            "喜欢你哦",
        )
        self.assertEqual(
            postprocess_cue_text("（在双耳边）喜欢你哦", "好きだよ"),
            "喜欢你哦",
        )
        self.assertEqual(
            postprocess_cue_text("（在耳后），喜欢你哦", "好きだよ"),
            "喜欢你哦",
        )

    def test_source_brackets_preserved(self) -> None:
        self.assertEqual(
            postprocess_cue_text("（在双耳边）喜欢你哦", "(両耳) 好きだよ"),
            "（在双耳边）喜欢你哦",
        )

    def test_staging_only_translation_keeps_original(self) -> None:
        self.assertEqual(
            postprocess_cue_text("（在耳后）", "好きだよ"),
            "（在耳后）",
        )

    def test_normal_parentheses_preserved(self) -> None:
        self.assertEqual(
            postprocess_cue_text("（笑）喜欢你", "好きだよ"),
            "（笑）喜欢你",
        )

    def test_multiline_cleaning_preserves_lines(self) -> None:
        self.assertEqual(
            postprocess_cue_text("，第一行\n，第二行", "hello"),
            "第一行\n第二行",
        )

    def test_window_id_removed_at_cue_boundary(self) -> None:
        self.assertEqual(
            postprocess_cue_text("你好 [#12]", "hello"),
            "你好",
        )

    def test_slash_alternative_repairs_japanese_or_track_side(self) -> None:
        self.assertEqual(
            postprocess_cue_text("ツルペタオセックス/光溜溜的做爱", "ツルペタオセックス?"),
            "光溜溜的做爱",
        )
        self.assertEqual(
            postprocess_cue_text("Track/音轨", "トラック"),
            "音轨",
        )
        self.assertEqual(
            postprocess_cue_text("吉田先生/女士", "吉田さん"),
            "吉田先生/女士",
        )
        self.assertEqual(postprocess_cue_text("1/2", "分数"), "1/2")


if __name__ == "__main__":
    unittest.main()
