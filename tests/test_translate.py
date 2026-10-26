"""翻译：SRT 时间轴/多行/BOM/换行、文本分块、失败汇总。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.config import TranslateSettings  # noqa: E402
from vanitas.errors import EngineError, TranslateError  # noqa: E402
from vanitas.translate import (  # noqa: E402
    TranslationEngine,
    TranslationOptions,
    chunk_text,
    parse_srt,
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


class TranslationEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompt_dir = ROOT / "config" / "prompts"

    def make_engine(self, client, **options) -> TranslationEngine:
        return TranslationEngine(
            client,
            self.prompt_dir,
            "系统提示",
            options=TranslationOptions(**options),
        )

    def test_context_sentinel_and_fifo(self) -> None:
        client = FakeTranslationClient()
        stats = []
        engine = TranslationEngine(
            client,
            self.prompt_dir,
            "系统提示",
            options=TranslationOptions(context_window=2),
            on_stat=stats.append,
        )
        engine.translate("alpha")
        engine.translate("beta")
        engine.translate("gamma")
        engine.translate("delta")

        first_prompt = client.calls[0]["messages"][-1]["content"]
        self.assertIn("（无前文，故事开端）", first_prompt)
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
