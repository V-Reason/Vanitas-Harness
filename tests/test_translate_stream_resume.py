"""P4 流式早停、KV 前缀保热与断点续传。

覆盖四件事：
1. ``StreamingAbortDetector`` 的四类早停判据（越界编号/空行/死锁/超长）；
2. 流式聚合与阻塞路径的等价性（文本、tokens、tps 来源）；
3. 早停把窗口降级为逐句，而不是把整个文件顶掉；
4. ``.part.srt`` 快照的写入、校验、续传与终态清理。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.config import TranslateSettings  # noqa: E402
from vanitas.errors import OutputTruncated  # noqa: E402
from vanitas.translate import (  # noqa: E402
    CritiqueBudget,
    StreamingAbortDetector,
    T1Budget,
    TranslationEngine,
    TranslationOptions,
    collect_translate_inputs,
    parse_srt,
    partial_snapshot_path,
    translate_file,
    translate_srt,
)

PROMPT_DIR = ROOT / "config" / "prompts"

OPTIONS = TranslationOptions(
    n_ctx=8192,
    context_window=5,
    context_summary_chars=200,
    min_output_tokens=512,
    output_ratio=2.5,
    prompt_safety_tokens=64,
    temperature=0.2,
    max_split_depth=3,
)
SETTINGS = TranslateSettings(
    source_suffixes=(".txt", ".md", ".srt"),
    chunk_chars=1500,
    output_suffix=".zh",
)

SOURCES = [f"これは{index}番目の台詞です" for index in range(1, 13)]
ZH = [f"这是第{index}句台词" for index in range(1, 13)]


def srt_text(texts: list[str]) -> str:
    blocks = []
    for index, text in enumerate(texts, start=1):
        start = f"00:00:{index * 2 - 1:02d},000"
        end = f"00:00:{index * 2:02d},000"
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


def batch_line(ids_and_text: list[tuple[int, str]]) -> str:
    return "\n".join(f"[#{index}] {text}" for index, text in ids_and_text)


class StreamingClient:
    """假客户端：实现 stream_chat，按脚本产出 delta/done 事件。"""

    def __init__(
        self,
        scripts: list[list[str]] | None = None,
        *,
        tokens: int = 7,
        tps: float = 12.5,
        tps_source: str = "server.timings",
        finish_reason: str = "stop",
        finish_reasons: list[str] | None = None,
    ) -> None:
        self.scripts = list(scripts or [])
        self.tokens = tokens
        self.tps = tps
        self.tps_source = tps_source
        self.finish_reason = finish_reason
        self.finish_reasons = list(finish_reasons or [])
        self.calls: list[str] = []
        self.closed = 0

    def stream_chat(self, messages, *, max_tokens, temperature=0.2, **kwargs):
        self.calls.append(str(messages[-1]["content"]))
        chunks = self.scripts.pop(0) if self.scripts else []
        reason = (
            self.finish_reasons.pop(0)
            if self.finish_reasons
            else self.finish_reason
        )
        try:
            for chunk in chunks:
                yield {"type": "delta", "content": chunk}
            yield {
                "type": "done",
                "finish_reason": reason,
                "tokens": self.tokens,
                "tps": self.tps,
                "tps_source": self.tps_source,
                "usage": {"completion_tokens": self.tokens},
                "timings": {"predicted_per_second": self.tps},
                "ttft_ms": 1.0,
                "elapsed_ms": 10.0,
            }
        finally:
            # 早停必须走到这里：调用方 close() 会抛 GeneratorExit 穿过 yield。
            self.closed += 1


class ScriptedChat:
    """按脚本返回整段文本；用异常对象模拟崩溃。"""

    def __init__(
        self, steps: list[object], *, task_template: str | None = None
    ) -> None:
        self.steps = list(steps)
        self.calls: list[str] = []
        if task_template is not None:
            self.task_template = task_template

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        step = self.steps.pop(0) if self.steps else ""
        if isinstance(step, BaseException):
            raise step
        return str(step)


class DetectorTests(unittest.TestCase):
    """早停判据必须只拦“确定救不回来”的形态；越界编号是软停机。"""

    def test_normal_text_never_aborts(self) -> None:
        detector = StreamingAbortDetector(7)
        self.assertTrue(detector.feed("[#1] 今天天气真好\n"))
        self.assertFalse(detector.aborted)

    def test_second_window_first_line_does_not_abort(self) -> None:
        # 第 2 窗编号是 [#7]…[#12]，下一个期望编号是 13：合法首行不得触发。
        detector = StreamingAbortDetector(13)
        for chunk in ("[#7] 一\n", "[#8] 二\n", "[#12] 六\n"):
            self.assertTrue(detector.feed(chunk), chunk)
        self.assertFalse(detector.aborted)
        self.assertFalse(detector.soft_stop)

    def test_out_of_range_tag_soft_stops(self) -> None:
        detector = StreamingAbortDetector(7)
        self.assertTrue(detector.feed("[#1] 一\n[#2] 二\n"))
        self.assertFalse(detector.feed("[#7] 七"))
        self.assertTrue(detector.soft_stop)
        self.assertFalse(detector.aborted)
        self.assertEqual(detector.buffer[detector.soft_offset :], "[#7] 七")

    def test_blank_line_flood_aborts(self) -> None:
        detector = StreamingAbortDetector(7)
        self.assertFalse(detector.feed("译文\n\n\n\n下一行"))
        self.assertIn("空行", detector.abort_reason)
        self.assertFalse(detector.soft_stop)

    def test_phrase_deadlock_aborts(self) -> None:
        detector = StreamingAbortDetector(7)
        unit = "あ" * 12
        self.assertFalse(detector.feed(unit * 3))
        self.assertIn("死锁", detector.abort_reason)

    def test_char_limit_aborts(self) -> None:
        detector = StreamingAbortDetector(7, max_chars=100)
        self.assertFalse(detector.feed("あ" * 200))
        self.assertIn("字符数", detector.abort_reason)


class StreamingEngineTests(unittest.TestCase):
    """流式聚合与早停的引擎层行为。"""

    def engine(self, client) -> TranslationEngine:
        return TranslationEngine(client, PROMPT_DIR, "", options=OPTIONS)

    def test_streaming_aggregation_matches_plain_text(self) -> None:
        client = StreamingClient([["你好", "，", "世界"]])
        stats: list = []
        engine = TranslationEngine(
            client, PROMPT_DIR, "", options=OPTIONS, on_stat=stats.append
        )
        self.assertEqual(engine("こんにちは"), "你好，世界")
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0].tokens, 7)
        self.assertEqual(stats[0].tps_source, "server.timings")

    def test_soft_stop_keeps_valid_prefix(self) -> None:
        # 模型多吐了一句越界编号：保留前面合法译文，不抛异常、不整窗作废。
        client = StreamingClient([["[#1] 一", "\n[#2] 二", "\n[#3] 三"]])
        engine = self.engine(client)
        result = engine("[#1] あ\n[#2] い")
        self.assertEqual(result, "[#1] 一\n[#2] 二")
        self.assertNotIn("[#3]", result)
        self.assertEqual(client.closed, 1)

    def test_abort_closes_stream_and_raises(self) -> None:
        client = StreamingClient([["[#1] 好", "\n\n\n\n"]])
        engine = self.engine(client)
        with self.assertRaises(OutputTruncated):
            engine("[#1] こんにちは")
        self.assertEqual(client.closed, 1)

    def test_streaming_length_reason_splits_input(self) -> None:
        # finish_reason=length 时走二分重译：1 次截断 + 2 次分片 = 3 次请求。
        client = StreamingClient(
            [["截断"], ["甲"], ["乙"]],
            finish_reasons=["length", "stop", "stop"],
        )
        engine = self.engine(client)
        self.assertEqual(engine("一句。二句。"), "甲乙")
        self.assertEqual(len(client.calls), 3)

    def test_window_abort_degrades_to_per_cue(self) -> None:
        # 批次流被早停掐断后，必须整窗回退逐句，而不是把文件顶掉。
        client = StreamingClient(
            [
                ["[#1] 一\n", "\n\n\n\n"],
                ["译一"],
                ["译二"],
            ]
        )
        engine = self.engine(client)
        failures: list = []
        output = translate_srt(
            engine,
            srt_text(SOURCES[:2]),
            chunk_chars=800,
            source="abort.srt",
            eol="\n",
            failures=failures,
            window_size=2,
            max_window_chars=1000,
        )
        self.assertEqual(failures, [])
        self.assertEqual(
            [block.text for block, _ in parse_srt(output, "\n")],
            ["译一", "译二"],
        )
        self.assertEqual(len(client.calls), 3)


class CheckpointTests(unittest.TestCase):
    """快照写入、崩溃残留、续传与清理。"""

    def test_partial_snapshot_survives_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "long.srt"
            source.write_text(srt_text(SOURCES), encoding="utf-8")
            first = batch_line(list(zip(range(1, 7), ZH[:6])))
            chat = ScriptedChat([first, RuntimeError("模拟崩溃")])

            with self.assertRaises(RuntimeError):
                translate_file(chat, source, SETTINGS, window_size=6)

            snapshot = partial_snapshot_path(
                tmp_path / "long.zh.srt"
            )
            self.assertTrue(snapshot.is_file())
            blocks = parse_srt(snapshot.read_text(encoding="utf-8"), "\n")
            self.assertEqual(
                [block.text for block, _ in blocks], ZH[:6]
            )
            self.assertFalse((tmp_path / "long.zh.srt").exists())

    def test_resume_skips_completed_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "long.srt"
            source.write_text(srt_text(SOURCES), encoding="utf-8")
            first = batch_line(list(zip(range(1, 7), ZH[:6])))
            crashing = ScriptedChat([first, RuntimeError("模拟崩溃")])
            with self.assertRaises(RuntimeError):
                translate_file(crashing, source, SETTINGS, window_size=6)

            second = batch_line(list(zip(range(7, 13), ZH[6:])))
            resuming = ScriptedChat([second])
            outcome = translate_file(
                resuming, source, SETTINGS, window_size=6, resume=True
            )

            self.assertEqual(len(resuming.calls), 1)
            self.assertEqual(outcome.failures, [])
            self.assertEqual(outcome.segments, 12)
            produced = outcome.output.read_text(encoding="utf-8")
            blocks = parse_srt(produced, "\n")
            self.assertEqual([block.text for block, _ in blocks], ZH)
            snapshot = partial_snapshot_path(outcome.output)
            self.assertFalse(snapshot.is_file())

    def test_stale_snapshot_is_ignored_without_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "long.srt"
            source.write_text(srt_text(SOURCES[:6]), encoding="utf-8")
            target = tmp_path / "long.zh.srt"
            # 造一个时间轴对不上的旧快照：续传必须整体丢弃而不是错位回填。
            snapshot = partial_snapshot_path(target)
            snapshot.write_text(
                "1\n00:09:59,000 --> 00:09:59,500\n旧快照文本\n",
                encoding="utf-8",
            )
            chat = ScriptedChat(
                [batch_line(list(zip(range(1, 7), ZH[:6])))]
            )
            outcome = translate_file(
                chat, source, SETTINGS, window_size=6, resume=True
            )
            self.assertEqual(len(chat.calls), 1)
            self.assertEqual(
                [block.text for block, _ in parse_srt(
                    outcome.output.read_text(encoding="utf-8"), "\n"
                )],
                ZH[:6],
            )
            self.assertFalse(snapshot.is_file())

    def test_resume_single_cue_path_skips_completed(self) -> None:
        # 默认 subtitle 预设是 window_size=1：这条路径的跳过快照必须单独覆盖。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "one.srt"
            source.write_text(srt_text(SOURCES[:4]), encoding="utf-8")
            snapshot = partial_snapshot_path(tmp_path / "one.zh.srt")
            snapshot.write_text(srt_text(ZH[:2]), encoding="utf-8")
            chat = ScriptedChat([ZH[2], ZH[3]])
            outcome = translate_file(
                chat, source, SETTINGS, window_size=1, resume=True
            )
            produced = outcome.output.read_text(encoding="utf-8")

        self.assertEqual(len(chat.calls), 2)
        self.assertEqual(outcome.failures, [])
        self.assertEqual(
            [block.text for block, _ in parse_srt(produced, "\n")], ZH[:4]
        )
        self.assertFalse(snapshot.is_file())

    def test_resume_with_complete_snapshot_makes_no_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "done.srt"
            source.write_text(srt_text(SOURCES[:4]), encoding="utf-8")
            snapshot = partial_snapshot_path(tmp_path / "done.zh.srt")
            snapshot.write_text(srt_text(ZH[:4]), encoding="utf-8")
            chat = ScriptedChat([])
            outcome = translate_file(
                chat, source, SETTINGS, window_size=1, resume=True
            )
            produced = outcome.output.read_text(encoding="utf-8")

        self.assertEqual(chat.calls, [])
        self.assertEqual(
            [block.text for block, _ in parse_srt(produced, "\n")], ZH[:4]
        )
        self.assertFalse(snapshot.is_file())

    def test_snapshot_is_not_collected_as_input(self) -> None:
        settings = TranslateSettings(
            source_suffixes=(".srt",), chunk_chars=800, output_suffix=".zh"
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "a.srt").write_text("1\n", encoding="utf-8")
            (tmp_path / "a.zh.srt.part.srt").write_text("1\n", encoding="utf-8")
            sources = collect_translate_inputs([tmp_path], settings)
            self.assertEqual([item.path.name for item in sources], ["a.srt"])


class PrefixHeatingTests(unittest.TestCase):
    """KV 保热的前提：动态内容全在模板尾部。"""

    def test_dynamic_placeholders_are_at_the_tail(self) -> None:
        from vanitas import prompts

        template = prompts.load_task(PROMPT_DIR, "translate_asmr")
        frozen = template.find("{context_history}")
        self.assertGreater(frozen, 0)
        self.assertGreater(frozen, int(len(template) * 0.9))
        self.assertEqual(template.find("{input}"), template.rfind("{input}"))
        self.assertGreater(template.find("{input}"), frozen)

    def test_rendered_prefix_is_identical_across_windows(self) -> None:
        from vanitas import prompts

        template = prompts.load_task(PROMPT_DIR, "translate_asmr")
        frozen = template[: template.find("{context_history}")]
        engine = TranslationEngine(
            StreamingClient([[], []]),
            PROMPT_DIR,
            "SYSTEM",
            options=OPTIONS,
            task_template="translate_asmr",
        )
        engine.restore_history((("前一句原文", "前一句译文"),))
        first = engine._render("[#1] 第一句")[0][-1]["content"]
        engine.restore_history((("另一句原文", "另一句译文"),))
        second = engine._render("[#1] 第二句")[0][-1]["content"]
        self.assertEqual(first[: len(frozen)], frozen)
        self.assertEqual(second[: len(frozen)], frozen)


class ThreadingComplianceTests(unittest.TestCase):
    """红线：translate 子包内不得出现任何线程/进程池。"""

    def test_no_threading_in_translate_package(self) -> None:
        package = ROOT / "vanitas" / "translate"
        banned = ("import threading", "ThreadPoolExecutor", "multiprocessing")
        offenders: list[str] = []
        for path in sorted(package.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for needle in banned:
                if needle in text:
                    offenders.append(f"{path.name}: {needle}")
        self.assertEqual(offenders, [])


class ResumeAuditIsolationTests(unittest.TestCase):
    """续传回填不能污染 T1 审计看到的邻行源文（P5-F1）。"""

    def test_resume_keeps_japanese_neighbours_for_invented_tool(self) -> None:
        texts = [
            "綿棒でお耳をきれいにしましょう",
            "はい、お願いします",
            "それでは始めますね",
            "気持ちいいですか",
        ]
        done = ["用棉棒把耳朵清理干净吧", "好的，麻烦你了"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "asmr.srt"
            source.write_text(srt_text(texts), encoding="utf-8")
            snapshot = partial_snapshot_path(tmp_path / "asmr.zh.srt")
            snapshot.write_text(srt_text(done), encoding="utf-8")
            chat = ScriptedChat(
                [batch_line([(3, "那就用掏耳勺开始吧"), (4, "舒服吗")])],
                task_template="translate_asmr",
            )
            failures: list = []
            output = translate_srt(
                chat,
                srt_text(texts),
                chunk_chars=800,
                source=str(source),
                eol="\n",
                failures=failures,
                window_size=2,
                max_window_chars=1000,
                asmr_guard=True,
                t1_budget=T1Budget(),
                checkpoint_path=snapshot,
                resume=True,
            )

        self.assertEqual(failures, [])
        # 第 1 条的「綿棒」必须仍以日文出现在 nearby 里；否则 invented_tool
        # 会误报并多发一次单句请求。
        self.assertEqual(len(chat.calls), 1, chat.calls)
        self.assertIn("掏耳勺", output)


class CheckpointWarningTests(unittest.TestCase):
    """丢弃未完成快照必须出声（P5-F6）。"""

    def test_discarding_stale_snapshot_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "a.srt"
            source.write_text(srt_text(SOURCES[:6]), encoding="utf-8")
            target = tmp_path / "a.zh.srt"
            snapshot = partial_snapshot_path(target)
            snapshot.write_text(srt_text(ZH[:2]), encoding="utf-8")
            chat = ScriptedChat(
                [batch_line(list(zip(range(1, 7), ZH[:6])))]
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                outcome = translate_file(
                    chat, source, SETTINGS, window_size=6
                )

        messages = [str(item.message) for item in caught]
        self.assertTrue(
            any("断点快照" in text and "--resume" in text for text in messages),
            messages,
        )
        self.assertFalse(snapshot.is_file())
        self.assertEqual(outcome.segments, 6)


class CritiquePathWarningTests(unittest.TestCase):
    """window_size=1 时 P3 复审不生效，必须告警而不是静默（P5-F11）。"""

    def test_critique_budget_on_single_cue_path_warns(self) -> None:
        chat = ScriptedChat([ZH[0]])
        failures: list = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            translate_srt(
                chat,
                srt_text(SOURCES[:1]),
                chunk_chars=800,
                source="one.srt",
                eol="\n",
                failures=failures,
                window_size=1,
                critique_budget=CritiqueBudget(),
            )

        messages = [str(item.message) for item in caught]
        self.assertTrue(any("P3 复审" in text for text in messages), messages)


if __name__ == "__main__":
    unittest.main()
