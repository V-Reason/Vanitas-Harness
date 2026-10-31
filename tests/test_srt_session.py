"""SrtTranslationSession：把原闭包拆成可单独构造、单独断言的生命周期。

这些用例直接驱动会话对象（不是只跑 translate_srt），覆盖预处理、单句/
窗口路径、续传、局部回退与实体引擎卸载。
"""

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.errors import TranslateError  # noqa: E402
from vanitas.translate import (  # noqa: E402
    SrtTranslationSession,
    T1Budget,
    WindowAlignment,
    parse_srt,
    translate_srt,
)

ID_RE = re.compile(r"^\[#(\d+)\]", re.MULTILINE)

SOURCES = [
    "朝の光が窓を抜けていく",
    "海辺の風は少し塩辛い",
    "古い本には花が挟まっている",
    "列車は長いトンネルを抜ける",
]
PHRASES = [
    "清晨的光线穿过百叶窗",
    "海边的风带着淡淡咸味",
    "旧书页里夹着干枯的花",
    "列车驶过长长的隧道",
]


def srt_text(texts: list[str]) -> str:
    blocks = []
    for index, text in enumerate(texts, start=1):
        start = f"00:00:{index * 2 - 1:02d},000"
        end = f"00:00:{index * 2:02d},000"
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


class ScriptedChat:
    """无状态脚本：按编号或源文返回确定性译文；可记录历史/实体钩子。"""

    def __init__(
        self,
        *,
        task_template: str | None = None,
        with_history: bool = False,
        with_entity: bool = False,
    ) -> None:
        self.calls: list[str] = []
        self.entity_calls: list[object] = []
        self.history: list[tuple[str, str]] = []
        self._mapping = {
            source: PHRASES[index]
            for index, source in enumerate(SOURCES)
        }
        if task_template is not None:
            self.task_template = task_template
        if with_history:
            self.history_snapshot = lambda: tuple(self.history)
            self.reset_history = self.history.clear
            self.restore_history = self._restore
        if with_entity:
            self.set_entity_engine = self.entity_calls.append

    def _restore(self, snapshot: tuple[tuple[str, str], ...]) -> None:
        self.history[:] = list(snapshot)

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        ids = [int(match.group(1)) for match in ID_RE.finditer(prompt)]
        if ids:
            return "\n".join(
                f"[#{index}] {PHRASES[(index - 1) % len(PHRASES)]}"
                for index in ids
            )
        return self._mapping.get(prompt.strip(), PHRASES[0])


def build_session(chat, text: str, **kwargs) -> SrtTranslationSession:
    kwargs.setdefault("chunk_chars", 800)
    kwargs.setdefault("source", "unit.srt")
    kwargs.setdefault("eol", "\n")
    kwargs.setdefault("failures", [])
    return SrtTranslationSession(chat, text, **kwargs)


class FinalizeStageTests(unittest.TestCase):
    def test_finalize_applies_data_rules(self) -> None:
        session = build_session(ScriptedChat(), srt_text(SOURCES[:1]))
        session._preprocess()
        self.assertEqual(
            session._finalize_cue(
                "你的小穴里面也要好好检查", "お耳の中も失礼して"
            ),
            "你的耳朵里面也要好好检查",
        )

    def test_finalize_respects_trailing_period_switch(self) -> None:
        session = build_session(ScriptedChat(), srt_text(SOURCES[:1]))
        session._preprocess()
        source = "窓を閉めてください"
        self.assertEqual(
            session._finalize_cue("请把窗户关上。", source),
            "请把窗户关上",
        )
        no_strip = build_session(
            ScriptedChat(),
            srt_text(SOURCES[:1]),
            strip_trailing_periods=False,
        )
        no_strip._preprocess()
        self.assertEqual(
            no_strip._finalize_cue("请把窗户关上。", source),
            "请把窗户关上。",
        )

    def test_nearby_sources_exclude_current_cue(self) -> None:
        chat = ScriptedChat()
        session = build_session(chat, srt_text(SOURCES))
        session._preprocess()
        nearby = session._nearby_sources_for(2)
        self.assertNotIn(SOURCES[1], nearby)
        self.assertIn(SOURCES[0], nearby)
        self.assertIn(SOURCES[2], nearby)


class PreprocessStageTests(unittest.TestCase):
    def test_asr_hallucination_is_filtered_from_translatable(self) -> None:
        texts = [*SOURCES[:3], "ご視聴ありがとうございました"]
        session = build_session(ScriptedChat(), srt_text(texts))
        session._preprocess()
        self.assertEqual(session.total, 4)
        self.assertEqual(session.filtered, 1)
        self.assertEqual(len(session.translatable), 3)
        self.assertEqual(
            [cue for cue, _text in session.all_cues], [1, 2, 3, 4]
        )

    def test_document_text_keeps_document_order(self) -> None:
        session = build_session(ScriptedChat(), srt_text(SOURCES))
        session._preprocess()
        self.assertEqual(
            session.document_text.splitlines(), SOURCES
        )


class PathEquivalenceTests(unittest.TestCase):
    def session_output(self, texts: list[str], **kwargs) -> str:
        chat = ScriptedChat(task_template="translate_asmr")
        session = build_session(
            chat, srt_text(texts), asmr_guard=True, **kwargs
        )
        return session.execute()

    def test_single_cue_matches_translate_srt(self) -> None:
        expected = self.session_output(SOURCES)
        chat = ScriptedChat(task_template="translate_asmr")
        failures: list = []
        actual = translate_srt(
            chat,
            srt_text(SOURCES),
            chunk_chars=800,
            source="unit.srt",
            eol="\n",
            failures=failures,
            asmr_guard=True,
            t1_budget=T1Budget(),
        )
        self.assertEqual(actual, expected)
        self.assertEqual(failures, [])

    def test_window_path_matches_translate_srt(self) -> None:
        expected = self.session_output(
            SOURCES, window_size=2, t1_budget=T1Budget()
        )
        chat = ScriptedChat(task_template="translate_asmr")
        actual = translate_srt(
            chat,
            srt_text(SOURCES),
            chunk_chars=800,
            source="unit.srt",
            eol="\n",
            failures=[],
            window_size=2,
            max_window_chars=1000,
            asmr_guard=True,
            t1_budget=T1Budget(),
        )
        self.assertEqual(actual, expected)


class ResumeStageTests(unittest.TestCase):
    def test_resume_backfills_without_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            part = Path(tmp) / "unit.part.srt"
            first_chat = ScriptedChat()
            first = build_session(
                first_chat,
                srt_text(SOURCES),
                checkpoint_path=part,
                window_size=1,
            )
            produced = first.execute()
            self.assertTrue(part.is_file())

            resumed_chat = ScriptedChat()
            resumed = build_session(
                resumed_chat,
                srt_text(SOURCES),
                checkpoint_path=part,
                resume=True,
            )
            output = resumed.execute()

        self.assertEqual(resumed_chat.calls, [])
        self.assertEqual(resumed.done, len(SOURCES))
        self.assertEqual(output, produced)


class WindowStageTests(unittest.TestCase):
    def session(self, **kwargs) -> SrtTranslationSession:
        chat = ScriptedChat(task_template="translate_asmr", with_entity=True)
        session = build_session(
            chat,
            srt_text(SOURCES),
            asmr_guard=True,
            window_size=2,
            t1_budget=T1Budget(),
            **kwargs,
        )
        session._preprocess()
        return session

    def test_partial_staging_falls_back_when_half_corrupt(self) -> None:
        session = self.session()
        session._build_entity_engine()
        window = session.translatable[:2]
        alignment = WindowAlignment(
            is_catastrophic=False,
            aligned={"1": "清晨的光线穿过百叶窗"},
            missing_ids=["2"],
        )
        self.assertIsNone(
            session._partial_window_staging(window, alignment, ())
        )

    def test_partial_staging_accepts_single_corruption(self) -> None:
        session = self.session()
        session._build_entity_engine()
        window = session.translatable[:2]
        alignment = WindowAlignment(
            is_catastrophic=False,
            aligned={
                "1": "清晨的光线穿过百叶窗",
                "2": "海边的风带着淡淡咸味",
            },
            missing_ids=[],
        )
        self.assertEqual(
            session._partial_window_staging(window, alignment, ()),
            ["清晨的光线穿过百叶窗", "海边的风带着淡淡咸味"],
        )

    def test_entity_engine_is_detached_after_run(self) -> None:
        session = self.session()
        session.execute()
        self.assertTrue(session.chat.entity_calls)
        self.assertIsNone(session.chat.entity_calls[-1])


class GuardTests(unittest.TestCase):
    def test_missing_asmr_budget_raises(self) -> None:
        with self.assertRaises(TranslateError):
            translate_srt(
                ScriptedChat(),
                srt_text(SOURCES[:1]),
                chunk_chars=800,
                source="unit.srt",
                eol="\n",
                failures=[],
                asmr_guard=True,
            )

    def test_progress_reaches_total_in_single_mode(self) -> None:
        progress: list[tuple[int, int]] = []
        texts = [*SOURCES[:3], "ご視聴ありがとうございました"]
        chat = ScriptedChat()
        session = build_session(
            chat,
            srt_text(texts),
            on_segment=lambda done, total: progress.append((done, total)),
        )
        session.execute()
        self.assertEqual(progress[-1], (4, 4))

    def test_parsed_timeline_is_unchanged(self) -> None:
        chat = ScriptedChat()
        session = build_session(chat, srt_text(SOURCES))
        output = session.execute()
        source_blocks = parse_srt(srt_text(SOURCES), "\n")
        output_blocks = parse_srt(output, "\n")
        self.assertEqual(
            [block.timing for block, _ in output_blocks],
            [block.timing for block, _ in source_blocks],
        )


if __name__ == "__main__":
    unittest.main()
