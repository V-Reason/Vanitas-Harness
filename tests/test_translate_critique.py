"""P3 两阶段定向反思：分流精度、预算熔断、隔离与负优化拦截。

全部用例都走真实 ``translate_srt`` 或真实 ``execute_critique_repair``，
只用脚本化 chat 替身控制模型输出，不 mock 被测逻辑。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import prompts  # noqa: E402
from vanitas.errors import EngineError  # noqa: E402
from vanitas.translate import (  # noqa: E402
    CritiqueBudget,
    build_critique_input,
    execute_critique_repair,
    parse_critique_response,
    parse_srt,
    translate_srt,
    triage_cue_risk,
)

SOURCES = [
    "今日はいい天気ですね",
    "お客様から私たちに触れることは禁止となっております",
    "窓を閉めてください",
    "コーヒーを入れます",
    "駅まで歩きます",
    "本を読みます",
]
DRAFTS = [
    "今天天气真好",
    "我们可以触碰客人",
    "请把窗户关上",
    "泡一杯咖啡",
    "走到车站去",
    "看看书",
]


def srt_text(texts: list[str]) -> str:
    blocks = []
    for index, text in enumerate(texts, start=1):
        start = f"00:00:{index * 2 - 1:02d},000"
        end = f"00:00:{index * 2:02d},000"
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


class CritiqueChat:
    """脚本化 chat：翻译与复审分开计数，并暴露历史快照接口。"""

    def __init__(
        self,
        drafts: list[str],
        critiques: list[str] | None = None,
        *,
        fail_critique: bool = False,
    ) -> None:
        self.drafts = list(drafts)
        self.critiques = list(critiques or [])
        self.fail_critique = fail_critique
        self.calls: list[str] = []
        self.critique_calls: list[str] = []
        self.history: list[tuple[str, str]] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        content = self.drafts.pop(0) if self.drafts else ""
        self.history.append((prompt, content))
        return content

    def critique(self, prompt: str, *, max_tokens: int = 96) -> str:
        self.critique_calls.append(prompt)
        if self.fail_critique:
            raise EngineError("模拟复审请求失败")
        if not self.critiques:
            raise AssertionError("并未预期发生复审调用")
        return self.critiques.pop(0)

    def history_snapshot(self) -> tuple[tuple[str, str], ...]:
        return tuple(self.history)

    def reset_history(self) -> None:
        self.history.clear()

    def restore_history(self, snapshot: tuple[tuple[str, str], ...]) -> None:
        self.history[:] = list(snapshot)


class TriageTests(unittest.TestCase):
    """分流器只做“要不要复审”，不做质量判断。"""

    def test_fast_pass_skips_onomatopoeia_and_short_greetings(self) -> None:
        for source in ("チュッ", "はむはむ", "あっ", "はい", "沙沙"):
            self.assertEqual(
                triage_cue_risk(source, "嗯"), set(), source
            )

    def test_fast_pass_never_hides_long_risky_lines(self) -> None:
        codes = triage_cue_risk(
            "お客様から私たちに触れることは禁止となっております",
            "我们可以触碰客人",
        )
        self.assertEqual(codes, {"customer_direction"})

    def test_five_risk_families_are_detected(self) -> None:
        cases = {
            "customer_direction": (
                "お客様から私たちに触れることは禁止となっております",
                "我们可以触碰客人",
            ),
            "negative_question": (
                "私のこと、覚えていませんか",
                "你还记得我吗",
            ),
            "first_person": (
                "下の名前で呼んであげて",
                "请叫我的名字",
            ),
        }
        for expected, (source, draft) in cases.items():
            self.assertIn(expected, triage_cue_risk(source, draft), expected)

        hanging = triage_cue_risk(
            "明日はもっと頑張るから", "明天我会更努力。", ("次も見てね",)
        )
        self.assertIn("hanging_conjunction", hanging)

    def test_completed_sentence_is_not_hanging(self) -> None:
        codes = triage_cue_risk(
            "明日はもっと頑張るから", "因为明天还要更努力，", ("次も見てね",)
        )
        self.assertNotIn("hanging_conjunction", codes)


class CritiqueResponseTests(unittest.TestCase):
    """复审输出的解析契约。"""

    def test_keep_markers_do_not_change_draft(self) -> None:
        for raw in ("【保持】", "保持", "KEEP", "", "   "):
            changed, text = parse_critique_response(raw, "初译草稿")
            self.assertFalse(changed, raw)
            self.assertEqual(text, "初译草稿")

    def test_fenced_repair_is_unwrapped(self) -> None:
        changed, text = parse_critique_response(
            "```\n客人不能触碰我们\n```", "我们可以触碰客人"
        )
        self.assertTrue(changed)
        self.assertEqual(text, "客人不能触碰我们")

    def test_identical_text_counts_as_kept(self) -> None:
        changed, _text = parse_critique_response(
            "我们可以触碰客人。", "我们可以触碰客人"
        )
        self.assertFalse(changed)

    def test_input_bundle_contains_all_sections(self) -> None:
        bundle = build_critique_input(
            "お客様",
            "客人",
            (("前の文", "前文译文"),),
            {"customer_direction"},
        )
        for marker in (
            "【前文参考】",
            "【当前日文】",
            "【初译草稿】",
            "【焦点风险】",
            "customer_direction",
        ):
            self.assertIn(marker, bundle)


class CritiqueBudgetTests(unittest.TestCase):
    """双层配额：整包与单文件都不能被突破。"""

    def test_file_budget_stops_after_quota(self) -> None:
        budget = CritiqueBudget(max_total=2)
        budget.begin_file("a.srt", 100)
        # 单文件按 10% 算得 10，但不得超过整包额度 2。
        self.assertEqual(budget.per_file, 2)
        self.assertTrue(budget.can_critique())
        budget.record_attempt()
        budget.record_attempt()
        self.assertFalse(budget.can_critique())

    def test_for_corpus_scales_and_has_floor(self) -> None:
        self.assertEqual(CritiqueBudget.for_corpus(0).max_total, 15)
        self.assertEqual(CritiqueBudget.for_corpus(None).max_total, 15)
        self.assertEqual(CritiqueBudget.for_corpus(500).max_total, 40)
        self.assertEqual(CritiqueBudget.for_corpus(50).max_total, 10)

    def test_begin_file_clamps_per_file_to_total(self) -> None:
        budget = CritiqueBudget(max_total=4)
        budget.begin_file("a.srt", 1000)
        self.assertEqual(budget.per_file, 4)


class CritiquePipelineTests(unittest.TestCase):
    """端到端：只在命中的 cue 上多一次请求，其余与 P2 完全一致。"""

    def run_window(
        self,
        chat: CritiqueChat,
        *,
        texts: list[str] | None = None,
        budget: CritiqueBudget | None = None,
        file_cues: int | None = None,
    ):
        failures: list = []
        output = translate_srt(
            chat,
            srt_text(texts or SOURCES),
            chunk_chars=800,
            source="critique.srt",
            eol="\n",
            failures=failures,
            window_size=6,
            max_window_chars=2000,
            critique_budget=budget,
            file_cues=file_cues,
        )
        return output, failures

    def batch(self) -> str:
        return "\n".join(
            f"[#{index}] {text}"
            for index, text in enumerate(DRAFTS, start=1)
        )

    def test_single_risky_cue_triggers_exactly_one_critique(self) -> None:
        chat = CritiqueChat([self.batch()], ["客人不能触碰我们"])
        budget = CritiqueBudget(max_total=10, per_file=5)
        output, failures = self.run_window(chat, budget=budget)

        self.assertEqual(failures, [])
        self.assertEqual(len(chat.calls), 1)
        self.assertEqual(len(chat.critique_calls), 1)
        self.assertEqual([e.outcome for e in budget.file_events], ["repaired"])
        texts = [block.text for block, _ in parse_srt(output, "\n")]
        self.assertEqual(texts[1], "客人不能触碰我们")
        self.assertEqual(
            [texts[index] for index in (0, 2, 3, 4, 5)],
            [DRAFTS[index] for index in (0, 2, 3, 4, 5)],
        )

    def test_default_is_off(self) -> None:
        chat = CritiqueChat([self.batch()])
        output, _failures = self.run_window(chat, budget=None)
        self.assertEqual(chat.critique_calls, [])
        texts = [block.text for block, _ in parse_srt(output, "\n")]
        self.assertEqual(texts, DRAFTS)

    def test_budget_blocks_extra_critiques_in_one_window(self) -> None:
        risky = [
            "今日はいい天気ですね",
            "お客様から私たちに触れることは禁止となっております",
            "私のこと、覚えていませんか",
            "まだ覚えていませんか",
            "明日はもっと頑張るから",
            "本を読みます",
        ]
        drafts = [
            "今天天气真好",
            "我们可以触碰客人",
            "你还记得我吗",
            "你还记得吗",
            "明天我会更努力。",
            "看看书",
        ]
        chat = CritiqueChat(
            ["\n".join(f"[#{i}] {t}" for i, t in enumerate(drafts, 1)),
             "你不记得我了吗"],
            ["客人不能触碰我们", "你不记得我了吗"],
        )
        budget = CritiqueBudget(max_total=2)
        output, failures = self.run_window(
            chat, texts=risky, budget=budget, file_cues=20
        )

        self.assertEqual(failures, [])
        self.assertEqual(len(chat.critique_calls), 2)
        outcomes = [event.outcome for event in budget.file_events]
        self.assertEqual(outcomes.count("repaired"), 2)
        self.assertEqual(outcomes.count("budget"), 2)
        self.assertEqual(budget.total, 2)
        blocks = parse_srt(output, "\n")
        self.assertEqual(len(blocks), len(risky))

    def test_keep_and_negative_optimization_keep_draft(self) -> None:
        for reply in ("【保持】", "こんにちは", "先生/女士"):
            chat = CritiqueChat([self.batch()], [reply])
            budget = CritiqueBudget(max_total=10, per_file=5)
            output, failures = self.run_window(chat, budget=budget)

            self.assertEqual(failures, [], reply)
            texts = [block.text for block, _ in parse_srt(output, "\n")]
            self.assertEqual(texts, DRAFTS, reply)
            outcomes = [event.outcome for event in budget.file_events]
            self.assertEqual(outcomes, ["kept"] if reply == "【保持】" else ["rejected"], reply)

    def test_critique_error_degrades_without_failure(self) -> None:
        chat = CritiqueChat([self.batch()], fail_critique=True)
        budget = CritiqueBudget(max_total=10, per_file=5)
        output, failures = self.run_window(chat, budget=budget)

        self.assertEqual(failures, [])
        self.assertEqual([e.outcome for e in budget.file_events], ["error"])
        texts = [block.text for block, _ in parse_srt(output, "\n")]
        self.assertEqual(texts, DRAFTS)

    def test_history_is_isolated_from_critique(self) -> None:
        chat = CritiqueChat([self.batch()], ["客人不能触碰我们"])
        budget = CritiqueBudget(max_total=10, per_file=5)
        self.run_window(chat, budget=budget)

        snapshot = chat.history_snapshot()
        self.assertEqual(len(snapshot), len(SOURCES))
        self.assertFalse(
            [source for source, _ in snapshot if "【初译草稿】" in source]
        )

    def test_structure_and_timeline_are_preserved(self) -> None:
        chat = CritiqueChat([self.batch()], ["客人不能触碰我们"])
        budget = CritiqueBudget(max_total=10, per_file=5)
        output, _failures = self.run_window(chat, budget=budget)

        blocks = parse_srt(output, "\n")
        source = parse_srt(srt_text(SOURCES), "\n")
        self.assertEqual(
            [block.index for block, _ in blocks],
            [block.index for block, _ in source],
        )
        self.assertEqual(
            [block.timing for block, _ in blocks],
            [block.timing for block, _ in source],
        )
        self.assertNotIn("[#", output)


class ExecuteCritiqueRepairTests(unittest.TestCase):
    """直接驱动修复单元，覆盖候选验收与失败降级。"""

    def call(self, chat, draft: str, **kwargs):
        budget = kwargs.pop(
            "budget", CritiqueBudget(max_total=10, per_file=5)
        )
        budget.begin_file("unit.srt")
        return (
            execute_critique_repair(
                chat,
                cue=1,
                source="お客様から私たちに触れることは禁止です",
                draft=draft,
                risk_codes={"customer_direction"},
                critique_budget=budget,
                finalize=lambda text, _source: text,
                accept=kwargs.pop("accept", lambda *args: True),
                **kwargs,
            ),
            budget,
        )

    def test_candidate_rejected_by_accept_hook(self) -> None:
        chat = CritiqueChat([], ["客人不能触碰我们"])
        text, budget = self.call(chat, "我们可以触碰客人", accept=lambda *a: False)
        self.assertEqual(text, "我们可以触碰客人")
        self.assertEqual([e.outcome for e in budget.file_events], ["rejected"])

    def test_budget_exhaustion_skips_request(self) -> None:
        chat = CritiqueChat([], ["客人不能触碰我们"])
        budget = CritiqueBudget(max_total=0, per_file=0)
        budget.begin_file("unit.srt")
        text = execute_critique_repair(
            chat,
            cue=1,
            source="お客様",
            draft="我们可以触碰客人",
            risk_codes={"customer_direction"},
            critique_budget=budget,
            finalize=lambda value, _source: value,
            accept=lambda *args: True,
        )
        self.assertEqual(text, "我们可以触碰客人")
        self.assertEqual(chat.critique_calls, [])
        self.assertEqual([e.outcome for e in budget.file_events], ["budget"])

    def test_no_critique_channel_degrades(self) -> None:
        text, budget = self.call(lambda prompt: prompt, "我们可以触碰客人")
        self.assertEqual(text, "我们可以触碰客人")
        self.assertEqual([e.outcome for e in budget.file_events], ["error"])


class CritiqueTemplateTests(unittest.TestCase):
    """模板契约：只认 {input}，且能通过 prompts 的加载校验。"""

    def test_template_contract(self) -> None:
        prompt_dir = ROOT / "config" / "prompts"
        text = prompts.load_task(prompt_dir, "translate_critique")
        self.assertEqual(sorted(prompts.placeholders(text)), ["input"])
        self.assertIn("【保持】", text)
        self.assertNotIn("task_translate_asmr", text)


if __name__ == "__main__":
    unittest.main()
