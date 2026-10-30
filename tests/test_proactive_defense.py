"""系统性"打提前枪"防御的离线单测：五道防线各一组。

覆盖范围与设计口径：

1. 协议残渣：不测"某一种写法被修好了"，而是把整类形态（单边/全角/Markdown/
   列表/粘连标点）一次性钉住，并**反向**钉住"不许过度清洗"的边界——
   正文里的年份括号与无井号单边残渣必须原样保留。
2. 标点语流：收敛方向 + 长度分界 + 引号内侧 + 关闭开关。
3. ASR 音近词族：表结构不变量（forbidden 不得是 preferred 子串）与真实收敛。
4. 元数据先验：提取、注入、锁定、以及"普通同音词绝不能被收成专名"的反向断言。
5. 评测抗噪：多轮均值/标准差、跨轮翻转识别、N=1 时的噪声声明。

全部用例不触网、不写仓库文件。
"""

from __future__ import annotations

import hashlib
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import qc_srt  # noqa: E402

from tests.benchmarks import bench_semantic_quality as bench  # noqa: E402
from vanitas.quality import Glossary  # noqa: E402
from vanitas.translate import (  # noqa: E402
    DynamicEntityEngine,
    WINDOW_ID_ANY_RE,
    extract_metadata_entity_seeds,
    normalize_punctuation,
    postprocess_cue_text,
    strip_window_id_prefix,
    translate_srt,
)

GLOSSARY = ROOT / "config" / "glossary" / "asmr.tsv"
PROMPT_TEMPLATE = ROOT / "config" / "prompts" / "task_translate_asmr.md"


def srt(*cues: str) -> str:
    blocks = []
    for index, text in enumerate(cues, start=1):
        start = f"00:00:{index * 2 - 1:02d},000"
        end = f"00:00:{index * 2:02d},000"
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


def parse_batch(prompt: str) -> list[tuple[str, str]]:
    """从批次 prompt 里抠出 ``[#N] 原文`` 条目；假引擎共用这一个解析器。"""
    items: list[tuple[str, str]] = []
    current: str | None = None
    buffer: list[str] = []
    for line in prompt.splitlines():
        match = re.match(r"^\[#([^\]]+)\]\s*(.*)$", line)
        if match:
            if current is not None:
                items.append((current, "\n".join(buffer).strip()))
            current = match.group(1)
            buffer = [match.group(2)]
        elif current is not None:
            buffer.append(line)
    if current is not None:
        items.append((current, "\n".join(buffer).strip()))
    return items


# --------------------------------------------------------------------------- #
# 防线 1：协议残渣全形态
# --------------------------------------------------------------------------- #
RESIDUE_FORMS = (
    "[#84]",
    "#84]",
    "[#84",
    "(#84)",
    "【#84】",
    "［#84］",
    "〔#84〕",
    "**[#84]**",
    "*#84]*",
    "1. [#84]",
    "- [#84]",
    "[#84]:",
    "[#84]：",
    "#84.",
    "[84]",
)


class ProtocolResidueTests(unittest.TestCase):
    def test_every_residue_form_is_stripped(self) -> None:
        for form in RESIDUE_FORMS:
            with self.subTest(form=form):
                text = postprocess_cue_text(f"{form} 你好", "こんにちは")
                self.assertEqual(text, "你好", form)
                self.assertIsNone(WINDOW_ID_ANY_RE.search(text), form)

    def test_residue_inside_body_is_removed(self) -> None:
        text = postprocess_cue_text("你好[#84]世界", "こんにちは")
        self.assertEqual(text, "你好世界")

    def test_residue_on_separate_line_does_not_join_lines(self) -> None:
        text = postprocess_cue_text("第一行\n[#84] 第二行", "こんにちは")
        self.assertEqual(text, "第一行\n第二行")

    def test_bare_number_is_only_residue_at_line_start(self) -> None:
        # 行首成对括号的纯数字是批次前缀
        self.assertEqual(
            postprocess_cue_text("[84] 你好", "こんにちは"), "你好"
        )
        # 正文里的括号数字是内容，不得吞掉（过度清洗 = 丢字）
        self.assertEqual(
            postprocess_cue_text("那是（2024）年的事", "それは2024年の話"),
            "那是（2024）年的事",
        )
        # 无井号的单边残渣与正文数字不可区分，刻意不纳入清洗
        self.assertEqual(
            postprocess_cue_text("84] 你好", "こんにちは"), "84] 你好"
        )

    def test_strip_window_id_prefix_covers_variants(self) -> None:
        for form in RESIDUE_FORMS:
            with self.subTest(form=form):
                self.assertEqual(strip_window_id_prefix(f"{form} 开头"), "开头")
        self.assertEqual(
            strip_window_id_prefix("译文 [#12] 内容 [#13]"), "译文内容"
        )

    def test_residue_only_cue_becomes_empty(self) -> None:
        # 真实语料（V2 · 05_口交绘画 · cue 83）：整行内容就是一个 "#84]"。
        # 清洗后必须为空——回退成原文会让残渣落盘，并且"非空原文→空译文"
        # 的重译/QC 信号永远不会触发。
        self.assertEqual(postprocess_cue_text("#84]", "なにか"), "")
        self.assertEqual(postprocess_cue_text("[#12]", "なにか"), "")

    def test_markdown_wrapped_id_does_not_leave_markers(self) -> None:
        self.assertEqual(
            postprocess_cue_text("**[#7]** 谢谢", "ありがとう"), "谢谢"
        )
        self.assertEqual(
            postprocess_cue_text("*#7]* 谢谢", "ありがとう"), "谢谢"
        )


class QcResidueTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def pair(self, source: str, translated: str) -> qc_srt.FileQC:
        src = self.dir / "a.srt"
        zh = self.dir / "a.zh.srt"
        src.write_text(source, encoding="utf-8")
        zh.write_text(translated, encoding="utf-8")
        return qc_srt.check_pair(src, zh)

    def test_qc_flags_every_residue_form(self) -> None:
        for form in RESIDUE_FORMS:
            with self.subTest(form=form):
                result = self.pair(
                    srt("こんにちは"), srt(f"{form} 你好")
                )
                codes = {item.code for item in result.errors}
                self.assertIn("window_id", codes, form)

    def test_qc_accepts_clean_translation(self) -> None:
        result = self.pair(srt("こんにちは"), srt("你好"))
        self.assertEqual(result.errors, [])


# --------------------------------------------------------------------------- #
# 防线 2：标点语流归一
# --------------------------------------------------------------------------- #
class PunctuationTests(unittest.TestCase):
    def test_ellipsis_forms_converge_to_standard(self) -> None:
        for raw in ("嗯。。。", "嗯...", "嗯‥‥", "嗯…", "嗯……"):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_punctuation(raw), "嗯……")

    def test_double_terminal_mark_drops_period(self) -> None:
        cases = {
            "好舒服……。": "好舒服……",
            "不要—。": "不要—",
            "好开心~。": "好开心~",
            "真的吗？。": "真的吗？",
            "太好了！。": "太好了！",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_punctuation(raw), expected)

    def test_repeated_connectors_fold(self) -> None:
        # 只关掉"剥短句句号"，单纯验证折叠方向。
        def fold(raw: str) -> str:
            return normalize_punctuation(raw, strip_trailing_period=False)

        cases = {
            "好舒服，，快点": "好舒服，快点",
            "等等、、": "等等、",
            "嗯；；好吧": "嗯；好吧",
            "舒服。。": "舒服。",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(fold(raw), expected)
        # 折叠与剥离串联时：重复句号先折叠，再由短句规则剥掉最后一个。
        self.assertEqual(normalize_punctuation("舒服。。"), "舒服")

    def test_tilde_normalized_to_half_width(self) -> None:
        self.assertEqual(normalize_punctuation("呼〜", ), "呼~")
        self.assertEqual(normalize_punctuation("这个是1～2秒"), "这个是1~2秒")

    def test_short_line_trailing_period_stripped(self) -> None:
        self.assertEqual(
            postprocess_cue_text("转过来看着我嘛。", "こっち向いて"), "转过来看着我嘛"
        )
        self.assertEqual(
            postprocess_cue_text("呐，看着我嘛。", "見て"), "呐，看着我嘛"
        )

    def test_long_line_trailing_period_kept(self) -> None:
        long_line = (
            "她把头发轻轻撩到耳后，慢慢凑近你的耳边，"
            "用几乎听不见的声音说了一句话，然后就再也没有开口，"
            "只是安静地等着你的回应。"
        )
        self.assertGreater(len(long_line), 30)
        self.assertEqual(
            postprocess_cue_text(long_line, "こっち向いて"), long_line
        )

    def test_quoted_line_trailing_period_stripped(self) -> None:
        self.assertEqual(
            postprocess_cue_text("「转过来看着我嘛。」", "こっち向いて"),
            "「转过来看着我嘛」",
        )
        self.assertEqual(
            postprocess_cue_text("『不要嘛。』", "だめ"),
            "『不要嘛』",
        )

    def test_strip_can_be_switched_off(self) -> None:
        self.assertEqual(
            normalize_punctuation("转过来看着我嘛。", strip_trailing_period=False),
            "转过来看着我嘛。",
        )
        self.assertEqual(
            postprocess_cue_text(
                "转过来看着我嘛。", "こっち向いて", strip_trailing_period=False
            ),
            "转过来看着我嘛。",
        )

    def test_deliberate_emphasis_is_not_folded(self) -> None:
        for raw in ("不行！！", "是吗？？", "啊、啊、啊"):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_punctuation(raw), raw)

    def test_punctuation_only_cue_is_not_wiped(self) -> None:
        # 归一后只剩空串时，后处理必须回退到原文，不能凭空把 cue 写空。
        self.assertEqual(postprocess_cue_text("。", "こっち向いて"), "。")


# --------------------------------------------------------------------------- #
# 防线 3：ASR 音近词族
# --------------------------------------------------------------------------- #
class PhoneticLexiconTests(unittest.TestCase):
    def setUp(self) -> None:
        self.glossary = Glossary.load(GLOSSARY)

    def test_table_is_three_columns_without_duplicate_keys(self) -> None:
        raw = GLOSSARY.read_text(encoding="utf-8")
        keys: list[str] = []
        for line_no, line in enumerate(raw.splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cells = line.split("\t")
            self.assertEqual(len(cells), 3, f"第 {line_no} 行不是 3 列")
            keys.append(cells[0].strip())
        self.assertEqual(len(keys), len(set(keys)))

    def test_preferred_never_contains_forbidden(self) -> None:
        """子串自碰撞会让正确译文被改坏（"不是旅馆" → "不不是旅馆"）。"""
        for entry in self.glossary.entries:
            for forbidden in entry.forbidden:
                with self.subTest(source=entry.source, forbidden=forbidden):
                    self.assertNotIn(forbidden, entry.preferred)

    def test_shiofuki_converges(self) -> None:
        self.assertEqual(
            self.glossary.apply("塩吹きが止まらない", "又在撒盐了，停不下来"),
            "又在潮吹了，停不下来",
        )
        self.assertEqual(
            self.glossary.apply("塩吹きが止まらない", "又在放盐了"),
            "又在潮吹了",
        )

    def test_ear_cleaning_term_converges(self) -> None:
        self.assertEqual(
            self.glossary.apply("耳かきしましょうね", "来掏耳朵吧"),
            "来采耳吧",
        )

    def test_common_words_are_untouched(self) -> None:
        # 源文没命中词条时不得改动；命中但译文本来就对时也不得改动。
        self.assertEqual(
            self.glossary.apply("今日はいい天気", "又在撒盐了"), "又在撒盐了"
        )
        self.assertEqual(
            self.glossary.apply("塩吹き", "今天潮吹了"), "今天潮吹了"
        )

    def test_politeness_is_not_stripped_by_glossary(self) -> None:
        # 「您」的口径由 quality.apply_pronoun_policy 按文档语域决定，
        # 词表层不得抢跑（服务语境要保留敬称）。
        self.assertEqual(
            self.glossary.apply("お耳が敏感ですね", "您的耳朵很敏感呢"),
            "您的耳朵很敏感呢",
        )

    def test_rejected_negation_entry_stays_out(self) -> None:
        # 否定句不能用子串替换表达：preferred 含 forbidden 会产出"不不是"。
        sources = {entry.source for entry in self.glossary.entries}
        self.assertNotIn("ホテルじゃ", sources)


# --------------------------------------------------------------------------- #
# 防线 4：文件名元数据先验
# --------------------------------------------------------------------------- #
class MetadataSeedTests(unittest.TestCase):
    def test_filename_bracket_yields_name(self) -> None:
        self.assertEqual(
            extract_metadata_entity_seeds(
                Path("<WORKSPACE>/ASMR/朝のひとりごと(茜音編).srt")
            ),
            ["茜音"],
        )

    def test_parent_directory_seed_is_used(self) -> None:
        self.assertEqual(
            extract_metadata_entity_seeds(Path("<WORKSPACE>/ASMR/【双叶姐妹】/track01.srt")),
            ["双叶"],
        )

    def test_bare_stem_with_suffix_yields_name(self) -> None:
        self.assertEqual(
            extract_metadata_entity_seeds(Path("<WORKSPACE>/ASMR/茜音編.wav")), ["茜音"]
        )
        self.assertEqual(
            extract_metadata_entity_seeds(Path("<WORKSPACE>/ASMR/RJ123456_あかね編.srt")),
            ["あかね"],
        )

    def test_junk_is_rejected(self) -> None:
        for raw in (
            "<WORKSPACE>/ASMR/【本編】/01_track.srt",
            "<WORKSPACE>/ASMR/おはよう.srt",
            "<WORKSPACE>/ASMR/効果音【実演】.srt",
            "<WORKSPACE>/ASMR/plain.srt",
            "<WORKSPACE>/ASMR/01_口交绘画.srt",
        ):
            with self.subTest(raw=raw):
                self.assertEqual(extract_metadata_entity_seeds(Path(raw)), [])


class SeedInjectionTests(unittest.TestCase):
    def test_seed_is_prepended_and_locked(self) -> None:
        engine = DynamicEntityEngine(
            ["赤音ちゃん、おはよう", "赤音ちゃん、聞こえる？"]
        )
        accepted = engine.seed_prior_entities(["茜音"])

        self.assertEqual(accepted, ["茜音"])
        self.assertEqual(engine._canonical_order[0], "茜音")
        self.assertEqual(engine.locked_translations["茜音"], "茜音")
        self.assertIn("赤音", engine._variants["茜音"])
        self.assertEqual(
            engine.enforce_locks("赤音，早上好"), "茜音，早上好"
        )

    def test_common_homophone_is_never_captured(self) -> None:
        # 「声音」与「茜音」同长且编辑距离 1，但它出现在普通词位置，
        # 一旦被收成变体就会把"声音"全篇改成"茜音"。
        engine = DynamicEntityEngine(["声音が好きだな", "この声音、いいね"])
        engine.seed_prior_entities(["茜音"])

        self.assertEqual(engine._variants["茜音"], {"茜音"})
        self.assertEqual(engine.enforce_locks("我听见了声音"), "我听见了声音")

    def test_kana_seed_is_registered_but_not_locked(self) -> None:
        engine = DynamicEntityEngine(["あかねちゃん、おはよう"])
        engine.seed_prior_entities(["あかね"])

        self.assertIn("あかね", engine._canonical_order)
        self.assertNotIn("あかね", engine.locked_translations)
        # 假名不能当锁定译名，否则会把假名注回中文字幕
        self.assertEqual(engine.enforce_locks("早上好"), "早上好")

    def test_seed_does_not_override_existing_entity(self) -> None:
        engine = DynamicEntityEngine(["梅地様、おはよう"])
        self.assertIn("梅地", engine.canonical_map)
        accepted = engine.seed_prior_entities(["梅地"])

        self.assertEqual(accepted, [])

    def test_stopword_seed_is_ignored(self) -> None:
        engine = DynamicEntityEngine(["こんにちは"])
        self.assertEqual(
            engine.seed_prior_entities(["セックス", "", "  "]), []
        )


class SeedPipelineTests(unittest.TestCase):
    """端到端：文件名种子必须能纠正 ASR 的同音错字。

    注意作用域：文档级实体只有在合流窗口（``window_size>1``）才建立，
    单句路径刻意 ``set_entity_engine(None)``。因此这里用窗口 2 走生产路径。
    """

    SOURCE = srt("赤音ちゃん、おはよう", "赤音ちゃん、聞こえる？")

    @staticmethod
    def chat(prompt: str) -> str:
        rendered = []
        for cue_id, text in parse_batch(prompt):
            translation = "赤音，能听见吗？" if "聞こえる" in text else "赤音，早上好"
            rendered.append(f"[#{cue_id}] {translation}")
        return "\n".join(rendered) or "赤音，早上好"

    def run_with(self, source_name: str) -> str:
        failures: list = []
        return translate_srt(
            self.chat,
            self.SOURCE,
            chunk_chars=800,
            source=source_name,
            eol="\n",
            failures=failures,
            window_size=2,
        )

    def test_filename_seed_fixes_asr_name(self) -> None:
        output = self.run_with("朝のひとりごと(茜音編).srt")
        self.assertIn("茜音", output)
        self.assertNotIn("赤音", output)

    def test_without_seed_behaviour_is_unchanged(self) -> None:
        output = self.run_with("plain.srt")
        self.assertIn("赤音", output)
        self.assertNotIn("茜音", output)


# --------------------------------------------------------------------------- #
# 防线 5：评测抗噪网格
# --------------------------------------------------------------------------- #
class _BenchChat:
    """按 [#N] 回放的假引擎：把源文换成预设译文，未登记的原样返回。"""

    task_template = "translate_asmr"
    glossary = None
    join_rules = ()

    def __init__(self, script: dict[str, str]) -> None:
        self.script = dict(script)

    def __call__(self, prompt: str) -> str:
        items = parse_batch(prompt)
        if not items:
            # 单句重试通道：返回空串，写回硬校验会拒绝，保留原译。
            return ""
        return "\n".join(
            f"[#{cue_id}] {self.script.get(text, text)}"
            for cue_id, text in items
        )


BENCH_CUES = (
    bench.GoldenCue(
        id="CAT_A_901",
        category="direction",
        source="お耳をマッサージしますね",
        context_cues=(),
        expected_pattern="耳朵按摩",
        forbidden_pattern="",
        reference_zh="好，给你做耳朵按摩",
    ),
    bench.GoldenCue(
        id="CAT_A_902",
        category="direction",
        source="お客様、ありがとうございます",
        context_cues=(),
        expected_pattern="客人",
        forbidden_pattern="",
        reference_zh="谢谢客人",
    ),
)
BENCH_SCRIPT = {
    "お耳をマッサージしますね": "好，给你做耳朵按摩",
    "お客様、ありがとうございます": "谢谢客人",
}
BENCH_DEGRADED = {
    "お耳をマッサージしますね": "好，给你做耳朵按摩",
    "お客様、ありがとうございます": "完全无关的内容",
}


class RepeatGridTests(unittest.TestCase):
    def run_repeated(self, scripts: list[dict[str, str]]) -> bench.RepeatReport:
        calls: list[int] = []

        def factory(on_stat):
            calls.append(len(calls))
            script = scripts[len(calls) - 1]
            return _BenchChat(script)

        return bench.run_benchmark_repeated(
            BENCH_CUES, factory, repeat=len(scripts), mode="baseline", window=2
        )

    def test_stable_run_has_zero_spread(self) -> None:
        report = self.run_repeated([BENCH_SCRIPT, BENCH_SCRIPT])

        self.assertEqual(report.repeat, 2)
        self.assertEqual(report.run_rates, [1.0, 1.0])
        self.assertAlmostEqual(report.mean_rate, 1.0)
        self.assertAlmostEqual(report.std_rate, 0.0)
        self.assertEqual(report.unstable_cues(), [])

    def test_degraded_run_is_captured_as_spread(self) -> None:
        report = self.run_repeated([BENCH_SCRIPT, BENCH_DEGRADED])

        self.assertEqual(report.run_rates, [1.0, 0.5])
        self.assertAlmostEqual(report.mean_rate, 0.75)
        self.assertAlmostEqual(report.std_rate, 0.35355, places=4)
        rates = report.cue_pass_rates()
        self.assertEqual(rates["CAT_A_901"], 1.0)
        self.assertEqual(rates["CAT_A_902"], 0.5)
        self.assertEqual(
            [cue_id for cue_id, _rate in report.unstable_cues()],
            ["CAT_A_902"],
        )

    def test_repeat_must_be_positive(self) -> None:
        with self.assertRaises(bench.BenchDataError):
            bench.run_benchmark_repeated(
                BENCH_CUES, lambda _on_stat: _BenchChat(BENCH_SCRIPT), repeat=0
            )

    def test_summary_states_noise_floor(self) -> None:
        single = self.run_repeated([BENCH_SCRIPT])
        text = bench.render_repeat_summary(single)
        self.assertIn("±6pp", text)
        self.assertIn("--repeat 3", text)

    def test_summary_reports_mean_and_spread(self) -> None:
        report = self.run_repeated([BENCH_SCRIPT, BENCH_DEGRADED])
        text = bench.render_repeat_summary(report)
        self.assertIn("Mean Pass Rate: 75.0%", text)
        self.assertIn("(N=2)", text)
        self.assertIn("跨轮翻转的 cue 1 条", text)

        payload = bench.repeat_payload(report)
        self.assertEqual(payload["n"], 2)
        self.assertAlmostEqual(payload["mean_rate"], 0.75)
        self.assertEqual(payload["unstable_cues"], ["CAT_A_902"])
        self.assertEqual(len(payload["runs"]), 2)


# --------------------------------------------------------------------------- #
# 红线：零线程 / 生产模板未被本批改动
# --------------------------------------------------------------------------- #
class RedLineTests(unittest.TestCase):
    def test_translate_layer_has_no_threading(self) -> None:
        """DoD 口径：``vanitas/translate/`` 内不得出现任何线程/进程并发。

        仓库其它层历史上有 REPL watchdog 的 ``threading.Thread``，本批不得
        新增，也不得把它带进翻译管线——这里只锁本批负责的层。
        """
        forbidden = (
            "threading.Thread",
            "import threading",
            "concurrent.futures",
            "multiprocessing",
        )
        offenders: list[str] = []
        for path in (ROOT / "vanitas" / "translate").rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path.name}: {token}")
        for path in (
            ROOT / "scripts" / "qc_srt.py",
            ROOT / "tests" / "benchmarks" / "bench_semantic_quality.py",
        ):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path.name}: {token}")
        self.assertEqual(offenders, [])

    def test_production_prompt_template_unchanged(self) -> None:
        raw = PROMPT_TEMPLATE.read_bytes()
        self.assertEqual(len(raw), 12784)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest()[:8].upper(), "CC2B4C82"
        )


if __name__ == "__main__":
    unittest.main()
