"""提示词模板加载与变量替换。"""

from __future__ import annotations

import hashlib
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import prompts  # noqa: E402
from vanitas.config import AGENT_ACTION_NAMES  # noqa: E402
from vanitas.errors import ConfigError  # noqa: E402


class PromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompt_dir = ROOT / "config" / "prompts"
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.temp_dir = Path(self._tmp.name)

    def test_all_templates_load(self) -> None:
        self.assertTrue(prompts.system_prompt(self.prompt_dir).strip())
        self.assertEqual(
            prompts.task_names(self.prompt_dir),
            [
                "agent",
                "chat",
                "code",
                "translate",
                "translate_asmr",
                "translate_asmr_experimental",
                "translate_critique",
            ],
        )
        for name in prompts.task_names(self.prompt_dir):
            text = prompts.load_task(self.prompt_dir, name)
            self.assertIn("{input}", text)

    def test_asmr_prompt_environments_are_byte_locked(self) -> None:
        production = (
            self.prompt_dir / "task_translate_asmr.md"
        ).read_bytes()
        experimental = (
            self.prompt_dir / "task_translate_asmr_experimental.md"
        ).read_bytes()
        self.assertEqual(len(production), 12427)
        self.assertEqual(
            hashlib.sha256(production).hexdigest().upper(),
            "A5E058802CC073E9D75BC0253CD378FD43983019D7415BCBAEFA6C5624233383",
        )
        self.assertEqual(len(experimental), 5399)
        self.assertEqual(
            hashlib.sha256(experimental).hexdigest().upper(),
            "7E589E421627645D622E353309039047FFD56F020824FB73DA3EA22F8B66C795",
        )
        self.assertNotEqual(production, experimental)
        self.assertEqual(
            prompts.placeholders(
                prompts.load_task(self.prompt_dir, "translate_asmr_experimental")
            ),
            {"input", "context_history"},
        )

    def test_asmr_template_helper_is_single_source(self) -> None:
        self.assertEqual(
            prompts.asmr_template("production"), "translate_asmr"
        )
        self.assertEqual(
            prompts.asmr_template("experimental"),
            "translate_asmr_experimental",
        )
        self.assertTrue(prompts.is_asmr_task("translate_asmr"))
        self.assertTrue(prompts.is_asmr_task("translate_asmr_experimental"))
        self.assertFalse(prompts.is_asmr_task("translate"))
        self.assertFalse(prompts.is_asmr_task(None))
        with self.assertRaises(ConfigError):
            prompts.asmr_template("bogus")

    def test_render_substitutes_all_placeholders(self) -> None:
        template = "日期 {date} / 模型 {model} / 上文 {context}\n{input}"
        rendered = prompts.render(
            template,
            {
                "date": "2026-10-25",
                "model": "demo",
                "context": "上一轮",
                "input": "你好",
            },
        )
        self.assertEqual(rendered, "日期 2026-10-25 / 模型 demo / 上文 上一轮\n你好")

    def test_render_defaults_context_to_empty(self) -> None:
        rendered = prompts.render("【{context}】{input}", {"input": "x"})
        self.assertEqual(rendered, "【】x")

    def test_render_default_date_has_second_precision(self) -> None:
        rendered = prompts.render("{date}")
        self.assertRegex(rendered, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

    def test_render_context_history_placeholder(self) -> None:
        rendered = prompts.render(
            "{context_history}|{input}",
            {"context_history": "前文", "input": "当前"},
        )
        self.assertEqual(rendered, "前文|当前")

    def test_render_cwd_placeholder(self) -> None:
        rendered = prompts.render(
            "目录 {cwd} / {input}",
            {"cwd": r"<WORKSPACE>\demo", "input": "x"},
        )
        self.assertEqual(rendered, "目录 T:\\demo / x")

    def test_task_chat_is_minimal(self) -> None:
        text = prompts.load_task(self.prompt_dir, "chat")
        self.assertIn("当前日期：{date}", text)
        self.assertIn("当前模型：{model}", text)
        self.assertIn("请直接回答", text)
        self.assertNotIn("复述", text)

    def test_asmr_prompt_contract(self) -> None:
        text = prompts.load_task(self.prompt_dir, "translate_asmr")
        self.assertIn("{input}", text)
        self.assertIn("{context_history}", text)
        self.assertIn("【ASR 瑕疵容错与语流校正】", text)
        self.assertIn("【拟声拟态词与动作音效规范】", text)
        self.assertIn("【强制输入/输出序号契约】", text)
        self.assertIn("チュッ", text)
        self.assertIn("はぁ", text)
        self.assertIn("[#序号]", text)

    def test_asmr_prompt_has_general_language_and_window_guard(self) -> None:
        text = prompts.load_task(self.prompt_dir, "translate_asmr")
        # V4 提示词重写后，语言/人称分节的措辞整体换过。这里断言的是
        # **现在真正生效**的条款，而不是旧版文案。
        # 已随重写消失、且**没有替代条款**的三条（审计记录，不静默删除）：
        #   1) 「默认使用简体中文 / 避免繁体」——V4 未再约束书写系统；
        #   2) 「专名全篇复用」——只剩「茜音必须译为茜音」这类单名约束；
        #   3) 「任何位置不得出现 [#N]」——并入 io 分节的「译文正文内绝对严禁残留」。
        # 前两条若仍要保证，应加回提示词或由确定性后处理兜底（TODO，未实现）。
        for phrase in (
            "<language_and_voice>",
            "彻底驱逐央视书面腔",
            "严禁出现“非常”",
            "全篇 100% 严禁出现“您”",
            "肉棒",                       # 器官归属常识
            "〜んですね",                 # 语法锚定：向对方求证而非自述
            "茜音",                       # 汉字专名优先
            "正文严禁合并，只译该编号内容",       # 跨行不预支（原「不借用相邻行内容」）
            "严禁无中生有增添原文没有的动作",      # 原「不要把泛指动作具体化」的替代
            "严禁音译为多字怪名",                 # 原「不音译成无意义的人名」的替代
        ):
            self.assertIn(phrase, text)
        # 提示词保持通用：不写死单部音频才出现的规则/专名。
        self.assertNotIn("禁止您", text)
        self.assertNotIn("美咲日和", text)
        self.assertNotIn("箱川诺亚", text)

    def test_asmr_prompt_has_sensory_and_crossline_rules(self) -> None:
        text = prompts.load_task(self.prompt_dir, "translate_asmr")
        for phrase in (
            "<sensory_immersion>",
            "贴耳",
            "化掉",
            "麻酥酥",
            "热气呼到耳边",
            "让人上瘾",
            "扑倒",
            "使坏",
            "接续词",            # V4 措辞：接续助词 → 接续词
            "原文没有提示",       # V4 措辞：原文没有任何括号 → 原文没有提示
        ):
            # 「不预支」已并入「正文严禁合并，只译该编号内容」，不再单独出现。

            self.assertIn(phrase, text)
        self.assertEqual(
            prompts.placeholders(text), {"input", "context_history"}
        )

    def test_asmr_prompt_has_term_direction_and_quote_rules(self) -> None:
        text = prompts.load_task(self.prompt_dir, "translate_asmr")
        # 「アマアマ（甘々）→甜腻，不是アマチュア」已在 V2 重写时移出提示词，
        # 该词义现由 config/glossary/asmr.tsv 的 `アマアマ→甜腻` 确定性兜底
        # （见 tests/test_quality.py::GlossaryTests）。此处只断言仍在提示词里的方向条款。
        for phrase in (
            "お客様から私たちに触れることは禁止",
            "是客人碰我们",              # V4 措辞（原「是客人→我们」）
            "私のことは雫でいいって",
            "转述第三人称",              # V4 措辞（原「引用他人」）
        ):
            self.assertIn(phrase, text)
        self.assertEqual(
            prompts.placeholders(text), {"input", "context_history"}
        )

    def test_asmr_prompt_rejects_common_katakana_and_artifacts(self) -> None:
        text = prompts.load_task(self.prompt_dir, "translate_asmr")
        # V4 精简时删掉了「普通片假名词不是专名」「コショコショ」「斜杠并列」三条文案。
        # 行为侧仍有确定性兜底：假名/SEXPRA 由 validate_cue_translation 拦截、
        # 中日文斜杠由 salvage_slash_translation 收敛（各有单测）。
        # 「普通片假名不是专名」这条**没有兜底**，只靠 lexicon 的逐词映射（TODO）。
        for phrase in (
            "SEXPAR",        # 严禁音译为 SEXPAR
            "Track",         # トラック / Track → 音轨 / 曲目
            "严禁音译",       # 机械音译黑名单的统称
        ):
            self.assertIn(phrase, text)
        self.assertNotIn("美咲日和", text)
        self.assertNotIn("箱川诺亚", text)

    def test_task_agent_enforces_sandbox_and_tool_priority(self) -> None:
        text = prompts.load_task(self.prompt_dir, "agent")
        self.assertIn("当前工作目录：{cwd}", text)
        self.assertIn("【安全沙箱】", text)
        self.assertIn("只能读写当前工作目录内的路径", text)
        self.assertIn("系统目录、其他盘符", text)
        self.assertIn("【工具选择】", text)
        self.assertIn("write_file", text)
        self.assertIn("read_file", text)
        self.assertIn("list_dir", text)
        self.assertIn("move_file", text)
        self.assertIn("delete_path", text)
        self.assertIn("delete_file", text)
        self.assertIn("python -m unittest", text)
        self.assertIn("run_shell_ro", text)
        self.assertIn("绝对不能尝试使用 shell 命令", text)
        self.assertIn("JSON 路径转义", text)
        self.assertIn("Invalid \\escape", text)
        self.assertIn("不要编造", text)
        self.assertIn("<tool_result>", text)
        self.assertIn("普通中文文本", text)
        self.assertIn("【输出格式强约束】", text)
        self.assertIn("<tool_call>", text)
        self.assertIn("必须且只能", text)
        self.assertIn("【任务逻辑纠偏】", text)
        self.assertIn("严禁", text)
        self.assertIn("【截断防御】", text)
        self.assertIn("如果写入内容过长", text)
        self.assertIn("【严禁前置闲聊】", text)
        self.assertIn("【修改文件必须调用 write_file 或 patch_file】", text)
        self.assertIn("patch_file", text)
        self.assertIn("read_file_range", text)
        self.assertIn("grep_workspace", text)
        self.assertIn("单轮批量提交", text)
        self.assertNotIn("原子", text)
        self.assertIn("git apply", text)
        self.assertIn("改扩展名", text)
        self.assertIn("move_file(src", text)

    def test_agent_continue_template_loads(self) -> None:
        text = prompts.load_template(self.prompt_dir, "agent_continue")
        self.assertIn("<tool_call>", text)
        self.assertIn("{cwd}", text)
        self.assertIn("{input}", text)
        self.assertIn("move_file", text)
        self.assertIn("patch_file", text)

    def test_system_prompt_is_static_and_rejects_placeholders(self) -> None:
        system = prompts.system_prompt(self.prompt_dir)
        self.assertEqual(prompts.placeholders(system), set())
        (self.temp_dir / "system_default.md").write_text(
            "静态占位符 {date}\n", encoding="utf-8"
        )
        with self.assertRaises(ConfigError) as ctx:
            prompts.system_prompt(self.temp_dir)
        self.assertIn("全静态", str(ctx.exception))

    def test_agent_prompt_matches_action_contract(self) -> None:
        text = prompts.load_task(self.prompt_dir, "agent")
        required = {
            "read_file": ("path",),
            "write_file": ("path", "content"),
            "patch_file": ("path", "search", "replace"),
            "read_file_range": ("path",),
            "grep_workspace": ("pattern",),
            "move_file": ("src", "dst"),
            "delete_path": ("path",),
            "delete_file": ("path",),
            "list_dir": (),
            "run_shell_ro": ("command",),
        }
        self.assertEqual(set(required), set(AGENT_ACTION_NAMES))
        for name, fields in required.items():
            self.assertIn(name, text)
            for field in fields:
                self.assertIn(field, text)
        for optional in ("start_line", "end_line", "max_results"):
            self.assertIn(optional, text)

    def test_agent_prompt_avoids_false_semantics(self) -> None:
        text = prompts.load_task(self.prompt_dir, "agent")
        for phrase in ("_reason", "recursive", "原子", "二次确认", "100%"):
            self.assertNotIn(phrase, text)
        self.assertIn("一次确认", text)
        self.assertIn("逐项执行", text)
        self.assertIn("不回滚", text)

    def test_dynamic_values_are_at_tail(self) -> None:
        cases = {
            "chat": ("{date}", "{model}"),
            "code": ("{date}", "{model}"),
            "agent": ("{cwd}", "{date}", "{model}"),
        }
        for name, variables in cases.items():
            with self.subTest(task=name):
                text = prompts.load_task(self.prompt_dir, name)
                self.assertIn("<runtime_context>", text)
                runtime = text.index("<runtime_context>")
                input_index = text.index("{input}")
                for variable in variables:
                    index = text.index(variable)
                    self.assertGreater(index, runtime)
                    self.assertLess(index, input_index)

    def test_asmr_prompt_uses_item_and_blank_id_contract(self) -> None:
        text = prompts.load_task(self.prompt_dir, "translate_asmr")
        for phrase in (
            "[#N]",
            "[#序号]",
            "编号项",
            "空",
            "不新增",
            "不合并",
            "不重排",
        ):
            self.assertIn(phrase, text)
        self.assertNotIn("输入 N 行", text)

    def test_translate_prompt_has_context_insulation_and_perspective_rules(self) -> None:
        text = prompts.load_task(self.prompt_dir, "translate")
        for phrase in (
            "参考上下文绝缘",
            "第一行",
            "第三人称",
            "豚汁",
            "红烧猪肉",
        ):
            self.assertIn(phrase, text)

    def test_asmr_prompt_has_negative_and_punctuation_rules(self) -> None:
        text = prompts.load_task(self.prompt_dir, "translate_asmr")
        for phrase in (
            "咕噜",
            "嗯姆",
            "哧溜",
            "……。",
            "買わないで",
            "（轻吻）",
        ):
            # 「離れないで」在 V4 精简 ASR 分节时被删（只剩「買わないで」一例）。
            # 同为挽留情境的近音还原，建议加回；当前无兜底。

            self.assertIn(phrase, text)

    def test_asmr_domain_lexicon_and_subject_direction(self) -> None:
        text = prompts.load_task(self.prompt_dir, "translate_asmr")
        for phrase in (
            "<asmr_domain_lexicon>",
            "梵天（ぼんてん）→ 梵天",
            "焚天",
            "膝枕（ひざまくら）→ 膝枕",
            "给你膝枕哦",
            "坐在膝枕上",
            "垫膝盖",
            "添寝",
            "甘えん坊",
            "爱撒娇",
            "オマケ",
            "特典",
            "お騒がせ",
            "给您添麻烦",
        ):
            # V4 的 lexicon 精简掉了「浅いところ/耳道浅处」「はむはむ/轻抿」
            # 「耳ふー/耳边轻吹」三组采耳方位与互动细目，也删了「不要过度意译」
            # 「不要加括号解释材质」。这些是真实覆盖损失，已记入 docs/项目现状 §52。

            self.assertIn(phrase, text)
        self.assertNotIn("羽毛绒球", text)
        self.assertNotIn("白毛球", text)
        self.assertEqual(
            prompts.placeholders(text), {"input", "context_history"}
        )
        self.assertLess(
            text.index("<asmr_domain_lexicon>"),
            text.index("<asr_repair_protocol>"),
        )

    def test_asmr_prompt_has_lookahead_insulation(self) -> None:
        text = prompts.load_task(self.prompt_dir, "translate_asmr")
        # V4 把后文绝缘写在 <reference_context> 的注释里，措辞从
        # 「禁止翻译 / 禁止写入当前编号」换成「后文仅供理解，不把示例整句…带入」。
        for phrase in ("后文参考", "后文仅供理解", "带入当前译文"):
            self.assertIn(phrase, text)
        self.assertEqual(
            prompts.placeholders(text), {"input", "context_history"}
        )

    def test_unknown_placeholder_in_template_rejected(self) -> None:
        (self.temp_dir / "bad.md").write_text("hi {unknown}\n", encoding="utf-8")
        with self.assertRaises(ConfigError) as ctx:
            prompts.load_template(self.temp_dir, "bad")
        self.assertIn("未知占位符", str(ctx.exception))

    def test_task_template_requires_input(self) -> None:
        (self.temp_dir / "task_x.md").write_text("只有 {date}\n", encoding="utf-8")
        with self.assertRaises(ConfigError) as ctx:
            prompts.load_template(self.temp_dir, "task_x")
        self.assertIn("{input}", str(ctx.exception))

    def test_render_unknown_variable_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            prompts.render("{input}", {"bogus": "x"})

    def test_template_name_traversal_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            prompts.load_template(self.temp_dir, "../settings")
        with self.assertRaises(ConfigError):
            prompts.load_template(self.temp_dir, "Bad-Name")

    def test_missing_template_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            prompts.load_template(self.temp_dir, "nope")


if __name__ == "__main__":
    unittest.main()
