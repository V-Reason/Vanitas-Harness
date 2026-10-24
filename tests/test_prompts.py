"""提示词模板加载与变量替换。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import prompts  # noqa: E402
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
            ["agent", "chat", "code", "translate"],
        )
        for name in prompts.task_names(self.prompt_dir):
            text = prompts.load_task(self.prompt_dir, name)
            self.assertIn("{input}", text)

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
