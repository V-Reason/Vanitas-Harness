"""运行环境解析：production/experimental 与 preset 模板隔离。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import environments  # noqa: E402
from vanitas.errors import ConfigError  # noqa: E402


class EnvironmentTests(unittest.TestCase):
    def test_named_profiles_and_seam_fields(self) -> None:
        self.assertEqual(
            environments.get_environment("production").name, "production"
        )
        self.assertEqual(
            environments.get_environment("experimental").name, "experimental"
        )
        self.assertEqual(
            environments.get_environment(None).name, "production"
        )
        self.assertEqual(
            environments.ENVIRONMENT_NAMES,
            frozenset({"production", "experimental"}),
        )
        # 未来 pipeline 替换的 seam 当前保持默认，不影响现有管线。
        for profile in environments.ENVIRONMENTS:
            self.assertEqual(profile.pipeline, "translate")
            self.assertIsNone(profile.pipeline_factory)
            self.assertEqual(profile.hooks, ())

    def test_invalid_environment_is_readable_config_error(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            environments.get_environment("bogus")
        self.assertIn("未知运行环境", str(ctx.exception))

    def test_production_keeps_all_preset_templates(self) -> None:
        profile = environments.PRODUCTION
        for preset, template in (
            ("subtitle", "translate"),
            ("prose", "translate"),
            ("general", "translate"),
            ("asmr", "translate_asmr"),
        ):
            with self.subTest(preset=preset):
                self.assertEqual(
                    profile.resolve_task_template(preset, template), template
                )

    def test_experimental_overrides_asmr_only(self) -> None:
        profile = environments.EXPERIMENTAL
        self.assertEqual(
            profile.resolve_task_template("asmr", "translate_asmr"),
            "translate_asmr_experimental",
        )
        for preset in ("subtitle", "prose", "general"):
            with self.subTest(preset=preset):
                with self.assertRaises(ConfigError) as ctx:
                    profile.resolve_task_template(preset, "translate")
                self.assertIn(
                    "experimental 环境未实现", str(ctx.exception)
                )
