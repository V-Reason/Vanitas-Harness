"""配置加载与校验。"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.config import load_config  # noqa: E402
from vanitas.errors import ConfigError  # noqa: E402

ENV_KEYS = ("VANITAS_PORT", "VANITAS_PROFILE", "VANITAS_CONFIG_DIR")


@contextmanager
def clean_env(**values):
    saved = {key: os.environ.get(key) for key in ENV_KEYS}
    for key in ENV_KEYS:
        os.environ.pop(key, None)
    for key, value in values.items():
        os.environ[key] = value
    try:
        yield
    finally:
        for key in ENV_KEYS:
            if saved[key] is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = saved[key]


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.config_dir = self.root / "config"
        shutil.copytree(ROOT / "config", self.config_dir)
        self.models = json.loads((self.config_dir / "models.json").read_text(encoding="utf-8"))
        self.settings = json.loads(
            (self.config_dir / "settings.json").read_text(encoding="utf-8")
        )

    def load(self):
        return load_config(self.config_dir, base_dir=self.root)

    def write_models(self) -> None:
        (self.config_dir / "models.json").write_text(
            json.dumps(self.models, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def write_settings(self) -> None:
        (self.config_dir / "settings.json").write_text(
            json.dumps(self.settings, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------ #

    def test_repo_config_loads(self) -> None:
        with clean_env():
            config = load_config()
        self.assertEqual(len(config.models), 3)
        self.assertEqual(
            [config.effective_port(model) for model in config.models],
            [4060, 4061, 4062],
        )
        qwen = config.model("qwen7b")
        self.assertEqual(qwen.shards, 2)
        self.assertEqual(qwen.path.name, "qwen2.5-7b-instruct-q6_k-00001-of-00002.gguf")
        self.assertEqual(qwen.alias, "qwen2.5-7b-instruct-q6_k")
        self.assertEqual(config.resolve_profile(qwen).key, "1")
        self.assertEqual(config.model_by_role("translate").id, "hymt")
        self.assertTrue(config.models[0].template_kwargs["enable_thinking"] is False)
        self.assertEqual(qwen.repeat_penalty, 1.0)

    def test_defaults_come_from_settings(self) -> None:
        with clean_env():
            config = load_config()
        self.assertEqual(config.settings.repl.max_context_tokens, 8192)
        self.assertEqual(config.settings.repl.context_reserve_tokens, 512)
        self.assertEqual(config.settings.repl.min_output_tokens, 256)
        self.assertEqual(config.settings.translate.chunk_chars, 1500)
        self.assertEqual(config.settings.translate.context_window, 5)
        self.assertEqual(config.settings.translate.min_output_tokens, 512)
        self.assertEqual(config.settings.translate.output_ratio, 2.5)
        self.assertEqual(config.settings.translate.prompt_safety_tokens, 64)
        self.assertEqual(config.settings.translate.monitor_lines, 5)
        self.assertTrue(config.settings.tools.exec_enabled)
        self.assertTrue(config.settings.tools.exec_require_confirm)
        self.assertIn("ren", config.settings.tools.exec_whitelist)

    def test_new_budget_fields_are_optional(self) -> None:
        del self.settings["repl"]["context_reserve_tokens"]
        del self.settings["repl"]["min_output_tokens"]
        del self.settings["translate"]["context_window"]
        del self.settings["translate"]["context_summary_chars"]
        del self.settings["translate"]["min_output_tokens"]
        del self.settings["translate"]["output_ratio"]
        del self.settings["translate"]["prompt_safety_tokens"]
        del self.settings["translate"]["monitor_lines"]
        self.write_settings()
        with clean_env():
            config = self.load()
        self.assertEqual(config.settings.repl.context_reserve_tokens, 512)
        self.assertEqual(config.settings.repl.min_output_tokens, 256)
        self.assertEqual(config.settings.translate.context_window, 5)
        self.assertEqual(config.settings.translate.context_summary_chars, 200)
        self.assertEqual(config.settings.translate.min_output_tokens, 512)
        self.assertEqual(config.settings.translate.output_ratio, 2.5)
        self.assertEqual(config.settings.translate.prompt_safety_tokens, 64)
        self.assertEqual(config.settings.translate.monitor_lines, 5)

    def test_repeat_penalty_below_one_rejected(self) -> None:
        self.models["models"][0]["repeat_penalty"] = 0.9
        self.write_models()
        with clean_env():
            with self.assertRaises(ConfigError):
                self.load()

    def test_duplicate_port_rejected(self) -> None:
        self.models["models"][1]["port"] = self.models["models"][0]["port"]
        self.write_models()
        with clean_env():
            with self.assertRaises(ConfigError) as ctx:
                self.load()
        self.assertIn("端口", str(ctx.exception))

    def test_unknown_field_rejected(self) -> None:
        self.models["models"][0]["ctx"] = 4096
        self.write_models()
        with clean_env():
            with self.assertRaises(ConfigError):
                self.load()

    def test_shard_entry_must_be_first(self) -> None:
        self.models["models"][0]["path"] = "models/qwen2.5-7b-instruct-q6_k-00002-of-00002.gguf"
        self.write_models()
        with clean_env():
            with self.assertRaises(ConfigError) as ctx:
                self.load()
        self.assertIn("第 1 个分片", str(ctx.exception))

    def test_missing_required_field(self) -> None:
        del self.models["models"][0]["alias"]
        self.write_models()
        with clean_env():
            with self.assertRaises(ConfigError):
                self.load()

    def test_bad_kv_type_rejected(self) -> None:
        self.models["models"][0]["profiles"]["1"]["kv_k"] = "q3_0"
        self.write_models()
        with clean_env():
            with self.assertRaises(ConfigError) as ctx:
                self.load()
        self.assertIn("KV 类型", str(ctx.exception))

    def test_bad_settings_number(self) -> None:
        self.settings["request"]["timeout_sec"] = 0
        self.write_settings()
        with clean_env():
            with self.assertRaises(ConfigError):
                self.load()

    def test_profile_lookup_and_unknown(self) -> None:
        with clean_env():
            config = load_config()
        qwen = config.model("qwen7b")
        profile = config.resolve_profile(qwen, "2")
        self.assertEqual((profile.ctx, profile.kv_k), (8192, "q4_0"))
        with self.assertRaises(ConfigError):
            config.resolve_profile(qwen, "9")

    def test_env_port_override(self) -> None:
        with clean_env():
            config = load_config()
        qwen = config.model("qwen7b")
        with clean_env(VANITAS_PORT="5050"):
            self.assertEqual(config.effective_port(qwen), 5050)
        with clean_env(VANITAS_PORT="abc"):
            with self.assertRaises(ConfigError):
                config.effective_port(qwen)

    def test_env_profile_override(self) -> None:
        with clean_env():
            config = load_config()
        qwen = config.model("qwen7b")
        with clean_env(VANITAS_PROFILE="2"):
            self.assertEqual(config.resolve_profile(qwen).key, "2")

    def test_env_config_dir(self) -> None:
        with clean_env(VANITAS_CONFIG_DIR=str(self.config_dir)):
            config = load_config()
        self.assertEqual(config.config_dir, self.config_dir.resolve())

    def test_missing_config_dir(self) -> None:
        with clean_env():
            with self.assertRaises(ConfigError):
                load_config(self.root / "nope", base_dir=self.root)

    # ------------------------------------------------------------------ #
    # settings.ui
    # ------------------------------------------------------------------ #

    def test_ui_default_theme_loads(self) -> None:
        with clean_env():
            config = load_config()
        self.assertEqual(config.settings.ui.color, "auto")
        self.assertEqual(config.settings.ui.theme.muted, "bright_black")
        self.assertEqual(config.settings.ui.theme.accent, "magenta")

    def test_ui_bad_color_mode_rejected(self) -> None:
        self.settings["ui"]["color"] = "yes"
        self.write_settings()
        with clean_env():
            with self.assertRaises(ConfigError) as ctx:
                self.load()
        self.assertIn("auto/always/never", str(ctx.exception))

    def test_ui_bad_color_name_rejected(self) -> None:
        self.settings["ui"]["theme"]["error"] = "purple"
        self.write_settings()
        with clean_env():
            with self.assertRaises(ConfigError) as ctx:
                self.load()
        self.assertIn("purple", str(ctx.exception))

    def test_ui_unknown_theme_key_rejected(self) -> None:
        self.settings["ui"]["theme"]["extra"] = "red"
        self.write_settings()
        with clean_env():
            with self.assertRaises(ConfigError) as ctx:
                self.load()
        self.assertIn("extra", str(ctx.exception))

    def test_ui_section_optional(self) -> None:
        # 旧 settings.json 没有 ui 段时不得报错，必须回落到内置默认主题
        del self.settings["ui"]
        self.write_settings()
        with clean_env():
            config = self.load()
        self.assertEqual(config.settings.ui.color, "auto")
        self.assertEqual(config.settings.ui.theme.success, "green")

    def test_ui_incomplete_theme_rejected(self) -> None:
        del self.settings["ui"]["theme"]["warn"]
        self.write_settings()
        with clean_env():
            with self.assertRaises(ConfigError) as ctx:
                self.load()
        self.assertIn("warn", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
