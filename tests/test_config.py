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
        self.assertEqual(len(config.models), 4)
        self.assertEqual(
            [config.effective_port(model) for model in config.models],
            [4060, 4063, 4061, 4062],
        )
        qwen = config.model("qwen7b")
        self.assertEqual(qwen.shards, 2)
        self.assertEqual(qwen.path.name, "qwen2.5-7b-instruct-q6_k-00001-of-00002.gguf")
        self.assertEqual(qwen.alias, "qwen2.5-7b-instruct-q6_k")
        self.assertEqual(config.resolve_profile(qwen).key, "1")
        self.assertEqual(config.model_by_role("translate").id, "hymt")
        self.assertEqual(config.model_by_role("general").id, "qwen_coder")
        coder = config.model("qwen_coder")
        self.assertEqual(coder.shards, 1)
        self.assertEqual(coder.repeat_penalty, 1.05)
        self.assertEqual(
            [coder.profiles[key].ctx for key in coder.profile_keys],
            [8192, 16384, 32768],
        )
        self.assertEqual(coder.profiles["1"].kv_k, "q8_0")
        self.assertEqual(coder.profiles["2"].kv_k, "q4_0")
        self.assertTrue(config.models[0].template_kwargs["enable_thinking"] is False)
        self.assertEqual(qwen.repeat_penalty, 1.1)

    def test_default_model_routes_and_custom_values(self) -> None:
        with clean_env():
            config = self.load()
        self.assertEqual(config.default_model("agent").id, "qwen_coder")
        self.assertEqual(config.default_model("chat").id, "qwen7b")
        self.assertEqual(config.default_model("translate").id, "hymt")
        self.assertEqual(config.default_model("batch").id, "qwen7b")

        self.settings["defaults"]["chat"] = "qwen_coder"
        self.settings["defaults"]["agent"] = "qwen7b"
        self.write_settings()
        with clean_env():
            config = self.load()
        self.assertEqual(config.default_model("agent").id, "qwen7b")
        self.assertEqual(config.default_model("chat").id, "qwen_coder")

    def test_unknown_default_model_is_rejected(self) -> None:
        self.settings["defaults"]["chat"] = "missing-model"
        self.write_settings()
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        self.assertIn("settings.defaults.chat", str(ctx.exception))
        self.assertIn("missing-model", str(ctx.exception))

    def test_missing_defaults_falls_back_to_roles(self) -> None:
        self.settings.pop("defaults")
        self.write_settings()
        with clean_env():
            config = self.load()
        self.assertEqual(config.default_model("agent").id, "qwen_coder")
        self.assertEqual(config.default_model("chat").id, "qwen_coder")
        self.assertEqual(config.default_model("translate").id, "hymt")
        self.assertEqual(config.default_model("batch").id, "qwen_coder")

    def test_new_operational_settings_are_validated(self) -> None:
        self.settings["agent"]["stagnation_limit"] = 1
        self.write_settings()
        with self.assertRaises(ConfigError):
            self.load()

        self.settings["agent"]["stagnation_limit"] = 3
        self.settings["repl"]["ctx_warn_pct"] = 80
        self.settings["repl"]["ctx_danger_pct"] = 60
        self.write_settings()
        with self.assertRaises(ConfigError):
            self.load()

        self.settings["repl"]["ctx_warn_pct"] = 60
        self.settings["repl"]["ctx_danger_pct"] = 80
        self.settings["tools"]["max_capture_bytes"] = 512
        self.write_settings()
        with self.assertRaises(ConfigError):
            self.load()

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
        self.assertFalse(hasattr(config.settings.tools, "exec_whitelist"))
        self.assertEqual(config.settings.chat.temperature, 0.2)
        self.assertEqual(config.settings.agent.max_steps, 10)
        self.assertEqual(config.settings.agent.max_batch_actions, 100)
        self.assertEqual(config.settings.agent.max_feedback_chars, 8000)
        self.assertEqual(
            config.settings.agent.confirm_actions,
            frozenset(
                {
                    "write_file",
                    "patch_file",
                    "delete_file",
                    "delete_path",
                    "move_file",
                }
            ),
        )
        self.assertIn("read_file_range", config.settings.agent.auto_actions)
        self.assertIn("grep_workspace", config.settings.agent.auto_actions)
        self.assertEqual(config.settings.agent.action_aliases["shell"], "run_shell_ro")
        self.assertEqual(config.settings.tools.max_read_bytes, 262144)
        self.assertEqual(config.settings.tools.max_list_entries, 200)
        self.assertEqual(config.settings.tools.max_patch_bytes, 262144)
        self.assertEqual(config.settings.tools.max_range_lines, 400)
        self.assertEqual(config.settings.tools.max_grep_results, 50)
        self.assertEqual(config.settings.tools.max_grep_file_bytes, 1048576)
        self.assertIn(".git", config.settings.tools.grep_ignore_dirs)
        self.assertEqual(config.settings.repl.context_collapse_ratio, 0.65)
        self.assertEqual(config.settings.repl.context_collapse_keep_rounds, 2)
        self.assertEqual(config.settings.repl.context_collapse_min_chars, 1000)
        self.assertEqual(config.settings.repl.ctx_warn_pct, 60)
        self.assertEqual(config.settings.repl.ctx_danger_pct, 80)
        self.assertEqual(config.settings.repl.status_bar_width, 20)
        self.assertEqual(config.settings.agent.max_parse_retries, 2)
        self.assertEqual(config.settings.agent.stagnation_limit, 3)
        self.assertEqual(config.settings.agent.feedback_head_lines, 20)
        self.assertEqual(config.settings.agent.feedback_tail_lines, 40)
        self.assertEqual(config.settings.tools.max_capture_bytes, 524288)
        self.assertEqual(config.settings.defaults.agent, "qwen_coder")
        self.assertEqual(config.settings.defaults.chat, "qwen7b")
        self.assertEqual(config.settings.defaults.translate, "hymt")
        self.assertEqual(config.settings.defaults.batch, "qwen7b")
        self.assertEqual(
            {
                mode: config.default_model(mode).id
                for mode in ("agent", "chat", "translate", "batch")
            },
            {
                "agent": "qwen_coder",
                "chat": "qwen7b",
                "translate": "hymt",
                "batch": "qwen7b",
            },
        )
        self.assertEqual(config.settings.tools.move_illegal_chars, '#<>"|?*')
        self.assertIn("git status", config.settings.tools.manual_exec_prefixes)
        self.assertIn("del", config.settings.tools.destructive_commands)
        self.assertEqual(config.settings.batch.max_tokens_limit, 8192)
        self.assertEqual(config.settings.translate.max_split_depth, 3)
        self.assertEqual(config.settings.translate.default_preset, "subtitle")
        self.assertEqual(
            config.settings.translate.presets["general"]["task_template"],
            "translate",
        )
        self.assertEqual(
            config.settings.translate.presets["asmr"]["task_template"],
            "translate_asmr",
        )
        self.assertEqual(
            config.settings.translate.presets["subtitle"]["profile"], "2"
        )
        self.assertEqual(
            config.settings.translate.presets["subtitle"]["context_window"], 12
        )
        self.assertEqual(
            config.settings.translate.presets["subtitle"]["chunk_chars"], 800
        )
        self.assertEqual(
            config.settings.translate.presets["prose"]["context_window"], 10
        )
        self.assertEqual(
            config.settings.translate.presets["general"]["context_window"], 5
        )
        self.assertEqual(config.settings.request.progress_interval_ms, 200)
        self.assertEqual(config.settings.launcher.log_tail_lines, 15)
        self.assertEqual(config.settings.ui.spinner_interval_ms, 300)

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

    def test_new_operational_fields_are_optional(self) -> None:
        for section in ("chat", "agent", "batch"):
            del self.settings[section]
        for key in (
            "health_timeout_sec",
            "status_probe_timeout_sec",
            "progress_interval_ms",
        ):
            del self.settings["request"][key]
        del self.settings["launcher"]["log_tail_lines"]
        for key in (
            "max_read_bytes",
            "max_list_entries",
            "move_illegal_chars",
            "manual_exec_prefixes",
            "destructive_commands",
            "readonly_shell_prefixes",
            "readonly_blocked_flags",
            "readonly_blocked_prefixes",
            "git_blocked_options",
            "git_blocked_prefixes",
        ):
            del self.settings["tools"][key]
        for key in ("temperature", "max_split_depth"):
            del self.settings["translate"][key]
        del self.settings["ui"]["spinner_interval_ms"]
        self.write_settings()
        with clean_env():
            config = self.load()
        self.assertEqual(config.settings.agent.max_steps, 10)
        self.assertEqual(config.settings.chat.temperature, 0.2)
        self.assertEqual(config.settings.batch.max_tokens_limit, 8192)
        self.assertEqual(config.settings.tools.max_read_bytes, 262144)
        self.assertEqual(config.settings.translate.max_split_depth, 3)
        self.assertEqual(config.settings.request.health_timeout_sec, 3.0)
        self.assertEqual(config.settings.launcher.log_tail_lines, 15)
        self.assertEqual(config.settings.ui.spinner_interval_ms, 300)

    def test_translate_presets_are_optional(self) -> None:
        del self.settings["translate"]["default_preset"]
        del self.settings["translate"]["presets"]
        self.write_settings()
        with clean_env():
            config = self.load()
        self.assertEqual(config.settings.translate.default_preset, "general")
        self.assertEqual(config.settings.translate.presets, {})

    def test_translate_presets_are_validated(self) -> None:
        for presets in (
            "translate",
            {"bad": "not-a-mapping"},
            {"bad": {"task_template": 3}},
            {"bad": {"unknown": "x"}},
        ):
            with self.subTest(presets=presets):
                self.settings["translate"]["presets"] = presets
                self.write_settings()
                with clean_env():
                    with self.assertRaises(ConfigError):
                        self.load()

    def test_translate_preset_window_fields(self) -> None:
        with clean_env():
            config = load_config()
        asmr = config.settings.translate.presets["asmr"]
        self.assertEqual(asmr["window_size"], 6)
        self.assertEqual(asmr["max_window_chars"], 250)

        self.settings["translate"]["presets"]["asmr"]["window_size"] = 0
        self.write_settings()
        with clean_env():
            with self.assertRaises(ConfigError):
                self.load()

        self.settings["translate"]["presets"]["asmr"]["window_size"] = 6
        self.settings["translate"]["presets"]["asmr"]["max_window_chars"] = 10
        self.write_settings()
        with clean_env():
            with self.assertRaises(ConfigError):
                self.load()

    def test_translate_scene_presets_and_limits(self) -> None:
        with clean_env():
            config = self.load()
        presets = config.settings.translate.presets
        self.assertEqual(presets["subtitle"]["profile"], "2")
        self.assertEqual(presets["subtitle"]["context_window"], 12)
        self.assertEqual(presets["subtitle"]["chunk_chars"], 800)
        self.assertEqual(presets["prose"]["profile"], "2")
        self.assertEqual(presets["prose"]["context_window"], 10)
        self.assertEqual(presets["asmr"]["window_size"], 6)
        self.assertEqual(presets["general"]["profile"], "1")
        self.assertEqual(presets["general"]["context_window"], 5)
        self.assertEqual(presets["general"]["chunk_chars"], 1500)

        for key, value in (
            ("context_window", 65),
            ("context_window", -1),
            ("context_summary_chars", 39),
            ("chunk_chars", 49),
            ("profile", 2),
            ("profile", ""),
        ):
            with self.subTest(key=key, value=value):
                with clean_env():
                    config = self.load()
                broken = json.loads(
                    (self.config_dir / "settings.json").read_text(
                        encoding="utf-8"
                    )
                )
                broken["translate"]["presets"]["subtitle"][key] = value
                (self.config_dir / "settings.json").write_text(
                    json.dumps(broken, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                with clean_env():
                    with self.assertRaises(ConfigError):
                        self.load()
                self.write_settings()

    def test_agent_action_policy_must_cover_all_actions(self) -> None:
        self.settings["agent"]["confirm_actions"].remove("delete_file")
        self.write_settings()
        with clean_env():
            with self.assertRaises(ConfigError) as ctx:
                self.load()
        self.assertIn("缺少动作", str(ctx.exception))

    def test_agent_action_policy_rejects_overlap_and_unknown(self) -> None:
        self.settings["agent"]["auto_actions"].append("write_file")
        self.write_settings()
        with clean_env():
            with self.assertRaises(ConfigError) as ctx:
                self.load()
        self.assertIn("重叠", str(ctx.exception))

        self.settings["agent"]["auto_actions"].remove("write_file")
        self.settings["agent"]["auto_actions"].append("not_an_action")
        self.write_settings()
        with clean_env():
            with self.assertRaises(ConfigError) as ctx:
                self.load()
        self.assertIn("未知动作", str(ctx.exception))

    def test_agent_alias_and_limits_are_validated(self) -> None:
        self.settings["agent"]["max_steps"] = 0
        self.write_settings()
        with clean_env():
            with self.assertRaises(ConfigError):
                self.load()

        self.settings["agent"]["max_steps"] = 10
        self.settings["agent"]["action_aliases"]["shell"] = "nope"
        self.write_settings()
        with clean_env():
            with self.assertRaises(ConfigError):
                self.load()

    def test_batch_and_temperature_limits_are_validated(self) -> None:
        self.settings["batch"]["default_max_tokens"] = 9000
        self.write_settings()
        with clean_env():
            with self.assertRaises(ConfigError) as ctx:
                self.load()
        self.assertIn("不能大于", str(ctx.exception))

        self.settings["batch"]["default_max_tokens"] = 512
        self.settings["chat"]["temperature"] = 3.0
        self.write_settings()
        with clean_env():
            with self.assertRaises(ConfigError):
                self.load()

    def test_security_floor_cannot_be_removed(self) -> None:
        self.settings["tools"]["destructive_commands"].remove("del")
        self.write_settings()
        with clean_env():
            with self.assertRaises(ConfigError) as ctx:
                self.load()
        self.assertIn("安全底线", str(ctx.exception))

    def test_tool_list_and_move_chars_validated(self) -> None:
        self.settings["tools"]["manual_exec_prefixes"] = []
        self.write_settings()
        with clean_env():
            with self.assertRaises(ConfigError):
                self.load()

        self.settings["tools"]["manual_exec_prefixes"] = ["echo"]
        self.settings["tools"]["move_illegal_chars"] = 1
        self.write_settings()
        with clean_env():
            with self.assertRaises(ConfigError):
                self.load()

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
        self.models["models"][1]["path"] = "models/qwen2.5-7b-instruct-q6_k-00002-of-00002.gguf"
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

    def test_exec_whitelist_is_retired(self) -> None:
        self.settings["tools"]["exec_whitelist"] = ["echo"]
        self.write_settings()
        with clean_env():
            with self.assertRaises(ConfigError) as ctx:
                self.load()
        self.assertIn("exec_whitelist", str(ctx.exception))

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
