"""ui 模块：着色开关、颜色名映射、符号前缀。不碰业务模块，只测字符串契约。"""

from __future__ import annotations

import io
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import ui  # noqa: E402
from vanitas.config import COLOR_NAMES as CONFIG_COLOR_NAMES  # noqa: E402
from vanitas.config import DEFAULT_THEME  # noqa: E402


class PaletteTests(unittest.TestCase):
    def test_paint_disabled_returns_text(self) -> None:
        palette = ui.Palette(DEFAULT_THEME, enabled=False)
        self.assertEqual(palette.paint("hello", "red"), "hello")

    def test_paint_enabled_wraps_with_ansi(self) -> None:
        palette = ui.Palette(DEFAULT_THEME, enabled=True)
        painted = palette.paint("hello", "red")
        self.assertIn("\x1b[31m", painted)
        self.assertTrue(painted.endswith("\x1b[0m"))
        self.assertEqual(painted, "\x1b[31mhello\x1b[0m")

    def test_paint_rejects_unknown_color(self) -> None:
        palette = ui.Palette(DEFAULT_THEME, enabled=True)
        with self.assertRaises(ValueError):
            palette.paint("hello", "purple")

    def test_symbol_prefixes_are_plain_when_disabled(self) -> None:
        palette = ui.Palette(DEFAULT_THEME, enabled=False)
        self.assertEqual(palette.ok("done"), "✓ done")
        self.assertEqual(palette.err("bad"), "✗ bad")
        self.assertEqual(palette.warn("careful"), "⚠ careful")
        self.assertEqual(palette.spin("loading"), "⋯ loading")
        self.assertEqual(palette.meta("note"), "· note")
        self.assertEqual(palette.hint("try /help"), "· try /help")
        self.assertEqual(palette.accent("plain"), "plain")

    def test_theme_is_used_when_enabled(self) -> None:
        theme = dict(DEFAULT_THEME)
        theme["success"] = "cyan"
        palette = ui.Palette(theme, enabled=True)
        self.assertIn("\x1b[36m", palette.ok("done"))
        self.assertIn("\x1b[35m", palette.accent("deco"))

    def test_resolve_enabled_modes(self) -> None:
        stream = io.StringIO()  # 非 TTY
        self.assertFalse(ui.resolve_enabled("never", stream))
        self.assertTrue(ui.resolve_enabled("always", stream))
        self.assertFalse(ui.resolve_enabled("auto", stream))
        with self.assertRaises(ValueError):
            ui.resolve_enabled("yes", stream)

    def test_resolve_enabled_auto_uses_tty(self) -> None:
        class Tty(io.StringIO):
            def isatty(self) -> bool:  # 模拟控制台
                return True

        # auto 还要求 VT 可用；把 enable_vt 固定为 True，避免依赖宿主机控制台状态
        original = ui.enable_vt
        ui.enable_vt = lambda: True
        try:
            self.assertTrue(ui.resolve_enabled("auto", Tty()))
        finally:
            ui.enable_vt = original

    def test_enable_vt_is_idempotent(self) -> None:
        first = ui.enable_vt()
        second = ui.enable_vt()
        self.assertEqual(first, second)
        self.assertIsInstance(first, bool)

    def test_enable_vt_accepts_modern_terminal_env(self) -> None:
        original = ui._vt_state
        try:
            ui._vt_state = None
            with mock.patch.dict(os.environ, {"WT_SESSION": "1"}, clear=True):
                self.assertTrue(ui.enable_vt())
        finally:
            ui._vt_state = original

    def test_symbols_table_complete(self) -> None:
        self.assertEqual(set(ui.SYMBOLS), {"ok", "err", "warn", "spin", "meta", "prompt"})
        for name, symbol in ui.SYMBOLS.items():
            self.assertTrue(symbol.strip(), name)

    def test_color_names_match_config(self) -> None:
        # 两处重复定义是刻意的（避免 config → ui 依赖），必须逐字一致
        self.assertEqual(ui.COLOR_NAMES, CONFIG_COLOR_NAMES)

    def test_palette_with_empty_theme_falls_back(self) -> None:
        palette = ui.Palette({}, enabled=False)
        self.assertEqual(palette.ok("x"), "✓ x")

    def test_semantic_paints_use_theme_colors(self) -> None:
        palette = ui.Palette(DEFAULT_THEME, enabled=True)
        self.assertEqual(palette.success("ok"), "\x1b[32mok\x1b[0m")
        self.assertEqual(palette.error("bad"), "\x1b[31mbad\x1b[0m")
        self.assertEqual(palette.warning("warn"), "\x1b[33mwarn\x1b[0m")
        self.assertEqual(palette.muted("muted"), "\x1b[90mmuted\x1b[0m")
        self.assertEqual(palette.info("info"), "\x1b[36minfo\x1b[0m")

    def test_bold_only_adds_ansi_when_enabled(self) -> None:
        enabled = ui.Palette(DEFAULT_THEME, enabled=True)
        disabled = ui.Palette(DEFAULT_THEME, enabled=False)
        self.assertEqual(enabled.bold("id"), "\x1b[1mid\x1b[0m")
        self.assertEqual(disabled.bold("id"), "id")

    def test_semantic_paints_follow_custom_theme(self) -> None:
        theme = dict(DEFAULT_THEME)
        theme.update({"success": "cyan", "warn": "magenta", "muted": "red"})
        palette = ui.Palette(theme, enabled=True)
        self.assertEqual(palette.success("a"), "\x1b[36ma\x1b[0m")
        self.assertEqual(palette.warning("b"), "\x1b[35mb\x1b[0m")
        self.assertEqual(palette.muted("c"), "\x1b[31mc\x1b[0m")

    def test_semantic_paints_plain_when_disabled(self) -> None:
        palette = ui.Palette(DEFAULT_THEME, enabled=False)
        self.assertEqual(palette.success("ok"), "ok")
        self.assertEqual(palette.error("bad"), "bad")
        self.assertEqual(palette.warning("warn"), "warn")
        self.assertEqual(palette.muted("muted"), "muted")
        self.assertEqual(palette.info("info"), "info")

    def test_strip_ansi_removes_palette_sequences(self) -> None:
        palette = ui.Palette(DEFAULT_THEME, enabled=True)
        painted = palette.bold(palette.accent("qwen7b"))
        self.assertIn("\x1b[", painted)
        self.assertEqual(ui.strip_ansi(painted), "qwen7b")


if __name__ == "__main__":
    unittest.main()
