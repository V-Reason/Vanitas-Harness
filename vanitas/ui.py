"""终端着色与符号前缀：本项目唯一的「颜色名 → ANSI 码」映射处。

依赖方向：业务模块 → ui。本模块不 import 任何 vanitas 业务模块，也不调用
print——写到 stdout 还是 stderr 由调用方决定，便于测试替换流。

Windows 控制台默认不解析 ANSI，需先用 SetConsoleMode 打开
ENABLE_VIRTUAL_TERMINAL_PROCESSING；enable_vt() 幂等，失败即整体降级为无色，
不抛异常、不影响主流程。
"""

from __future__ import annotations

import os
import re

# 与 config.py::COLOR_NAMES 必须完全一致（config 是基础层，不反向依赖本模块）。
_ANSI = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
    "bright_black": 90,
    "bright_red": 91,
    "bright_green": 92,
    "bright_yellow": 93,
    "bright_blue": 94,
    "bright_magenta": 95,
    "bright_cyan": 96,
    "bright_white": 97,
    "default": 39,
}

COLOR_NAMES: frozenset[str] = frozenset(_ANSI)

SYMBOLS: dict[str, str] = {
    "ok": "✓",
    "err": "✗",
    "warn": "⚠",
    "spin": "⋯",
    "meta": "·",
    "prompt": "›",
}

COLOR_MODES: frozenset[str] = frozenset({"auto", "always", "never"})

RESET = "\x1b[0m"
_ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
_STD_OUTPUT_HANDLE = -11
_STD_ERROR_HANDLE = -12

# 传入空 theme 时（例如测试里的 Palette({}, enabled=False)）按此补齐，
# 保证 Palette 永远有可用的颜色名；config 侧会用同一组默认值生成 UiSettings。
FALLBACK_THEME: dict[str, str] = {
    "prompt": "cyan",
    "success": "green",
    "error": "red",
    "warn": "yellow",
    "info": "cyan",
    "muted": "bright_black",
    "accent": "magenta",
}

_ANSI_CSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_vt_state: bool | None = None


def strip_ansi(text: str) -> str:
    """剥离 Palette 生成的 ANSI SGR 序列，供降级输入与非 TTY 输出复用。"""
    return _ANSI_CSI_RE.sub("", text)


def _try_enable_vt() -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetStdHandle.argtypes = [ctypes.c_uint32]
        kernel32.GetStdHandle.restype = ctypes.c_void_p
        kernel32.GetConsoleMode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        kernel32.GetConsoleMode.restype = ctypes.c_int
        kernel32.SetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetConsoleMode.restype = ctypes.c_int
    except (ImportError, AttributeError, OSError):
        return False

    enabled = False
    for handle_id in (_STD_OUTPUT_HANDLE, _STD_ERROR_HANDLE):
        handle = kernel32.GetStdHandle(ctypes.c_uint32(handle_id & 0xFFFFFFFF))
        if not handle:
            continue
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            continue  # 该流被重定向，没有控制台模式可设
        if kernel32.SetConsoleMode(
            handle, mode.value | _ENABLE_VIRTUAL_TERMINAL_PROCESSING
        ):
            enabled = True
    return enabled


def enable_vt() -> bool:
    """打开当前进程的 VT 解析；现代终端环境变量存在时直接放行。幂等。"""
    global _vt_state
    if _vt_state is None:
        if os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM"):
            _vt_state = True
        else:
            try:
                _vt_state = _try_enable_vt()
            except OSError:  # 极少数被策略拦截的宿主：降级而不是崩溃
                _vt_state = False
    return _vt_state


def resolve_enabled(mode: str, stream: object) -> bool:
    """按 settings.ui.color 决定是否着色：never 关、always 开、auto 看 TTY + VT。"""
    if mode == "never":
        return False
    if mode == "always":
        return True
    if mode not in COLOR_MODES:
        raise ValueError(f"未知颜色模式 {mode!r}；允许：{sorted(COLOR_MODES)}")
    try:
        is_tty = bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False
    if not is_tty:
        return False
    return enable_vt()


class Palette:
    """持有一份 theme 与 enabled 标志；每个方法只做「符号 + 着色」的纯字符串拼接。"""

    def __init__(self, theme: dict[str, str], enabled: bool) -> None:
        merged = dict(FALLBACK_THEME)
        for key, value in theme.items():
            if key in merged:
                merged[key] = value
        self._theme = merged
        self.enabled = enabled

    def paint(self, text: str, color: str) -> str:
        if color not in COLOR_NAMES:
            raise ValueError(f"未知颜色名 {color!r}；允许：{sorted(COLOR_NAMES)}")
        if not self.enabled:
            return text
        return f"\x1b[{_ANSI[color]}m{text}{RESET}"

    def _with_symbol(self, symbol: str, text: str, color_key: str) -> str:
        return self.paint(f"{SYMBOLS[symbol]} {text}", self._theme[color_key])

    def ok(self, text: str) -> str:
        return self._with_symbol("ok", text, "success")

    def err(self, text: str) -> str:
        return self._with_symbol("err", text, "error")

    def warn(self, text: str) -> str:
        return self._with_symbol("warn", text, "warn")

    def spin(self, text: str) -> str:
        return self._with_symbol("spin", text, "warn")

    def meta(self, text: str) -> str:
        return self._with_symbol("meta", text, "muted")

    def hint(self, text: str) -> str:
        return self._with_symbol("meta", text, "muted")

    def accent(self, text: str) -> str:
        return self.paint(text, self._theme["accent"])

    # ------------------------------------------------------------------ #
    # 语义着色：只上色、不附加符号，供表格状态列与紧凑摘要使用。
    # ------------------------------------------------------------------ #

    def _theme_paint(self, text: str, key: str) -> str:
        return self.paint(text, self._theme[key])

    def success(self, text: str) -> str:
        return self._theme_paint(text, "success")

    def error(self, text: str) -> str:
        return self._theme_paint(text, "error")

    def warning(self, text: str) -> str:
        return self._theme_paint(text, "warn")

    def muted(self, text: str) -> str:
        return self._theme_paint(text, "muted")

    def info(self, text: str) -> str:
        return self._theme_paint(text, "info")

    def bold(self, text: str) -> str:
        if not self.enabled:
            return text
        return f"\x1b[1m{text}{RESET}"
