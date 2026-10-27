"""路径沙箱：唯一的工作区基准解析与边界校验。

所有工具层的路径入参必须经过此模块。依赖仅限标准库与 ..errors。
"""
from __future__ import annotations

import os
from pathlib import Path

from ..errors import ToolDenied


def _workspace_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()
    # bat 会先 cd 到仓库根；Agent 的“当前工作区”必须回到用户原 CWD。
    original = os.environ.get("VANITAS_ORIG_CWD")
    if original:
        return Path(original).expanduser().resolve()
    return Path.cwd().resolve()


def _path_text(path: str | os.PathLike[str]) -> str:
    try:
        raw = os.fspath(path)
    except TypeError as exc:
        raise ToolDenied("路径必须是字符串") from exc
    if not isinstance(raw, str):
        raise ToolDenied("路径必须是字符串")
    raw = raw.strip()
    if not raw or "\x00" in raw:
        raise ToolDenied("安全限制：路径不能为空")
    return raw


def _resolve_candidate(path_str: str, root: Path | str | None = None) -> Path:
    base = _workspace_root(root)
    expanded = os.path.expandvars(os.path.expanduser(str(path_str)))
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def is_inside_workspace(
    path: str | os.PathLike[str],
    root: Path | str | None = None,
    *,
    raise_on_error: bool = True,
) -> bool:
    """规范化路径并判断是否位于工作区内。

    默认在越界、空路径或路径无法解析时抛出 `ToolDenied`；需要布尔查询时传
    `raise_on_error=False`。Windows 通过 `normcase` + `commonpath` 做大小写
    不敏感比较，跨盘符同样判为越界。
    """
    try:
        raw = _path_text(path)
        # 独立调用严格以进程 CWD 为根；AgentTools 会显式传入原 CWD。
        base = (
            Path.cwd().resolve()
            if root is None
            else Path(root).expanduser().resolve()
        )
        resolved = _resolve_candidate(raw, base)
        base_key = os.path.normcase(str(base))
        target_key = os.path.normcase(str(resolved))
        inside = os.path.commonpath([base_key, target_key]) == base_key
    except (OSError, RuntimeError, TypeError, ValueError, ToolDenied) as exc:
        inside = False
        raw = str(path) if "raw" not in locals() else raw
        detail = str(exc) or "无法解析路径"
    else:
        detail = "路径不在工作区内"
    if inside:
        return True
    if raise_on_error:
        raise ToolDenied(f"安全限制：越界操作已被拦截：{raw}（{detail}）")
    return False


def _require_path(
    path: str | os.PathLike[str], root: Path | str | None = None
) -> Path:
    raw = _path_text(path)
    # 与 is_inside_workspace 一致：未显式给根时使用进程 CWD；
    # AgentTools 永远显式传入原 CWD，避免 bat 的仓库根劫持。
    base = (
        Path.cwd().resolve()
        if root is None
        else Path(root).expanduser().resolve()
    )
    is_inside_workspace(raw, base)
    return _resolve_candidate(raw, base)


def _is_safe_path(path_str: str, root: Path | str | None = None) -> bool:
    """兼容旧调用：返回路径是否落在工作区内，不抛异常。"""
    return is_inside_workspace(path_str, root, raise_on_error=False)


def _is_within_cwd(path: str, cwd: Path) -> bool:
    """兼容旧调用：路径解析后是否落在 cwd 内。"""
    return _is_safe_path(path, cwd)
