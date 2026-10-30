"""工作区内文件工具：读写、移动、删除、列表、补丁、行区间读取、跨文件检索。

所有路径入参必须经 sandbox._require_path / sandbox.is_inside_workspace 校验；
本模块不直接使用 subprocess、不决定进程生命周期。
"""
from __future__ import annotations

import os
import re
import shutil
from functools import lru_cache
from pathlib import Path

from ..errors import ToolDenied
from . import sandbox


def read_file(
    path: str, *, root: Path | str | None = None, max_bytes: int
) -> str:
    target = sandbox._require_path(path, root)
    try:
        if not target.is_file():
            raise ToolDenied(f"不是可读取的文件：{path}")
        size = target.stat().st_size
        if size > max_bytes:
            raise ToolDenied(
                f"文件超过 {max_bytes // 1024} KiB，已拒绝读取：{path}"
            )
        raw = target.read_bytes()
    except ToolDenied:
        raise
    except OSError as exc:
        raise ToolDenied(f"读取失败：{path}；{exc.strerror or exc}") from exc
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ToolDenied(f"文件不是 UTF-8：{path}") from exc


def write_file(path: str, content: str, *, root: Path | str | None = None) -> int:
    if not isinstance(content, str):
        raise ToolDenied("write_file 的 content 必须是字符串")
    target = sandbox._require_path(path, root)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        target.write_bytes(data)
    except OSError as exc:
        raise ToolDenied(f"写入失败：{path}；{exc.strerror or exc}") from exc
    return len(data)


def _reject_illegal_move_text(value: str, illegal_chars: str) -> None:
    """模型常把 `#`、通配符等路径污染带进 move_file；提前给出可读拒绝。"""
    if any(char in illegal_chars or ord(char) < 32 for char in value):
        raise ToolDenied("源文件不存在或路径非法")


def move_file(
    src: str,
    dst: str,
    *,
    root: Path | str | None = None,
    illegal_chars: str,
) -> str:
    source = sandbox._require_path(src, root)
    target = sandbox._require_path(dst, root)
    _reject_illegal_move_text(src, illegal_chars)
    _reject_illegal_move_text(dst, illegal_chars)
    if source == target:
        raise ToolDenied(f"源和目标相同：{src}")
    try:
        if not source.is_file():
            raise ToolDenied("源文件不存在或路径非法")
        if target.exists():
            raise ToolDenied(f"目标已存在：{dst}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    except ToolDenied:
        raise
    except OSError as exc:
        raise ToolDenied(f"移动失败：{src} -> {dst}；{exc.strerror or exc}") from exc
    return f"重命名 {src} -> {dst}"


def delete_path(path: str, *, root: Path | str | None = None) -> str:
    """删除文件或递归删除目录；严禁删除工作区根目录自身。

    候选路径故意不先 resolve：符号链接必须按“链接本身”处理，不能跟随到
    链接目标后再删除；边界检查仍通过 is_inside_workspace 解析复核。
    """
    raw = sandbox._path_text(path)
    base_root = sandbox._workspace_root(root)
    expanded = os.path.expandvars(os.path.expanduser(raw))
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = base_root / candidate
    candidate = Path(os.path.abspath(candidate))
    sandbox.is_inside_workspace(candidate, base_root)
    if candidate.resolve() == base_root:
        raise ToolDenied("安全限制：禁止删除工作区根目录自身")
    try:
        if candidate.is_symlink():
            candidate.unlink()
            return f"删除链接 {path}"
        if candidate.is_dir():
            shutil.rmtree(candidate)
            return f"递归删除目录 {path}"
        if candidate.is_file():
            candidate.unlink()
            return f"删除文件 {path}"
        raise ToolDenied(f"路径不存在或不是可删除的文件/目录：{path}")
    except ToolDenied:
        raise
    except OSError as exc:
        raise ToolDenied(f"删除失败：{path}；{exc.strerror or exc}") from exc


# 向后兼容旧名称；动作层同时保留 delete_file 别名。
delete_file = delete_path


def list_dir(
    path: str = ".",
    *,
    root: Path | str | None = None,
    max_entries: int,
) -> list[str]:
    target = sandbox._require_path(path, root)
    if not target.is_dir():
        raise ToolDenied(f"不是目录：{path}")
    entries: list[str] = []
    truncated = False
    try:
        with os.scandir(target) as iterator:
            for entry in iterator:
                if len(entries) >= max_entries:
                    truncated = True
                    break
                suffix = "/" if entry.is_dir(follow_symlinks=False) else ""
                entries.append(entry.name + suffix)
    except OSError as exc:
        raise ToolDenied(f"目录读取失败：{path}；{exc.strerror or exc}") from exc
    entries.sort(key=lambda item: (not item.endswith("/"), item.casefold()))
    if truncated:
        entries.append("…")
    return entries


def patch_file(
    path: str,
    search: str,
    replace: str,
    *,
    root: Path | str | None = None,
    max_bytes: int = 262144,
) -> str:
    """对已有文件做唯一片段替换；保留原 BOM/换行字节，避免全量重写。"""
    if not isinstance(search, str) or not search:
        raise ToolDenied("patch_file 的 search 匹配串必须是非空字符串")
    if not isinstance(replace, str):
        raise ToolDenied("patch_file 的 replace 替换串必须是字符串")
    target = sandbox._require_path(path, root)
    try:
        if not target.is_file():
            raise ToolDenied(
                f"目标文件不存在：{path}。"
                "若目标是重命名或改扩展名，请改用 move_file(src, dst)；"
                "patch_file 只修改已有文件的内容。"
            )
        size = target.stat().st_size
        if size > max_bytes:
            raise ToolDenied(
                f"文件超过 {max_bytes // 1024} KiB，已拒绝补丁：{path}"
            )
        raw = target.read_bytes()
    except ToolDenied:
        raise
    except OSError as exc:
        raise ToolDenied(f"读取文件失败：{path}；{exc.strerror or exc}") from exc

    has_bom = raw.startswith(b"\xef\xbb\xbf")
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ToolDenied(f"文件不是 UTF-8 编码，无法打补丁：{path}") from exc

    count = content.count(search)
    if count == 0:
        raise ToolDenied(
            f"未在文件 {path} 中找到完全匹配的代码片段。"
            "请先调用 read_file_range 核对最新内容；"
            "若目标是重命名或改扩展名，请改用 move_file(src, dst)。"
        )
    if count > 1:
        raise ToolDenied(
            f"在文件 {path} 中找到 {count} 处匹配片段，定位不唯一。"
            "请扩展 search 前后的上下文行以唯一定位。"
        )

    patched = content.replace(search, replace, 1)
    encoding = "utf-8-sig" if has_bom else "utf-8"
    try:
        target.write_bytes(patched.encode(encoding))
    except OSError as exc:
        raise ToolDenied(f"写入补丁失败：{path}；{exc.strerror or exc}") from exc
    return f"补丁应用成功：{path}（替换 1 处匹配）"


def read_file_range(
    path: str,
    start_line: int = 1,
    end_line: int = 100,
    *,
    root: Path | str | None = None,
    max_bytes: int = 262144,
    max_lines: int = 400,
) -> str:
    """按 1-based 闭区间读取文件行，防止大文件撑爆上下文。"""
    if start_line < 1:
        raise ToolDenied("start_line 必须 ≥ 1")
    if end_line < start_line:
        raise ToolDenied(
            f"end_line ({end_line}) 必须 ≥ start_line ({start_line})"
        )
    if end_line - start_line + 1 > max_lines:
        raise ToolDenied(
            f"请求行数超过上限 {max_lines}；请缩小 start_line/end_line 范围"
        )
    target = sandbox._require_path(path, root)
    try:
        if not target.is_file():
            raise ToolDenied(f"文件不存在：{path}")
        size = target.stat().st_size
        if size > max_bytes:
            raise ToolDenied(
                f"文件超过 {max_bytes // 1024} KiB，已拒绝读取：{path}"
            )
        raw = target.read_bytes()
    except ToolDenied:
        raise
    except OSError as exc:
        raise ToolDenied(f"读取文件失败：{path}；{exc.strerror or exc}") from exc
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ToolDenied(f"文件不是 UTF-8 编码：{path}") from exc

    lines = content.splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        return f"[{path}] 共 0 行：\n"
    if start_line > total_lines:
        raise ToolDenied(
            f"start_line {start_line} 超出文件总行数 {total_lines}：{path}"
        )
    actual_start = max(1, start_line)
    actual_end = min(end_line, total_lines)
    selected = lines[actual_start - 1 : actual_end]
    formatted = [
        f"{idx:4d} | {line}"
        for idx, line in enumerate(selected, start=actual_start)
    ]
    header = (
        f"[{path}] 第 {actual_start}–{actual_end} 行"
        f"（共 {total_lines} 行）：\n"
    )
    return header + "\n".join(formatted)


@lru_cache(maxsize=64)
def _compile_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def grep_workspace(
    pattern: str,
    path: str = ".",
    *,
    root: Path | str | None = None,
    max_results: int = 50,
    max_file_bytes: int = 1048576,
    ignore_dirs: tuple[str, ...] = (".git", "__pycache__", ".venv", "node_modules"),
) -> list[str]:
    """在工作区内做正则检索；跳过隐藏目录、符号链接与超大文件。"""
    if not pattern:
        raise ToolDenied("检索 pattern 不能为空")
    if max_results < 1:
        raise ToolDenied("max_results 必须 ≥ 1")
    target_dir = sandbox._require_path(path, root)
    if not target_dir.is_dir():
        raise ToolDenied(f"检索路径不是目录：{path}")
    try:
        regex = _compile_pattern(pattern)
    except re.error as exc:
        raise ToolDenied(f"无效的正则表达式 {pattern!r}：{exc}") from exc

    base_root = sandbox._workspace_root(root)
    ignored = frozenset(ignore_dirs)
    matches: list[str] = []
    for cur_root, dirs, files in os.walk(target_dir, followlinks=False):
        dirs[:] = [
            item
            for item in dirs
            if not item.startswith(".") and item not in ignored
        ]
        for file in files:
            if file.startswith("."):
                continue
            file_path = Path(cur_root) / file
            if file_path.is_symlink():
                continue
            try:
                resolved = file_path.resolve()
                if not sandbox.is_inside_workspace(
                    resolved, base_root, raise_on_error=False
                ):
                    continue
                if resolved.stat().st_size > max_file_bytes:
                    continue
                text = resolved.read_text(encoding="utf-8", errors="ignore")
            except (OSError, RuntimeError, ValueError):
                continue
            try:
                rel_path = resolved.relative_to(base_root).as_posix()
            except ValueError:
                continue
            for line_idx, line in enumerate(text.splitlines(), start=1):
                if not regex.search(line):
                    continue
                matches.append(f"{rel_path}:{line_idx}: {line.strip()[:120]}")
                if len(matches) >= max_results:
                    matches.append("…（结果超出上限，已截断）")
                    return matches
    return matches or ["未找到匹配文本"]
