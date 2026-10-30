"""ASR 跨 cue 断词修复 CLI（默认 dry-run，只写独立 out-dir）。

用法：
    python scripts/repair_asr_srt.py --source-dir <SRT目录> --out-dir <输出目录>
    python scripts/repair_asr_srt.py a.srt b.srt --out-dir out --write
    python scripts/repair_asr_srt.py --source-dir SRT --out-dir out --check

退出码：
- 0 成功（dry-run 也返回 0）；
- 1 ``--check`` 发现待合并；
- 2 合并表坏行、IO 失败、out-dir 等于源目录、试图覆盖 ``.zh.srt``、
  输出文件冲突/已存在。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas.asr_repair import (  # noqa: E402
    JoinRule,
    Skip,
    load_joins,
    parse_cues,
    render_srt,
    repair_cues,
)
from vanitas.errors import TranslateError  # noqa: E402


@dataclass(frozen=True)
class Plan:
    source: Path
    target: Path


@dataclass
class FileReport:
    source: str
    target: str
    merged: int
    skipped: list[Skip]
    written: bool
    pending: bool

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "merged": self.merged,
            "skipped": [
                {
                    "left": item.left,
                    "right": item.right,
                    "reason": item.reason,
                }
                for item in self.skipped
            ],
            "written": self.written,
            "pending": self.pending,
        }


@dataclass
class RunReport:
    files: list[FileReport] = field(default_factory=list)
    write: bool = False
    check: bool = False

    @property
    def merged(self) -> int:
        return sum(item.merged for item in self.files)

    @property
    def skipped(self) -> int:
        return sum(len(item.skipped) for item in self.files)

    @property
    def pending(self) -> bool:
        return any(item.pending for item in self.files)

    @property
    def written(self) -> int:
        return sum(1 for item in self.files if item.written)

    def as_dict(self) -> dict:
        return {
            "summary": {
                "files": len(self.files),
                "merged": self.merged,
                "skipped": self.skipped,
                "written": self.written,
                "pending": self.pending,
                "write": self.write,
                "check": self.check,
            },
            "files": [item.as_dict() for item in self.files],
        }


def _collect_source_dir(directory: Path, out_dir: Path) -> list[Path]:
    if not directory.is_dir():
        raise TranslateError(f"源目录不存在或不是目录：{directory}")
    out_key = out_dir.resolve()
    found: list[Path] = []
    for candidate in sorted(
        directory.rglob("*.srt"), key=lambda item: str(item).lower()
    ):
        if not candidate.is_file():
            continue
        if candidate.name.lower().endswith(".zh.srt"):
            continue
        try:
            candidate.resolve().relative_to(out_key)
        except ValueError:
            pass
        else:
            # 不把本轮输出目录里的产物再次当输入。
            continue
        found.append(candidate)
    if not found:
        raise TranslateError(f"源目录中没有可修复的 .srt：{directory}")
    return found


def _collect_sources(
    files: list[str], source_dirs: list[str], out_dir: Path
) -> list[Path]:
    collected: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path.resolve()).casefold()
        if key in seen:
            return
        seen.add(key)
        collected.append(path)

    for raw in source_dirs:
        for path in _collect_source_dir(Path(raw), out_dir):
            add(path)
    for raw in files:
        path = Path(raw)
        if not path.is_file():
            raise TranslateError(f"文件不存在：{path}")
        if path.suffix.lower() != ".srt":
            raise TranslateError(f"只接受 .srt：{path}")
        add(path)
    if not collected:
        raise TranslateError("没有输入；请给出 --source-dir 或 .srt 文件列表")
    return collected


def _plan_sources(sources: list[Path], out_dir: Path, *, write: bool) -> list[Plan]:
    if out_dir.exists() and not out_dir.is_dir():
        raise TranslateError(f"--out-dir 不是目录：{out_dir}")
    plans: list[Plan] = []
    targets: dict[str, Path] = {}
    for source in sources:
        if source.name.lower().endswith(".zh.srt"):
            raise TranslateError(
                f"拒绝覆盖译文文件（.zh.srt）：{source}"
            )
        if source.parent.resolve() == out_dir.resolve():
            raise TranslateError(
                f"--out-dir 不能等于源目录（会原地覆盖）：{source.parent}"
            )
        stem = source.stem
        if stem.lower().endswith(".zh"):
            raise TranslateError(
                f"输出会覆盖 .zh.srt：{source}；请确认输入是源 SRT"
            )
        target = out_dir / f"{stem}.srt"
        if target.name.lower().endswith(".zh.srt"):
            raise TranslateError(
                f"输出会覆盖 .zh.srt：{target}；请确认输入是源 SRT"
            )
        key = str(target.resolve()).casefold()
        if key in targets:
            raise TranslateError(
                f"输出路径冲突：{targets[key]} 与 {source} 都会写到 {target}"
            )
        targets[key] = source
        if write and target.exists():
            raise TranslateError(
                f"目标已存在，拒绝覆盖：{target}；请换 --out-dir 或先清理"
            )
        plans.append(Plan(source=source, target=target))
    return plans


def _read_source(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TranslateError(
            f"读取失败：{path}；{exc.strerror or exc}"
        ) from exc
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TranslateError(
            f"文件不是 UTF-8：{path}；第 {exc.start} 字节，请先转码"
        ) from exc


def _render_report(report: RunReport, out_dir: Path) -> str:
    mode = "check" if report.check else ("write" if report.write else "dry-run")
    lines = [
        f"· ASR 断词修复 · {len(report.files)} 个文件 · "
        f"合并 {report.merged} · 跳过 {report.skipped} · {mode}"
    ]
    for item in report.files:
        status = "已写" if item.written else ("待合并" if item.pending else "无变化")
        lines.append(
            f"  [{status}] {item.source} → {item.target} · "
            f"merged {item.merged} · skipped {len(item.skipped)}"
        )
        for skip in item.skipped:
            lines.append(
                f"    - skip {skip.left}/{skip.right}：{skip.reason}"
            )
    if report.check and report.pending:
        lines.append(f"✗ --check：仍有待合并，输出目录 {out_dir}")
    elif report.check:
        lines.append("✓ --check：没有待合并")
    elif not report.write:
        lines.append("· dry-run：未写文件；确认后加 --write")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(
        description="ASR 跨 cue 断词修复（默认 dry-run）"
    )
    parser.add_argument("files", nargs="*", help="一个或多个源 .srt")
    parser.add_argument(
        "--source-dir",
        action="append",
        default=[],
        help="递归收集源 .srt（可重复）",
    )
    parser.add_argument(
        "--joins",
        default=str(ROOT / "config" / "asr_joins.tsv"),
        help="合并表路径（默认 config/asr_joins.tsv）",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="独立输出目录")
    parser.add_argument("--write", action="store_true", help="真正写盘")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查待合并；有待合并返回 1，不写盘",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    try:
        rules: tuple[JoinRule, ...] = load_joins(Path(args.joins))
        sources = _collect_sources(args.files, args.source_dir, args.out_dir)
        plans = _plan_sources(
            sources, args.out_dir, write=bool(args.write and not args.check)
        )
        report = RunReport(write=bool(args.write), check=bool(args.check))
        for plan in plans:
            text = _read_source(plan.source)
            result = repair_cues(parse_cues(text), rules)
            written = False
            if args.write and not args.check:
                try:
                    plan.target.parent.mkdir(parents=True, exist_ok=True)
                    plan.target.write_text(
                        render_srt(result.cues),
                        encoding="utf-8",
                        newline="\n",
                    )
                except OSError as exc:
                    raise TranslateError(
                        f"写入失败：{plan.target}；{exc.strerror or exc}"
                    ) from exc
                written = True
            report.files.append(
                FileReport(
                    source=str(plan.source),
                    target=str(plan.target),
                    merged=result.merged,
                    skipped=list(result.skipped),
                    written=written,
                    pending=result.changed,
                )
            )
    except (TranslateError, OSError) as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(_render_report(report, args.out_dir))

    if args.check and report.pending:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
