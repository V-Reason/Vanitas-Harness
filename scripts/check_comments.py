"""注释门：和编译门同级的一道确定性检查，拦“注释腐化”。

为什么需要这道门：编译门只保证语法能过，静态红线只查端口/依赖/标签；
没有任何东西阻止注释长成下面这样——

- 把取证写进注释（批次号、行号、“某轮测出多少条”），半年后没人知道
  那串编号是什么、属于哪份文件；
- 注释里引用的符号已经被删掉（``# 由某个已退役开关统一控制``），读者会
  去找一个不存在的东西；
- 注释指向的文档锚在文档重构后已经不存在；
- 把测试已经钉死的值抄进注释，改产物时注释跟着过期。

这四类都是**可确定性判定**的，所以做成门；判不了的（注释说得对不对）不做。

用法：
    python scripts/check_comments.py            # 全仓检查，违规返回 1
    python scripts/check_comments.py --json     # 机器可读
    python scripts/check_comments.py --selftest # 验证每条规则都还能报警

失败关闭：任何文件 ``ast.parse`` 失败都返回 2，避免"解析不了就当通过"。

豁免：只允许行内 ``gate-allow: <理由>``（必须行首），不设全局白名单文件；
每次运行会打印豁免条数，超过 10 条视为腐化信号。
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path

ROOT_DEFAULT = Path(__file__).resolve().parents[1]

# G1 取证外泄：这些形态只在“当时的观察”里成立，写进注释就会过期。
# 分两档：生产代码（vanitas/、scripts/）全档；测试里的编号常指向**合成夹具**
# （同一文件里自定义的 file1/number2/3），那是本地可验证的引用，不算取证，
# 所以测试只查最强的三类：run stamp、轮次标签、实测条数。
FORENSIC_PATTERNS = (
    (re.compile(r"\b\d{8}-\d{6}\b"), "run stamp"),
    (re.compile(r"\bfix\d+[a-z]?\b"), "会话/轮次标签"),
    (re.compile(r"实测\s*\d+\s*条"), "实测条数"),
    (re.compile(r"cue\s*[#]?\d+", re.IGNORECASE), "cue 号"),
    (re.compile(r"第\s*\d+\s*条"), "cue 序号"),
)
PRODUCTION_ROOTS = ("vanitas", "scripts")

# G2 幽灵符号：注释里反引号包住的标识符必须真的还在仓库里。
BACKTICK_IDENT = re.compile(r"`([A-Za-z_][A-Za-z0-9_]{2,})`")
# 这些词不是本仓库符号，或含义太宽，跳过。
GHOST_SKIP = frozenset(
    {
        "true", "false", "none", "python", "json", "utf", "srt", "wav", "md",
        "bat", "asmr", "asr", "api", "cli", "http", "https", "sse", "kv",
        "cpu", "gpu", "oom", "lru", "eof", "llama", "server", "git", "utf-8",
        "tool_result", "tool_call", "raw_decode", "type_ignore", "asr_joins",
        # 内建异常/常量：解释器自带，不需要在本仓库定义。
        "generatorexit", "stopiteration", "keyboardinterrupt", "exception",
        "notimplemented", "notimplementederror", "ellipsis", "baseexception",
        "systemexit", "oserror", "valueerror", "typeerror", "keyerror",
        "attributeerror", "importerror", "runtimeerror",
    }
)

# G3 悬空文档锚：§n 必须能在文档里解析到。
DOC_ANCHOR = re.compile(r"§(\d+)")
ANCHOR_FILES = ("docs/变更历史.md", "docs/项目现状.md")

# G4 复读钉死值：值不手写，从测试里读，提示词一改门自动跟着改。
PINNED_SOURCE = "tests/test_prompts.py"
PIN_BYTES_RE = re.compile(r"len\(production\),\s*(\d+)")
PIN_SHA_RE = re.compile(r'"([0-9A-F]{64})"')

# W1 未归档待办：这类标记必须指到 §n，否则就是没人跟的债。
TODO_RE = re.compile(r"\b(TODO|FIXME|XXX)\b")

EXEMPT_MARK = "gate-allow:"
EXEMPT_SOFT_LIMIT = 10


def is_exempt(text: str) -> bool:
    """行内豁免必须以它开头；否则“文档里提到这个标记”会被误当成豁免。"""
    return text.lstrip("# \t").lower().startswith(EXEMPT_MARK)

SCAN_ROOTS = ("vanitas", "tests", "scripts")


def comment_sites(path: Path) -> list[tuple[int, str]]:
    """返回 [(行号, 文本)]：只含文档字符串与 # 注释，不含代码。"""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)  # 语法错交给调用方按失败关闭处理
    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            sites.append((body[0].lineno, body[0].value.value))
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            sites.append((token.start[0], token.string))
    return sites


def build_corpus(root: Path) -> str:
    """“符号是否存在”的判定语料：只看代码标识符与配置，不看注释与文档。

    注释/文档不进语料是必须的：否则一个只出现在注释里的符号会“自己证明
    自己存在”，G2 永远不会触发。另外把代码里的**字符串常量**里的全大写
    标识也收进来（`"SEXPAR"` 这类常量常被注释按名字引用）。
    """
    names: set[str] = set()
    for folder in SCAN_ROOTS:
        for path in (root / folder).rglob("*.py"):
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            try:
                for token in tokenize.generate_tokens(
                    io.StringIO(source).readline
                ):
                    if token.type == tokenize.NAME:
                        names.add(token.string)
                    elif token.type == tokenize.STRING:
                        # 代码里的字符串常量常被注释按名字引用
                        # （"short_duplicate"、"latin"、"SEXPAR" 这类 code 值）。
                        bare = token.string.strip("\"'")
                        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{2,}", bare):
                            names.add(bare)
            except (tokenize.TokenError, SyntaxError):
                continue
    for pattern in ("config/**/*.tsv", "config/**/*.json"):
        for path in root.glob(pattern):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            names.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    return "\n".join(sorted(names))


def doc_anchors(root: Path) -> set[str]:
    anchors: set[str] = set()
    for name in ANCHOR_FILES:
        path = root / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            matched = re.match(r"^#{2,3}\s+(\d+)\.", line)
            if matched:
                anchors.add(matched.group(1))
    return anchors


def pinned_literals(root: Path) -> set[str]:
    path = root / PINNED_SOURCE
    if not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8")
    values = set(PIN_BYTES_RE.findall(text))
    for digest in PIN_SHA_RE.findall(text):
        values.add(digest)
        values.add(digest[:8])
    return values


def scan_text(
    text: str,
    *,
    corpus: str,
    anchors: set[str],
    pins: set[str],
    production: bool = True,
) -> list[tuple[str, str]]:
    """对一段注释文本跑四条硬规则 + 一条软规则，返回 [(code, 说明)]。"""
    found: list[tuple[str, str]] = []
    limit = len(FORENSIC_PATTERNS) if production else 3
    for pattern, label in FORENSIC_PATTERNS[:limit]:
        matched = pattern.search(text)
        if matched:
            found.append(("G1", f"{label}「{matched.group(0)}」→ 搬到 变更历史"))
    for name in BACKTICK_IDENT.findall(text):
        if name.lower() in GHOST_SKIP:
            continue
        if not re.search(rf"\b{re.escape(name)}\b", corpus):
            found.append(("G2", f"幽灵符号 `{name}`"))
    for anchor in DOC_ANCHOR.findall(text):
        if anchor not in anchors:
            found.append(("G3", f"悬空文档锚 §{anchor}"))
    for pin in pins:
        if pin and pin in text:
            found.append(("G4", f"复读钉死值 {pin} → 改指向测试"))
    if TODO_RE.search(text) and not DOC_ANCHOR.search(text):
        found.append(("W1", "TODO 未指向 §n"))
    return found


def scan_repo(root: Path) -> tuple[list[dict], list[str], int]:
    corpus = build_corpus(root)
    anchors = doc_anchors(root)
    pins = pinned_literals(root)
    findings: list[dict] = []
    parse_errors: list[str] = []
    exempted = 0
    for folder in SCAN_ROOTS:
        for path in sorted((root / folder).rglob("*.py")):
            try:
                sites = comment_sites(path)
            except (SyntaxError, ValueError) as exc:
                parse_errors.append(f"{path.relative_to(root).as_posix()}: {exc}")
                continue
            for lineno, text in sites:
                if is_exempt(text):
                    exempted += 1
                    continue
                for code, message in scan_text(
                    text,
                    corpus=corpus,
                    anchors=anchors,
                    pins=pins,
                    production=folder in PRODUCTION_ROOTS,
                ):
                    findings.append(
                        {
                            "file": path.relative_to(root).as_posix(),
                            "line": lineno,
                            "code": code,
                            "message": message,
                            "snippet": " ".join(text.split())[:80],
                        }
                    )
    return findings, parse_errors, exempted


SELFTEST_CASES = (
    ("G1", "# S2/V1 实测 6 条 duplicate", True),
    ("G1", "# 上限 11，为什么不是 8 见 §52", False),
    ("G2", "# 由 `removed_helper_xyz` 统一控制", True),
    ("G2", "# 由 `quality.py` 统一控制", False),
    ("G3", "# 细节见 §99", True),
    ("G3", "# 细节见 §54", False),
    ("G4", "# 提示词 13108 B", True),
    ("W1", "# TODO 还没做", True),
)


def selftest(root: Path) -> int:
    # 语料里必须排除本文件：否则用例样本自身会让“幽灵符号”看起来存在。
    corpus = build_corpus(root).replace("removed_helper_xyz", "")
    anchors = doc_anchors(root)
    pins = pinned_literals(root) | {"13108", "74324996"}
    failures = 0
    for expected, sample, should_fire in SELFTEST_CASES:
        codes = [
            code
            for code, _ in scan_text(
                sample, corpus=corpus, anchors=anchors, pins=pins
            )
        ]
        fired = expected in codes
        if fired != should_fire:
            failures += 1
            print(f"FAIL {expected} {sample!r}: expected fire={should_fire}, got {codes}")
    print(f"selftest: {len(SELFTEST_CASES)} 例，失败 {failures}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="注释门：拦注释腐化")
    parser.add_argument("--root", default=str(ROOT_DEFAULT))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.selftest:
        return selftest(root)
    findings, parse_errors, exempted = scan_repo(root)
    if args.json:
        print(json.dumps(
            {
                "findings": findings,
                "parse_errors": parse_errors,
                "exempted": exempted,
            },
            ensure_ascii=False,
            indent=1,
        ))
        return 2 if parse_errors else (1 if findings else 0)
    for item in findings:
        print(f"{item['file']}:{item['line']}: [{item['code']}] {item['message']}")
        print(f"    {item['snippet']}")
    for error in parse_errors:
        print(f"PARSE ERROR {error}")
    hard = [item for item in findings if item["code"] != "W1"]
    warnings = [item for item in findings if item["code"] == "W1"]
    print(
        f"注释门：硬违规 {len(hard)} · 警告 {len(warnings)} · 豁免 {exempted}"
        f"（软上限 {EXEMPT_SOFT_LIMIT}）"
    )
    if exempted > EXEMPT_SOFT_LIMIT:
        print("豁免过多：请把对应注释改写成结论 + §n 指针，而不是逐条豁免")
    if parse_errors:
        return 2
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
