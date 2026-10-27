"""只读 CLI 冒烟：不启动 llama-server，只验证命令分派与配置加载。

用法：`python -m tests.smoke_cli`
退出码：0 全通过 / 1 任一失败
"""
from __future__ import annotations

import subprocess
import sys


CASES: tuple[tuple[str, int], ...] = (
    ("--version", 0),
    ("--help", 0),
    ("models", 0),
    ("status", 0),
    ("agent --help", 0),
    ("chat --help", 0),
    ("translate --help", 0),
    ("batch --help", 0),
    ("serve --help", 0),
    ("stop --help", 0),
)


def run_one(args: str) -> tuple[int, str, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "vanitas", *args.split()],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode, completed.stdout, completed.stderr


def main() -> int:
    failures = 0
    for args, expected in CASES:
        code, _stdout, stderr = run_one(args)
        ok = code == expected
        status = "OK" if ok else "FAIL"
        print(f"[{status}] vanitas {args} → exit {code} (expected {expected})")
        if not ok:
            failures += 1
            if stderr.strip():
                print(f"  stderr: {stderr.strip()[:300]}")
    if failures:
        print(f"\n{failures} / {len(CASES)} 失败")
        return 1
    print(f"\n全部 {len(CASES)} 项通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
