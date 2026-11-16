"""公开仓库边界：身份、许可证、私有路径与模型资产不得回潮。"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA = ROOT / "tests" / "data"


class PublicReleaseTests(unittest.TestCase):
    def _tracked_files(self) -> list[str]:
        raw = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT
        ).decode("utf-8")
        return [item for item in raw.split("\0") if item]

    def test_public_benchmarks_are_explicitly_synthetic(self) -> None:
        for name in ("golden_asmr_benchmark.tsv", "qc_golden.tsv"):
            text = (PUBLIC_DATA / name).read_text(encoding="utf-8")
            self.assertIn("SYNTHETIC", text)
            self.assertNotIn("origin=" + "corpus", text)

    def test_models_binaries_and_private_corpus_are_not_tracked(self) -> None:
        forbidden_prefixes = ("bin/", "models/", "tests/private/", "tests/reports/")
        tracked = self._tracked_files()
        self.assertEqual(
            [path for path in tracked if path.startswith(forbidden_prefixes)],
            [],
        )

    def test_private_path_markers_are_absent(self) -> None:
        forbidden = (
            "T:" + "\\杂七杂八",
            "T:" + "\\Open-Source\\Vanitas-Harness",
            "D:" + "/ASMR",
            "Vanitas" + "-AGENT",
            "origin=" + "corpus",
        )
        hits: list[str] = []
        for relative in self._tracked_files():
            if relative == "tests/test_public_release.py":
                continue
            path = ROOT / relative
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for marker in forbidden:
                if marker in text:
                    hits.append(f"{relative}: {marker}")
        self.assertEqual(hits, [])

    def test_release_identity_and_license_are_present(self) -> None:
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Copyright (c) 2026 V-Reason", license_text)
        self.assertIn("# Vanitas-Harness", readme)
        self.assertIn("MIT License", readme)


if __name__ == "__main__":
    unittest.main()
