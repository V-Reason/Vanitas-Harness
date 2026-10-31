"""文档契约护栏：阻止“旧值冒充当前值”和“历史被精简掉”。

为什么只做这 4 条：
1. 生产提示词是唯一被测试钉死字节的产物；文档里的字节数/指纹必须与真实模板一致，
   否则读者会把旧版当当前版，Agent 也会按旧契约行动。
2. 变更历史必须保留全量批次；没有下限，下一次“整理”就可能把 53 节压成一页摘要。
3. 旧指纹/旧字节只允许活在历史文件里；出现在当前文档就是以现在时陈述旧事实。
4. AGENTS 的规则区不应再长出按日期追加的批次小标题；批次叙事应迁到变更历史。
这些是结构断言，不是覆盖率；不测试 Markdown 排版，也不复制业务逻辑。
"""

from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "docs" / "项目现状.md"
HISTORY = ROOT / "docs" / "变更历史.md"
AGENTS = ROOT / "AGENTS.md"
PROMPT = ROOT / "config" / "prompts" / "task_translate_asmr.md"
HISTORICAL_FILES = {
    "变更历史.md",
    "ASMR语义质量冲刺_开发流程与实验复盘.md",
    "ASMR字幕生成与测试.md",
}


class DocsContractTests(unittest.TestCase):
    def test_status_prompt_fingerprint_matches_template(self) -> None:
        text = STATUS.read_text(encoding="utf-8")
        match = re.search(
            r"\|\s*生产提示词\s*\|\s*`?([\d,]+)\s*B\s*/\s*([0-9A-Fa-f]{8})",
            text,
        )
        self.assertIsNotNone(match, "项目现状.md 缺少“生产提示词”速查行")
        documented_bytes = int(match.group(1).replace(",", ""))
        documented_sha = match.group(2).upper()
        raw = PROMPT.read_bytes()
        self.assertEqual(documented_bytes, len(raw))
        self.assertEqual(documented_sha, hashlib.sha256(raw).hexdigest().upper()[:8])

    def test_history_keeps_all_sections(self) -> None:
        self.assertTrue(HISTORY.is_file(), "缺少 docs/变更历史.md")
        text = HISTORY.read_text(encoding="utf-8")
        self.assertIn("## 53.", text)
        self.assertGreaterEqual(len(text.splitlines()), 1000)

    def test_old_values_only_live_in_history(self) -> None:
        offenders: list[str] = []
        for path in ROOT.rglob("*.md"):
            if path.name in HISTORICAL_FILES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for old in ("CC2B4C82", "12,784"):
                if old in text:
                    offenders.append(f"{path.relative_to(ROOT)}: {old}")
        self.assertEqual(offenders, [], "旧值出现在历史文件之外")

    def test_agents_has_no_dated_batch_headings(self) -> None:
        text = AGENTS.read_text(encoding="utf-8")
        dated = [
            line
            for line in text.splitlines()
            if re.match(r"^#{1,6}\s", line) and re.search(r"2026-10-2|（2026-", line)
        ]
        self.assertEqual(dated, [], "AGENTS 中出现了日期型批次标题")

    def test_production_discipline_is_pinned(self) -> None:
        """生产纪律必须留在 AGENTS 正文明里，且 §7 的门禁第 4 步归所有者。

        这一条是纪律的机器护栏：删掉/软化 §5.1 或把真实冒烟写回“Agent 执行”
        都会红，防止下一次“顺手重跑一遍”又被当成正常流程。
        """
        agents = AGENTS.read_text(encoding="utf-8")
        self.assertIn("### 5.1 生产纪律", agents)
        for phrase in ("`translate` / `serve` / `stop`", "由所有者手动执行", "只读验收"):
            self.assertIn(phrase, agents)
        self.assertIn("真实冒烟（由项目所有者执行", agents)
        self.assertNotIn("# 4) 真实冒烟（至少一次）", agents)
        status = STATUS.read_text(encoding="utf-8")
        self.assertIn("生产纪律", status)


if __name__ == "__main__":
    unittest.main()
