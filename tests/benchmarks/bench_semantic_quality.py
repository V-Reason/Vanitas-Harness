"""ASMR 语义黄金靶场评测器（80 cue，双正则确定性裁决）。

用法::

    python tests/benchmarks/bench_semantic_quality.py --mode baseline
    python tests/benchmarks/bench_semantic_quality.py --mode critique -v
    python tests/benchmarks/bench_semantic_quality.py --model minicpm --port 4061 --mode baseline

设计边界（为什么和"跑一遍文件"不同）：

1. **初译只跑一遍**。``--mode critique`` 在同一批初译上做定向复审，而不是
   重跑第二遍翻译：同参数两次运行的 cue 级差异可达 60%，两遍对比会把采样
   噪声误算成"修复/误伤"。这样"反思改了什么"才是 100% 可归因的。
2. **复审走生产通道**：``triage_cue_risk`` 分流 + ``execute_critique_repair``
   隔离复审 + 生产同源的 ``finalize`` 顺序（清洗→术语→人称→窄修复）。
   唯一的差别是省略 ``DynamicEntityEngine.enforce_locks``：靶场没有文档级
   实体表；真残留日文名仍会被 ``validate_cue_translation`` 的假名规则拦住。
3. **不管理服务端**。默认只探 ``/health``；``--ensure`` 才调用
   ``python -m vanitas serve``（会回收现有 llama-server 实例）。
4. **context_cues 只喂分流器**，不塞进待译文档：塞进去每条都要多一次请求，
   而它对"翻译"影响小、对"是否悬空没说完"的判断影响大。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import __version__ as VANITAS_VERSION  # noqa: E402
from vanitas import prompts  # noqa: E402
from vanitas.client import LlamaClient  # noqa: E402
from vanitas.config import load_config  # noqa: E402
from vanitas.quality import (  # noqa: E402
    Glossary,
    TranslationProfile,
    apply_pronoun_policy,
    detect_register,
    normalize_compare,
    repair_first_person_name,
    repair_quote_private,
)
from vanitas.translate import (  # noqa: E402
    CritiqueBudget,
    T1Budget,
    execute_critique_repair,
    make_engine_chat,
    parse_srt,
    postprocess_cue_text,
    translate_srt,
    triage_cue_risk,
    validate_cue_translation,
)

DEFAULT_TSV = ROOT / "tests" / "data" / "golden_asmr_benchmark.tsv"
DEFAULT_REPORT_DIR = ROOT / "tests" / "reports"
DEFAULT_WINDOW = 5
COLUMNS = 7
CATEGORY_ORDER = ("direction", "negation", "pronoun", "conjunction", "lexicon")
CATEGORY_LABELS = {
    "direction": "CAT-A 授受/使役主客倒装",
    "negation": "CAT-B 否定疑问反诘语气",
    "pronoun": "CAT-C 隐晦代词引用归属",
    "conjunction": "CAT-D 跨行接续语流省略",
    "lexicon": "CAT-E 专名一致文化保真",
}
ID_PREFIX = {
    "direction": "CAT_A_",
    "negation": "CAT_B_",
    "pronoun": "CAT_C_",
    "conjunction": "CAT_D_",
    "lexicon": "CAT_E_",
}
CRITIQUE_LENGTH_LIMIT = 1.8
EXIT_OK = 0
EXIT_TRANSPORT = 1
EXIT_DATA = 2
EXIT_SERVICE = 3


class BenchDataError(Exception):
    """TSV 结构/正则/金标自检失败；带行号。"""


@dataclass(frozen=True)
class GoldenCue:
    id: str
    category: str
    source: str
    context_cues: tuple[str, ...]
    expected_pattern: str
    forbidden_pattern: str
    reference_zh: str

    def matches_expected(self, text: str) -> bool:
        return re.search(self.expected_pattern, text) is not None

    def hits_forbidden(self, text: str) -> bool:
        return bool(self.forbidden_pattern) and (
            re.search(self.forbidden_pattern, text) is not None
        )


@dataclass
class CueEvalResult:
    cue: GoldenCue
    draft_text: str
    final_text: str
    transition: str
    passed_draft: bool
    passed_final: bool
    reason_draft: str | None
    reason_final: str | None
    elapsed_s: float = 0.0


@dataclass
class BenchmarkReport:
    mode: str
    window: int
    model: str
    profile: str
    port: int
    elapsed_s: float
    tokens: int = 0
    stat_elapsed_s: float = 0.0
    critique_used: int = 0
    critique_cap: int = 0
    critique_requests: int = 0
    notes: list[str] = field(default_factory=list)
    results: list[CueEvalResult] = field(default_factory=list)

    # ---------------- 聚合 ----------------
    def subset(self, category: str) -> list[CueEvalResult]:
        return [item for item in self.results if item.cue.category == category]

    @staticmethod
    def _rate(hit: int, total: int) -> float | None:
        return None if total <= 0 else hit / total

    def category_row(self, category: str) -> dict[str, Any]:
        items = self.subset(category)
        passed_draft = sum(1 for item in items if item.passed_draft)
        passed_final = sum(1 for item in items if item.passed_final)
        return {
            "category": category,
            "label": CATEGORY_LABELS[category],
            "cues": len(items),
            "passed_draft": passed_draft,
            "passed_final": passed_final,
            "draft_rate": self._rate(passed_draft, len(items)),
            "final_rate": self._rate(passed_final, len(items)),
            "repaired": sum(1 for item in items if item.transition == "repaired"),
            "degraded": sum(1 for item in items if item.transition == "degraded"),
        }

    def rows(self) -> list[dict[str, Any]]:
        return [self.category_row(cat) for cat in CATEGORY_ORDER if self.subset(cat)]

    @property
    def total_cues(self) -> int:
        return len(self.results)

    @property
    def passed_draft(self) -> int:
        return sum(1 for item in self.results if item.passed_draft)

    @property
    def passed_final(self) -> int:
        return sum(1 for item in self.results if item.passed_final)

    @property
    def repaired(self) -> int:
        return sum(1 for item in self.results if item.transition == "repaired")

    @property
    def degraded(self) -> int:
        return sum(1 for item in self.results if item.transition == "degraded")

    @property
    def repair_rate(self) -> float | None:
        failed = self.total_cues - self.passed_draft
        return self._rate(self.repaired, failed)

    @property
    def degrade_rate(self) -> float | None:
        return self._rate(self.degraded, self.passed_draft)

    @property
    def tps(self) -> float:
        return 0.0 if self.stat_elapsed_s <= 0 else self.tokens / self.stat_elapsed_s

    def hard_defects(self) -> dict[str, int]:
        counted: dict[str, int] = {}
        for item in self.results:
            reason = validate_cue_translation(item.final_text)
            if reason is None:
                continue
            counted[reason] = counted.get(reason, 0) + 1
        return counted


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def _unescape(cell: str) -> str:
    return cell.replace("\\n", "\n")


def load_golden_benchmark(tsv_path: Path) -> list[GoldenCue]:
    """严格解析 TSV；任何结构/正则/金标问题都抛 BenchDataError（带行号）。"""
    path = Path(tsv_path)
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise BenchDataError(f"{path}: 读取失败：{exc}") from exc
    cues: list[GoldenCue] = []
    seen: set[str] = set()
    for line_no, line in enumerate(raw.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cells = line.split("\t")
        if len(cells) != COLUMNS:
            raise BenchDataError(
                f"{path}:{line_no}: 需要 {COLUMNS} 列，实得 {len(cells)} 列"
            )
        cid, category, source, context, expected, forbidden, reference = (
            cell.strip() for cell in cells
        )
        missing = [
            name
            for name, value in (
                ("id", cid),
                ("category", category),
                ("source", source),
                ("expected_pattern", expected),
                ("reference_zh", reference),
            )
            if not value
        ]
        if missing:
            raise BenchDataError(f"{path}:{line_no}: 字段为空：{missing}")
        if category not in CATEGORY_ORDER:
            raise BenchDataError(
                f"{path}:{line_no}: 未知 category：{category}"
            )
        if not cid.startswith(ID_PREFIX[category]):
            raise BenchDataError(
                f"{path}:{line_no}: id 前缀与 category 不一致：{cid}/{category}"
            )
        if cid in seen:
            raise BenchDataError(f"{path}:{line_no}: id 重复：{cid}")
        seen.add(cid)
        for label, pattern in (("expected", expected), ("forbidden", forbidden)):
            try:
                re.compile(pattern)
            except re.error as exc:
                raise BenchDataError(
                    f"{path}:{line_no}: {label}_pattern 非法：{exc}"
                ) from exc
        if not re.search(r"[\u3040-\u30ff]", source):
            raise BenchDataError(f"{path}:{line_no}: source 不含假名，疑似非日文")
        contexts = ()
        if context.upper() != "NONE":
            contexts = tuple(
                part for part in _unescape(context).split("\n") if part.strip()
            )
        cue = GoldenCue(
            id=cid,
            category=category,
            source=_unescape(source),
            context_cues=contexts,
            expected_pattern=expected,
            forbidden_pattern=forbidden,
            reference_zh=reference,
        )
        reason = validate_cue_translation(cue.reference_zh)
        if reason is not None:
            raise BenchDataError(
                f"{path}:{line_no}: reference_zh 未过硬校验（{reason}）"
            )
        if not cue.matches_expected(cue.reference_zh):
            raise BenchDataError(
                f"{path}:{line_no}: reference_zh 不满足 expected_pattern"
            )
        if cue.hits_forbidden(cue.reference_zh):
            raise BenchDataError(
                f"{path}:{line_no}: reference_zh 命中 forbidden_pattern"
            )
        cues.append(cue)
    if not cues:
        raise BenchDataError(f"{path}: 没有任何数据行")
    return cues


# --------------------------------------------------------------------------- #
# Oracle
# --------------------------------------------------------------------------- #
def evaluate_cue(text: str, cue: GoldenCue) -> tuple[bool, str | None]:
    """确定性裁决：硬校验 → 必要特征 → 禁忌特征。"""
    if not text or not text.strip():
        return False, "译文为空"
    hard_reason = validate_cue_translation(text)
    if hard_reason is not None:
        return False, f"硬校验失败：{hard_reason}"
    if not cue.matches_expected(text):
        return False, f"未匹配必要特征：{cue.expected_pattern}"
    if cue.hits_forbidden(text):
        return False, f"触发禁忌特征：{cue.forbidden_pattern}"
    return True, None


# --------------------------------------------------------------------------- #
# 合成 SRT
# --------------------------------------------------------------------------- #
def _timestamp(ms: int) -> str:
    hour, rest = divmod(ms, 3_600_000)
    minute, rest = divmod(rest, 60_000)
    second, milli = divmod(rest, 1000)
    return f"{hour:02d}:{minute:02d}:{second:02d},{milli:03d}"


def build_benchmark_srt(cues: Sequence[GoldenCue]) -> str:
    """把靶场 cue 拼成一段合法 SRT；序号即 1..N，供回读按 index 对齐。"""
    blocks: list[str] = []
    for position, cue in enumerate(cues, start=1):
        start = position * 3000
        end = start + 2000
        body = cue.source.replace("\n", "\n")
        blocks.append(
            f"{position}\n{_timestamp(start)} --> {_timestamp(end)}\n{body}\n"
        )
    return "\n".join(blocks)


def _index_translations(srt_text: str) -> dict[int, str]:
    table: dict[int, str] = {}
    for subtitle, _separator in parse_srt(srt_text, "\n"):
        try:
            index = int(subtitle.index)
        except (TypeError, ValueError):
            continue
        table[index] = subtitle.text
    return table


# --------------------------------------------------------------------------- #
# 生产同源后处理 / 候选验收
# --------------------------------------------------------------------------- #
def make_finalize(
    glossary: Glossary | None, profile: TranslationProfile
) -> Callable[[str, str], str]:
    """复刻 ``translate_srt.finalize`` 的顺序，仅省略 entity.enforce_locks。"""

    def finalize(zh_text: str, source_text: str) -> str:
        text = postprocess_cue_text(zh_text, source_text)
        if glossary is not None:
            text = glossary.apply(source_text, text)
        text = apply_pronoun_policy(text, profile)
        text = repair_first_person_name(source_text, text)
        return repair_quote_private(source_text, text)

    return finalize


def make_accept() -> Callable[[str, str, str, Iterable[str]], bool]:
    """复审候选验收：硬校验 + 1.8× 源文长度 + 与初译规范化后不同。"""

    def accept(
        source: str, draft: str, candidate: str, _risk_codes: Iterable[str] = ()
    ) -> bool:
        if not candidate or not candidate.strip():
            return False
        if validate_cue_translation(candidate) is not None:
            return False
        source_norm = normalize_compare(source)
        cand_norm = normalize_compare(candidate)
        if not cand_norm:
            return False
        if source_norm and len(cand_norm) > CRITIQUE_LENGTH_LIMIT * len(source_norm):
            return False
        return cand_norm != normalize_compare(draft)

    return accept


def classify_transition(
    passed_draft: bool, passed_final: bool, changed: bool
) -> str:
    """把"初译/终译是否通过 + 复审是否改文"映射成状态流转标签。

    ``changed`` 是复审真的改写了文本（不是"发了请求"）；未改文时初译与终译
    必然同判，标签只用来区分"没被分流"与"复审后仍不合格"。
    """
    if changed:
        if passed_final and not passed_draft:
            return "repaired"
        if passed_draft and not passed_final:
            return "degraded"
        return "rejected"
    return "kept" if passed_draft else "rejected"


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_benchmark(
    cues: Sequence[GoldenCue],
    chat_factory: Callable[[Callable[[Any], None] | None], Any],
    *,
    mode: str = "baseline",
    window: int = DEFAULT_WINDOW,
    critique_budget_cap: int | None = None,
    force_critique: bool = False,
    verbose: bool = False,
    log: Callable[[str], None] | None = None,
) -> BenchmarkReport:
    """跑一次评测；``chat_factory`` 是唯一外部依赖注入点（真机/Mock 共用）。

    ``force_critique`` 只用于诊断：对**初译未通过**的 cue 无视
    ``triage_cue_risk`` 的空结果强行复审，用来度量"复审通道本身的上限"。
    它不代表 P3 的线上行为（线上严格受 triage 门禁与预算约束）。
    """
    if mode not in ("baseline", "critique"):
        raise BenchDataError(f"未知 mode：{mode}")
    if window < 1:
        raise BenchDataError("--window 必须 ≥ 1")
    collected: list[Any] = []

    def on_stat(stat: Any) -> None:
        collected.append(stat)

    chat = chat_factory(on_stat)
    profile = TranslationProfile(
        register=detect_register([cue.source for cue in cues]),
        pronoun="你" if detect_register([cue.source for cue in cues]) == "intimate" else "auto",
    )
    glossary = getattr(chat, "glossary", None) or Glossary()
    srt_text = build_benchmark_srt(cues)
    failures: list[Any] = []
    t1_budget = T1Budget.for_corpus(len(cues))
    started = time.perf_counter()
    output = translate_srt(
        chat,
        srt_text,
        chunk_chars=800,
        source="<golden-benchmark>",
        eol="\n",
        failures=failures,
        window_size=window,
        max_window_chars=250,
        join_rules=(),
        asmr_guard=True,
        t1_budget=t1_budget,
        file_cues=len(cues),
    )
    translations = _index_translations(output)
    drafts: dict[str, str] = {}
    for position, cue in enumerate(cues, start=1):
        drafts[cue.id] = translations.get(position, "")

    results: list[CueEvalResult] = []
    critique_used = 0
    critique_cap = 0
    critique_requests = 0
    for position, cue in enumerate(cues, start=1):
        draft = drafts[cue.id]
        passed_draft, reason_draft = evaluate_cue(draft, cue)
        results.append(
            CueEvalResult(
                cue=cue,
                draft_text=draft,
                final_text=draft,
                transition="unscored",
                passed_draft=passed_draft,
                passed_final=passed_draft,
                reason_draft=reason_draft,
                reason_final=reason_draft,
            )
        )

    notes: list[str] = []
    if mode == "critique":
        budget = (
            CritiqueBudget.for_corpus(len(cues))
            if critique_budget_cap is None
            else CritiqueBudget(
                max_total=critique_budget_cap, per_file=critique_budget_cap
            )
        )
        budget.begin_file("<golden-benchmark>", len(cues))
        if critique_budget_cap is not None:
            # begin_file 会按 10% 重算 per_file；显式覆盖时以调用方为准。
            budget.max_total = critique_budget_cap
            budget.per_file = critique_budget_cap
        critique_cap = min(budget.max_total, budget.per_file)
        finalize = make_finalize(glossary, profile)
        accept = make_accept()
        for result in results:
            cue = result.cue
            codes = triage_cue_risk(
                cue.source, result.draft_text, cue.context_cues
            )
            if force_critique and not result.passed_draft:
                codes = codes | {"forced"}
            if not codes:
                result.transition = "kept" if result.passed_draft else "rejected"
                continue
            cue_started = time.perf_counter()
            candidate = execute_critique_repair(
                chat,
                cue=cue.id,
                source=cue.source,
                draft=result.draft_text,
                context=(),
                risk_codes=codes,
                critique_budget=budget,
                finalize=finalize,
                accept=accept,
            )
            result.elapsed_s = time.perf_counter() - cue_started
            if candidate != result.draft_text:
                critique_used += 1
            result.final_text = candidate
            passed_final, reason_final = evaluate_cue(candidate, cue)
            result.passed_final = passed_final
            result.reason_final = reason_final
            result.transition = classify_transition(
                result.passed_draft,
                passed_final,
                candidate != result.draft_text,
            )
            if verbose and log is not None:
                log(
                    f"  {cue.id} {sorted(codes)} -> {result.transition}"
                )
        notes.append(
            "critique 只对 triage 命中的 cue 发请求；预算耗尽后直通初译。"
        )
        critique_requests = budget.total

    elapsed = time.perf_counter() - started
    tokens = sum(int(getattr(stat, "tokens", 0) or 0) for stat in collected)
    stat_elapsed = sum(float(getattr(stat, "elapsed_s", 0.0) or 0.0) for stat in collected)
    if failures:
        notes.append(f"translate_srt 记录 {len(failures)} 条失败（保留原文）。")
    return BenchmarkReport(
        mode=mode,
        window=window,
        model="",
        profile="",
        port=0,
        elapsed_s=elapsed,
        tokens=tokens,
        stat_elapsed_s=stat_elapsed,
        critique_used=critique_used,
        critique_cap=critique_cap,
        critique_requests=critique_requests,
        notes=notes,
        results=results,
    )


# --------------------------------------------------------------------------- #
# 看板
# --------------------------------------------------------------------------- #
def _pct(value: float | None) -> str:
    return "  —  " if value is None else f"{value * 100:5.1f}%"


def _rate_cell(hit: int, total: int) -> str:
    if total <= 0:
        return "  —  "
    return f"{hit} ({hit / total * 100:.0f}%)"


def render_dashboard(report: BenchmarkReport) -> str:
    title = "Vanitas-Harness 语义质量基准评测报告 (ASMR 80-cue)"
    in_critique = report.mode == "critique"

    def final_cell(row: dict[str, Any]) -> str:
        if not in_critique:
            return "  —  "
        return _rate_cell(row["passed_final"], row["cues"])

    def pair_cell(value: int, prefix: str = "") -> str:
        if not in_critique:
            return "—"
        return f"{prefix}{value}" if value else "0"

    lines = [
        "=" * 100,
        title.center(100),
        "=" * 100,
        (
            f"模型: {report.model} · 档位: {report.profile} · 模式: {report.mode}"
            f" · 样本数: {report.total_cues} · window: {report.window}"
            f" · 耗时: {report.elapsed_s:.1f}s ({report.tps:.1f} tok/s)"
        ),
        "-" * 100,
        f"{'战区分类':<26}{'样本数':>8}{'基线通过':>12}{'反思通过':>12}{'修复':>10}{'误伤':>10}",
        "-" * 100,
    ]
    for row in report.rows():
        lines.append(
            f"{row['label']:<26}{row['cues']:>8}"
            f"{_rate_cell(row['passed_draft'], row['cues']):>12}"
            f"{final_cell(row):>12}"
            f"{pair_cell(row['repaired'], '+'):>10}"
            f"{pair_cell(row['degraded']):>10}"
        )
    lines.append("-" * 100)
    lines.append(
        f"{'综合指标 (TOTAL)':<26}{report.total_cues:>8}"
        f"{_rate_cell(report.passed_draft, report.total_cues):>12}"
        f"{final_cell({'passed_final': report.passed_final, 'cues': report.total_cues}):>12}"
        f"{pair_cell(report.repaired, '+'):>10}"
        f"{pair_cell(report.degraded):>10}"
    )
    lines.append("-" * 100)
    if in_critique:
        lines.append(
            f"修复率 {_pct(report.repair_rate)} · 误伤率 {_pct(report.degrade_rate)}"
            f" · 复审请求 {report.critique_requests}/{report.critique_cap or '—'}"
        f" · 采纳改判 {report.critique_used}"
        )
    else:
        lines.append(
            "复审未启用（--mode baseline）：修复率/误伤率不适用；"
            "加上 --mode critique 可度量 P3 的净收益与误伤"
        )
    defects = report.hard_defects()
    if defects:
        detail = " · ".join(f"{name} {count}" for name, count in sorted(defects.items()))
    else:
        detail = "无"
    lines.append(f"硬性缺陷: {detail}")
    for note in report.notes:
        lines.append(f"· {note}")
    lines.append("=" * 100)
    delta = None
    if report.passed_draft or report.passed_final:
        delta = (
            report.passed_final - report.passed_draft
        ) / max(1, report.total_cues) * 100
    if report.mode == "critique" and delta is not None:
        lines.append(
            f"✓ 语义通过率变化 {delta:+.1f}pp"
            f" ({report.passed_draft / report.total_cues * 100:.1f}%"
            f" -> {report.passed_final / report.total_cues * 100:.1f}%)"
            f" · 误伤率 {_pct(report.degrade_rate).strip()}"
        )
    return "\n".join(lines)


def report_payload(report: BenchmarkReport) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "vanitas_version": VANITAS_VERSION,
        "model": report.model,
        "profile": report.profile,
        "port": report.port,
        "mode": report.mode,
        "window": report.window,
        "wall_clock_s": round(report.elapsed_s, 3),
        "tps": round(report.tps, 2),
        "critique_budget": {
            "requests": report.critique_requests,
            "adopted": report.critique_used,
            "cap": report.critique_cap,
        },
        "categories": report.rows(),
        "totals": {
            "cues": report.total_cues,
            "passed_draft": report.passed_draft,
            "passed_final": report.passed_final,
            "draft_rate": report._rate(report.passed_draft, report.total_cues),
            "final_rate": report._rate(report.passed_final, report.total_cues),
            "repaired": report.repaired,
            "degraded": report.degraded,
            "repair_rate": report.repair_rate,
            "degrade_rate": report.degrade_rate,
        },
        "hard_defects": report.hard_defects(),
        "notes": report.notes,
        "cues": [
            {
                "id": item.cue.id,
                "category": item.cue.category,
                "source": item.cue.source,
                "context_cues": list(item.cue.context_cues),
                "reference_zh": item.cue.reference_zh,
                "draft": item.draft_text,
                "final": item.final_text,
                "transition": item.transition,
                "passed_draft": item.passed_draft,
                "passed_final": item.passed_final,
                "reason_draft": item.reason_draft,
                "reason_final": item.reason_final,
            }
            for item in report.results
        ],
    }


# --------------------------------------------------------------------------- #
# 真机接线
# --------------------------------------------------------------------------- #
def build_chat_factory(
    *,
    model_id: str,
    profile_key: str,
    port: int | None,
    ensure: bool,
) -> tuple[Callable[[Callable[[Any], None] | None], Any], str, str, int]:
    config = load_config()
    model = config.model(model_id)
    profile = config.resolve_profile(model, profile_key)
    effective_port = port if port is not None else config.effective_port(model)
    client = LlamaClient.for_model(config, model, effective_port)
    if not client.health():
        if not ensure:
            raise SystemExit(
                f"[服务不可用] {model.id} 在 {client.endpoint} 无响应；"
                f"请先运行：python -m vanitas serve {model.id} --profile {profile.key}"
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "vanitas",
                "serve",
                model.id,
                "--profile",
                profile.key,
            ],
            cwd=str(ROOT),
            check=False,
        )
        if not client.health():
            raise SystemExit(f"[服务不可用] 启动后 {client.endpoint} 仍无响应")
    settings = config.settings.translate
    preset = settings.presets.get("asmr") or {}
    prompt_dir = config.config_dir / "prompts"
    system_text = prompts.system_prompt(prompt_dir)

    def factory(on_stat):
        return make_engine_chat(
            LlamaClient.for_model(config, model, effective_port),
            prompt_dir,
            system_text,
            n_ctx=profile.ctx,
            context_window=int(preset.get("context_window", settings.context_window)),
            context_summary_chars=int(
                preset.get("context_summary_chars", settings.context_summary_chars)
            ),
            min_output_tokens=settings.min_output_tokens,
            output_ratio=settings.output_ratio,
            prompt_safety_tokens=settings.prompt_safety_tokens,
            temperature=settings.temperature,
            max_split_depth=settings.max_split_depth,
            on_stat=on_stat,
            task_template="translate_asmr",
        )

    return factory, model.id, profile.key, effective_port


def _select(cues: list[GoldenCue], only: str | None) -> list[GoldenCue]:
    if not only:
        return cues
    wanted = {token.strip().upper() for token in only.split(",") if token.strip()}
    selector = {
        "CAT_A": "direction",
        "CAT_B": "negation",
        "CAT_C": "pronoun",
        "CAT_D": "conjunction",
        "CAT_E": "lexicon",
    }
    categories = {selector.get(token, token.lower()) for token in wanted}
    selected = [cue for cue in cues if cue.category in categories]
    if not selected:
        raise BenchDataError(f"--only {only} 没有匹配任何样本")
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ASMR 语义黄金靶场评测（80 cue 双正则裁决）"
    )
    parser.add_argument("--mode", choices=("baseline", "critique"), default="baseline")
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--port", type=int, default=None, help="服务端口，默认取配置")
    parser.add_argument("--model", default="hymt", help="模型 id，默认 hymt")
    parser.add_argument("--profile", default="2", help="档位编号，默认 2")
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="合流窗口大小；1 为逐句（慢 4~5 倍，仅调试）",
    )
    parser.add_argument(
        "--critique-budget",
        type=int,
        default=None,
        help="覆盖复审预算上限；默认按 for_corpus(8%) 推导",
    )
    parser.add_argument("--only", default=None, help="只跑某战区：CAT_A,CAT_B…")
    parser.add_argument(
        "--force-critique",
        action="store_true",
        help="诊断用：对未通过的 cue 强制复审，绕过 triage 门禁（非线上行为）",
    )
    parser.add_argument("--out", type=Path, default=None, help="JSON 报告路径")
    parser.add_argument("--ensure", action="store_true", help="服务未起时自动 serve")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    try:
        cues = _select(load_golden_benchmark(args.tsv), args.only)
    except BenchDataError as exc:
        print(f"✗ 数据错误：{exc}", file=sys.stderr)
        return EXIT_DATA

    try:
        factory, model_id, profile_key, port = build_chat_factory(
            model_id=args.model,
            profile_key=args.profile,
            port=args.port,
            ensure=args.ensure,
        )
    except SystemExit as exc:  # 服务不可用
        message = str(exc.code) if exc.code is not None else ""
        if message:
            print(message, file=sys.stderr)
        return EXIT_SERVICE

    try:
        report = run_benchmark(
            cues,
            factory,
            mode=args.mode,
            window=args.window,
            critique_budget_cap=args.critique_budget,
            force_critique=args.force_critique,
            verbose=args.verbose,
            log=print,
        )
    except BenchDataError as exc:
        print(f"✗ 数据错误：{exc}", file=sys.stderr)
        return EXIT_DATA
    except Exception as exc:  # noqa: BLE001 - 顶层兜底，转为可读错误
        print(f"✗ 评测中断：{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_TRANSPORT

    report.model = model_id
    report.profile = profile_key
    report.port = port
    print(render_dashboard(report))

    out_path = args.out
    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = DEFAULT_REPORT_DIR / f"semantic_eval_{stamp}.json"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report_payload(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"✗ JSON 落盘失败：{out_path}（{exc}）", file=sys.stderr)
        return EXIT_TRANSPORT
    print(f"产物留档: {out_path}")

    if args.verbose:
        for item in report.results:
            if item.passed_final:
                continue
            print(f"\n✗ {item.cue.id} [{item.cue.category}] {item.transition}")
            print(f"  源文：{item.cue.source}")
            print(f"  金标：{item.cue.reference_zh}")
            print(f"  初译：{item.draft_text}")
            print(f"  终译：{item.final_text}")
            print(f"  原因：{item.reason_final}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
