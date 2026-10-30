"""数据契约：全部 dataclass、预算对象、失败记录与事件结构。

本模块是 ``vanitas.translate`` 子包的绝对叶子，只依赖标准库。任何
``from .xxx import`` 都会立刻制造循环引用，因此这里不出现同级导入；
约定（字段名、默认值、frozen 与否）一旦发布即为外部契约。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

# ASR 平台套话/幻觉：命中即置空、不发请求。契约层持有该 pattern，是因为
# ``translate_srt``（快照前过滤与语域检测）与 QC 语义必须用同一份规则，
# 两处各写一份必然漂移。
ASR_HALLUCINATION_PATTERNS = re.compile(
    r"(?:ご視聴ありがとうございました|チャンネル登録|高評価|ベルマーク|Thank you for watching|Please subscribe|Sous-titres)",
    re.IGNORECASE,
)
BOM = b"\xef\xbb\xbf"


@dataclass
class TranslationFailure:
    source: str
    segment: str
    reason: str


@dataclass(frozen=True)
class T1Event:
    """一条 T1 守卫命中记录；P4 用 (source, cue) join 评委结果。"""

    cue: int | str
    code: str
    outcome: str  # accepted / rejected / error / budget
    detail: str = ""
    source: str = ""


@dataclass
class T1Budget:
    """候选级 T1 重译预算：整包 max_total，单文件 per_file。

    由 CLI 在候选开始时创建一个对象并跨文件复用；``translate_srt`` 每次
    文件开始调用 ``begin_file(source)`` 重置单文件计数与来源，``total``
    保留整包计数。``file_events`` 只保留当前文件的事件，供
    ``TranslationOutcome`` 回传；事件带 ``source`` 以区分同号 cue。
    """

    max_total: int = 5
    per_file: int = 2
    total: int = 0
    file_total: int = 0
    file_events: list[T1Event] = field(default_factory=list)
    current_source: str = ""

    @classmethod
    def for_corpus(cls, total_cues: int | None) -> "T1Budget":
        """按整包非空 cue 数缩放整包预算；未知/非正数回退 5/514 默认值。

        514 → round(5.14)=5（保持预注册 5/514）；1404 → round(14.04)=14。
        """
        try:
            cues = int(total_cues) if total_cues is not None else 0
        except (TypeError, ValueError):
            cues = 0
        if cues <= 0:
            return cls()
        return cls(max_total=max(5, round(cues * 0.01)))

    def begin_file(
        self, source: str = "", file_cues: int | None = None
    ) -> None:
        """重置单文件计数；file_cues 已知时按 1% 缩放 per_file。

        file_cues=712 → ceil(7.12)=8；149/51 → max(2, ceil(…))=2。
        未知/非正数保持旧默认 per_file=2。
        """
        self.file_total = 0
        self.file_events.clear()
        self.current_source = str(source)
        try:
            cues = int(file_cues) if file_cues is not None else 0
        except (TypeError, ValueError):
            cues = 0
        self.per_file = max(2, ceil(cues * 0.01)) if cues > 0 else 2

    def can_retry(self) -> bool:
        return self.total < max(0, int(self.max_total)) and self.file_total < max(
            0, int(self.per_file)
        )

    def record_attempt(self) -> None:
        self.total += 1
        self.file_total += 1

    def record_event(
        self,
        cue: int | str,
        codes: set[str] | frozenset[str],
        outcome: str,
        detail: str = "",
    ) -> None:
        for code in sorted(codes):
            self.file_events.append(
                T1Event(
                    cue=cue,
                    code=code,
                    outcome=outcome,
                    detail=detail,
                    source=self.current_source,
                )
            )


@dataclass
class TranslationOutcome:
    source: Path
    output: Path | None
    segments: int
    failures: list[TranslationFailure] = field(default_factory=list)
    t1_events: tuple[T1Event, ...] = ()


@dataclass(frozen=True)
class TranslationSource:
    """一个待翻译文件；relative_parent 用于保留层级，translated_stem 为译名。"""

    path: Path
    relative_parent: Path = field(default_factory=Path)
    translated_stem: str | None = None


@dataclass(frozen=True)
class TranslationPlan:
    source: TranslationSource
    out_dir: Path | None
    target: Path

    @property
    def label(self) -> str:
        parent = self.source.relative_parent
        if parent in (Path(), Path(".")):
            return self.source.path.name
        return (parent / self.source.path.name).as_posix()


@dataclass(frozen=True)
class NameMask:
    text: str
    literals: list[str]


@dataclass(frozen=True)
class NameTranslation:
    name: str
    fallback_reason: str | None = None


@dataclass(frozen=True)
class NameTarget:
    path: Path
    is_dir: bool
    translated_name: str | None = None


@dataclass(frozen=True)
class NameRename:
    old: Path
    new: Path


@dataclass
class Subtitle:
    index: str
    timing: str
    text: str


@dataclass(frozen=True)
class TranslationOptions:
    n_ctx: int
    context_window: int
    context_summary_chars: int
    min_output_tokens: int
    output_ratio: float
    prompt_safety_tokens: int
    temperature: float
    max_split_depth: int
