"""文件与路径规划：目录递归、层级保留、输出冲突阻断、文件名占位符遮罩与改名。

职责边界：只回答“哪些文件要处理、产物写到哪、名字怎么改”，
不读取字幕语义、不调用模型、不做译文质量判断（唯一例外是改名用的
``sanitize_filename_stem``：它校验的是“能不能当文件名”，不是译文质量）。
"""

from __future__ import annotations

import os
import re
import unicodedata
import uuid
from pathlib import Path

from .cleaner import FENCE_RE
from ..config import TranslateSettings
from ..errors import TranslateError
from .format import _detect_eol
from .types import (
    BOM,
    NameMask,
    NameRename,
    NameTarget,
    TranslationPlan,
    TranslationSource,
)

FILENAME_PROMPT = (
    "翻译下面的文件名主体。[[K0]]、[[K1]] 等占位符必须原样保留。"
    "只输出结果，不要解释、不要加引号。\n"
    "文件名：{name}"
)
FILENAME_SYSTEM_PROMPT = (
    "你是文件名翻译器。只翻译文件名中的普通英文/日文词，已有中文保持原样。"
    "[[K0]]、[[K1]] 等占位符代表必须原样保留的编号或符号，输出时原样保留。"
    "示例：Hello_World → 你好_世界；Chapter_01 → 第01章；"
    "photos_2024 → 照片_2024；トラック① → 曲目①；"
    "report_临时 → 报告_临时；The Clock → 时钟。"
    "只输出文件名主体，不解释、不加引号。"
)
FILENAME_MAX_TOKENS = 96
FILENAME_MAX_RETRIES = 1
FILENAME_MAX_CHARS = 80
FILENAME_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
FILENAME_PREFIX = re.compile(
    r"^(?:(?:文件名|file name|翻译结果|翻译|译文|译名)\s*[:：]\s*)",
    re.IGNORECASE,
)
FILENAME_EXPLANATION = re.compile(
    r"(?:翻译|译文|译名|file name)\s*[:：]\s*", re.IGNORECASE
)
FILENAME_PLACEHOLDER = re.compile(r"\[\[K(\d+)\]\]")
FILENAME_KANA = re.compile(r"[\u3040-\u30ff]")
FILENAME_TOKEN = re.compile(r"[A-Za-z0-9+#&]+")
FILENAME_CODE_ID = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]{1,4}\d{2,}(?![A-Za-z0-9])"
)
FILENAME_CODE_VERSION = re.compile(
    r"(?<![A-Za-z0-9])[vV]\d+(?![A-Za-z0-9])"
)
FILENAME_CODE_HYPHEN = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]{1,4}[-_]\d+(?![A-Za-z0-9])"
)
FILENAME_CODE_NUMERIC = re.compile(r"(?<!\d)\d{2,}(?:[-_/.]\d{2,})+(?!\d)")
FILENAME_MARKER = re.compile(
    r"(?i)_(?:临时|final|draft|temp|copy|old|new|备份|草稿|保留)"
)
FILENAME_TECHNICAL_STEMS = frozenset(
    {"wav", "mp3", "mp4", "png", "jpg", "jpeg", "flac", "m4a", "txt"}
)
FILENAME_SYMBOL_CONNECTOR = re.compile(
    r"([♪♡❤💗✩★☆])(?:是|则|就)(?=[一-鿿])"
)

# 文件名元数据先验：音声作品的主角名几乎总写在文件名/目录名的括号里
# （``朝のひとりごと(茜音編).wav``），而 ASR 从气声里听出正确汉字写法的概率
# 极低（必然倾向常见字，把「茜音」听成「赤音」）。用文件系统既有元数据反哺
# 实体引擎，是全篇专名一致性成本最低的一环。括号解析见 ``_bracket_contents``
# （必须支持嵌套：`『晨间独白(茜音篇)』`）。
# 作品名里的通用后缀：剥掉后剩下的才是候选人名。
METADATA_SUFFIXES = (
    "ロングバージョン", "ショートバージョン", "ボイスドラマ", "ボイス",
    "ルート", "パート", "バージョン", "完全版", "限定版", "通常版",
    "版本", "姐妹", "姉妹", "姊妹",
    "編", "篇", "役", "ver", "Ver", "VER",
)
# 文件名主体本身当作候选的唯一理由：它带 RJ 编号或作品后缀。否则
# 「おはよう」「大好き」这类普通词也会被当成主角名锁进实体表。
METADATA_STEM_SUFFIXES = ("編", "篇", "役")
# 裸文件名的候选长度上限（含 RJ 编号与后缀），超出说明是整句作品标题。
METADATA_STEM_MAX_CHARS = 24
# 括号里常见但不是人名的标注；命中即丢弃，避免把场景词锁成专名。
METADATA_STOPWORDS = frozenset(
    {
        "本編", "前編", "後編", "完結編", "最終編", "新作", "最新", "作品",
        "音声", "実演", "効果音", "効果", "両耳", "左右", "右耳", "左耳",
        "字幕", "翻訳", "体験版", "無修正", "修正版", "初回", "特典",
        "おまけ", "オマケ", "ASMR", "asmr", "DL", "同梱", "差分",
        "笑", "溜息", "ため息", "喘ぎ", "吐息", "耳元", "耳舐め",
    }
)
METADATA_ID_RE = re.compile(r"(?i)(?<![A-Za-z])RJ\s*\d{4,}(?!\d)")
METADATA_JUNK_RE = re.compile(r"(?:第\s*\d+\s*[話话]|[Vv]ol\.?\s*\d+|part\s*\d+)")
METADATA_NAME_CHAR_RE = re.compile(r"^[\u3041-\u30ff\u4e00-\u9fa5]+$")
METADATA_MIN_CHARS = 2
METADATA_MAX_CHARS = 6
METADATA_OPEN = "（(【[〔「『"
METADATA_CLOSE = "）)】]〕」』"
METADATA_MAX_CANDIDATES = 8


def _bracket_contents(text: str) -> list[str]:
    """按括号配对取出全部括号内容，支持嵌套（内层优先）。

    非重叠正则会在真实文件名上失效：`特典4.语音①『晨间独白(茜音篇)』` 里外层
    `『』` 把内层 `(茜音篇)` 整段吞掉，只能拿到"晨间独白(茜音篇)"（含括号，
    字符集校验必挂），真正的人名反而取不到。这里用栈做配对，内外层都产出候选。
    """
    stack: list[int] = []
    found: list[tuple[int, int, str]] = []
    for index, char in enumerate(text):
        if char in METADATA_OPEN:
            stack.append(index)
        elif char in METADATA_CLOSE and stack:
            start = stack.pop()
            found.append((start, index, text[start + 1 : index]))
    # 内层（起点更靠后）优先；同起点时更靠内的闭合优先。
    found.sort(key=lambda item: (-item[0], item[1]))
    return [content for _start, _end, content in found][:METADATA_MAX_CANDIDATES]


def _clean_metadata_seed(candidate: str) -> str:
    """把一段括号文本收敛成候选人名；不可用时返回空串。"""
    text = candidate.strip()
    if not text:
        return ""
    text = METADATA_ID_RE.sub("", text)
    text = METADATA_JUNK_RE.sub("", text)
    text = text.strip(" 　・-—―_·.,、:：")
    changed = True
    while changed and text:
        changed = False
        for suffix in METADATA_SUFFIXES:
            if len(text) > len(suffix) and text.endswith(suffix):
                text = text[: -len(suffix)].strip(" 　・-—―_·")
                changed = True
                break
    if not METADATA_NAME_CHAR_RE.match(text):
        return ""
    if not METADATA_MIN_CHARS <= len(text) <= METADATA_MAX_CHARS:
        return ""
    if text in METADATA_STOPWORDS:
        return ""
    return text


def extract_metadata_entity_seeds(path: Path) -> list[str]:
    """从文件名与父目录名提取高置信度主角名种子。

    只认两种来源，宁可漏也不猜：
    1. 括号内文本（``(茜音編)`` / ``【双叶】`` / ``［あかね］``），剥掉通用后缀；
    2. 文件名主体本身就是 ``2~8`` 字且以 ``編/篇/役`` 结尾的短名（``茜音編.wav``）。

    返回去重后的列表（文件名优先于目录名）；无候选时返回空列表，
    调用方据此跳过预热，不额外分支。
    """
    path = Path(path)
    seeds: list[str] = []
    for index, raw in enumerate((path.stem, path.parent.name)):
        if not raw:
            continue
        candidates = _bracket_contents(raw)
        if not candidates and index == 0 and (
            METADATA_ID_RE.search(raw)
            or raw.rstrip().endswith(METADATA_STEM_SUFFIXES)
        ) and len(raw) <= METADATA_STEM_MAX_CHARS:
            candidates = [raw]
        for candidate in candidates:
            seed = _clean_metadata_seed(candidate)
            if seed and seed not in seeds:
                seeds.append(seed)
    return seeds


def _is_mask_symbol(char: str) -> bool:
    """遮罩装饰符号，但保留下划线/点/连字符作为文件名分隔符。"""
    if "\u2460" <= char <= "\u2473":  # ①..⑳
        return True
    if char in "_.-" or char.isalnum() or char.isspace():
        return False
    category = unicodedata.category(char)
    return category.startswith("P") or category.startswith("S")


def _mask_name_literals(stem: str) -> NameMask:
    spans: list[tuple[int, int, str]] = []

    def add_span(start: int, end: int, text: str) -> None:
        if start < end:
            spans.append((start, end, text))

    for pattern in (
        FILENAME_CODE_ID,
        FILENAME_CODE_VERSION,
        FILENAME_CODE_HYPHEN,
        FILENAME_CODE_NUMERIC,
    ):
        for match in pattern.finditer(stem):
            add_span(match.start(), match.end(), match.group(0))
    for token_match in FILENAME_TOKEN.finditer(stem):
        token = token_match.group(0)
        letters = [char for char in token if char.isalpha()]
        uppercase = sum(1 for char in letters if char.isupper())
        has_internal_upper = any(char.isupper() for char in token[1:])
        if uppercase >= 2 or has_internal_upper or any(
            char in "+#&" for char in token
        ):
            add_span(token_match.start(), token_match.end(), token)
    for match in FILENAME_MARKER.finditer(stem):
        add_span(match.start(), match.end(), match.group(0))
    leading_dots = re.match(r"^\.+", stem)
    if leading_dots:
        add_span(0, leading_dots.end(), leading_dots.group(0))
    trailing_dots = re.search(r"\.+$", stem)
    if trailing_dots:
        add_span(trailing_dots.start(), trailing_dots.end(), trailing_dots.group(0))
    for index, char in enumerate(stem):
        if _is_mask_symbol(char):
            add_span(index, index + 1, char)

    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: list[tuple[int, int, str]] = []
    for start, end, text in spans:
        if selected and start < selected[-1][1]:
            continue
        selected.append((start, end, text))

    pieces: list[str] = []
    literals: list[str] = []
    cursor = 0
    for start, end, text in selected:
        pieces.append(stem[cursor:start])
        pieces.append(f"[[K{len(literals)}]]")
        literals.append(text)
        cursor = end
    pieces.append(stem[cursor:])
    return NameMask(text="".join(pieces), literals=literals)


def _has_translatable_text(masked: str) -> bool:
    remainder = FILENAME_PLACEHOLDER.sub(" ", masked)
    return bool(re.search(r"[A-Za-z]{2,}|[\u3040-\u30ff]", remainder))


def _placeholder_failure(text: str, literals: list[str]) -> str | None:
    expected = [f"[[K{index}]]" for index in range(len(literals))]
    for token in expected:
        if text.count(token) != 1:
            return f"占位符缺失或重复：{token}"
    found = FILENAME_PLACEHOLDER.findall(text)
    if len(found) != len(expected):
        return "出现未知占位符"
    return None


def _restore_literals(text: str, literals: list[str]) -> str:
    for index, literal in enumerate(literals):
        text = text.replace(f"[[K{index}]]", literal, 1)
    return text


def _strip_filename_explanation(raw: str) -> str:
    text = raw.strip()
    match = FILENAME_EXPLANATION.search(text)
    if match:
        after = text[match.end() :].strip()
        if after:
            return after
    return text


def sanitize_filename_stem(raw: str, source_suffix: str = "") -> str:
    """把模型输出收敛成安全的单段文件名（不含扩展名）。"""
    text = raw.strip()
    for _ in range(3):
        match = FENCE_RE.match(text)
        if not match:
            break
        text = match.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    text = _strip_filename_explanation(lines[0])
    text = FILENAME_PREFIX.sub("", text)
    if source_suffix and text.lower().endswith(source_suffix.lower()):
        text = text[: -len(source_suffix)]
    text = FILENAME_INVALID_CHARS.sub("_", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or text in {".", ".."} or not text.strip("._- "):
        return ""
    return text


def _read_text(path: Path) -> tuple[str, bool, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TranslateError(f"读取失败：{path}；{exc.strerror or exc}") from exc
    has_bom = raw.startswith(BOM)
    try:
        text = raw[len(BOM):].decode("utf-8") if has_bom else raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TranslateError(
            f"文件不是 UTF-8：{path}；第 {exc.start} 字节，请先转码"
        ) from exc
    return text, has_bom, _detect_eol(text)


def output_path(
    source: Path,
    out_dir: Path | None,
    suffix: str,
    stem: str | None = None,
) -> Path:
    if stem is not None:
        base = sanitize_filename_stem(stem, source.suffix) or source.stem
    else:
        base = source.stem
    name = f"{base}{suffix}{source.suffix}"
    return (out_dir / name) if out_dir is not None else (source.parent / name)


def _path_key(path: Path) -> str:
    """跨 Windows 大小写/符号链接比较路径；目标文件通常尚不存在。"""
    return os.path.normcase(os.path.realpath(os.path.abspath(str(path))))


def _is_generated_output(path: Path, output_suffix: str) -> bool:
    suffix = (output_suffix or "").lower()
    return bool(suffix) and path.stem.lower().endswith(suffix)


def _is_partial_snapshot(path: Path) -> bool:
    """断点快照 `<output>.part.srt` 不是输入；否则续传残片会被再翻一遍。"""
    return path.name.lower().endswith(".part.srt")


def _is_excluded_dir(candidate: Path, excluded_key: str | None) -> bool:
    """仅剪掉 --out-dir 本身，避免把上次产物再次当输入。"""
    if not excluded_key:
        return False
    return _path_key(candidate) == excluded_key


def collect_translate_inputs(
    inputs: list[Path],
    settings: TranslateSettings,
    *,
    exclude_dir: Path | None = None,
) -> list[TranslationSource]:
    """把文件/目录输入展开为去重后的文件列表。

    - 目录递归遍历，只收 settings.source_suffixes，显式文件不按后缀过滤；
    - 目录中的 ``*.zh.md`` 等已有产物跳过，避免自我递归；
    - 同名文件按真实路径去重，首次出现的目录层级决定输出层级；
    - 空目录与不可遍历目录在启动模型前报错；显式文件仍由 translate_file 判定。
    """
    allowed = frozenset(item.lower() for item in settings.source_suffixes)
    excluded_key = _path_key(exclude_dir) if exclude_dir is not None else None
    collected: list[TranslationSource] = []
    seen: set[str] = set()

    def add(path: Path, relative_parent: Path) -> bool:
        key = _path_key(path)
        if key in seen:
            return False
        seen.add(key)
        collected.append(TranslationSource(path=path, relative_parent=relative_parent))
        return True

    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            matched = 0
            errors: list[OSError] = []

            def on_error(exc: OSError) -> None:
                errors.append(exc)

            for root, dirs, files in os.walk(
                path, topdown=True, onerror=on_error, followlinks=False
            ):
                root_path = Path(root)
                dirs[:] = sorted(
                    (
                        name
                        for name in dirs
                        if not _is_excluded_dir(root_path / name, excluded_key)
                    ),
                    key=str.lower,
                )
                files.sort(key=str.lower)
                for name in files:
                    candidate = root_path / name
                    if candidate.suffix.lower() not in allowed:
                        continue
                    if _is_generated_output(candidate, settings.output_suffix):
                        continue
                    if _is_partial_snapshot(candidate):
                        continue
                    if not candidate.is_file():
                        continue
                    matched += 1
                    add(candidate, candidate.parent.relative_to(path))
            if errors:
                first = errors[0]
                detail = first.strerror or str(first)
                raise TranslateError(
                    f"遍历目录失败：{first.filename or path}；{detail}"
                ) from first
            if matched == 0:
                suffixes = "、".join(settings.source_suffixes)
                raise TranslateError(
                    f"目录中没有可翻译文件：{path}；支持：{suffixes}"
                )
        else:
            # 显式文件保持旧契约：存在性、后缀与可读性仍由 translate_file 判定。
            # 这里不能提前拒绝，否则调用方替换/延迟写盘的场景会被误伤。
            add(path, Path())

    if not collected:
        raise TranslateError("没有可翻译的输入")
    return collected


def collect_name_targets(inputs: list[Path]) -> list[NameTarget]:
    """收集只改名模式的目标：显式文件本身 + 目录下的文件/子目录。

    传入的根目录本身不重命名，避免把用户当前目录或仓库根整体改掉；
    隐藏文件和隐藏目录不进入递归，防止误改 ``.git`` 等。
    """
    targets: list[NameTarget] = []
    seen: set[str] = set()

    def add(path: Path, is_dir: bool) -> None:
        key = _path_key(path)
        if key in seen:
            return
        seen.add(key)
        targets.append(NameTarget(path=path, is_dir=is_dir))

    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            found = 0
            errors: list[OSError] = []

            def on_error(exc: OSError) -> None:
                errors.append(exc)

            for root, dirs, files in os.walk(
                path, topdown=True, onerror=on_error, followlinks=False
            ):
                root_path = Path(root)
                dirs[:] = sorted(
                    (name for name in dirs if not name.startswith(".")),
                    key=str.lower,
                )
                files = sorted(
                    (name for name in files if not name.startswith(".")),
                    key=str.lower,
                )
                for name in dirs:
                    add(root_path / name, True)
                    found += 1
                for name in files:
                    child = root_path / name
                    if child.is_file() or child.is_symlink():
                        add(child, False)
                        found += 1
            if errors:
                first = errors[0]
                detail = first.strerror or str(first)
                raise TranslateError(
                    f"遍历目录失败：{first.filename or path}；{detail}"
                ) from first
            if found == 0:
                raise TranslateError(f"目录中没有可重命名的文件或子目录：{path}")
        elif path.is_file() or path.is_symlink():
            add(path, False)
        else:
            raise TranslateError(f"路径不存在或不是文件/目录：{path}")

    if not targets:
        raise TranslateError("没有可重命名的文件或目录")
    return targets


def plan_name_renames(targets: list[NameTarget]) -> list[NameRename]:
    """生成并校验只改名计划；不触碰文件内容。"""
    renames: list[NameRename] = []
    new_keys: dict[str, NameRename] = {}
    old_keys = {_path_key(target.path) for target in targets}
    moved_old: set[str] = set()

    for target in targets:
        if target.translated_name is None:
            raise TranslateError(f"缺少译名：{target.path}")
        old = target.path
        if target.is_dir:
            new_name = sanitize_filename_stem(target.translated_name, "")
        else:
            base = sanitize_filename_stem(target.translated_name, old.suffix)
            new_name = f"{base}{old.suffix}" if base else ""
        if not new_name:
            raise TranslateError(f"译名为空或不可用：{old}")
        if new_name == old.name:
            continue
        new_path = old.with_name(new_name)
        key = _path_key(new_path)
        if key in new_keys:
            raise TranslateError(
                f"译名冲突：{new_keys[key].old} 与 {old} 都会变成 {new_path}"
            )
        rename = NameRename(old=old, new=new_path)
        new_keys[key] = rename
        moved_old.add(_path_key(old))
        renames.append(rename)

    for rename in renames:
        key = _path_key(rename.new)
        if key in old_keys and key not in moved_old:
            raise TranslateError(
                f"译名会占用未移动的源名称：{rename.old} → {rename.new}；"
                "请调整译名或先移动同名目标"
            )
        if rename.new.exists() and key not in moved_old:
            raise TranslateError(
                f"目标名称已存在：{rename.new}；请调整译名或先移动同名目标"
            )
    return renames


def _rename_temp(path: Path) -> Path:
    for _ in range(10):
        candidate = path.with_name(f".vanitas-rename-{uuid.uuid4().hex}")
        if not candidate.exists():
            return candidate
    raise TranslateError(f"无法生成临时文件名：{path}")


def apply_name_renames(renames: list[NameRename]) -> None:
    """按深度从深到浅、每层两阶段重命名，支持交换与纯大小写变化。"""
    groups: dict[int, list[NameRename]] = {}
    for rename in renames:
        groups.setdefault(len(rename.old.parts), []).append(rename)

    for depth in sorted(groups, reverse=True):
        group = sorted(groups[depth], key=lambda item: str(item.old).lower())
        staged: list[tuple[NameRename, Path]] = []
        try:
            for rename in group:
                temp = _rename_temp(rename.old)
                rename.old.rename(temp)
                staged.append((rename, temp))
        except OSError as exc:
            for rename, temp in reversed(staged):
                try:
                    temp.rename(rename.old)
                except OSError:
                    pass
            raise TranslateError(
                f"重命名失败：{exc.filename or exc}；已尝试回滚当前层"
            ) from exc

        completed: list[tuple[NameRename, Path]] = []
        try:
            for rename, temp in staged:
                temp.rename(rename.new)
                completed.append((rename, temp))
        except OSError as exc:
            for rename, temp in reversed(completed):
                try:
                    rename.new.rename(temp)
                except OSError:
                    pass
            for rename, temp in reversed(staged):
                try:
                    if temp.exists():
                        temp.rename(rename.old)
                except OSError:
                    pass
            raise TranslateError(
                f"重命名失败：{exc.filename or exc}；已尝试回滚当前层"
            ) from exc


def plan_translation_outputs(
    sources: list[TranslationSource],
    settings: TranslateSettings,
    out_dir: Path | None,
    *,
    output_suffix: str | None = None,
) -> list[TranslationPlan]:
    """计算输出路径并在写盘前拦截冲突，避免静默覆盖。

    --out-dir 存在时，目录输入保留相对层级；直接指定的文件仍平铺在输出根。
    output_suffix 为 None 时使用 settings.output_suffix；传空串可用于“译名不加 .zh”。
    """
    suffix = settings.output_suffix if output_suffix is None else output_suffix
    plans: list[TranslationPlan] = []
    targets: dict[str, TranslationSource] = {}
    source_keys = {_path_key(item.path) for item in sources}

    for source in sources:
        target_dir = out_dir / source.relative_parent if out_dir is not None else None
        target = output_path(
            source.path,
            target_dir,
            suffix,
            source.translated_stem,
        )
        key = _path_key(target)
        previous = targets.get(key)
        if previous is not None:
            raise TranslateError(
                f"输出路径冲突：{previous.path} 与 {source.path} 都会写到 {target}；"
                "请调整 --out-dir 或输入目录"
            )
        if key in source_keys:
            raise TranslateError(
                f"输出会覆盖输入文件：{source.path} → {target}；"
                "请更换 --out-dir 或输出后缀"
            )
        targets[key] = source
        plans.append(
            TranslationPlan(source=source, out_dir=target_dir, target=target)
        )
    return plans
