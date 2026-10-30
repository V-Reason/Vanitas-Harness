"""批量翻译：.txt / .md / .srt。

- 文本按空行分段，段内超 chunk_chars 再按句末标点切，拼回时保留原空行结构；
- SRT 只翻译字幕文本，序号、时间轴、空行与换行风格原样保留；
- 任一段失败不中断，最终汇总失败段并由 CLI 返回非零退出码。

本包由原单文件 ``vanitas/translate.py`` 按职责正交切分而来，对外仍是
**一个模块**：本文件是唯一的公开门面，``from vanitas.translate import X``
以及 ``mock.patch("vanitas.translate.X")`` 的既有用法必须保持可用。

内部分层（只允许向下依赖，禁止回流）：

    pipeline → engine → {window, io, entity, format, cleaner} → types

- ``types``     纯数据契约，不 import 同级模块；
- ``cleaner``   确定性清洗与写回硬校验（0 Token）；
- ``format``    SRT/纯文本结构解析、分块、EOL 归一化；
- ``window``    微批次 [#N] 协议与窗口错位熔断；
- ``entity``    文档级实体倒排与锁定名；
- ``io``        输入展开、输出规划、文件名遮罩与改名；
- ``engine``    单段翻译驱动（滑动上下文/预算/截断重译/文件名翻译）；
- ``pipeline``  端到端串联（文件、字幕窗口、审计、T1 预算）。

注意：本文件只做 re-export，**不是**内部调用的查找点。要给内部函数打
补丁，必须指向定义模块（如 ``vanitas.translate.pipeline.translate_srt``）；
patch 门面上的同名属性不会影响子模块内部的全局查找，会静默失效。
"""

from .cleaner import (
    ARTIFACT_LATIN_RE,
    BLANK_LINE_RUN_RE,
    FENCE_RE,
    LATIN_ALLOW,
    LATIN_TOKEN_RE,
    LEADING_INVALID_PUNCT_RE,
    QUOTE_CHARS,
    QUOTE_PAIRS,
    REPEATED_PUNCT_RE,
    SLASH_ALTERNATIVE_RE,
    SOURCE_HAS_BRACKETS_RE,
    STAGING_BRACKETS_RE,
    TRADITIONAL_CHARS,
    TRADITIONAL_TO_SIMPLIFIED,
    TRADITIONAL_TO_SIMPLIFIED_MAP,
    UNTRANSLATED_KANA_RE,
    _clip,
    _collapse_blank_lines,
    _has_dialogue_quotes,
    postprocess_cue_text,
    repair_slash_alternatives,
    salvage_slash_translation,
    sanitize_translation_output,
    validate_cue_translation,
)
from .engine import TranslationEngine, build_chat_for, extract_content, make_engine_chat
from .entity import (
    ENTITY_CONTEXT_CHARS,
    ENTITY_EXAMPLE_CHARS,
    ENTITY_HONORIFIC_SUFFIXES,
    ENTITY_MAX_CLUSTERS,
    ENTITY_MAX_EXAMPLES,
    ENTITY_MAX_VARIANTS,
    ENTITY_STOPWORDS,
    EXPLICIT_NAME_RE,
    HIRAGANA_HONORIFIC_RE,
    HIRAGANA_NAME_CHAR,
    HIRAGANA_TERM_RE,
    KANJI_CHUNK_RE,
    KANJI_KATAKANA_HONORIFIC_RE,
    KANJI_KATAKANA_NAME_CHAR,
    KATAKANA_NICKNAME_RE,
    KATAKANA_TERM_RE,
    NAME_BOUNDARY,
    NAME_CHAR_CLASS,
    DynamicEntityEngine,
    _base_of_variant,
    _common_prefix,
    _edit_distance,
    _explicit_alias,
    _is_name_variant,
    _kana_key,
    _seed_related,
)
from .format import (
    BLOCK_SEP,
    SENTENCE_ENDERS,
    SRT_TIME,
    _detect_eol,
    _normalize_eol,
    _split_for_retry,
    chunk_text,
    parse_srt,
    rebuild_srt,
    split_prose,
)
from .io import (
    FILENAME_CODE_HYPHEN,
    FILENAME_CODE_ID,
    FILENAME_CODE_NUMERIC,
    FILENAME_CODE_VERSION,
    FILENAME_EXPLANATION,
    FILENAME_INVALID_CHARS,
    FILENAME_KANA,
    FILENAME_MARKER,
    FILENAME_MAX_CHARS,
    FILENAME_MAX_RETRIES,
    FILENAME_MAX_TOKENS,
    FILENAME_PLACEHOLDER,
    FILENAME_PREFIX,
    FILENAME_PROMPT,
    FILENAME_SYMBOL_CONNECTOR,
    FILENAME_SYSTEM_PROMPT,
    FILENAME_TECHNICAL_STEMS,
    FILENAME_TOKEN,
    _has_translatable_text,
    _is_excluded_dir,
    _is_generated_output,
    _is_mask_symbol,
    _mask_name_literals,
    _path_key,
    _placeholder_failure,
    _read_text,
    _rename_temp,
    _restore_literals,
    _strip_filename_explanation,
    apply_name_renames,
    collect_name_targets,
    collect_translate_inputs,
    output_path,
    plan_name_renames,
    plan_translation_outputs,
    sanitize_filename_stem,
)
from .pipeline import (
    SINGLE_RETRY_HINT,
    T1_NEARBY_RADIUS,
    _translate_chunks,
    _translate_single_cue,
    translate_file,
    translate_prose,
    translate_srt,
)
from .types import (
    ASR_HALLUCINATION_PATTERNS,
    BOM,
    NameMask,
    NameRename,
    NameTarget,
    NameTranslation,
    Subtitle,
    T1Budget,
    T1Event,
    TranslationFailure,
    TranslationOptions,
    TranslationOutcome,
    TranslationPlan,
    TranslationSource,
)
from .window import (
    WINDOW_ID_ANY_RE,
    WINDOW_ID_RE,
    WindowAlignment,
    _copies_history,
    _normalize_for_compare,
    _parse_window_items,
    _source_texts_are_similar,
    _translation_is_too_short,
    _window_translation_is_suspicious,
    align_window_response,
    identify_suspicious_cues,
    parse_window_response,
    recover_window_response,
    strip_window_id_prefix,
)

__all__ = [
    # types
    "ASR_HALLUCINATION_PATTERNS", "BOM", "NameMask", "NameRename", "NameTarget",
    "NameTranslation", "Subtitle", "T1Budget", "T1Event", "TranslationFailure",
    "TranslationOptions", "TranslationOutcome", "TranslationPlan",
    "TranslationSource",
    # cleaner
    "ARTIFACT_LATIN_RE", "BLANK_LINE_RUN_RE", "FENCE_RE", "LATIN_ALLOW",
    "LATIN_TOKEN_RE", "LEADING_INVALID_PUNCT_RE", "QUOTE_CHARS", "QUOTE_PAIRS",
    "REPEATED_PUNCT_RE", "SLASH_ALTERNATIVE_RE", "SOURCE_HAS_BRACKETS_RE",
    "STAGING_BRACKETS_RE", "TRADITIONAL_CHARS", "TRADITIONAL_TO_SIMPLIFIED",
    "TRADITIONAL_TO_SIMPLIFIED_MAP", "UNTRANSLATED_KANA_RE", "_clip",
    "_collapse_blank_lines", "_has_dialogue_quotes", "postprocess_cue_text",
    "repair_slash_alternatives", "salvage_slash_translation",
    "sanitize_translation_output", "validate_cue_translation",
    # format
    "BLOCK_SEP", "SENTENCE_ENDERS", "SRT_TIME", "_detect_eol", "_normalize_eol",
    "_split_for_retry", "chunk_text", "parse_srt", "rebuild_srt", "split_prose",
    # window
    "WINDOW_ID_ANY_RE", "WINDOW_ID_RE", "_copies_history",
    "_normalize_for_compare", "_parse_window_items", "_source_texts_are_similar",
    "_translation_is_too_short", "_window_translation_is_suspicious",
    "WindowAlignment", "align_window_response", "identify_suspicious_cues",
    "parse_window_response", "recover_window_response", "strip_window_id_prefix",
    # entity
    "ENTITY_CONTEXT_CHARS", "ENTITY_EXAMPLE_CHARS", "ENTITY_HONORIFIC_SUFFIXES",
    "ENTITY_MAX_CLUSTERS", "ENTITY_MAX_EXAMPLES", "ENTITY_MAX_VARIANTS",
    "ENTITY_STOPWORDS", "EXPLICIT_NAME_RE", "HIRAGANA_HONORIFIC_RE",
    "HIRAGANA_NAME_CHAR", "HIRAGANA_TERM_RE", "KANJI_CHUNK_RE",
    "KANJI_KATAKANA_HONORIFIC_RE", "KANJI_KATAKANA_NAME_CHAR",
    "KATAKANA_NICKNAME_RE", "KATAKANA_TERM_RE", "NAME_BOUNDARY",
    "NAME_CHAR_CLASS", "DynamicEntityEngine", "_base_of_variant",
    "_common_prefix", "_edit_distance", "_explicit_alias", "_is_name_variant",
    "_kana_key", "_seed_related",
    # io
    "FILENAME_CODE_HYPHEN", "FILENAME_CODE_ID", "FILENAME_CODE_NUMERIC",
    "FILENAME_CODE_VERSION", "FILENAME_EXPLANATION", "FILENAME_INVALID_CHARS",
    "FILENAME_KANA", "FILENAME_MARKER", "FILENAME_MAX_CHARS",
    "FILENAME_MAX_RETRIES", "FILENAME_MAX_TOKENS", "FILENAME_PLACEHOLDER",
    "FILENAME_PREFIX", "FILENAME_PROMPT", "FILENAME_SYMBOL_CONNECTOR",
    "FILENAME_SYSTEM_PROMPT", "FILENAME_TECHNICAL_STEMS", "FILENAME_TOKEN",
    "_has_translatable_text", "_is_excluded_dir", "_is_generated_output",
    "_is_mask_symbol", "_mask_name_literals", "_path_key", "_placeholder_failure",
    "_read_text", "_rename_temp", "_restore_literals",
    "_strip_filename_explanation", "apply_name_renames", "collect_name_targets",
    "collect_translate_inputs", "output_path", "plan_name_renames",
    "plan_translation_outputs", "sanitize_filename_stem",
    # engine
    "TranslationEngine", "build_chat_for", "extract_content", "make_engine_chat",
    # pipeline
    "SINGLE_RETRY_HINT", "T1_NEARBY_RADIUS", "_translate_chunks",
    "_translate_single_cue", "translate_file", "translate_prose", "translate_srt",
]
