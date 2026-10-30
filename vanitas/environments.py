"""运行环境（production/experimental）与 preset→task_template 解析。

environment 独立于 ASMR 与 preset：production 保持所有 preset 的现有模板；
experimental 目前只实现 ASMR 提示词覆盖，其它 preset 明确报错，不静默
fallback。``EnvironmentProfile`` 的 ``pipeline``/``pipeline_factory``/
``hooks`` 是后续整条管线替换的 seam；当前两者都走现有 TranslationEngine。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import prompts
from .errors import ConfigError

PipelineFactory = Callable[..., object]


@dataclass(frozen=True)
class EnvironmentProfile:
    """一个运行环境；当前只影响 task_template 解析。

    ``pipeline``/``pipeline_factory``/``hooks`` 保留给未来 experimental
    管线，不在当前行为中使用；调用方可以据 profile 选择构造器/钩子。
    """

    name: str
    pipeline: str = "translate"
    pipeline_factory: PipelineFactory | None = None
    hooks: tuple[str, ...] = ()

    def resolve_task_template(self, preset: str, base_template: str) -> str:
        """把 preset 的 production 模板解析为该环境的最终模板。

        production：原样返回；experimental：仅 ASMR 覆盖，其它 preset
        直接 ConfigError，绝不回退到生产模板。
        """
        if self.name == "production":
            return base_template
        if self.name == "experimental":
            if prompts.is_asmr_task(base_template):
                return prompts.asmr_template("experimental")
            raise ConfigError(
                f"experimental 环境未实现 preset {preset!r}；"
                "当前仅 ASMR 有实验模板，请使用 --env production"
            )
        raise ConfigError(
            f"未知运行环境 {self.name!r}；可用：production、experimental"
        )


PRODUCTION = EnvironmentProfile("production")
EXPERIMENTAL = EnvironmentProfile("experimental")
ENVIRONMENTS = (PRODUCTION, EXPERIMENTAL)
ENVIRONMENT_NAMES = frozenset(profile.name for profile in ENVIRONMENTS)
DEFAULT_ENVIRONMENT = PRODUCTION


def get_environment(name: str | None) -> EnvironmentProfile:
    """按名称取环境 profile；未知值给可读 ConfigError（不回退）。"""
    if name is None:
        return DEFAULT_ENVIRONMENT
    for profile in ENVIRONMENTS:
        if profile.name == name:
            return profile
    choices = "、".join(profile.name for profile in ENVIRONMENTS)
    raise ConfigError(
        f"未知运行环境 {name!r}；可用：{choices}"
    )
