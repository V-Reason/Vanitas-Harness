"""Agent 规划模式：复用 REPL，但固定 task_agent 模板。

安全边界：本模块**不执行**模型返回的 action，也不实现自动工具循环；
执行仍由用户在 REPL 中通过 `/exec` 逐次确认。这样 `vanitas agent`
有可观测的规划入口，同时不越过项目的安全红线。
"""

from __future__ import annotations

from .config import AppConfig, ModelSpec
from .repl import Repl
from .ui import Palette


def run_agent(
    config: AppConfig,
    model: ModelSpec | None = None,
    profile: str | None = None,
    *,
    system: str | None = None,
    palette: Palette | None = None,
    verbose: bool = False,
) -> int:
    paint = palette or Palette({}, enabled=False)
    print(
        paint.hint(
            "Agent 规划模式：仅输出 JSON 动作，不自动执行；"
            "执行请用 /exec（逐次确认）"
        )
    )
    repl = Repl(
        config,
        model,
        profile,
        auto_start=True,
        palette=paint,
        verbose=verbose,
    )
    repl.configure(system=system, task="agent")
    return repl.run()
