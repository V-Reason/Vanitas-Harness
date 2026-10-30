"""统一异常类型。

约定：所有对外可见的失败都抛本模块的子类，由 cli 统一映射为退出码；
禁止空 catch，也禁止把异常吞成「退出 1」——调用方必须能从异常里读到可执行的上下文。
"""

from __future__ import annotations


class VanitasError(Exception):
    """所有可预期失败的基类；消息必须自带可执行的上下文。"""


class ConfigError(VanitasError):
    """配置缺失、格式非法或语义不合法（CLI 退出码 2）。"""


class GuardError(VanitasError):
    """守卫拒绝启动；code 沿用 0/1/2/3/4/9 契约，messages 为待打印的行。"""

    def __init__(self, code: int, messages: list[str]) -> None:
        super().__init__("；".join(messages) or f"守卫失败：code={code}")
        self.code = code
        self.messages = list(messages)


class EngineError(VanitasError):
    """HTTP/协议层失败：连接、超时、4xx/5xx、SSE 中断（CLI 退出码 5）。

    retryable=False 表示错误由请求本身引起（4xx/协议不合法），重试只会重复失败。
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class LaunchError(VanitasError):
    """llama-server 无法启动或未在超时内就绪（CLI 退出码 5）。"""


class TranslateError(VanitasError):
    """翻译流水线的结构性失败（文件不可读、分段解析失败等）。"""


class OutputTruncated(TranslateError):
    """模型输出在长度上限处被截断；调用方可拆分输入重试。"""


class ToolDenied(VanitasError):
    """/exec 被配置、白名单或用户确认拒绝。"""
