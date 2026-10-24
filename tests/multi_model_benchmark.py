#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vanitas-Harness 多模型验收基准。

只依赖标准库。性能指标优先读 llama-server 返回的 timings（服务端真实 token 计数），
客户端分片数只用于流式协议检查——SSE 分片不等于 token，拿它算 t/s 会系统性失真。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

DEFAULT_TIMEOUT = 60.0
REPORT_DIR = Path(__file__).resolve().parent / "reports"

TRANSLATION_TERMS = ("零拷贝", "内存映射", "缓冲区", "分配")
TRANSLATION_PROMPT = (
    "Translate the following technical paragraph into Simplified Chinese. "
    "Output ONLY the Chinese translation, no English, no notes:\n"
    "'The zero-copy streaming pipeline avoids intermediate buffer allocations "
    "via memory-mapped I/O.'"
)
AGENT_SYSTEM = (
    "Respond ONLY with a valid JSON object matching this schema: "
    '{"action": "query_database", "parameters": {"sql": string}}'
)
AGENT_USER = "Retrieve the top 5 highest-latency API endpoints."
AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["query_database"]},
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
    },
    "required": ["action", "parameters"],
    "additionalProperties": False,
}
PERF_PROMPT = (
    "Explain the difference between optimistic concurrency control and "
    "pessimistic locking in a database engine."
)


class ProbeError(RuntimeError):
    """带上下文的探测失败；调用方直接打印，不向用户泄露堆栈。"""


def _build_opener() -> urllib.request.OpenerDirector:
    # 必须绕开系统代理：HTTP_PROXY 会把 127.0.0.1 也劫持到代理，导致误判服务未启动
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


OPENER = _build_opener()


def request_json(
    port: int,
    path: str,
    payload: dict | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    host: str = "127.0.0.1",
) -> dict:
    url = f"http://{host}:{port}{path}"
    data = (
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if payload is not None
        else None
    )
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(
        url, data=data, headers=headers, method="POST" if data else "GET"
    )
    try:
        with OPENER.open(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise ProbeError(f"HTTP {exc.code} {path}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ProbeError(f"无法连接 {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ProbeError(f"{path} 超时（{timeout}s）") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"{path} 返回非 JSON: {body[:200]!r}") from exc


def wait_ready(port: int, host: str, wait_s: float) -> float:
    """轮询 /health 直到模型加载完成；加载期 503 属于正常中间态，不能当失败。"""
    started = time.perf_counter()
    deadline = started + wait_s
    last_state = "未响应"
    while time.perf_counter() < deadline:
        try:
            with OPENER.open(f"http://{host}:{port}/health", timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") == "ok":
                return time.perf_counter() - started
            last_state = f"status={payload.get('status')}"
        except urllib.error.HTTPError as exc:
            last_state = f"HTTP {exc.code}（模型加载中）"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_state = str(exc)
        time.sleep(1.0)
    raise ProbeError(f"等待 {wait_s:.0f}s 后服务仍未就绪（最后状态: {last_state}）")


def read_model_id(port: int, host: str) -> str:
    data = request_json(port, "/v1/models", host=host)
    items = data.get("data") or []
    if not items:
        raise ProbeError("/v1/models 返回空列表")
    return str(items[0].get("id", "UNKNOWN"))


def chat(
    port: int,
    host: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.2,
    response_format: dict | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    payload: dict = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    return request_json(port, "/v1/chat/completions", payload, timeout=timeout, host=host)


def measure_performance(
    port: int, host: str, runs: int, timeout: float
) -> dict:
    # 预热把 CUDA 图与 KV 分配打进稳态，否则首轮样本会把 TTFT 拉高
    chat(port, host, [{"role": "user", "content": "ping"}], max_tokens=8, timeout=timeout)

    samples: list[dict] = []
    for index in range(runs):
        # 每次追加唯一探针号，避免命中 llama-server 的前缀缓存把 prefill 速率抬高几个数量级
        probe = f"{PERF_PROMPT}\n\n[probe {index + 1}-{time.perf_counter_ns()}]"
        started = time.perf_counter()
        result = chat(
            port,
            host,
            [{"role": "user", "content": probe}],
            max_tokens=128,
            timeout=timeout,
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        timings = result.get("timings") or {}
        usage = result.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)

        server_decode = float(timings.get("predicted_per_second") or 0.0)
        server_prompt = float(timings.get("prompt_per_second") or 0.0)
        # 服务端没给 timings 时才退化为 usage/wall 的近似值，并在报告里标明来源
        client_decode = (
            completion_tokens / (wall_ms / 1000.0)
            if completion_tokens > 0 and wall_ms > 0
            else 0.0
        )
        samples.append(
            {
                "prompt_tps": server_prompt if server_prompt > 0 else 0.0,
                "decode_tps": server_decode if server_decode > 0 else client_decode,
                "completion_tokens": completion_tokens,
                "wall_ms": round(wall_ms, 1),
                "metric_source": "server.timings" if server_decode > 0 else "client.usage",
            }
        )

    decode_values = [sample["decode_tps"] for sample in samples]
    prompt_values = [sample["prompt_tps"] for sample in samples]
    return {
        "runs": len(samples),
        "decode_tps_median": round(statistics.median(decode_values), 2),
        "decode_tps_min": round(min(decode_values), 2),
        "decode_tps_max": round(max(decode_values), 2),
        "prompt_tps_median": round(statistics.median(prompt_values), 2),
        "completion_tokens": samples[-1]["completion_tokens"],
        "wall_ms_median": round(statistics.median(s["wall_ms"] for s in samples), 1),
        "metric_source": samples[-1]["metric_source"],
        "samples": samples,
    }


def measure_streaming(
    port: int, host: str, timeout: float, max_tokens: int
) -> dict:
    payload = {
        "messages": [
            {"role": "system", "content": "You are a precise and helpful assistant."},
            {"role": "user", "content": "List three primary colors, one per line."},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True,
    }
    url = f"http://{host}:{port}/v1/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    started = time.perf_counter()
    first_token_at: float | None = None
    chunks = 0
    done = False
    collected: list[str] = []
    try:
        with OPENER.open(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                content = (choices[0].get("delta") or {}).get("content") or ""
                if content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    chunks += 1
                    collected.append(content)
    except urllib.error.URLError as exc:
        raise ProbeError(f"流式请求失败: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ProbeError(f"流式请求超时（{timeout}s）") from exc

    end = time.perf_counter()
    return {
        "ttft_ms": round((first_token_at - started) * 1000.0, 1) if first_token_at else None,
        "chunks": chunks,
        "done": done,
        "wall_ms": round((end - started) * 1000.0, 1),
        "preview": "".join(collected)[:160],
    }


def _extract_json_object(text: str) -> tuple[str, bool]:
    """剥离 Markdown 围栏；fenced=True 是质量信号，不直接判失败。"""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1), True
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1], False
    return text, False


def run_agent_suite(port: int, host: str, timeout: float) -> dict:
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": AGENT_USER},
    ]
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "agent_action", "schema": AGENT_SCHEMA},
    }
    try:
        result = chat(
            port,
            host,
            messages,
            max_tokens=128,
            temperature=0.0,
            response_format=response_format,
            timeout=timeout,
        )
        format_mode = "json_schema"
    except ProbeError as exc:
        # 异构/旧版服务可能不认 json_schema；退到 json_object 但把降级写进报告
        if "HTTP 4" not in str(exc):
            raise
        result = chat(
            port,
            host,
            messages,
            max_tokens=128,
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=timeout,
        )
        format_mode = "json_object（fallback）"
    raw = (result["choices"][0]["message"]["content"] or "").strip()
    candidate, fenced = _extract_json_object(raw)

    failures: list[str] = []
    parsed: dict | None = None
    try:
        parsed_value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        failures.append(f"输出不是合法 JSON: {exc.msg}")
    else:
        if not isinstance(parsed_value, dict):
            failures.append("输出 JSON 顶层不是对象")
        else:
            parsed = parsed_value
            action = parsed.get("action")
            parameters = parsed.get("parameters")
            if not isinstance(action, str) or not action.strip():
                failures.append("action 缺失或不是非空字符串")
            if not isinstance(parameters, dict):
                failures.append("parameters 缺失或不是对象")

    return {
        "suite": "agent",
        "passed": not failures,
        "response_format": format_mode,
        "fenced": fenced,
        "failures": failures,
        "raw_preview": raw[:240],
        "parsed": parsed,
    }


def run_translation_suite(port: int, host: str, timeout: float) -> dict:
    result = chat(
        port,
        host,
        [
            {
                "role": "system",
                "content": "You are a professional technical translator. "
                "Return only the translation, without notes or explanations.",
            },
            {"role": "user", "content": TRANSLATION_PROMPT},
        ],
        max_tokens=160,
        temperature=0.0,
        timeout=timeout,
    )
    output = (result["choices"][0]["message"]["content"] or "").strip()
    # 小模型常在译文后追加 "Note: ..." 说明；只评第一段译文，但同时记录是否有多余段落
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", output) if part.strip()]
    body = paragraphs[0] if paragraphs else ""
    cjk = sum(1 for char in body if "\u4e00" <= char <= "\u9fff")
    cjk_ratio = round(cjk / len(body), 3) if body else 0.0
    hits = [term for term in TRANSLATION_TERMS if term in body]

    failures: list[str] = []
    if len(body) < 10:
        failures.append(f"译文过短（{len(body)} 字符）")
    if cjk_ratio < 0.2:
        failures.append(f"译文中文占比过低（{cjk_ratio:.1%}），可能未按目标语言输出")
    if not hits:
        failures.append("译文未命中任何领域术语（零拷贝/内存映射/缓冲区/分配）")

    return {
        "suite": "translation",
        "passed": not failures,
        "cjk_ratio": cjk_ratio,
        "term_hits": hits,
        "extra_paragraphs": max(len(paragraphs) - 1, 0),
        "failures": failures,
        "output": output[:240],
    }


def build_report(args: argparse.Namespace, model_id: str, ready_s: float, perf: dict, stream: dict, quality: dict) -> dict:
    protocol_failures: list[str] = []
    if perf["completion_tokens"] <= 0:
        protocol_failures.append("性能采样未产生 completion_tokens")
    if perf["decode_tps_median"] <= 0:
        protocol_failures.append("解码速率不可用（服务端 timings 与 usage 均为空）")
    if not stream["done"]:
        protocol_failures.append("流式响应未收到 data: [DONE]")
    if stream["chunks"] <= 0:
        protocol_failures.append("流式响应没有内容分片")
    quality_failures = list(quality["failures"])

    return {
        "tool": "vanitas-bench",
        "version": "1.0.0",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "endpoint": f"http://{args.host}:{args.port}",
        "model": model_id,
        "ready_seconds": round(ready_s, 2),
        "performance": perf,
        "streaming": stream,
        "quality": quality,
        "verdict": {
            "passed": not (protocol_failures or quality_failures),
            "protocol_failures": protocol_failures,
            "quality_failures": quality_failures,
        },
    }


def _write_report(report: dict, json_out: str) -> Path:
    if json_out:
        path = Path(json_out).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = REPORT_DIR / f"bench_{report['endpoint'].rsplit(':', 1)[-1]}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vanitas-Harness 多模型验收基准（OpenAI 兼容协议）")
    parser.add_argument("--port", type=int, default=8080, help="目标端口（8080/8081/8082 或自定义）")
    parser.add_argument("--host", default="127.0.0.1", help="目标主机，默认本机回环")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="单请求超时秒数")
    parser.add_argument("--wait", type=float, default=120.0, help="等待模型加载的最长秒数")
    parser.add_argument("--runs", type=int, default=3, help="性能采样次数（预热不计入）")
    parser.add_argument("--suite", choices=("auto", "agent", "translation"), default="auto")
    parser.add_argument("--json-out", default="", help="报告路径；省略则写入 tests/reports/")
    parser.add_argument("--stream-max-tokens", type=int, default=64)
    return parser.parse_args(argv)


def select_suite(requested: str, model_id: str, port: int) -> str:
    if requested != "auto":
        return requested
    hint = model_id.lower()
    if port == 8082 or "hunyuan" in hint or "mt" in hint:
        return "translation"
    return "agent"


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.runs < 1:
        print("[FATAL] --runs 必须 ≥ 1", file=sys.stderr)
        return 4

    # Windows 控制台默认 cp936，中文与表格混排会触发 UnicodeEncodeError
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    print("=" * 64)
    print(f"Vanitas-Harness 验收评测  http://{args.host}:{args.port}")
    print("=" * 64)

    try:
        ready_s = wait_ready(args.port, args.host, args.wait)
        model_id = read_model_id(args.port, args.host)
        suite = select_suite(args.suite, model_id, args.port)
        print(f"模型: {model_id}    就绪: {ready_s:.2f}s    suite: {suite}")
        print("-" * 64)

        perf = measure_performance(args.port, args.host, args.runs, args.timeout)
        stream = measure_streaming(args.port, args.host, args.timeout, args.stream_max_tokens)
        quality = (
            run_translation_suite(args.port, args.host, args.timeout)
            if suite == "translation"
            else run_agent_suite(args.port, args.host, args.timeout)
        )
    except ProbeError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1

    print(
        f"[性能] prompt {perf['prompt_tps_median']:.1f} t/s | decode {perf['decode_tps_median']:.1f} t/s "
        f"(min {perf['decode_tps_min']:.1f} / max {perf['decode_tps_max']:.1f}, n={perf['runs']}, {perf['metric_source']})"
    )
    ttft = f"{stream['ttft_ms']:.0f} ms" if stream["ttft_ms"] is not None else "未收到首字"
    print(f"[流式] TTFT {ttft} | 分片 {stream['chunks']} | DONE {'是' if stream['done'] else '否'}")
    if quality["suite"] == "translation":
        print(
            f"[质量] 中文占比 {quality['cjk_ratio']:.1%} | 术语命中 {len(quality['term_hits'])} "
            f"({', '.join(quality['term_hits']) or '无'})"
        )
    else:
        print(
            f"[质量] response_format={quality['response_format']} | "
            f"JSON 围栏包裹 {'是' if quality['fenced'] else '否'} | 结构校验"
        )

    report = build_report(args, model_id, ready_s, perf, stream, quality)
    verdict = report["verdict"]
    for failure in verdict["protocol_failures"] + verdict["quality_failures"]:
        print(f"[FAIL] {failure}")

    try:
        report_path = _write_report(report, args.json_out)
        print(f"[REPORT] {report_path}")
    except OSError as exc:
        print(f"[WARN] 报告写入失败（不影响验收结论）: {exc}", file=sys.stderr)

    print("-" * 64)
    if verdict["passed"]:
        print("[VERDICT] 通过：协议、性能与质量断言全部满足。")
        return 0
    print("[VERDICT] 未通过：详见上方 FAIL 列表。")
    # 2 = 协议/性能链路问题，3 = 模型服务质量不达标，便于上层脚本分流处置
    return 2 if verdict["protocol_failures"] else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
