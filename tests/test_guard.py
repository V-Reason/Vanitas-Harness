"""守卫 8 例：用临时目录构造，不碰真实权重、不杀真实进程。"""

from __future__ import annotations

import socket
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import guard  # noqa: E402
from vanitas.ui import Palette  # noqa: E402

RUNTIME = ("llama-server.exe", "llama.dll", "ggml-base.dll", "ggml-cuda.dll")
MIN_BYTES = 1024


class GuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "bin").mkdir()
        for name in RUNTIME:
            (self.root / "bin" / name).write_bytes(b"stub")
        (self.root / "models").mkdir()

    def make_model(self, name: str, magic: bytes = b"GGUF", size: int = 4096) -> Path:
        path = self.root / "models" / name
        path.write_bytes(magic + b"\0" * max(0, size - len(magic)))
        return path

    def evaluate(self, port, model, **kwargs):
        return guard.evaluate(
            port,
            model,
            base_dir=self.root,
            allow_kill=False,
            min_bytes=MIN_BYTES,
            **kwargs,
        )

    # 1) 正常
    def test_ok_single_file(self) -> None:
        result = self.evaluate(4062, self.make_model("base.gguf"))
        self.assertEqual(result.code, 0)
        self.assertEqual(result.stderr, [])
        self.assertTrue(any("环境核验通过" in line for line in result.stdout))
        self.assertTrue(any("单文件" in line for line in result.stdout))

    def test_ok_split_shards(self) -> None:
        self.make_model("demo-00001-of-00002.gguf")
        self.make_model("demo-00002-of-00002.gguf")
        result = self.evaluate(4060, self.root / "models" / "demo-00001-of-00002.gguf")
        self.assertEqual(result.code, 0)
        self.assertTrue(any("2 个分片" in line for line in result.stdout))

    # 2) 缺模型
    def test_missing_model(self) -> None:
        result = self.evaluate(4060, self.root / "models" / "nope.gguf")
        self.assertEqual(result.code, 2)
        self.assertIn("找不到模型文件", result.stderr[0])

    # 3) 缺分片
    def test_missing_shard(self) -> None:
        first = self.make_model("demo-00001-of-00002.gguf")
        result = self.evaluate(4060, first)
        self.assertEqual(result.code, 2)
        self.assertIn("分片缺失", result.stderr[0])
        self.assertIn("demo-00002-of-00002.gguf", result.stderr[0])

    # 4) 传错分片
    def test_wrong_shard_entry(self) -> None:
        self.make_model("demo-00001-of-00002.gguf")
        second = self.make_model("demo-00002-of-00002.gguf")
        result = self.evaluate(4060, second)
        self.assertEqual(result.code, 2)
        self.assertIn("启动入口错误", result.stderr[0])

    # 5) 体积不足
    def test_too_small(self) -> None:
        path = self.root / "models" / "tiny.gguf"
        path.write_bytes(b"GGUF")
        result = self.evaluate(4060, path)
        self.assertEqual(result.code, 2)
        self.assertIn("体积不足", result.stderr[0])

    # 6) 非 GGUF
    def test_bad_magic(self) -> None:
        result = self.evaluate(4060, self.make_model("fake.gguf", magic=b"NOPE"))
        self.assertEqual(result.code, 2)
        self.assertIn("不是有效的 GGUF", result.stderr[0])

    # 7) 参数缺失 / 非法
    def test_usage_errors(self) -> None:
        model = self.make_model("base.gguf")
        self.assertEqual(self.evaluate(4060, "").code, 9)
        self.assertEqual(self.evaluate("abc", model).code, 9)
        self.assertEqual(self.evaluate(70000, model).code, 9)

    # 8) 端口被占
    def test_port_busy_reports_pid(self) -> None:
        model = self.make_model("base.gguf")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            result = self.evaluate(port, model)
        self.assertEqual(result.code, 1)
        self.assertTrue(
            result.stderr[0].startswith(f"✗ 端口 {port} 被 PID "),
            result.stderr[0],
        )
        self.assertIn("占用", result.stderr[0])

    # 附加：运行时缺失 → 3
    def test_missing_runtime(self) -> None:
        (self.root / "bin" / "ggml-cuda.dll").unlink()
        result = self.evaluate(4060, self.make_model("base.gguf"))
        self.assertEqual(result.code, 3)
        self.assertIn("缺少运行组件", result.stderr[0])

    # 附加：消息不含任何方括号标签（批 2 契约）
    def test_messages_contain_no_bracket_tags(self) -> None:
        cases = [
            self.evaluate(4060, self.root / "models" / "nope.gguf"),
            self.evaluate(4060, self.make_model("fake.gguf", magic=b"NOPE")),
            self.evaluate("abc", self.make_model("base.gguf")),
            self.evaluate(4060, self.make_model("base.gguf")),
        ]
        joined = "\n".join(
            line for result in cases for line in result.stdout + result.stderr
        )
        self.assertNotIn("[GUARD", joined)
        for line in joined.splitlines():
            self.assertFalse(line.lstrip().startswith("["), line)

    # 附加：成功消息带对勾
    def test_success_message_has_checkmark(self) -> None:
        result = self.evaluate(4060, self.make_model("base.gguf"))
        self.assertEqual(result.code, 0)
        self.assertIn("✓", " ".join(result.stdout))

    # 附加：format_for_cli 只上色、不加前缀
    def test_format_for_cli_only_colorizes(self) -> None:
        result = self.evaluate(4060, self.root / "models" / "nope.gguf")
        plain_out, plain_err = guard.format_for_cli(result)
        self.assertEqual(plain_err[0], result.stderr[0])  # 无色 palette 原样返回
        _out, colored = guard.format_for_cli(result, Palette({}, enabled=True))
        self.assertTrue(colored[0].startswith("\x1b[31m✗ 找不到模型文件"))
        self.assertTrue(colored[0].endswith("\x1b[0m"))
        self.assertNotIn("[GUARD-CODE", colored[0])


if __name__ == "__main__":
    unittest.main()
