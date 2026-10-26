"""Agent 工具调用解析：标签、裸 JSON、多对象重组与流式抑制。"""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vanitas import agent  # noqa: E402
from vanitas.config import load_config  # noqa: E402

AGENT_SETTINGS = load_config().settings.agent


def parse_actions(text: str):
    return agent._parse_actions(
        text,
        max_batch_actions=AGENT_SETTINGS.max_batch_actions,
        action_aliases=AGENT_SETTINGS.action_aliases,
    )


def tagged(payload: str) -> str:
    return f"<tool_call>\n{payload}\n</tool_call>"


class ExtractToolCallsTests(unittest.TestCase):
    def test_tagged_array(self) -> None:
        calls = agent.extract_tool_calls(
            tagged(
                '[{"action":"list_dir","parameters":{"path":"."}},'
                '{"action":"read_file","parameters":{"path":"a.txt"}}]'
            )
        )
        self.assertEqual([item["action"] for item in calls or []], ["list_dir", "read_file"])

    def test_tagged_single_object_becomes_list(self) -> None:
        calls = agent.extract_tool_calls(
            tagged('{"action":"list_dir","parameters":{"path":"."}}')
        )
        self.assertEqual(len(calls or []), 1)
        self.assertEqual(calls[0]["action"], "list_dir")  # type: ignore[index]

    def test_bare_array(self) -> None:
        calls = agent.extract_tool_calls(
            '[{"action":"list_dir","parameters":{}}]'
        )
        self.assertEqual(len(calls or []), 1)

    def test_prose_then_object(self) -> None:
        calls = agent.extract_tool_calls('好的，将执行 {"action":"a"}')
        self.assertEqual(calls, [{"action": "a"}])

    def test_multiple_objects_are_reassembled(self) -> None:
        calls = agent.extract_tool_calls('{"action":"a"},{"action":"b"}')
        self.assertEqual(calls, [{"action": "a"}, {"action": "b"}])

    def test_raw_decode_extracts_prefix_before_trailing_text(self) -> None:
        calls = agent.extract_tool_calls(
            tagged('{"action":"list_dir","parameters":{}} trailing')
        )
        self.assertEqual(len(calls or []), 1)
        self.assertEqual(calls[0]["action"], "list_dir")  # type: ignore[index]

    def test_multiple_tags_are_merged(self) -> None:
        raw = (
            tagged('{"action":"a"}')
            + "\n"
            + tagged('[{"action":"b"},{"action":"c"}]')
        )
        calls = agent.extract_tool_calls(raw)
        self.assertEqual([item["action"] for item in calls or []], ["a", "b", "c"])

    def test_brackets_inside_string_do_not_break_balancing(self) -> None:
        raw = tagged(
            '{"action":"write_file","parameters":'
            '{"path":"a.txt","content":"} [x] {"}}'
        )
        calls = agent.extract_tool_calls(raw)
        self.assertEqual(len(calls or []), 1)
        self.assertEqual(calls[0]["parameters"]["content"], "} [x] {")  # type: ignore[index]

    def test_windows_backslashes_in_tagged_path_are_repaired(self) -> None:
        raw = tagged(
            r'{"action":"list_dir","parameters":{"path":"<WORKSPACE>"}}'
        )
        calls = agent.extract_tool_calls(raw)
        self.assertEqual(calls[0]["parameters"]["path"], r"<WORKSPACE>")  # type: ignore[index]

    def test_multiple_objects_with_raw_backslashes_are_recovered(self) -> None:
        raw = (
            r'{"action":"move_file","parameters":'
            r'{"src":"T:\TMP\a.md","dst":"T:\TMP\a.srt"}},'
            r'{"action":"move_file","parameters":'
            r'{"src":"T:\TMP\b.md","dst":"T:\TMP\b.srt"}}'
        )
        calls = agent.extract_tool_calls(raw)
        self.assertEqual(len(calls or []), 2)
        self.assertEqual(calls[1]["parameters"]["src"], r"T:\TMP\b.md")  # type: ignore[index]

    def test_plain_json_without_action_is_not_a_tool_call(self) -> None:
        self.assertIsNone(agent.extract_tool_calls('{"result":"ok"}'))
        self.assertIsNone(agent.extract_tool_calls("结果是 [1, 2]"))

    def test_empty_array_is_recognized_as_empty(self) -> None:
        self.assertEqual(agent.extract_tool_calls("[]"), [])
        self.assertEqual(agent.extract_tool_calls(tagged("[]")), [])

    def test_malformed_or_non_string_returns_none(self) -> None:
        self.assertIsNone(agent.extract_tool_calls('{"action":'))
        self.assertIsNone(agent.extract_tool_calls("<tool_call></tool_call>"))
        self.assertIsNone(agent.extract_tool_calls(None))  # type: ignore[arg-type]


class ParseActionsTests(unittest.TestCase):
    def test_tagged_batch_becomes_actions(self) -> None:
        actions = parse_actions(
            tagged(
                '[{"action":"list_dir","parameters":{}},'
                '{"action":"delete_file","parameters":{"path":"a"}}]'
            )
        )
        self.assertEqual(
            [item.name for item in actions or []], ["list_dir", "delete_file"]
        )

    def test_prose_then_tagged_action(self) -> None:
        actions = parse_actions(
            "现在开始。\n"
            + tagged('{"action":"read_file","parameters":{"path":"a.txt"}}')
        )
        self.assertEqual(actions[0].parameters["path"], "a.txt")  # type: ignore[index]

    def test_malformed_tag_raises_action_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_actions("<tool_call>{\"action\":</tool_call>")
        self.assertIn("动作", str(ctx.exception))

    def test_plain_json_returns_none(self) -> None:
        self.assertIsNone(parse_actions('{"result":"ok"}'))

    def test_empty_and_oversized_arrays_raise(self) -> None:
        with self.assertRaises(ValueError):
            parse_actions("[]")
        oversized = "[" + ",".join(
            '{"action":"list_dir","parameters":{}}'
            for _ in range(AGENT_SETTINGS.max_batch_actions + 1)
        ) + "]"
        with self.assertRaises(ValueError):
            parse_actions(oversized)

    def test_unknown_action_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_actions(
                tagged('{"action":"nope","parameters":{}}')
            )
        self.assertIn("未知动作", str(ctx.exception))

    def test_alias_and_batch_limit_are_injected(self) -> None:
        actions = agent._parse_actions(
            tagged('{"action":"sh","parameters":{"path":"a.txt"}}'),
            max_batch_actions=1,
            action_aliases={"sh": "read_file"},
        )
        self.assertEqual(actions[0].name, "read_file")  # type: ignore[index]
        with self.assertRaises(ValueError):
            agent._parse_actions(
                tagged(
                    '[{"action":"list_dir","parameters":{}},'
                    '{"action":"list_dir","parameters":{}}]'
                ),
                max_batch_actions=1,
                action_aliases={},
            )

    def test_minicpm_function_xml_with_cdata(self) -> None:
        raw = (
            '<function name="write_file">'
            '<param name="path">a.py</param>'
            '<param name="content"><![CDATA[import json\nprint("ok")]]></param>'
            "</function>"
        )
        calls = agent.extract_tool_calls(raw)
        self.assertEqual(calls[0]["action"], "write_file")  # type: ignore[index]
        self.assertEqual(calls[0]["parameters"]["path"], "a.py")  # type: ignore[index]
        self.assertEqual(
            calls[0]["parameters"]["content"],  # type: ignore[index]
            'import json\nprint("ok")',
        )
        actions = parse_actions(raw)
        self.assertEqual(actions[0].name, "write_file")  # type: ignore[index]
        self.assertEqual(
            actions[0].parameters["content"],  # type: ignore[index]
            'import json\nprint("ok")',
        )


class TaggedStreamTests(unittest.TestCase):
    def test_pure_tag_is_not_emitted(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            stream = agent._AgentAnswerStream()
            stream.write("<tool_call>\n")
            stream.write('[{"action":"list_dir","parameters":{}}]\n')
            stream.write("</tool_call>")
            self.assertFalse(stream.emitted)
            self.assertTrue(stream.suppressed)
        self.assertEqual(out.getvalue(), "")

    def test_prose_then_tag_then_text(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            stream = agent._AgentAnswerStream()
            stream.write("好的。\n<tool_call>\n")
            stream.write('[{"action":"list_dir","parameters":{}}]\n')
            stream.write("</tool_call>\n完成")
            self.assertTrue(stream.emitted)
        self.assertEqual(out.getvalue(), "好的。\n\n完成")

    def test_suppressed_tag_can_be_restored(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            stream = agent._AgentAnswerStream()
            stream.write("<tool_call>不是 JSON</tool_call>")
            self.assertTrue(stream.suppressed)
            stream.flush_suppressed()
        self.assertIn("<tool_call>不是 JSON</tool_call>", out.getvalue())

    def test_tag_split_across_chunks(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            stream = agent._AgentAnswerStream()
            for chunk in ("<tool", "_call>", "[", "]", "</tool", "_call>"):
                stream.write(chunk)
        self.assertEqual(out.getvalue(), "")
        self.assertTrue(stream.suppressed)

    def test_function_xml_is_suppressed(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            stream = agent._AgentAnswerStream()
            stream.write('前文\n<function name="read_file">')
            stream.write('<param name="path">a.txt</param>')
            self.assertTrue(stream.suppressed)
            stream.write("</function>\n后文")
        value = out.getvalue()
        self.assertIn("前文", value)
        self.assertIn("后文", value)
        self.assertNotIn("<function", value)

        results = io.StringIO()
        with contextlib.redirect_stdout(results):
            stream = agent._AgentAnswerStream()
            stream.write("<function_results>ok</function_results>完成")
        self.assertEqual(results.getvalue(), "完成")

    def test_legacy_line_start_json_is_suppressed(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            stream = agent._AgentAnswerStream()
            stream.write("说明\n")
            stream.write('{"action":"list_dir","parameters":{}}')
        self.assertEqual(out.getvalue(), "说明\n")
        self.assertTrue(stream.suppressed)


if __name__ == "__main__":
    unittest.main()
