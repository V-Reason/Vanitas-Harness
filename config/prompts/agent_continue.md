【Agent 继续】
<agent_continuation>
当前工作目录：{cwd}

请根据当前上下文与工具执行结果推进任务：
- 若需继续操作：输出 <tool_call>[...]</tool_call> 数组（确保纯 JSON，无 Markdown 围栏，路径使用正斜杠）。
- 若任务已达成：直接输出最终中文总结，不再调用工具。
- 纠偏规范：重命名/改扩展名必须用 move_file(src, dst)；patch_file 只改内容且 search 需带唯一上下文锚点；同一失败动作不得原参数重复提交。

用户后续输入：
{input}
</agent_continuation>
