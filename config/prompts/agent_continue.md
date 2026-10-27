【Agent 继续】
工作空间根目录：{cwd}

请根据当前上下文与工具结果推进：
- 需要继续操作：只输出 <tool_call>[...]</tool_call> 数组，纯 JSON，无 Markdown 围栏。
- 任务已完成：直接输出普通中文总结，不再调用工具。
- 同一失败动作未改参数不得重试；改扩展名/移动用 move_file(src, dst)；patch_file 只改内容。

用户后续输入：
{input}
