from enum import Enum, auto


class SystemPrompt(Enum):
    DEFAULT = auto()
    WITH_TOOLS = auto()
    ONLY_SUMMARY = auto()
    # 필요 시 계속 추가


PROMPT_TEXT = {
    SystemPrompt.DEFAULT: "You are a helpful assistant.",
    SystemPrompt.WITH_TOOLS: (
        "You are a helpful assistant with access to these tools: \n\n"
        "{tools_description}\n"
        "Choose the appropriate tool based on the user's question in sequential thinking.\n\n"
        "You should structure your responses as follows:\n"
        "1. Think through your reasoning process inside <think>...</think> tags\n"
        "2. If you need to use a tool, respond with the exact JSON object inside <tool_call> tags:\n"
        "<tool_call>\n"
        "{{\n"
        '  "tool" : "tool-name",\n'
        '  "arguments" : {{\n'
        '       "argument-name" : "value",\n'
        "   }}\n"
        "}}\n"
        "</tool_call>\n\n"
        "3. Your final response to the user should be provided inside <answer>...</answer> tags\n\n"
        "You can repeat this thinking-tool call-answer cycle multiple times if necessary.\n"
        "The flow can repeat as: thinking → tool call → thinking → answer (whatever if you need), or thinking → answer.\n\n"
        "Multi-tool execution strategy:\n"
        "- If you need to run multiple tools in sequence, analyze the result from the first tool in <think> tags\n"
        "- Then decide if another tool call is needed or provide your final answer in <answer> tags\n"
        "- You can use up to 10 tools in sequence if needed\n\n"
        "Please use only the tools that are explicitly defined above. Answer should include the tool name and the result and be in Korean and markdown format. "
    ),
    SystemPrompt.ONLY_SUMMARY: "Answer with a short summary only.",
}
