"""Step04 · 工具调用（Function Calling）—— 给模型装上"手脚"。

与 Step03 手写 ReAct 循环不同，本场景展示标准的工具调用交互：
1. 在提示词中声明工具清单（名称、参数、用途）—— 即最简 Tool Schema；
2. 模型判断需要工具时，返回结构化调用请求 {"tool": "...", "arguments": {...}}；
3. 智能体执行工具，把结果作为新消息回填，再次请求模型生成最终回答。

生产中 OpenAI 的 tools/tool_calls 协议、MCP 协议，本质都是这三步。
"""
import json
import re

from ..agent import Agent, jstr, try_parse_json, tr
from ..llm import Message
from .step03_react_agent import MockWeather

SYSTEM_PROMPT = """你是一个可以调用工具的智能体。可用工具：
- get_weather(city): 查询指定城市的实时天气
- calculator(expression): 计算四则运算，如 12*8

决策规则：
- 需要工具时，只输出 JSON：{"tool": "工具名", "arguments": {"参数名": "值"}}
- 不需要工具时，直接用自然语言回答，不要输出 JSON。
"""

_CALC_PTN = re.compile(r"\s*(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*", re.ASCII)


class ToolCallAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._last_trace = []

    def name(self) -> str:
        return "Step04 · 工具调用（Function Calling）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()
        messages = [Message.system(SYSTEM_PROMPT), Message.user(input)]

        # 第一轮：模型决定"要不要用工具"
        first = self._model.chat(messages).strip()
        decision = try_parse_json(first)

        if isinstance(decision, dict) and jstr(decision, "tool") is not None:
            tool = jstr(decision, "tool")
            args = json.dumps(decision.get("arguments"), ensure_ascii=False, separators=(",", ":"))
            self._last_trace.append(tr("tool", "模型发起工具调用", tool + " " + args))

            # 第二步：执行工具
            result = self._execute(tool, args)
            self._last_trace.append(tr("observation", "工具返回结果", result))

            # 第三步：结果回填，让模型基于事实作答
            messages.append(Message.assistant(first))
            messages.append(Message.user("工具 " + tool + " 的执行结果: " + result))
            answer = self._model.chat(messages)
            self._last_trace.append(tr("final", "基于工具结果生成回答",
                                       "模型拿到 Observation 后组织最终回答"))
            return answer

        # 模型选择不调用工具，直接回答
        self._last_trace.append(tr("final", "无需工具，直接回答", "模型判断自身知识足够"))
        return first

    def _execute(self, tool: str, args_json: str) -> str:
        try:
            args = json.loads(args_json if args_json is not None else "{}")
            if tool == "get_weather":
                return MockWeather.query(args.get("city"))
            if tool == "calculator":
                m = _CALC_PTN.fullmatch(str(args.get("expression")))
                if not m:
                    return "无法解析表达式"
                a, b = float(m.group(1)), float(m.group(3))
                op = m.group(2)
                if op == "+":
                    return str(a + b)
                if op == "-":
                    return str(a - b)
                if op == "*":
                    return str(a * b)
                if op == "/":
                    return "除数不能为 0" if b == 0 else str(a / b)
                return "不支持的运算符"
            return "未知工具: " + tool
        except Exception as e:
            return "工具执行失败: " + str(e)

    def trace(self) -> list:
        return list(self._last_trace)
