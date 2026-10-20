"""Step09 · MCP 智能体 —— 工具的标准化接口。

Step04 的工具是"写死在智能体里"的。MCP（Model Context Protocol）
把工具侧拆成一个独立服务：工具在 Server 中统一注册，Client 通过
两个标准方法使用它们 —— tools/list（发现）与 tools/call（调用）。
从此工具与智能体解耦：换智能体不用重写工具，接新工具不用改智能体。

本场景用几个小类模拟这一协议（真实 MCP 是 JSON-RPC over stdio/HTTP）。
"""
import json
from collections import namedtuple
from datetime import datetime

from ..agent import Agent, try_parse_json, tr
from ..llm import Message
from .step03_react_agent import MockWeather

# 工具描述符：MCP 世界里工具的标准"名片"。
ToolDescriptor = namedtuple("ToolDescriptor", ["name", "description", "parameters"])


class McpServer:
    """MCP Server 接口：工具的标准化宿主（真实实现为独立进程，JSON-RPC 通信）。"""

    def tools(self):
        raise NotImplementedError

    def call(self, tool_name: str, args_json: str) -> str:
        raise NotImplementedError


class WeatherTimeServer(McpServer):
    """示例 Server：挂载天气与时间两个工具。"""

    def tools(self):
        return [
            ToolDescriptor("get_weather", "查询城市实时天气", "city"),
            ToolDescriptor("get_time", "获取当前时间", ""),
        ]

    def call(self, tool_name: str, args_json: str) -> str:
        try:
            args = json.loads(args_json if args_json is not None else "{}")
            if tool_name == "get_weather":
                return MockWeather.query(args.get("city"))
            if tool_name == "get_time":
                return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return "未知工具"
        except Exception as e:
            return "调用失败: " + str(e)


class McpClient:
    """MCP Client：负责 tools/list 与 tools/call 两个标准方法。"""

    def __init__(self, server: McpServer):
        self._server = server

    def list_tools(self):
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._server.tools()
        ]

    def call_tool(self, name: str, args_json: str) -> str:
        if not any(t.name == name for t in self._server.tools()):
            return "MCP 错误: 工具不存在 " + name
        return self._server.call(name, args_json)


class McpAgent(Agent):

    def __init__(self, model):
        self._model = model
        # 一个 MCP Server 挂载两个工具，Client 负责协议交互
        self._client = McpClient(WeatherTimeServer())
        self._last_trace = []

    def name(self) -> str:
        return "Step09 · MCP 协议（工具标准化接口）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()

        # 1. 工具发现：从 MCP Server 拉取工具清单，拼进提示词
        tools = self._client.list_tools()
        self._last_trace.append(tr(
            "mcp", "tools/list 工具发现",
            "从 MCP Server 动态获取 " + str(len(tools)) + " 个工具: "
            + ", ".join(t["name"] for t in tools)))

        tool_docs = ""
        for t in tools:
            tool_docs += f"- {t['name']}({t['parameters']}): {t['description']}\n"

        decide_prompt = (
            "你是可以调用 MCP 工具的智能体。可用工具：\n"
            + tool_docs
            + '需要工具时只输出 JSON：{"method": "tools/call", "params": {"name": "工具名", "arguments": {"参数": "值"}}}\n'
            + "否则直接自然语言回答。\n")

        # 2. 模型决策
        raw = self._model.chat([Message.system(decide_prompt), Message.user(input)]).strip()
        decision = try_parse_json(raw)

        if isinstance(decision, dict) and decision.get("method") == "tools/call":
            params = decision.get("params") or {}
            name = str(params.get("name"))
            args = json.dumps(params.get("arguments"), ensure_ascii=False, separators=(",", ":"))
            self._last_trace.append(tr("tool", "tools/call 工具调用", name + " " + args))

            # 3. 经由 MCP Client 标准化调用
            result = self._client.call_tool(name, args)
            self._last_trace.append(tr("observation", "Server 返回", result))

            answer = self._model.chat([
                Message.system("基于工具返回的事实回答，简洁准确。"),
                Message.user(input + "\n工具结果: " + result)])
            self._last_trace.append(tr("final", "生成回答",
                                       "工具与智能体已解耦：Server 换实现，智能体零改动"))
            return answer

        self._last_trace.append(tr("final", "直接回答", "本次无需工具"))
        return raw

    def trace(self) -> list:
        return list(self._last_trace)
