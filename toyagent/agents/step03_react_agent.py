"""Step03 · ReAct 智能体 —— 让模型学会"边想边做"。

ReAct = Reasoning + Acting。模型不再一次作答，而是输出结构化的
Thought（思考）→ Action（行动）→ 等待 Observation（观察结果），
智能体执行工具后把结果喂回模型，循环直到模型给出 Final Answer。
这是几乎所有智能体框架（LangChain、Agent Loop）的内核。
"""
import re
from datetime import datetime

from ..agent import Agent, jstr, strip_code_fence, tr
from ..llm import Message

MAX_STEPS = 5

# 用系统提示词约定 ReAct 的输出协议（JSON 形式，便于解析）。
SYSTEM_PROMPT = """你是一个使用 ReAct 模式解决问题的智能体。
每一轮只输出一个 JSON 对象，格式二选一：
1. 继续行动：{"thought": "你的思考", "action": "工具名", "action_input": "参数"}
2. 给出答案：{"thought": "你的思考", "final": "最终答案"}
可用工具：
- get_weather(city): 查询城市天气
- calculator(expression): 计算四则运算，如 12*8
- get_time(): 获取当前时间
不要输出 JSON 以外的内容。
"""

_CALC_PTN = re.compile(r"\s*(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*", re.ASCII)


class MockWeather:
    """本地 Mock 天气数据（后续多个场景复用）。"""

    DATA = {
        "北京": "晴，26℃", "上海": "多云，28℃", "广州": "雷阵雨，31℃",
        "深圳": "阵雨，30℃", "杭州": "晴，27℃", "成都": "阴，22℃",
    }

    @staticmethod
    def query(city) -> str:
        key = (city or "").strip()
        return MockWeather.DATA.get(key, "未收录城市 " + key + " 的数据")


def calc(expr: str) -> float:
    """教学用四则运算求值（仅支持 a op b）。"""
    m = _CALC_PTN.fullmatch(expr)
    if not m:
        raise ValueError("无法解析表达式: " + expr)
    a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0:
            raise ZeroDivisionError("除数不能为 0")
        return a / b
    raise ValueError("不支持的运算符 " + op)


class ReActAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._last_trace = []

    def name(self) -> str:
        return "Step03 · ReAct 智能体（思考-行动-观察循环）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()
        messages = [Message.system(SYSTEM_PROMPT), Message.user(input)]

        for step in range(1, MAX_STEPS + 1):
            # 1. 模型思考并决策
            raw = self._model.chat(messages).strip()
            raw = strip_code_fence(raw)
            try:
                import json
                parsed = json.loads(raw)
            except Exception:
                # 模型没按协议输出 → 视为最终答案直接返回
                self._last_trace.append(tr("final", "Final Answer", raw))
                return raw

            thought = jstr(parsed, "thought")
            self._last_trace.append(tr("thought", "Thought " + str(step), thought or ""))

            # 2. 有 final → 循环结束
            fin = jstr(parsed, "final")
            if fin is not None:
                self._last_trace.append(tr("final", "Final Answer", fin))
                return fin

            # 3. 有 action → 执行工具，把观察结果喂回模型
            action = jstr(parsed, "action") or ""
            action_input = jstr(parsed, "action_input") or ""
            self._last_trace.append(tr("action", "Action " + str(step), action + "(" + action_input + ")"))

            observation = self._execute_tool(action, action_input)
            self._last_trace.append(tr("observation", "Observation " + str(step), observation))

            messages.append(Message.assistant(raw))
            messages.append(Message.user("Observation: " + observation))

        msg = "已达最大步数 " + str(MAX_STEPS) + "，循环终止（生产系统必须有这个保险丝）。"
        self._last_trace.append(tr("guard", "达到最大步数", msg))
        return msg

    def _execute_tool(self, action: str, input: str) -> str:
        """工具注册表：本场景仅 3 个本地工具，执行器就是一个分支。"""
        try:
            if action == "get_weather":
                return MockWeather.query(input)
            if action == "calculator":
                return str(calc(input))
            if action == "get_time":
                return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return "未知工具: " + action
        except Exception as e:
            return "工具执行失败: " + str(e)

    def trace(self) -> list:
        return list(self._last_trace)
