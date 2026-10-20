"""Step15 · Loop 运行时智能体 —— 生产级 Agent Loop 的骨架。

Step03 的 ReAct 循环是"裸奔"的：没有保险丝、没有安全检查。
真实的智能体运行时（Runtime）在循环外面包了一层工程化外壳：
- 输入守卫 —— 拦截敏感/恶意请求（Prompt 注入防护）；
- 最大步数 —— 防止模型陷入死循环烧钱；
- 输出守卫 —— 对最终回答做合规过滤；
- 异常兜底 —— 工具失败不拖垮整个循环。

模型负责聪明，运行时负责可靠。
"""
import re

from ..agent import Agent, jstr, try_parse_json, tr
from ..llm import Message
from .step03_react_agent import MockWeather

MAX_STEPS = 4
MAX_INPUT_CHARS = 500

# 敏感词黑名单（生产中应为规则引擎 + 分类模型）。
BLOCKED = ["密码", "攻击", "hack", "漏洞", "暴力", "炸药"]

SYSTEM_PROMPT = """你是运行在受控运行时里的智能体。每轮只输出 JSON：
{"thought": "思考", "action": "get_weather|calculator", "action_input": "参数"}
或 {"thought": "思考", "final": "最终答案"}
可用工具：get_weather(city)、calculator(expression)
"""

_CALC_FULL = re.compile(r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)", re.ASCII)


class LoopAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._last_trace = []

    def name(self) -> str:
        return "Step15 · Agent Loop 运行时（守卫 + 保险丝）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()

        # ---------- 输入守卫 ----------
        if len(input) > MAX_INPUT_CHARS:
            self._last_trace.append(tr(
                "guard", "⛔ 输入守卫拦截",
                "输入超过 " + str(MAX_INPUT_CHARS) + " 字符上限"))
            return "输入过长，已拦截（防止上下文爆炸与注入攻击面扩大）。"
        for word in BLOCKED:
            if word in input.lower():
                self._last_trace.append(tr(
                    "guard", "⛔ 输入守卫拦截",
                    "命中敏感词: " + word + "（Prompt 注入/安全防护）"))
                return "该请求涉及受限内容，已被输入守卫拦截。"
        self._last_trace.append(tr(
            "guard", "🛡️ 输入守卫通过",
            "长度 " + str(len(input)) + " 字符，未命中敏感词"))

        # ---------- Agent Loop ----------
        messages = [Message.system(SYSTEM_PROMPT), Message.user(input)]

        for step in range(1, MAX_STEPS + 1):
            try:
                raw = self._model.chat(messages).strip()
            except Exception as e:
                self._last_trace.append(tr(
                    "warn", "⚠️ 异常兜底", "模型调用失败: " + str(e)))
                return "运行时异常已兜底，请稍后重试。"
            parsed = try_parse_json(raw)
            thought = jstr(parsed, "thought") if isinstance(parsed, dict) else None
            if thought is not None:
                self._last_trace.append(tr("thought", "Thought " + str(step), thought))

            fin = jstr(parsed, "final") if isinstance(parsed, dict) else None
            if fin is not None:
                # ---------- 输出守卫 ----------
                for word in BLOCKED:
                    if word in fin.lower():
                        self._last_trace.append(tr(
                            "guard", "⛔ 输出守卫拦截",
                            "回答命中敏感词，已替换为安全话术"))
                        return "回答未通过输出合规检查，已按安全策略处理。"
                self._last_trace.append(tr(
                    "final", "✅ 输出守卫通过 · Final", "循环共执行 " + str(step) + " 步"))
                return fin

            action = jstr(parsed, "action") if isinstance(parsed, dict) else None
            action_input = jstr(parsed, "action_input") if isinstance(parsed, dict) else None
            action = action or ""
            action_input = action_input or ""
            self._last_trace.append(tr(
                "action", "Action " + str(step), action + "(" + action_input + ")"))

            # 工具异常不拖垮循环
            try:
                observation = self._execute(action, action_input)
            except Exception as e:
                observation = "工具执行异常: " + str(e)
                self._last_trace.append(tr("warn", "⚠️ 工具异常", observation))
            self._last_trace.append(tr(
                "observation", "Observation " + str(step), observation))

            messages.append(Message.assistant(raw))
            messages.append(Message.user("Observation: " + observation))

        self._last_trace.append(tr(
            "guard", "🔥 保险丝熔断",
            "达到最大步数 " + str(MAX_STEPS) + "，强制终止循环"))
        return "已达到最大执行步数（" + str(MAX_STEPS) + "），为控制成本与风险强制终止。"

    def _execute(self, action: str, input: str) -> str:
        if action == "get_weather":
            return MockWeather.query(input)
        if action == "calculator":
            m = _CALC_FULL.fullmatch(input or "")
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
        return "未知工具: " + action

    def trace(self) -> list:
        return list(self._last_trace)
