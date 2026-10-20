"""Step06 · 意图路由智能体 —— Agent 的大脑：先识别意图，再决策走向。

复杂系统不会把所有问题都丢给一个循环：先让模型做一次"意图识别"
（输出 JSON 分类），再由决策中枢把请求路由到不同处理器 ——
天气走工具、算式走计算器、闲聊走对话。这正是 Agent Loop 中
Router/Dispatcher 的原型，也是多智能体系统编排的前置能力。
"""
import json
import math
import re

from ..agent import Agent, jstr, try_parse_json, tr
from ..llm import Message
from .step03_react_agent import MockWeather

ROUTER_PROMPT = """你是意图识别路由器。分析用户输入，只输出 JSON：
{"intent": "weather|math|chat", "reason": "一句话理由"}
- weather: 查询天气
- math: 四则运算
- chat: 其他一切对话
"""

_EXPR_FIND = re.compile(r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)", re.ASCII)

_CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都"]


class RouterAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._last_trace = []

    def name(self) -> str:
        return "Step06 · 意图识别与决策中枢（路由）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()

        # 1. 意图识别
        raw = self._model.chat([
            Message.system(ROUTER_PROMPT),
            Message.user(input)]).strip()
        decision = try_parse_json(raw)
        intent = jstr(decision, "intent") if isinstance(decision, dict) else None
        if intent is None:
            intent = "chat"
        reason = jstr(decision, "reason") if isinstance(decision, dict) else None
        self._last_trace.append(tr(
            "intent", "意图识别",
            intent + "（" + (reason or "") + "）"))

        # 2. 按意图路由到不同处理器
        if intent == "weather":
            city = self._extract_city(input)
            self._last_trace.append(tr("action", "路由 → 天气工具", "get_weather(" + city + ")"))
            result = MockWeather.query(city)
            self._last_trace.append(tr("observation", "工具返回", result))
            return "「" + city + "」现在" + result + "。"
        if intent == "math":
            self._last_trace.append(tr("action", "路由 → 计算器", input))
            return "计算结果：" + _calc(input) + "。"
        self._last_trace.append(tr("action", "路由 → 对话模型", "无工具直答"))
        return self._model.chat([
            Message.system("你是一个友好的中文聊天助手。"),
            Message.user(input)])

    def _extract_city(self, text: str) -> str:
        for city in _CITIES:
            if city in text:
                return city
        return "北京"

    def trace(self) -> list:
        return list(self._last_trace)


def _calc(expr: str) -> str:
    m = _EXPR_FIND.search(expr)
    if not m:
        return "未识别的算式"
    a, b = float(m.group(1)), float(m.group(3))
    op = m.group(2)
    if op == "+":
        r = a + b
    elif op == "-":
        r = a - b
    elif op == "*":
        r = a * b
    elif op == "/":
        r = math.nan if b == 0 else a / b
    else:
        r = math.nan
    if math.isnan(r):
        return "无法计算"
    return str(int(r)) if r == math.floor(r) else str(r)
