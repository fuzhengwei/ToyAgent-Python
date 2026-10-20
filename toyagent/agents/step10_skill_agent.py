"""Step10 · Skills 技能智能体 —— 工具的组合与复用。

单个工具解决单步问题，真实任务往往是多步组合。
Skill 是"可沉淀、可复用的能力包"，按抽象层级分三层：
- L0 提示词技能 —— 纯 Prompt 模板，零工具；
- L1 工具组合 —— 固定调用一组工具并拼装结果；
- L2 子流程 —— 一个完整的 ReAct 循环被封成一项技能。

模型按语义选择技能，智能体负责执行 —— 这就是 Claude Skills /
插件体系的编排思想。
"""
from datetime import datetime

from ..agent import Agent, jstr, try_parse_json, tr
from ..llm import Message
from .step03_react_agent import MockWeather

SELECT_PROMPT = """你是技能选择器。当前已注册技能：
- travel_plan(L1): 旅行攻略 —— 组合天气查询与时间工具生成出行建议
- weather_report(L1): 天气播报 —— 调用天气工具
- free_chat(L0): 自由聊天 —— 纯提示词技能，直接回答

根据用户输入只输出 JSON：{"skill": "技能名", "reason": "一句话理由"}
"""


class SkillAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._last_trace = []

    def name(self) -> str:
        return "Step10 · 技能编排（Skills L0/L1/L2）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()

        # 1. 技能选择
        raw = self._model.chat([Message.system(SELECT_PROMPT), Message.user(input)]).strip()
        decision = try_parse_json(raw)
        skill = jstr(decision, "skill") if isinstance(decision, dict) else None
        if skill is None:
            skill = "free_chat"
        reason = jstr(decision, "reason") if isinstance(decision, dict) else None
        self._last_trace.append(tr(
            "skill", "技能选择", skill + "（" + (reason or "") + "）"))

        # 2. 技能执行
        if skill == "travel_plan":
            return self._travel_plan(input)        # L1: 多工具组合
        if skill == "weather_report":
            return self._weather_report(input)     # L1: 单工具
        return self._free_chat(input)              # L0: 纯提示词

    def _travel_plan(self, input: str) -> str:
        """L1 技能：组合天气 + 时间两个工具，产出结构化攻略。"""
        if "杭州" in input:
            city = "杭州"
        elif "上海" in input:
            city = "上海"
        elif "成都" in input:
            city = "成都"
        elif "广州" in input:
            city = "广州"
        elif "深圳" in input:
            city = "深圳"
        else:
            city = "北京"
        self._last_trace.append(tr("action", "L1 组合工具 1/2", "get_weather(" + city + ")"))
        weather = MockWeather.query(city)
        self._last_trace.append(tr("observation", "工具返回", weather))
        self._last_trace.append(tr("action", "L1 组合工具 2/2", "get_time()"))
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._last_trace.append(tr("observation", "工具返回", now))

        answer = self._model.chat([
            Message.system(
                "你是旅行顾问。基于给定的天气与时间信息，输出简短出行攻略（3 条以内要点）。\n"),
            Message.user(input + "\n天气: " + weather + "\n当前时间: " + now)])
        self._last_trace.append(tr("final", "L1 技能完成", "两件工具的结果已拼装进提示词"))
        return answer

    def _weather_report(self, input: str) -> str:
        """L1 技能：单工具播报。"""
        city = "上海" if "上海" in input else ("杭州" if "杭州" in input else "北京")
        self._last_trace.append(tr("action", "L1 调用工具", "get_weather(" + city + ")"))
        weather = MockWeather.query(city)
        self._last_trace.append(tr("final", "播报完成", weather))
        return "「" + city + "」" + weather + "。"

    def _free_chat(self, input: str) -> str:
        """L0 技能：纯提示词，不碰工具。"""
        self._last_trace.append(tr("final", "L0 提示词技能", "无工具，直接对话"))
        return self._model.chat([
            Message.system("你是友好的中文助手，回答保持简洁。"),
            Message.user(input)])

    def trace(self) -> list:
        return list(self._last_trace)
