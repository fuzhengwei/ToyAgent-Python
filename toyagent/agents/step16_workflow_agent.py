"""Step16 · 工作流智能体 —— LangGraph 式的状态机编排。

ReAct 是"模型说了算"的自由循环，但很多业务需要"流程说了算"的
确定性编排。把智能体拆成节点（Node）与边（Edge）：
每个节点是一段能力（分类/工具/问答/润色），条件边决定流转，
状态（State）在节点间传递 —— 这就是 LangGraph 的核心思想，
也是生产系统中"可控性"与"灵活性"的平衡点。
"""
import json
import math
import re

from ..agent import Agent, jstr, try_parse_json, tr
from ..llm import Message
from .step03_react_agent import MockWeather

_EXPR_FIND = re.compile(r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)", re.ASCII)


def _clip(s: str, n: int) -> str:
    import re as _re
    one = _re.sub(r"\s+", " ", s, flags=_re.ASCII).strip()
    return one if len(one) <= n else one[:n] + "…"


class WorkflowAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._last_trace = []

    def name(self) -> str:
        return "Step16 · 工作流状态机（LangGraph 思想）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()

        # State：在节点之间流转的共享状态
        state = {"input": input, "answer": "", "observation": ""}

        # 固定入口：classify 节点
        self._run_node("classify", state)

        # 条件边：按分类结果路由
        next_node = str(state.get("next", "faq"))
        self._last_trace.append(tr(
            "edge", "条件边",
            "classify --" + next_node + "--> "
            + ("tool 节点" if next_node == "tool" else "faq 节点")))

        self._run_node(next_node, state)
        self._run_node("polish", state)   # 汇合点：统一润色
        self._last_trace.append(tr(
            "edge", "END",
            "执行路径: classify → " + next_node + " → polish → END"))

        return str(state.get("answer"))

    def _run_node(self, node: str, state: dict):
        """节点执行器：每个节点在 State 上读输入、写产出。"""
        input = str(state.get("input"))
        node_prompt = "[节点:" + node + "] "
        if node == "classify":
            # 意图分类节点：模型输出路由决策
            raw = self._model.chat([
                Message.system(node_prompt
                               + '你是意图分类器，只输出 JSON：{"next": "faq|tool", "reason": "一句话"}\n'
                               + "涉及天气或四则运算 → tool，其他 → faq\n"),
                Message.user(input)]).strip()
            parsed = try_parse_json(raw)
            nxt = jstr(parsed, "next") if isinstance(parsed, dict) else None
            state["next"] = nxt if nxt is not None else "faq"
            self._last_trace.append(tr(
                "node", "🔹 节点 classify", "路由决策 → " + str(state.get("next"))))
        elif node == "tool":
            m = _EXPR_FIND.search(input)
            if m:
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
                observation = "计算结果 " + ("失败" if math.isnan(r)
                                           else str(int(r)) if r == math.floor(r) else str(r))
            else:
                if "上海" in input:
                    city = "上海"
                elif "杭州" in input:
                    city = "杭州"
                elif "成都" in input:
                    city = "成都"
                else:
                    city = "北京"
                observation = "「" + city + "」" + MockWeather.query(city)
            state["observation"] = observation
            self._last_trace.append(tr("node", "🔹 节点 tool", observation))
        elif node == "faq":
            answer = self._model.chat([
                Message.system(node_prompt + "你是 FAQ 问答节点，直接简洁回答常见问题。"),
                Message.user(input)])
            state["answer"] = answer
            self._last_trace.append(tr("node", "🔹 节点 faq", _clip(answer, 40)))
        elif node == "polish":
            # 汇合润色节点：tool 分支的 observation 在这里转为最终回答
            obs = str(state.get("observation"))
            base = str(state.get("answer")) if not obs else "根据查询结果：" + obs
            answer = self._model.chat([
                Message.system(node_prompt
                               + "你是润色节点。基于前序节点产出整理最终回答，保持事实不变、表达自然、50 字以内。\n前序产出: "
                               + base + "\n"),
                Message.user(input)])
            state["answer"] = answer
            self._last_trace.append(tr("node", "🔹 节点 polish", "最终回答已生成"))
        else:
            self._last_trace.append(tr("warn", "⚠️ 未知节点", node))

    def trace(self) -> list:
        return list(self._last_trace)
