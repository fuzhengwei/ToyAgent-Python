"""Step12 · 多智能体协作 —— 从一个大脑到一支团队。

单个智能体上下文有限、职责混杂。多智能体模式按角色拆分：
规划者（Planner）拆解任务 → 专家（Worker）各司其职 →
审查员（Reviewer）质量把关 → 主持者（Orchestrator）汇总交付。

每个角色都是一个独立的模型调用（不同的系统提示词），
协作的本质是：结构化的消息在角色之间流动。
"""
import re

from ..agent import Agent, jstr, try_parse_json, tr
from ..llm import Message


def _clip(s: str, n: int) -> str:
    one = re.sub(r"\s+", " ", s, flags=re.ASCII).strip()
    return one if len(one) <= n else one[:n] + "…"


class MultiAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._last_trace = []

    def name(self) -> str:
        return "Step12 · 多智能体协作（规划-执行-审查）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()

        # 1. 规划者：拆解任务
        plan_raw = self._model.chat([
            Message.system(
                "你是规划者（Planner）。把用户任务拆解为三个角色的执行计划，只输出 JSON：\n"
                '{"plan": [{"role": "研究员", "task": "..."}, {"role": "写手", "task": "..."}, {"role": "审查员", "task": "..."}]}\n'),
            Message.user(input)]).strip()
        plan_obj = try_parse_json(plan_raw)
        plan = plan_obj.get("plan") if isinstance(plan_obj, dict) else None
        plan = plan if isinstance(plan, list) else []
        self._last_trace.append(tr(
            "agent", "🧭 规划者",
            "产出 " + str(len(plan)) + " 步计划：" + self._describe(plan)))

        # 2. 研究员：收集素材
        notes = self._model.chat([
            Message.system("你是研究员（Researcher），围绕任务输出 3 条以内研究要点，每条一句话。"),
            Message.user(input)])
        self._last_trace.append(tr("agent", "🔬 研究员", _clip(notes, 60)))

        # 3. 写手：基于素材成稿
        draft = self._model.chat([
            Message.system(
                "你是写手（Writer）。基于研究笔记写一段 100 字以内的短文，主题如下。\n研究笔记: "
                + notes + "\n"),
            Message.user(input)])
        self._last_trace.append(tr("agent", "✍️ 写手", _clip(draft, 60)))

        # 4. 审查员：质量把关
        review_raw = self._model.chat([
            Message.system(
                "你是审查员（Reviewer）。审查短文，只输出 JSON：\n"
                '{"pass": true/false, "comment": "一句话点评"}\n'),
            Message.user(draft)]).strip()
        review = try_parse_json(review_raw)
        review = review if isinstance(review, dict) else {}
        passed = str(review.get("pass")).strip().lower() == "true"
        comment = jstr(review, "comment")
        self._last_trace.append(tr(
            "guard" if passed else "warn",
            "✅ 审查员：通过" if passed else "⚠️ 审查员：打回",
            comment or ""))

        # 5. 主持者汇总交付
        if passed:
            final_answer = draft
        else:
            final_answer = self._model.chat([
                Message.system("根据审查意见修改短文，直接输出修改后的版本。"),
                Message.user("原文: " + draft + "\n审查意见: " + (comment or ""))])
        self._last_trace.append(tr("final", "🏁 主持者汇总",
                                   "协作链路：规划 → 研究 → 写作 → 审查 → 交付"))
        return final_answer

    def _describe(self, plan) -> str:
        sb = ""
        for item in plan:
            if isinstance(item, dict):
                sb += str(item.get("role")) + "→"
        return sb[:-1] if sb else sb

    def trace(self) -> list:
        return list(self._last_trace)
