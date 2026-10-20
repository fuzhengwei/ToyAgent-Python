"""Step05 · 记忆智能体 —— 让对话"有来有回"。

Step01 每轮都是失忆的。本场景引入短期记忆：把历史消息一并放进
上下文窗口发给模型。当历史超过阈值时，用"摘要压缩"保护窗口 ——
滑动窗口 + 摘要，就是上下文工程（Context Engineering）的基础盘。

记忆的本质不是数据库，而是：每一次请求，都带上该带的历史。
"""
import re

from ..agent import Agent, tr
from ..llm import Message

# 保留的最大历史轮数（一条 user + 一条 assistant 算一轮）。
MAX_ROUNDS = 4

_CLIP_PTN = re.compile(r"\s+", re.ASCII)

SYSTEM_PROMPT = """你是一个有记忆的聊天智能体。请基于"对话历史 + 压缩摘要 + 新输入"回答，
用户提到的个人信息（如名字）要在后续对话中保持一致。
"""


def _clip40(s: str) -> str:
    one = _CLIP_PTN.sub(" ", s).strip()
    return one if len(one) <= 40 else one[:40] + "…"


class MemoryAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._history = []
        self._summary = ""
        self._last_trace = []

    def name(self) -> str:
        return "Step05 · 记忆系统（多轮记忆 + 上下文压缩）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()
        self._history.append(Message.user(input))

        # 历史超限 → 压缩：把最旧的两轮合并为一条摘要
        if len(self._history) > MAX_ROUNDS * 2:
            self._compress()

        # 组装上下文：系统提示词 + 摘要 + 历史
        context = [Message.system(SYSTEM_PROMPT)]
        if self._summary:
            context.append(Message.system("【历史摘要】" + self._summary))
        context.extend(self._history)

        self._last_trace.append(tr(
            "memory", "记忆窗口",
            "携带 " + str(len(self._history)) + " 条历史消息"
            + ("" if not self._summary else " + 1 条压缩摘要")))

        answer = self._model.chat(context)
        self._history.append(Message.assistant(answer))

        self._last_trace.append(tr(
            "final", "回答完成",
            "本条回答同时看到：系统提示词" + ("" if not self._summary else " + 摘要") + " + 全部历史 + 新输入"))
        return answer

    def _compress(self):
        """上下文压缩：把最旧两轮摘成一句话，历史重新滑窗。"""
        old1 = self._history.pop(0)
        old2 = self._history.pop(0)
        digest = _clip40(old1.content) + "；随后助手回答了 " + _clip40(old2.content)
        self._summary = digest if not self._summary else self._summary + " → " + digest
        self._last_trace.append(tr(
            "compress", "触发上下文压缩",
            "最旧 2 轮已摘要化，摘要长度 " + str(len(self._summary)) + " 字符"))

    def reset(self) -> None:
        self._history.clear()
        self._summary = ""

    def trace(self) -> list:
        return list(self._last_trace)
