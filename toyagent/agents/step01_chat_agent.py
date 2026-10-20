"""Step01 · 对话智能体 —— 智能体的最小 MVP。

一个接口，一个方法，没有策略、没有工具、没有记忆。
它证明了一件事：所谓智能体，起点只是「把用户的话交给模型，把模型的话还给用户」。
后续场景，都是往这个最小骨架里加入一种能力。
"""
from ..agent import Agent
from ..llm import Message


class ChatAgent(Agent):
    """最小 MVP 实现：每轮对话独立，输入 → 模型 → 输出。"""

    def __init__(self, model):
        self._model = model

    def name(self) -> str:
        return "Step01 · 对话智能体（最小 MVP）"

    def chat(self, input: str) -> str:
        # 全部实现就这一行核心逻辑
        return self._model.chat([Message.user(input)])
