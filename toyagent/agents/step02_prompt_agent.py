"""Step02 · 提示词工程 —— 给模型一份"角色说明书"。

与 Step01 唯一的区别：调用模型前，先塞入一条 system 消息。
系统提示词采用经典三层结构：角色（你是谁）→ 背景（约束与风格）→ 任务（怎么干）。
同一个模型，换一份提示词就换了一种"人格"与输出格式 —— 这就是提示词工程的杠杆。
"""
from ..agent import Agent, tr
from ..llm import Message

# 三层结构系统提示词：角色 → 背景 → 任务与约束。
SYSTEM_PROMPT = """# 角色
你是小傅哥的技术助理，专注讲清楚 AI Agent 的原理与工程实践。

# 背景
读者是正在学习智能体开发的 Java 工程师，熟悉后端技术，但对 AI 领域是新手。

# 任务与约束
1. 回答使用简体中文，口语化但专业。
2. 先给结论，再展开解释；涉及概念时给出生活化类比。
3. 若用户要求"用一句话"，必须严格只回答一句话。
4. 不确定的内容要如实说明，禁止编造。
"""


class PromptAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._last_trace = []

    def name(self) -> str:
        return "Step02 · 提示词工程（系统提示词三层结构）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()
        self._last_trace.append(tr(
            "prompt", "注入系统提示词",
            "角色/背景/任务 三层结构，长度 " + str(len(SYSTEM_PROMPT)) + " 字符"))
        answer = self._model.chat([
            Message.system(SYSTEM_PROMPT),
            Message.user(input)])
        self._last_trace.append(tr(
            "prompt", "模型按人设作答", "输出风格由 system 消息约束"))
        return answer

    def trace(self) -> list:
        return list(self._last_trace)

    @staticmethod
    def system_prompt() -> str:
        """暴露系统提示词，供展示。"""
        return SYSTEM_PROMPT
