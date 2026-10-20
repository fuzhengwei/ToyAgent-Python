"""Step07 · RAG 智能体 —— 检索增强生成。

模型的知识是静态的，私有知识它根本没见过。RAG 的思路：
先把问题在知识库里检索出最相关的片段，再把片段作为"参考资料"
塞进提示词，让模型"戴着资料说话"。回答有出处、知识可热更。

完整 RAG = 切分 → 向量化 → 相似检索 → 增强生成。本场景用
关键词重合度模拟向量相似度（教学简化），骨架完全一致 ——
换成 Embedding 检索即生产级实现。
"""
from ..agent import Agent, tr
from ..llm import Message

# 私有知识库：真实系统存向量库，这里用内存文档集合演示。
KNOWLEDGE = [
    "ToyAgent 是小傅哥的智能体教学项目，用 12 个渐进式场景讲清楚智能体实现，每个场景只有一个接口类。",
    "智能体 = 大模型（大脑）+ 工具（手脚）+ 记忆（延续）+ 循环（节拍）。ReAct 循环是智能体的经典执行模式。",
    "MCP（Model Context Protocol）是 Anthropic 提出的工具标准化协议，核心方法为 tools/list 与 tools/call，实现工具与智能体解耦。",
    "RAG（检索增强生成）先从知识库检索相关片段，再把片段注入提示词让模型基于事实回答，能缓解幻觉与知识过期问题。",
    "Function Calling 让模型输出结构化工具调用请求（工具名 + 参数），由外部程序执行后将结果回填给模型。",
    "上下文窗口是模型一次能处理的最大 Token 数，超出时需要用滑动窗口、摘要压缩等上下文工程手段保护关键信息。",
]

SYSTEM_PROMPT = """你是知识库问答智能体。回答规则：
1. 优先依据【参考资料】回答，并标注来源编号如 [1]。
2. 资料不足以回答时，如实说"知识库中暂无相关内容"，禁止编造。
"""


def score(text: str, query: str) -> float:
    """教学版相似度：字符 bigram 重合率。生产中替换为 Embedding 余弦相似度。"""
    if len(query) < 2:
        return 0.0
    grams = {query[i:i + 2] for i in range(len(query) - 1)}
    if not grams:
        return 0.0
    hit = sum(1 for g in grams if g in text)
    return hit / len(grams)


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"


class RagAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._last_trace = []

    def name(self) -> str:
        return "Step07 · RAG 检索增强生成"

    def chat(self, input: str) -> str:
        self._last_trace.clear()

        # 1. 检索：按关键词重合度给知识片段打分，取 Top2
        ranked = sorted(
            ((text, score(text, input)) for text in KNOWLEDGE),
            key=lambda h: h[1], reverse=True)[:2]
        hits = [(t, s) for t, s in ranked if s > 0]
        self._last_trace.append(tr(
            "retrieve", "知识库检索",
            "未命中任何片段（得分均为 0）" if not hits else
            " ｜ ".join(f"{s * 100:.0f} 分 · {_clip(t, 30)}" for t, s in hits)))

        # 2. 增强：命中片段注入提示词；未命中则要求拒答
        messages = [Message.system(SYSTEM_PROMPT)]
        if hits:
            ctx = "【参考资料】\n"
            for i, (t, _s) in enumerate(hits, 1):
                ctx += f"[{i}] {t}\n"
            messages.append(Message.user(ctx))
        messages.append(Message.user(input))

        # 3. 生成
        answer = self._model.chat(messages)
        self._last_trace.append(tr(
            "final", "增强生成完成",
            "模型按约定拒答" if not hits else "回答基于 " + str(len(hits)) + " 个知识片段"))
        return answer

    def trace(self) -> list:
        return list(self._last_trace)
