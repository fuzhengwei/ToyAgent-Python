"""Step19 · 人工介入 —— ask_user_question：把「异步的人」翻译成「可等待的答复」。

ToyAgent 是同步架构，教学版用「会话挂起 + 续答」实现人机协同：
1. 模型发现关键信息缺失时，调用 ask_user_question「工具」，turn 挂起，
   问题持久化在会话状态里返回给用户；
2. 用户的下一条消息不再进模型，而是作为人工答复回填挂起的工具位，
   turn 恢复继续执行。

生产版的差别只在「等待」的实现：用 Future 阻塞执行线程等人，
教学版把等待摊开成两次请求 —— 人机协同的语义完全一致。
"""
from ..agent import Agent, jstr, or_raw, try_parse_json, tr
from ..llm import Message

SYSTEM_PROMPT = """你是一个懂得「何时该问人」的智能体。
当任务缺少关键信息（如预订缺少出发地、时间），不要猜测，主动向人提问：
只输出 JSON：{"ask": "你的问题（含可选选项）"}
信息完整时直接完成任务，只输出 JSON：{"final": "最终答案"}。不要输出 JSON 以外内容。"""


class AskAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._history = []
        # 挂起中的问题：非 None 表示 turn 暂停，等待人工答复。
        self._pending_question = None
        self._last_trace = []

    def name(self) -> str:
        return "Step19 · 人工介入（ask_user_question：提问挂起 → 答复续跑）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()

        # ===== 分支一：有挂起问题 —— 本条输入是「人工答复」，不进模型 =====
        if self._pending_question is not None:
            q = self._pending_question
            self._pending_question = None
            self._last_trace.append(tr(
                "observation", "人工答复回填",
                "挂起问题「" + q + "」收到答复：" + input
                + "（生产版是 Future 阻塞-唤醒，教学版把等待摊开成两次请求）"))
            self._history.append(Message.user("人工答复：" + input))

            answer = self._complete(q)
            self._history.append(Message.assistant(answer))
            return answer

        # ===== 分支二：正常对话 —— 模型可发起 ask_user_question =====
        self._history.append(Message.user(input))
        req = [Message.system(SYSTEM_PROMPT)]
        req.extend(self._history)

        raw = self._model.chat(req).strip()
        parsed = try_parse_json(raw)

        question = jstr(parsed, "ask") if isinstance(parsed, dict) else None
        if question is not None:
            # 模型主动向人提问：挂起 turn，问题存入会话状态
            self._pending_question = question
            self._last_trace.append(tr(
                "action", "调用 ask_user_question",
                "turn 挂起，等待人工答复 —— 「异步的人」被翻译成了「可等待的工具结果」"))
            self._last_trace.append(tr(
                "guard", "会话挂起 PENDING_ASK",
                "问题已持久化在会话状态，下一条消息将作为答复回填"))
            return question + "\n\n（直接回复即可，我会带着你的答复继续任务。）"

        answer = or_raw(jstr(parsed, "final") if isinstance(parsed, dict) else None, raw)
        self._last_trace.append(tr("final", "直接作答", "信息完整，无需人工介入"))
        self._history.append(Message.assistant(answer))
        return answer

    def _complete(self, question: str) -> str:
        """答复回填后，带着完整信息让模型继续完成任务（turn 恢复）。"""
        req = [Message.system(SYSTEM_PROMPT + "\n（此前你曾向用户提问「" + question
                              + "」，用户已答复，见最后一条人工答复。现在信息完整，直接完成任务并作答。）")]
        req.extend(self._history)
        raw = self._model.chat(req).strip()
        parsed = try_parse_json(raw)
        fin = jstr(parsed, "final") if isinstance(parsed, dict) else None
        self._last_trace.append(tr(
            "final", "turn 恢复 · 完成",
            "人工答复注入上下文，任务继续执行 —— 人机协同闭环"))
        return or_raw(fin, raw) if isinstance(parsed, dict) else raw

    def reset(self) -> None:
        self._history.clear()
        self._pending_question = None

    def trace(self) -> list:
        return list(self._last_trace)
