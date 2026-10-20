"""Step13 · 子代理 —— 主智能体的「可派遣分身」。

主代理不必亲自做所有事：把子任务委托给带独立上下文的子代理，结果回填主会话。
两种派遣方式：
- spawn：全新实例，只有职责提示词，看不到父会话 —— 适合无状态专项任务；
- fork：复制父会话上下文再出发 —— 适合「接着当前话题继续做」。

与 Step12 多智能体的区别：Step12 是固定流水线（规划→研究→写作→审查），
角色间结构化协作；本步是主代理动态派遣—— 模型自己决定派谁、派几个、
带不带上下文，即 Claude Code 的 Task 工具。
"""
from ..agent import Agent, jstr, try_parse_json, tr
from ..llm import Message

# 子代理档案：职责提示词（真实场景里每个子代理还可挂自己的工具集）。
CHILDREN = {
    "researcher": "你是调研子代理。根据任务输出 3 条要点式调研结论，不写代码。",
    "writer": "你是写作子代理。根据任务输出一段 50 字以内的短文，不解释。",
}

SYSTEM_PROMPT_PREFIX = """你是主智能体，可以把子任务委托给子代理。可用子代理：
- researcher：调研汇总
- writer：文案写作
派遣方式：spawn = 全新实例（无父会话上下文）；fork = 继承父会话上下文。
需要委托时只输出 JSON：{"tool": "dispatch_subagent", "name": "researcher", "mode": "spawn", "task": "子任务描述"}
不需要委托时，直接用自然语言回答，不要输出 JSON。
"""


class SubagentAgent(Agent):

    def __init__(self, model):
        self._model = model
        # 父会话上下文：fork 时被子代理继承。
        self._parent_transcript = []
        self._last_trace = []

    def name(self) -> str:
        return "Step13 · 子代理（spawn 独立上下文 / fork 继承上下文）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()
        self._parent_transcript.append("用户: " + input)
        self._last_trace.append(tr(
            "tool", "主代理就绪",
            "可派遣子代理：[" + ", ".join(CHILDREN)
            + "]（父会话上下文 " + str(len(self._parent_transcript)) + " 条）"))

        messages = []
        ctx = "" if not self._parent_transcript else \
            "\n【父会话上下文】\n" + "\n".join(self._parent_transcript)
        messages.append(Message.system(SYSTEM_PROMPT_PREFIX + ctx))
        messages.append(Message.user(input))

        for _round in range(1, 4):
            raw = self._model.chat(messages).strip()
            decision = try_parse_json(raw)

            if isinstance(decision, dict) and jstr(decision, "tool") is not None:
                name = jstr(decision, "name")
                mode = jstr(decision, "mode")
                task = jstr(decision, "task")
                self._last_trace.append(tr(
                    "action", "主代理派遣子代理",
                    "subagent=" + name + "  mode=" + mode + "  task=" + task))

                result = self._dispatch(name, mode, task)
                self._last_trace.append(tr("observation", "子代理返回", result))

                messages.append(Message.assistant(raw))
                messages.append(Message.user("子代理 " + name + "(" + mode + ") 执行结果: " + result))
                continue

            self._last_trace.append(tr(
                "final", "最终回答", "主代理汇总子代理结果，向用户作答"))
            self._parent_transcript.append("助手: " + raw)
            return raw
        return "已达最大轮数，循环终止。"

    def _dispatch(self, name: str, mode: str, task: str) -> str:
        """派遣：spawn 给全新上下文；fork 复制父会话再出发。"""
        specialty = CHILDREN.get(name)
        if specialty is None:
            return "没有叫 " + name + " 的子代理。可用：[" + ", ".join(CHILDREN) + "]"
        fork = mode is not None and mode.lower() == "fork"

        sys_prompt = "【子代理身份】" + specialty
        if fork:
            sys_prompt += "\n【继承的父会话上下文】\n" + "\n".join(self._parent_transcript)
        else:
            sys_prompt += "\n（spawn 模式：全新实例，看不到父会话）"

        report = self._model.chat([
            Message.system(sys_prompt),
            Message.user(task)]).strip()
        self._last_trace.append(tr(
            "tool", "子代理运行（" + mode + "）",
            name + " 收到任务「" + task + "」，上下文 "
            + ("继承父会话 " + str(len(self._parent_transcript)) + " 条" if fork else "全新（0 条历史）")))
        return report

    def reset(self) -> None:
        self._parent_transcript.clear()

    def trace(self) -> list:
        return list(self._last_trace)
