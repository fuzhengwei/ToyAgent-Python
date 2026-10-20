"""Step20 · 审批门禁 —— 高危操作先过人，批准才放行。

四个教学点：
1. 风险分级：每个工具声明风险等级，安全工具（只读）自动放行，
   高危工具（写文件/发邮件/执行命令）必须过门禁；
2. 三档模式：AUTO_MANUAL_SAFE（本场景默认）/ ALWAYS_ASK / FULL_AUTO，决定门禁松紧；
3. 会话级免审记忆：「批准并记住」后，同工具本会话内自动放行 ——
   信任一次，不再重复打扰；
4. 超时默认 DENY：生产里审批有超时，超时即拒绝（宁可不执行，不可乱执行）。
"""
import enum

from ..agent import Agent, jstr, or_raw, try_parse_json, tr
from ..llm import Message


class Mode(enum.Enum):
    """审批模式三档。"""
    AUTO_MANUAL_SAFE = 0  # 安全自动放行，高危问人（默认）
    ALWAYS_ASK = 1        # 一切外部动作都要审批
    FULL_AUTO = 2         # 全自动（仅教学演示，生产慎用）


SYSTEM_PROMPT = """你是一个接入了多种工具的智能体。可用工具：
- get_weather(city)：查天气【安全·只读】
- send_email(to, content)：发邮件【高危·写操作】
- delete_file(path)：删文件【高危·写操作】
需要工具时只输出 JSON：{"tool": "工具名", "arguments": "参数摘要"}
无需工具时只输出 JSON：{"final": "最终答案"}。不要输出 JSON 以外内容。"""


class ApprovalAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._history = []
        # 会话级免审名单：批准并记住的高危工具。
        self._trusted_tools = set()
        # 挂起中的审批请求。
        self._pending_tool = None
        self._pending_args = None
        self._mode = Mode.AUTO_MANUAL_SAFE
        self._last_trace = []

    def name(self) -> str:
        return "Step20 · 审批门禁（高危操作先过人：批准 / 拒绝 / 批准并记住）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()

        # ===== 分支一：有挂起审批 —— 本条输入是人的决策，不进模型 =====
        if self._pending_tool is not None:
            tool = self._pending_tool
            args = self._pending_args
            self._pending_tool = None
            return self._resolve_approval(tool, args, input)

        # ===== 分支二：正常对话 =====
        self._history.append(Message.user(input))
        return self._decide_and_act(input)

    def _decide_and_act(self, input: str) -> str:
        req = [Message.system(SYSTEM_PROMPT)]
        req.extend(self._history)

        raw = self._model.chat(req).strip()
        parsed = try_parse_json(raw)
        if not isinstance(parsed, dict):
            self._history.append(Message.assistant(raw))
            self._last_trace.append(tr("final", "直接作答", "无工具需求"))
            return raw

        tool = jstr(parsed, "tool")
        if tool is None:
            answer = or_raw(jstr(parsed, "final"), raw)
            self._history.append(Message.assistant(answer))
            self._last_trace.append(tr("final", "直接作答", "无需工具"))
            return answer

        arg_val = parsed.get("arguments")
        args = "" if arg_val is None else str(arg_val)
        self._last_trace.append(tr(
            "action", "模型发起工具调用", tool + " " + args))

        # ===== 权限评估：风险分级 + 模式 + 会话免审记忆 =====
        dangerous = self._is_dangerous(tool)
        need_ask = (self._mode == Mode.ALWAYS_ASK
                    or (self._mode == Mode.AUTO_MANUAL_SAFE and dangerous
                        and tool not in self._trusted_tools))
        self._last_trace.append(tr(
            "thought", "权限矩阵评估",
            tool + " → " + ("高危" if dangerous else "安全")
            + " · 模式 " + self._mode.name
            + ((" · 命中会话免审" if tool in self._trusted_tools else ""))))

        if need_ask:
            self._pending_tool = tool
            self._pending_args = args
            self._last_trace.append(tr(
                "guard", "审批挂起 PENDING_APPROVAL",
                "高危操作等待人工决策。生产版有审批超时 —— 超时默认 DENY，宁可不执行"))
            return ("⚠️ 即将执行高危操作：" + tool + "\n参数：" + args + "\n\n"
                    + "请回复：「批准」执行 ｜「拒绝」取消 ｜「批准并记住」本会话内该工具免审")

        result = self._execute(tool, args)
        self._last_trace.append(tr(
            "observation",
            "自动放行（会话免审）" if (tool in self._trusted_tools and dangerous)
            else "自动放行（安全工具）",
            result))
        return self._finish(tool, result)

    def _resolve_approval(self, tool: str, args: str, decision: str) -> str:
        """处理人的审批决策。"""
        d = decision.strip()
        if d.startswith("批准并记住"):
            self._trusted_tools.add(tool)
            self._last_trace.append(tr(
                "guard", "人工决策 · 批准并记住",
                tool + " 加入会话级免审名单，本会话后续调用自动放行"))
        elif d.startswith("批准"):
            self._last_trace.append(tr(
                "guard", "人工决策 · 批准", "仅本次放行，下次仍会询问"))
        elif d.startswith("拒绝"):
            self._last_trace.append(tr(
                "guard", "人工决策 · 拒绝",
                "高危操作被拦截，任务终止（生产版超时未审批同样按 DENY 处理）"))
            self._history.append(Message.user("审批结果：" + tool + " 被拒绝"))
            return "已拒绝执行 " + tool + "。操作未发生 —— 审批门禁的意义就在于：人的否决权永远在模型之上。"
        else:
            self._pending_tool = tool
            self._pending_args = args
            self._last_trace.append(tr(
                "guard", "无法识别的决策",
                "审批仍挂起，请回复：批准 / 拒绝 / 批准并记住"))
            return "没看懂你的决定。请回复：「批准」｜「拒绝」｜「批准并记住」"
        result = self._execute(tool, args)
        self._last_trace.append(tr("observation", "审批放行 · 执行", result))
        self._history.append(Message.user("审批结果：" + tool + " 已批准并执行"))
        return self._finish(tool, result)

    def _finish(self, tool: str, result: str) -> str:
        """工具执行后，让模型基于结果收尾。"""
        req = [Message.system(SYSTEM_PROMPT + "\n（工具 " + tool + " 的执行结果：" + result + "。请据此作答。）")]
        req.extend(self._history)
        raw = self._model.chat(req).strip()
        parsed = try_parse_json(raw)
        answer = or_raw(jstr(parsed, "final") if isinstance(parsed, dict) else None, raw)
        self._history.append(Message.assistant(answer))
        return answer

    # ------------------------------------------------------- 风险与执行

    @staticmethod
    def _is_dangerous(tool: str) -> bool:
        """风险分级：写出去的操作都是高危，只读是安全。"""
        return tool in ("send_email", "delete_file", "shell_execute", "publish_article")

    @staticmethod
    def _execute(tool: str, args: str) -> str:
        """模拟执行（教学：不真发邮件、不真删文件）。"""
        if tool == "send_email":
            return "已发送（模拟）：" + args
        if tool == "delete_file":
            return "已删除（模拟）：" + args
        if tool == "get_weather":
            return "晴，26℃（模拟数据）"
        return "已执行（模拟）：" + args

    def reset(self) -> None:
        self._history.clear()
        self._trusted_tools.clear()
        self._pending_tool = None
        self._pending_args = None

    def trace(self) -> list:
        return list(self._last_trace)
