"""Step17 · ReAct 运行时 —— turn/step 两级循环 + 上下文裁剪 + TurnEndReason。

三个核心机制：
1. turn/step 两级循环：一次 chat() = 一个 turn；turn 内的循环是 step 循环。
   与 Step03 的区别：工具执行完不结束回合，而是续步（mid-turn continuation）——
   循环只能以 TurnEndReason 收场；
2. 上下文裁剪 truncate_to_budget：每次调模型前估算 token（≈字符数/4），
   超预算就从最老的非系统消息开始丢弃（保底留最近 4 条）。多轮对话后
   在轨迹里能亲眼看到「上下文裁剪」事件；
3. 步数保险丝：单回合最多 MAX_STEPS_PER_TURN 步，超限以
   TurnEndReason.MaxSteps 收场（发「死循环测试」可亲眼看到保险丝起跳）。

工具直接复用 Step11 的 ToolRegistry —— 运行时与工具协议天然解耦。
"""
import re

from ..agent import Agent, jstr, try_parse_json, tr
from ..llm import Message
from .step11_registry_agent import ToolDefinition, ToolRegistry

# 单回合步数保险丝（生产值通常为 50）。
MAX_STEPS_PER_TURN = 6
# 上下文 token 预算（教学值调小以便演示，约 4 轮触发；估算规则：token ≈ 字符数 / 4）。
BUDGET_TOKENS = 100
# 裁剪时保底保留的最近消息条数。
KEEP_MIN = 4

SYSTEM_PROMPT = """你是一个使用 ReAct 模式的智能体，运行在 turn/step 两级循环之上。
可用工具（由注册表生成）：
"""

_CALC_PTN = re.compile(r"\s*(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*", re.ASCII)


# ------------------------------------------------- 回合结束原因（TurnEndReason）
#
# 用封闭类型统一表达"循环为什么停"：Completed / MaxSteps / Error 三种结局，
# 下游（前端展示、审计、续跑决策）都需要知道"为什么停"。

class Completed:
    """正常完成：模型给出最终答案。"""

    def __init__(self, summary: str):
        self.summary = summary


class MaxSteps:
    """触达步数保险丝：单回合最多 MAX_STEPS 步，强制收敛。"""

    def __init__(self, steps: int):
        self.steps = steps


class Error:
    """异常终止。"""

    def __init__(self, message: str):
        self.message = message


# ------------------------------------------------- 工具（本场景自带 EchoTool）

class _WeatherTool(ToolDefinition):
    DATA = {
        "北京": "晴，26℃", "上海": "多云，28℃", "广州": "雷阵雨，31℃",
        "深圳": "阵雨，30℃", "杭州": "晴，27℃", "成都": "阴，22℃",
    }

    def name(self):
        return "get_weather"

    def description(self):
        return "查询指定城市的实时天气"

    def parameters(self):
        return {"city": "string，城市名"}

    def execute(self, args: dict) -> str:
        city = str(args.get("city", "")).strip()
        return _WeatherTool.DATA.get(city, "未收录城市 " + city + " 的数据")


class _CalculatorTool(ToolDefinition):

    def name(self):
        return "calculator"

    def description(self):
        return "计算四则运算，如 12*8"

    def parameters(self):
        return {"expression": "string，算式"}

    def execute(self, args: dict) -> str:
        m = _CALC_PTN.fullmatch(str(args.get("expression", "")))
        if not m:
            return "无法解析表达式"
        a, b = float(m.group(1)), float(m.group(3))
        op = m.group(2)
        if op == "+":
            return str(a + b)
        if op == "-":
            return str(a - b)
        if op == "*":
            return str(a * b)
        if op == "/":
            return "除数不能为 0" if b == 0 else str(a / b)
        return "不支持的运算符"


class EchoTool(ToolDefinition):
    """永远回显的工具：专供「死循环测试」演示保险丝。"""

    def name(self):
        return "echo"

    def description(self):
        return "回显输入文本（演示用）"

    def parameters(self):
        return {"text": "string，要回显的内容"}

    def execute(self, args: dict) -> str:
        return "echo: " + str(args.get("text", ""))


def _estimate_tokens(msgs) -> int:
    """估算 token：教学化近似 —— token ≈ 字符数 / 4。"""
    return sum(len(m.content) // 4 for m in msgs)


def _nz(s):
    return s if s is not None else ""


def _first_line(a, b) -> str:
    s = b if (a is None or not a.strip()) else a
    i = s.find("\n")
    return s[:i] if i > 0 else s


class RuntimeAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._registry = ToolRegistry()
        # 跨对话的会话历史：运行时的上下文管理正是作用在它上面。
        self._history = []
        self._last_trace = []
        self._last_answer = ""
        self._registry.register(_WeatherTool())
        self._registry.register(_CalculatorTool())
        self._registry.register(EchoTool())

    def name(self) -> str:
        return "Step17 · ReAct 运行时（turn/step + 上下文裁剪 + TurnEndReason）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()
        self._history.append(Message.user(input))

        # ===== 机制 2：调模型前先做上下文预算裁剪 =====
        self._trim_to_budget()

        # ===== 机制 1：turn 内的 step 循环，只能以 TurnEndReason 收场 =====
        reason = self._run_turn()

        # 回合结束原因入轨迹，并作为 user 消息回写历史（下一轮模型可见）
        if isinstance(reason, Completed):
            label = "Turn 结束 · COMPLETED —— " + reason.summary
            fused = False
        elif isinstance(reason, MaxSteps):
            label = "Turn 结束 · MAX_STEPS —— 达到保险丝 " + str(reason.steps) + " 步，强制收敛"
            fused = True
        else:
            label = "Turn 结束 · ERROR —— " + reason.message
            fused = False
        self._last_trace.append(tr(
            "guard" if fused else "final", label,
            "结束原因以 TurnEndReason 类型统一表达，逐一处理"))
        return self._last_answer

    def _run_turn(self):
        """一个 turn：循环执行 step，直到出现结束原因。"""
        for step in range(1, MAX_STEPS_PER_TURN + 1):
            # ---- 组装请求：系统提示词 + 会话历史（裁剪后） ----
            req = [Message.system(
                SYSTEM_PROMPT + self._registry.prompt_catalog()
                + "\n决策规则：\n- 需要工具时，只输出 JSON："
                + '{"thought": "思考", "action": "工具名", "action_input": "参数"}\n'
                + '- 信息足够时，只输出 JSON：{"thought": "思考", "final": "最终答案"}\n'
                + "不要输出 JSON 以外的内容。")]
            req.extend(self._history)

            try:
                raw = self._model.chat(req).strip()
            except Exception as e:
                return Error("模型调用失败: " + str(e))
            parsed = try_parse_json(raw)
            self._last_trace.append(tr(
                "thought", "Step " + str(step) + " · 思考",
                _first_line(jstr(parsed, "thought") if isinstance(parsed, dict) else None, raw)))

            if not isinstance(parsed, dict):
                # 模型失约：按协议约定视为最终答案
                self._last_answer = raw
                self._history.append(Message.assistant(raw))
                return Completed("模型直接给出答案")

            fin = jstr(parsed, "final")
            if fin is not None:
                self._last_answer = fin
                self._history.append(Message.assistant(raw))
                return Completed("模型给出最终答案")

            # ---- 工具调用：执行后续步（不结束回合，这就是"续步"） ----
            action = jstr(parsed, "action") or ""
            action_input = jstr(parsed, "action_input") or ""
            self._last_trace.append(tr(
                "action", "Step " + str(step) + " · 行动",
                action + "(" + action_input + ")"))

            tool = self._registry.lookup(action)
            if tool is not None:
                try:
                    observation = tool.execute({
                        "city": _nz(action_input),
                        "expression": _nz(action_input),
                        "text": _nz(action_input)})
                except Exception as e:
                    observation = "工具执行失败: " + str(e)
            else:
                observation = "未知工具: " + action + "。可用清单：\n" + self._registry.prompt_catalog()
            self._last_trace.append(tr(
                "observation", "Step " + str(step) + " · 观察", observation))

            # 观察结果进历史 —— 下一 step 模型自然"看见"，循环继续
            self._history.append(Message.assistant(raw))
            self._history.append(Message.user("Observation: " + observation))
            self._trim_to_budget()

        self._last_answer = ("已达最大步数 " + str(MAX_STEPS_PER_TURN)
                             + "，本回合被保险丝强制终止。生产系统的 ReAct 循环必须有这层兜底（默认 50 步）。")
        return MaxSteps(MAX_STEPS_PER_TURN)

    # ===== 机制 2：上下文预算裁剪（truncate_to_budget） =====

    def _trim_to_budget(self):
        if _estimate_tokens(self._history) <= BUDGET_TOKENS or len(self._history) <= KEEP_MIN:
            return
        # 从最老的消息开始丢（历史里没有系统消息，系统提示词在请求时单独加）
        drop = 0
        while _estimate_tokens(self._history) > BUDGET_TOKENS and len(self._history) > KEEP_MIN:
            self._history.pop(0)
            drop += 1
        self._last_trace.append(tr(
            "compress", "上下文裁剪 truncateToBudget",
            "预估 token 超过预算 " + str(BUDGET_TOKENS) + "，丢弃最早 " + str(drop)
            + " 条消息，保留最近 " + str(len(self._history)) + " 条，现估 "
            + str(_estimate_tokens(self._history)) + " tokens"))

    def reset(self) -> None:
        self._history.clear()
        self._registry.clear()
        self._registry.register(_WeatherTool())
        self._registry.register(_CalculatorTool())
        self._registry.register(EchoTool())

    def trace(self) -> list:
        return list(self._last_trace)
