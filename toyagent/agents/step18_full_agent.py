"""Step18 · 全流程智能体 —— 把前面的场景的能力串成一条完整链路。

前面的场景每次只演示一种能力，这一个类把它们全部组装到一起，
一次 chat() 完整走完生产级智能体的标准生命周期：

    输入守卫 → 读取记忆(画像+历史) → 组装上下文 → ReAct 循环(思考→工具→观察)
           → 保险丝 → 输出校验 → 回写记忆

注意：它依然只是一个类 + 一个 chat(input) 方法 ——
智能体再复杂，骨架不变；变厚的只是「策略层」。
这也是理解生产框架（LangGraph / Dify / 工业级 Harness）的钥匙：
框架只是把这些环节做成了可配置、可观测、可容错的工程件。
"""
import re
from datetime import datetime

from ..agent import Agent, jstr, strip_code_fence, tr
from ..llm import Message
from .step03_react_agent import MockWeather, calc

# 循环保险丝：最多工具轮数（Budget Guard 的一种）。
MAX_STEPS = 5
# 输入长度守卫阈值。
MAX_INPUT_LEN = 300
# 记忆窗口压缩阈值：历史超过该条数时触发压缩。
COMPRESS_THRESHOLD = 10

SYSTEM_PROMPT = """你是 ToyAgent 综合智能体，具备记忆与工具使用能力。
每一轮只输出一个 JSON 对象，格式二选一：
1. 继续行动：{"thought": "你的思考", "action": "工具名", "action_input": "参数"}
2. 给出答案：{"thought": "你的思考", "final": "最终答案"}
可用工具：
- get_weather(city): 查询城市天气
- calculator(expression): 计算四则运算，如 12*8
- get_time(): 获取当前时间
回答时自然融入用户画像与历史信息；不要输出 JSON 以外的内容。
"""

# 敏感意图黑名单（教学演示，生产环境用内容安全服务）。
SENSITIVE = re.compile("hack|攻击|爆破|破解|密码", re.IGNORECASE)

# 画像提取：记住「我叫 XX / 我喜欢 XX」（排除「我叫什么」这类疑问句）。
NAME_PTN = re.compile("我(?:叫|的名字(?:是|叫)?)\\s*(?!(?:什么|啥|谁))([\u4e00-\u9fa5A-Za-z0-9]{1,12})")
LIKE_PTN = re.compile("我(?:喜欢|爱)\\s*([\u4e00-\u9fa5A-Za-z0-9·\\s]{1,12})")


def _java_map_str(profile: dict) -> str:
    """以 Java Map.toString 风格渲染画像（Mock 剧本依赖 `名字={` 标记）。"""
    return "{" + ", ".join(f"{k}={v}" for k, v in profile.items()) + "}"


class FullAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._last_trace = []
        # 跨轮记忆：历史消息与用户画像（真实项目里这两样在记忆组件中）。
        self._history = []
        self._profile = {}

    def name(self) -> str:
        return "Step18 · 全流程智能体（守卫+记忆+ReAct+工具+保险丝）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()

        # ---------- 1. 输入守卫（Step15 的 Guard） ----------
        if len(input) > MAX_INPUT_LEN:
            self._last_trace.append(tr(
                "guard", "输入守卫",
                "输入长度 " + str(len(input)) + " 超过上限 " + str(MAX_INPUT_LEN)
                + "，直接拒绝，不进模型。"))
            return "输入过长，请精简后重试（守卫规则：长度 ≤ " + str(MAX_INPUT_LEN) + " 字符）。"
        sm = SENSITIVE.search(input)
        if sm:
            self._last_trace.append(tr(
                "guard", "输入守卫",
                "命中敏感意图黑名单「" + sm.group() + "」，拦截且不消耗模型预算。"))
            return "该请求涉及不安全操作，已被输入守卫拦截。智能体的第一道防线在模型之前。"
        self._last_trace.append(tr(
            "guard", "输入守卫", "长度与安全检查通过，放行进入后续流程。"))

        # ---------- 2. 记忆读取 + 画像提取（Step05 的 Memory） ----------
        self._extract_profile(input)
        self._last_trace.append(tr(
            "memory", "读取记忆",
            "历史 " + str(len(self._history)) + " 条；画像："
            + ("暂无" if not self._profile else _java_map_str(self._profile))
            + "（随请求一起组装进上下文）。"))

        # ---------- 3. 组装上下文（system + 历史 + 本轮输入） ----------
        messages = [Message.system(SYSTEM_PROMPT + self._user_portrait())]
        messages.extend(self._history)
        messages.append(Message.user(input))

        # ---------- 4. ReAct 循环（Step03/04 的 Thought→Action→Observation） ----------
        answer = self._run_re_act(messages)

        # ---------- 5. 输出校验（守卫不只管输入） ----------
        answer = self._output_guard(answer, input)

        # ---------- 6. 记忆回写 ----------
        self._history.append(Message.user(input))
        self._history.append(Message.assistant(answer))
        memo = "本轮对话已写入记忆（现共 " + str(len(self._history)) + " 条）。"
        if len(self._history) > COMPRESS_THRESHOLD:
            self._compress_history()
            memo += " 超过阈值 " + str(COMPRESS_THRESHOLD) + " 条 → 上下文压缩已执行（保留系统提示词与最近消息）。"
        self._last_trace.append(tr("memory", "回写记忆", memo))
        return answer

    def _run_re_act(self, messages) -> str:
        """ReAct 主体：与 Step03 相同的循环骨架，外层多了守卫与记忆。"""
        for step in range(1, MAX_STEPS + 1):
            raw = self._model.chat(messages).strip()
            raw = strip_code_fence(raw)
            try:
                import json
                parsed = json.loads(raw)
            except Exception:
                # 模型没按协议输出 → 当作最终答案兜底返回
                self._last_trace.append(tr("final", "Final Answer", raw))
                return raw

            thought = jstr(parsed, "thought")
            self._last_trace.append(tr(
                "thought", "Thought " + str(step), thought or ""))

            fin = jstr(parsed, "final")
            if fin is not None:
                self._last_trace.append(tr("final", "Final Answer", fin))
                return fin

            action = jstr(parsed, "action") or ""
            action_input = jstr(parsed, "action_input") or ""
            self._last_trace.append(tr(
                "action", "Action " + str(step), action + "(" + action_input + ")"))

            observation = self._execute_tool(action, action_input)
            self._last_trace.append(tr(
                "tool", "Tool · " + action, "工具执行完成，产出观察结果。"))
            self._last_trace.append(tr(
                "observation", "Observation " + str(step), observation))

            messages.append(Message.assistant(raw))
            messages.append(Message.user("Observation: " + observation))
        msg = "已达最大步数 " + str(MAX_STEPS) + "，保险丝熔断，循环终止（Budget Guard）。"
        self._last_trace.append(tr("guard", "保险丝", msg))
        return msg

    def _execute_tool(self, action: str, input: str) -> str:
        """工具注册表：本地分支执行器（真实项目里是 MCP/插件网关）。"""
        try:
            if action == "get_weather":
                return MockWeather.query(input)
            if action == "calculator":
                return str(calc(input))
            if action == "get_time":
                return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return "未知工具: " + action
        except Exception as e:
            return "工具执行失败: " + str(e)

    def _output_guard(self, answer, input: str) -> str:
        """输出守卫：拦截答案中的敏感内容，并在能回答画像问题时兜底。"""
        if answer is not None and SENSITIVE.search(answer):
            self._last_trace.append(tr(
                "guard", "输出守卫", "答案命中敏感词，替换为安全话术后返回。"))
            return "该内容不宜展示，已由输出守卫处理。"
        # 记忆兜底：模型看不到被压缩掉的历史时，画像仍能回答「我叫什么」
        if "我叫什么" in input or "我是谁" in input:
            name = self._profile.get("名字")
            if name is not None:
                self._last_trace.append(tr(
                    "memory", "画像兜底",
                    "命中画像「名字=" + name + "」，无需再问模型。"))
                return "你叫 " + name + "（来自画像记忆，不依赖模型）。"
        return answer

    def _extract_profile(self, input: str):
        """从输入中提取用户画像（生产环境由记忆组件用模型抽取）。"""
        n = NAME_PTN.search(input)
        if n:
            self._profile["名字"] = n.group(1)
        l = LIKE_PTN.search(input)
        if l:
            self._profile["喜好"] = l.group(1).strip()

    def _user_portrait(self) -> str:
        """画像注入提示词的片段。"""
        if not self._profile:
            return ""
        return "\n已知用户画像：" + _java_map_str(self._profile) + "（回答时自然使用，不要逐字复述）。"

    def _compress_history(self):
        """上下文压缩：保留系统提示词语义（此实现重建时自动携带）与最近 6 条。"""
        keep = min(6, len(self._history))
        self._history[:] = self._history[-keep:]

    def reset(self) -> None:
        self._history.clear()
        self._profile.clear()

    def trace(self) -> list:
        return list(self._last_trace)
