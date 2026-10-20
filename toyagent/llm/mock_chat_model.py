"""教学演示用 Mock 模型：不联网、不花钱，让 26 个场景开箱即跑。

它并非"真智能"，而是按场景预设了符合协议的返回（例如工具调用时
返回 JSON 决策、ReAct 时返回 Thought/Action），让学习者可以完整
观察智能体的「骨架」如何运转。配置真实 API Key 后骨架不变，
换上真大脑即可。

本文件与 Java 版 MockChatModel 逐剧本 1:1 对齐（正则、关键词、
返回文案照搬），是 Mock 冒烟测试验收的基准。
"""
import json
import re

from .chat_model import ChatModel
from .message import Message


def jdumps(value) -> str:
    """紧凑 JSON 序列化（等价 Java 自研 Json.write 的输出形态）。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _extract_after(text: str, marker: str) -> str:
    idx = text.rfind(marker)
    if idx < 0:
        return ""
    return re.sub(r"^[\s:：]+", "", text[idx + len(marker):]).strip()


def _clip(s: str) -> str:
    one = re.sub(r"\s+", " ", s, flags=re.ASCII).strip()
    return one if len(one) <= 60 else one[:60] + "…"


def _find_city(text: str) -> str:
    for city in CITIES:
        if city in text:
            return city
    return "北京"


def _count_history(all_text: str) -> int:
    return sum(1 for line in all_text.split("\n") if line.strip())


def _is_expr_sentence(text: str) -> bool:
    return re.fullmatch(r".*\d+\s*[+\-*/]\s*\d+.*", text, re.ASCII) is not None


def _find_expr(text: str, default: str = "1+1") -> str:
    m = re.search(r"(\d+\s*[+\-*/]\s*\d+)", text, re.ASCII)
    return m.group(1) if m else default


_NAME_RE = re.compile("我叫([\u4e00-\u9fa5A-Za-z0-9]{1,10})")

CITIES = {
    "北京": "晴，26℃", "上海": "多云，28℃", "广州": "雷阵雨，31℃",
    "深圳": "阵雨，30℃", "杭州": "晴，27℃", "成都": "阴，22℃",
}


class MockChatModel(ChatModel):

    def __init__(self, scenario: str):
        self._scenario = scenario

    def chat(self, messages) -> str:
        system = messages[0].content
        last = self._last_user_text(messages)
        all_text = "".join("\n" + m.content for m in messages)

        handler = {
            "step01": lambda: self._mock_chat(last),
            "step02": lambda: self._mock_prompt(last, system),
            "step03": lambda: self._mock_re_act(all_text, last),
            "step04": lambda: self._mock_tool_call(last),
            "step05": lambda: self._mock_memory(all_text, last),
            "step06": lambda: self._mock_router(last),
            "step09": lambda: self._mock_mcp(last),
            "step10": lambda: self._mock_skill(all_text, last),
            "step07": lambda: self._mock_rag(all_text, last),
            "step12": lambda: self._mock_multi(all_text, system),
            "step15": lambda: self._mock_loop(last, all_text),
            "step16": lambda: self._mock_workflow(all_text, last),
            "step18": lambda: self._mock_full(all_text, last),
            "step11": lambda: self._mock_registry(last),
            "step17": lambda: self._mock_runtime(all_text, last),
            "step19": lambda: self._mock_ask(all_text, last),
            "step20": lambda: self._mock_approval(last),
            "step21": lambda: self._mock_sandbox(last),
            "step22": lambda: self._mock_event(all_text, last),
            "step23": lambda: self._mock_plugin(all_text, last),
            "step25": lambda: self._mock_cli(all_text, last, system),
            "step13": lambda: self._mock_subagent(all_text, last, system),
            "step24": lambda: self._mock_hooks(all_text, last, system),
            "step14": lambda: self._mock_a2a(all_text, last, system),
            "step26": lambda: self._mock_schedule(all_text, last, system),
        }.get(self._scenario)
        return handler() if handler else "（Mock 模型）收到：" + last

    @staticmethod
    def _last_user_text(messages) -> str:
        for m in reversed(messages):
            if m.role == "user":
                return m.content
        return ""

    # ------------------------------------------------------------- step01 对话

    def _mock_chat(self, input: str) -> str:
        if "你好" in input or "hi" in input or "hello" in input:
            return "你好！我是 ToyAgent 里的最小智能体。我的全部实现只有一个方法：chat(input)。输入进来，模型处理，回答出去 —— 这就是智能体的最小 MVP。"
        return ("（Mock 模型）已收到你的消息：「" + _clip(input) + "」。\n"
                + "Step01 的实现没有策略、没有工具、没有记忆 —— 每次对话都是一次独立的「输入 → 模型 → 输出」。\n"
                + "配置真实 API Key 后，这句回复将由真实大模型生成。继续试试右边的示例问题吧。")

    # ------------------------------------------------------------- step02 提示词

    def _mock_prompt(self, input: str, system: str) -> str:
        if "一句话" in input:
            return "好的，遵守「一句话」约束：智能体就是一个用大模型做决策、用工具做执行的程序。"
        return ("好的，我是你的技术助理（角色由系统提示词设定）。「" + _clip(input) + "」可以这样理解：\n\n"
                + "1. 定义 —— 智能体 = 大模型(大脑) + 工具(手脚) + 循环(节拍)。\n"
                + "2. 原理 —— 模型根据系统提示词中的人设与约束来组织回答，本条回复就是模板化输出的效果。\n"
                + "3. 建议 —— 好提示词 = 明确角色 + 交代背景 + 给出任务与约束，这正是 Step02 代码里 systemPrompt 的三层结构。\n\n"
                + "（Mock 模型按系统提示词的格式约定作答；真实模型会表现出同样的角色倾向。）")

    # ------------------------------------------------------------- step18 全流程

    def _mock_full(self, all_text: str, last: str) -> str:
        portrait = ""
        if "名字={" in all_text:
            portrait = "，" + re.sub(r"\}.*", "", _extract_after(all_text, "名字={")) + "同学"
        elif "喜好={" in all_text:
            portrait = "，还记得你喜欢" + re.sub(r"\}.*", "", _extract_after(all_text, "喜好={"))
        if "Observation:" in all_text:
            obs = _extract_after(all_text, "Observation:")
            return jdumps({
                "thought": "工具结果已返回，结合记忆与观察结果组织最终回答",
                "final": _clip(obs) + " —— 本轮完整走过了：守卫 → 记忆 → 循环 → 工具 → 输出校验 → 回写记忆"
                         + ("。" if not portrait else "（" + portrait.replace("，", "", 1) + "）。")})
        if _is_expr_sentence(last):
            return jdumps({
                "thought": "有算式，调用计算器工具，不能靠心算",
                "action": "calculator",
                "action_input": _find_expr(last)})
        city = _find_city(last)
        if city != "北京" or "天气" in last:
            return jdumps({
                "thought": "问题需要实时数据，先调天气工具",
                "action": "get_weather",
                "action_input": city})
        return jdumps({
            "thought": "普通寒暄无需工具，直接结合记忆回答",
            "final": "你好" + ("！" if not portrait else portrait + "！")
                     + "我是全流程智能体：守卫已放行你的输入，记忆里" + ("还没有你的档案" if not portrait else "有你的画像")
                     + "，本轮无需调用工具即可作答。"})

    # ------------------------------------------------------------- step03 ReAct

    def _mock_re_act(self, all_text: str, last: str) -> str:
        if "Observation:" in all_text:
            obs = _extract_after(all_text, "Observation:")
            return jdumps({
                "thought": "已拿到工具观察结果，可以回答用户了",
                "final": "根据工具查询结果：" + _clip(obs) + " —— 这就是 ReAct：先想（Thought）、再做（Action）、看结果（Observation），直到能给出最终答案（Final Answer）。"})
        if _is_expr_sentence(last):
            return jdumps({
                "thought": "这是一个算式，我不能心算，应该调用计算器工具保证准确",
                "action": "calculator",
                "action_input": _find_expr(last)})
        city = _find_city(last)
        return jdumps({
            "thought": "用户询问了天气相关的问题，我需要先调用天气工具获取实时数据",
            "action": "get_weather",
            "action_input": city})

    # ------------------------------------------------------------- step04 工具调用

    def _mock_tool_call(self, last: str) -> str:
        # 工具结果回填轮：不再发起调用，基于事实生成回答
        if "执行结果" in last or "Observation" in last:
            fact = _extract_after(last, "执行结果")
            return "根据工具返回的结果：" + (_clip(last) if not fact.strip() else _clip(fact)) + "，这就是你要的答案 —— 工具调用链闭环完成。"
        if "天气" in last or "weather" in last:
            return jdumps({"tool": "get_weather", "arguments": {"city": _find_city(last)}})
        return jdumps({"tool": "calculator", "arguments": {"expression": _find_expr(last)}}) \
            if re.search(r"(\d+\s*[+\-*/]\s*\d+)", last, re.ASCII) \
            else ("这个问题不需要调用工具，我直接回答：「" + _clip(last) + "」—— 工具调用的关键在于：模型自己判断「要不要用工具、用哪个、参数是什么」。")

    # ------------------------------------------------------------- step05 记忆

    def _mock_memory(self, all_text: str, last: str) -> str:
        # 从完整对话历史里找"我叫X"，真实体现"记忆 = 把历史放进上下文"
        # 修正：排除「我叫什么/名字/谁/啥」疑问句（与 step22 剧本的排除式一致，
        # 否则「我叫什么」会被误当作用户自报姓名）
        remembered = None
        for m in _NAME_RE.finditer(all_text):
            n = m.group(1)
            if n.startswith("什么") or n.startswith("名字") or n.startswith("谁") or n.startswith("啥"):
                continue
            remembered = n

        if "我叫" in last and not re.search(r"我叫(?:什么|名字|谁|啥)", last):
            return "你好，" + remembered + "！我已把你的名字放进对话历史（注意看右侧的「记忆窗口」计数）。"
        if "叫什么" in last or "记得" in last or "谁" in last:
            return ("当然记得，你叫「" + remembered + "」。我没有数据库 —— 是因为之前那轮对话还在上下文窗口里，这就是短期记忆的全部秘密。"
                    if remembered is not None
                    else "目前的历史里还没有你的名字，先告诉我「我叫XX」吧。")
        if "压缩" in last or "总结" in last:
            return "已对历史对话做了压缩摘要：保留关键事实（如你的名字、讨论主题），丢弃寒暄与重复内容 —— 滑窗 + 摘要，就是上下文工程最常用的两板斧。"
        return "（Mock 模型）这一轮我带着 " + str(_count_history(all_text)) + " 条历史消息在工作。多轮对话 = 每次都把「系统提示词 + 全部历史 + 新输入」发给模型。"

    # ------------------------------------------------------------- step06 路由

    def _mock_router(self, last: str) -> str:
        if "天气" in last or "weather" in last:
            intent = "weather"
        elif _is_expr_sentence(last):
            intent = "math"
        else:
            intent = "chat"
        return jdumps({"intent": intent, "reason": "根据用户输入关键词与语义判断下一步走向"})

    # ------------------------------------------------------------- step09 MCP

    def _mock_mcp(self, last: str) -> str:
        # 工具结果回填轮：不再发起调用
        if "工具结果" in last or "Observation" in last:
            fact = _extract_after(last, "工具结果")
            return "基于 MCP Server 返回的事实：" + (_clip(last) if not fact.strip() else _clip(fact)) + " —— 标准协议下的调用链闭环完成。"
        if "天气" in last or "weather" in last:
            return jdumps({"method": "tools/call",
                           "params": {"name": "get_weather", "arguments": {"city": _find_city(last)}}})
        return "（Mock 模型）MCP 的核心是把「工具」标准化：统一注册、统一发现（tools/list）、统一调用（tools/call）。这个问题我直接回答即可，无需调用工具。"

    # ------------------------------------------------------------- step10 技能

    def _mock_skill(self, all_text: str, last: str) -> str:
        # 技能执行轮：travel_plan 技能内的模型调用按角色作答
        if "旅行顾问" in all_text:
            return "杭州出行攻略：① 当前晴，27℃，适宜户外；② 建议上午游西湖，下午避晒逛馆；③ 早晚温差小，轻装出行即可。（由 L1 技能组合天气与时间工具产出）"
        if "旅行" in last or "攻略" in last or "旅游" in last:
            skill = "travel_plan"
        elif "天气" in last:
            skill = "weather_report"
        else:
            skill = "free_chat"
        return jdumps({"skill": skill, "reason": "依据输入语义匹配已注册技能"})

    # ------------------------------------------------------------- step07 RAG

    def _mock_rag(self, all_text: str, last: str) -> str:
        ctx = _extract_after(all_text, "【参考资料】")
        if ctx:
            first_line = next((l for l in ctx.split("\n") if l.strip()), "相关资料")
            return ("根据检索到的资料：「" + _clip(first_line) + "」……\n\n"
                    + "回答基于知识库片段 [1][2] 生成 —— 模型只负责「戴着资料说话」，事实由检索保证，这就是 RAG 解决模型幻觉与知识过期的思路。")
        return "（Mock 模型）知识库中未命中相关片段时，我会如实回答不知道，而不是编造 —— 拒答也是 RAG 的重要能力。"

    # ------------------------------------------------------------- step12 多智能体

    def _mock_multi(self, all_text: str, system: str) -> str:
        if "规划者" in system:
            return jdumps({"plan": [
                {"role": "研究员", "task": "围绕主题收集要点"},
                {"role": "写手", "task": "把要点组织成一篇短文"},
                {"role": "审查员", "task": "检查事实与结构，不通过则打回"}]})
        if "研究员" in system:
            return "研究笔记：① 智能体的本质是模型驱动的循环；② 工具扩展了模型的行动边界；③ 工程化决定上限。"
        if "审查员" in system:
            return jdumps({"pass": True, "comment": "结构完整、表述准确，建议保留要点式风格"})
        if "研究笔记" in all_text:
            return "短文：智能体并不神秘 —— 它是一个让大模型「边想边做」的循环程序：模型负责决策，工具负责执行，记忆负责延续，工程负责兜底。（由写手智能体基于研究笔记产出）"
        return "（Mock 模型）已收到任务。"

    # ------------------------------------------------------------- step15 Loop + 守卫

    def _mock_loop(self, last: str, all_text: str) -> str:
        if "Observation:" in all_text:
            obs = _extract_after(all_text, "Observation:")
            return jdumps({"thought": "信息足够，输出最终答案", "final": "循环结束，最终回答：" + _clip(obs)})
        return jdumps({
            "thought": "进入运行时循环，先获取所需信息",
            "action": "get_weather",
            "action_input": _find_city(last)})

    # ------------------------------------------------------------- step16 工作流

    def _mock_workflow(self, all_text: str, last: str) -> str:
        if "[节点:classify]" in all_text:
            next_node = "tool" if ("天气" in last or re.fullmatch(r".*\d+\s*[+\-*/].*", last, re.ASCII)) else "faq"
            return jdumps({"next": next_node, "reason": "意图分类节点给出路由"})
        if "[节点:tool]" in all_text:
            if _is_expr_sentence(last):
                return jdumps({"tool": "calculator", "arguments": {"expression": _find_expr(last)}})
            return jdumps({"tool": "get_weather", "arguments": {"city": _find_city(last)}})
        if "Observation:" in all_text:
            return "综合以上节点结果，这里是最终回答：" + _clip(_extract_after(all_text, "Observation:")) + "（由 polish 节点整理输出）。"
        if "[节点:polish]" in all_text:
            base = next((l for l in _extract_after(all_text, "前序产出: ").split("\n") if l.strip()), "")
            return "润色完成：「" + _clip(last) + "」。" if not base else "润色完成：" + _clip(base)
        return "这是 FAQ 节点的直接回答。「" + _clip(last) + "」属于常见问题，无需工具，命中预设答案后进入 polish 润色。"

    # ------------------------------------------------------------- step11 工具注册表

    def _mock_registry(self, last: str) -> str:
        # 与 step04 同一套 {"tool","arguments"} 协议，但工具清单来自注册表。
        if "执行结果" in last:
            fact = _extract_after(last, "执行结果")
            return "根据工具返回的结果：" + (_clip(last) if not fact.strip() else _clip(fact)) + " —— 执行器全程只查注册表，不认识任何具体工具。"
        if "天气" in last or "weather" in last:
            return jdumps({"tool": "get_weather", "arguments": {"city": _find_city(last)}})
        m = re.search(r"(\d+\s*[+\-*/]\s*\d+)", last, re.ASCII)
        if m:
            return jdumps({"tool": "calculator", "arguments": {"expression": m.group(1)}})
        if "几点" in last or "时间" in last:
            return jdumps({"tool": "get_time", "arguments": {}})
        return "这个问题不需要调用工具，直接回答即可 —— 工具清单由注册表动态生成，注册即生效、注销即消失。"

    # ------------------------------------------------------------- step17 ReAct 运行时

    def _mock_runtime(self, all_text: str, last: str) -> str:
        # echo 的观察结果继续发起 echo —— 模拟失控场景，直到保险丝起跳
        if last.startswith("Observation: echo:"):
            return jdumps({
                "thought": "任务还没完成，继续调用 echo（模拟失控）",
                "action": "echo",
                "action_input": "tick"})
        if "死循环" in last or "循环测试" in last:
            return jdumps({
                "thought": "这个任务永远做不完（模拟失控场景）",
                "action": "echo",
                "action_input": "tick"})
        if "Observation:" in all_text:
            obs = _extract_after(all_text, "Observation:")
            return jdumps({
                "thought": "已拿到观察结果，信息足够，结束本回合",
                "final": "根据工具查询结果：" + _clip(obs) + "（turn 以 TurnEndReason.Completed 收场）"})
        m = re.search(r"(\d+\s*[+\-*/]\s*\d+)", last, re.ASCII)
        if m:
            return jdumps({
                "thought": "有算式，调用计算器，不能心算",
                "action": "calculator",
                "action_input": m.group(1)})
        if "天气" in last or "weather" in last:
            return jdumps({
                "thought": "需要实时数据，先调天气工具",
                "action": "get_weather",
                "action_input": _find_city(last)})
        return jdumps({
            "thought": "普通对话无需工具，直接作答",
            "final": "（Mock 模型）本轮 " + str(_count_history(all_text)) + " 条历史的上下文已按预算裁剪后发给模型。多聊几轮，可在轨迹里看到「上下文裁剪」事件。"})

    # ------------------------------------------------------------- step19 人工介入

    def _mock_ask(self, all_text: str, last: str) -> str:
        if "人工答复：" in all_text:
            ans = next((l for l in _extract_after(all_text, "人工答复：").split("\n") if l.strip()), "上海")
            if "上海" in ans:
                from_city = "上海"
            elif "杭州" in ans:
                from_city = "杭州"
            elif "广州" in ans:
                from_city = "广州"
            else:
                from_city = _clip(ans)
            return jdumps({
                "final": "已为你预订：" + from_city + " → 北京，明日 08:30 起飞，经济舱 ¥1,280（模拟出票）。"
                         + "刚才缺出发地时我没有瞎猜，而是问了人 —— 这就是 ask_user_question 的人机协同闭环。"})
        if "机票" in last or "订" in last or "出行" in last:
            return jdumps({
                "ask": "预订机票需要确认出发地 —— 请问从哪个城市出发？（可选：北京 / 上海 / 杭州 / 广州）"})
        return jdumps({
            "final": "（Mock 模型）信息完整的问题我直接回答。想看人工介入：发「帮我订一张去北京的机票」，我会停下来向你确认出发地。"})

    # ------------------------------------------------------------- step20 审批门禁

    def _mock_approval(self, last: str) -> str:
        if "执行结果" in last or "审批结果" in last:
            return jdumps({
                "final": "任务完成：" + _clip(last) + " —— 注意轨迹里的权限矩阵：安全工具自动放行，"
                         + "高危工具必须过审批门禁，这是生产级智能体与玩具的分水岭。"})
        if "邮件" in last or "发送" in last or "发一封" in last:
            return jdumps({"tool": "send_email", "arguments": "to=team@xiaofuge.cn, content=本周周报"})
        if "删除" in last or "清理" in last:
            return jdumps({"tool": "delete_file", "arguments": "path=workspace/tmp/old.log"})
        if "天气" in last:
            return jdumps({"tool": "get_weather", "arguments": "city=" + _find_city(last)})
        return jdumps({
            "final": "（Mock 模型）想看审批门禁：发「给团队发一封周报邮件」（高危，会请求审批），"
                     + "回复「批准 / 拒绝 / 批准并记住」观察三种走向。"})

    # ------------------------------------------------------------- step21 沙箱

    def _mock_sandbox(self, last: str) -> str:
        if "执行结果" in last or "已拦截" in last or "沙箱内" in last:
            return jdumps({"final": _clip(last)})
        return jdumps({
            "final": "（Mock 模型）沙箱采用纵深防御：策略 → 黑名单 → 边界 → 规范化四层拦截。"
                     + "发「执行：ls sandbox」看放行，或「执行：rm -rf /」看第 2 层拦截。"})

    # ------------------------------------------------------------- step22 事件溯源

    def _mock_event(self, all_text: str, last: str) -> str:
        remembered = None
        for m in _NAME_RE.finditer(all_text):
            n = m.group(1)
            if n.startswith("什么") or n.startswith("名字") or n.startswith("谁"):
                continue
            remembered = n
        if "我叫" in last:
            return jdumps({
                "final": "你好，" + remembered + "！这次对话连同你的名字已作为事件写入 JSONL 日志 ——"
                         + "我的记忆不是内存变量，而是从日志投影出来的视图，服务重启也不丢。"})
        if "叫什么" in last or "记得" in last:
            return (jdumps({"final": "你叫「" + remembered + "」。这条记忆是我刚才从事件日志回放重建的 —— 状态即日志。"})
                    if remembered is not None
                    else jdumps({"final": "日志里还没有你的名字，先说「我叫XX」吧。"}))
        return jdumps({
            "final": "（Mock 模型）本轮会话由事件日志回放重建。多聊几轮再看 events/step22-events.jsonl，"
                     + "每一条对话都是不可变事件；「重置」删掉日志，记忆归零。"})

    # ------------------------------------------------------------- step23 插件机制

    def _mock_plugin(self, all_text: str, last: str) -> str:
        # 插件机制剧本：安装后正常决策调用插件工具；未安装/已卸载时由注册表未命中路径兜底。
        if "执行结果" in last:
            fact = _extract_after(last, "执行结果")
            return ("根据插件工具返回的结果：" + (_clip(last) if not fact.strip() else _clip(fact))
                    + " —— 这个工具不在宿主代码里，而是插件经独立 ClassLoader 装进注册表的。")
        if "掷" in last or "硬币" in last or "coin" in last:
            return jdumps({"tool": "coin_flip", "arguments": {}})
        if "运行" in last or "多久" in last or "uptime" in last:
            return jdumps({"tool": "uptime", "arguments": {}})
        return "（Mock 模型）先「安装插件」，工具才会出现在我的清单里；「卸载插件」后我再也看不见它们 —— 能力即插即拔。"

    # ------------------------------------------------------------- step25 CLI 智能体

    def _mock_cli(self, all_text: str, last: str, system: str) -> str:
        # CLI 剧本：命令类输入返回 run_command 决策；其余兜底引导。
        if "执行结果" in last:
            fact = _extract_after(last, "执行结果")
            return ("命令结果：" + (_clip(last) if not fact.strip() else _clip(fact))
                    + " —— run_command 在受限工作区执行，Step21 的四层防御全程护航。")
        if "列出" in last or "文件" in last or "目录" in last or "ls" in last:
            return jdumps({"tool": "run_command", "arguments": {"command": "ls"}})
        if "几点" in last or "时间" in last or "日期" in last:
            return jdumps({"tool": "run_command", "arguments": {"command": "date"}})
        if "hello" in last:
            return jdumps({"tool": "run_command", "arguments": {"command": "cat hello.txt"}})
        return "我是终端里的 CLI 智能体。试试「运行：ls」，或让我「列出工作区文件」—— 命令都在 sandbox/step25 里执行，危险命令会被四层防御拦截。"

    # ------------------------------------------------------------- step13 子代理

    def _mock_subagent(self, all_text: str, last: str, system: str) -> str:
        # 子代理剧本：system 带【子代理身份】为子代理回合（fork 能看到父会话里的紫色）；否则为主代理回合。
        if "子代理身份" in system:
            fork_knows = "继承的父会话上下文" in system and "紫色" in system
            if "写作子代理" in system:
                return ("愿你的每一天都被喜爱的事物环绕 —— 像那一抹恰到好处的紫色。（fork：我从父会话记得你最喜欢紫色）"
                        if fork_knows
                        else "愿你所行皆坦途，所遇皆温暖，往后的每一天都值得期待。（spawn：我只知道任务本身）")
            return "Agent 调研要点：1) 工具标准化（MCP）成为默认范式；2) 运行时基座（审批/沙箱/事件）走向生产化；3) 多代理协作从固定流水线走向动态派遣。"
        if "执行结果" in last:
            fact = _extract_after(last, "执行结果")
            return "子代理已完成任务，结果如下：\n" + (_clip(last) if not fact.strip() else fact)
        if "fork" in last or "继承" in last:
            return jdumps({"tool": "dispatch_subagent", "name": "writer", "mode": "fork", "task": "写一句贺词"})
        if "spawn" in last or "全新" in last:
            return jdumps({"tool": "dispatch_subagent", "name": "writer", "mode": "spawn", "task": "写一句贺词"})
        if "调研" in last or "研究" in last or "趋势" in last:
            return jdumps({"tool": "dispatch_subagent", "name": "researcher", "mode": "spawn", "task": "调研 Agent 趋势"})
        if "记住" in last or "最喜欢" in last:
            return "好的，已记下。这条信息已进入父会话上下文 —— 之后 fork 出去的子代理都能看到它。"
        if "对比" in last:
            return "两次贺词的差异就是 spawn 与 fork 的差异：fork 的子代理继承了「最喜欢紫色」，贺词带了紫色；spawn 的子代理是全新实例，只会写通用贺词。"
        return "（Mock 模型）试试：「fork 一个子代理，让它写一句贺词」或「spawn 一个子代理写贺词」，体会两种派遣的上下文差异。"

    # ------------------------------------------------------------- step24 钩子体系

    def _mock_hooks(self, all_text: str, last: str, system: str) -> str:
        # 钩子讲解员：工具调用走宿主直连路由（查询：/发送：/钩子：），模型只负责讲解引导。
        if "执行结果" in last:
            fact = _extract_after(last, "执行结果")
            return "钩子管线说明：" + (_clip(last) if not fact.strip() else _clip(fact))
        return jdumps({"final":
                       "本场景的钩子挂在工具调用的 before / after 生命周期上。试试：\n"
                       + "·「查询：张三」—— after 脱敏钩子把手机号/邮箱打码\n"
                       + "·「发送：给 13812345678 发 密码是123456」—— before 风控钩子直接拦截\n"
                       + "·「钩子：关」再「查询：张三」—— 摘掉钩子看明文，对比出钩子的价值\n"
                       + "·「审计日志」—— before 审计钩子记下的每一次调用"})

    # ------------------------------------------------------------- step14 A2A 协作

    def _mock_a2a(self, all_text: str, last: str, system: str) -> str:
        # A2A 剧本：先发现名片，再按任务类型路由到 translator / weather。
        if "执行结果" in last:
            fact = _extract_after(last, "执行结果")
            return "外部代理已通过 A2A 回执：" + (_clip(last) if not fact.strip() else fact) \
                + " —— 对方有自己的模型与工具，我们只认 task_id 与回执。"
        if "发现" in last or "名片" in last or "附近" in last:
            return jdumps({"tool": "a2a_discover"})
        if "翻译" in last:
            return jdumps({"tool": "a2a_send", "agent": "translator", "task": "把「你好，智能体」翻译成英文"})
        if "天气" in last or "户外" in last:
            return jdumps({"tool": "a2a_send", "agent": "weather", "task": "明天适合户外运动吗？"})
        return jdumps({
            "final": "我是主代理，可以经 A2A 协议与外部代理协作。试试：「发现一下附近的代理」，"
                     + "或直接「让翻译代理把……翻译成英文」「问问天气代理明天适合户外运动吗」。"})

    # ------------------------------------------------------------- step26 定时工具

    def _mock_schedule(self, all_text: str, last: str, system: str) -> str:
        # 定时讲解员：调度走宿主直连路由（定时：/循环：/任务列表/取消：），模型只负责讲解引导。
        if "执行结果" in last:
            fact = _extract_after(last, "执行结果")
            return "定时任务说明：" + (_clip(last) if not fact.strip() else _clip(fact))
        return jdumps({"final":
                       "本场景给工具装上了时间维度。试试：\n"
                       + "·「定时：5秒 提醒我喝水」—— 一次性调度，拿 job_id\n"
                       + "·「任务列表」—— 看 pending / 已触发次数\n"
                       + "·「循环：每10秒 报一次时」—— 周期任务\n"
                       + "·「取消：job-1」「完成了什么」—— 管理与回看（触发结果落盘 events/step26-tasks.jsonl）"})
