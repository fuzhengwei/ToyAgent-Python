"""Step14 · A2A 协作 —— 代理之间的「标准化网络协议」。

Step12/23 的多代理都活在同一个进程里，主代理直接 new 出子代理。
A2A（Agent-to-Agent）解决的是跨进程、跨主人的协作：
- 名片发现（Agent Card）：每个代理对外公布一份 JSON 名片 —— 我是谁、住在哪个 URL、会什么技能；
- 标准化请求：协作不靠猜接口，靠统一信封 —— 任务描述 + task_id + 状态机
  （submitted → working → completed/failed）；
- 异步回执：发起方拿到 task_id 即可离开，结果异步取回。

与 Step13 子代理的区别：子代理是自家分身（进程内、共享宿主模型）；
A2A 是外部同行（网络调用、对方有自己的模型与工具），
类比：请同事帮忙 vs 外包给合作公司。
"""
import json
import uuid
from collections import namedtuple
from collections import OrderedDict

from ..agent import Agent, jstr, try_parse_json, tr
from ..llm import Message

# Agent Card（名片）：A2A 协议里的发现单元 ——
# 生产中托管在 https://host/.well-known/agent.json，这里教学化简为 namedtuple。
AgentCard = namedtuple("AgentCard", ["name", "url", "version", "skills", "description"])

SYSTEM_PROMPT = """你是 ToyAgent 主代理，可通过 A2A 协议与外部代理协作。协作前先发现名片。
查看目录时只输出 JSON：{"tool": "a2a_discover"}
委派任务时只输出 JSON：{"tool": "a2a_send", "agent": "translator|weather", "task": "任务描述"}
翻译类任务派给 translator，天气类任务派给 weather。拿到结果后用自然语言回答。"""


class A2AAgent(Agent):

    def __init__(self, model):
        self._model = model
        # 本地目录：模拟网络上已注册的两个外部代理（真实实现是 HTTP 拉取名片）。
        self._DIRECTORY = OrderedDict([
            ("translator", AgentCard("translator", "http://a2a.local/translator", "1.0.0",
                                     ["translate"], "中英互译代理")),
            ("weather", AgentCard("weather", "http://a2a.local/weather", "2.1.0",
                                  ["forecast"], "天气预报代理")),
        ])
        # 已发起的协作任务：id → 描述（生产中由协作控制器持久化并轮询状态）。
        self._tasks = OrderedDict()
        self._seq = 0
        self._last_trace = []

    def name(self) -> str:
        return "Step14 · A2A 协作（Agent Card 名片发现 · task_id 信封 · 异步回执）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()
        messages = [Message.system(SYSTEM_PROMPT), Message.user(input)]

        for _round in range(1, 4):
            raw = self._model.chat(messages).strip()
            decision = try_parse_json(raw)

            if isinstance(decision, dict) and jstr(decision, "tool") is not None:
                tool = jstr(decision, "tool")
                if tool == "a2a_discover":
                    result = self._discover()
                else:
                    result = self._send(jstr(decision, "agent"), jstr(decision, "task"))
                messages.append(Message.assistant(raw))
                messages.append(Message.user("工具执行结果: " + result))
                continue

            self._last_trace.append(tr("final", "最终回答", "汇总协作结果，向用户作答"))
            return raw
        return "已达最大轮数，循环终止。"

    def _discover(self) -> str:
        """a2a_discover：拉取目录里所有代理的名片（生产中 GET /.well-known/agent.json）。"""
        self._last_trace.append(tr(
            "action", "a2a_discover · 名片发现",
            "拉取 " + str(len(self._DIRECTORY)) + " 份 Agent Card（模拟网络请求）"))
        sb = "["
        first = True
        for card in self._DIRECTORY.values():
            if not first:
                sb += ", "
            first = False
            skills = "[" + ", ".join(card.skills) + "]"
            sb += ("\n  {\"name\": \"" + card.name + "\", \"url\": \"" + card.url
                   + "\", \"version\": \"" + card.version
                   + "\", \"skills\": " + skills + "}")
        cards = sb + "\n]"
        self._last_trace.append(tr(
            "observation", "收到名片", "translator（中英互译） · weather（天气预报）"))
        return cards

    def _send(self, agent: str, task: str) -> str:
        """a2a_send：按名片路由委派请求，模拟 submitted → completed 状态机并回执。"""
        card = self._DIRECTORY.get(agent)
        if card is None:
            self._last_trace.append(tr(
                "guard", "a2a_send · 路由失败",
                "目录中没有 " + agent + "，请先 a2a_discover"))
            return '{"error": "unknown agent: ' + agent + '"}'
        self._seq += 1
        task_id = "task-{:03d}-{}".format(self._seq, uuid.uuid4().hex[:4])
        self._tasks[task_id] = agent + ": " + task

        self._last_trace.append(tr(
            "action", "a2a_send · 发起协作",
            "→ " + card.name + " @ " + card.url + "  task=" + task))
        self._last_trace.append(tr(
            "thought", "状态机 · submitted → working",
            "对方已受理 task_id=" + task_id + "（信封标准化：发起方只认 task_id，不关心对方内部实现）"))

        reply = self._handle_remotely(card, task)
        self._last_trace.append(tr(
            "observation", "回执 · completed",
            "task_id=" + task_id + " 已完成：" + reply))
        return json.dumps({"task_id": task_id, "status": "completed", "result": reply},
                          ensure_ascii=False)

    @staticmethod
    def _handle_remotely(card: AgentCard, task: str) -> str:
        """模拟远端代理按 skill 处理（真实实现：POST JSON-RPC / HTTP task 请求）。"""
        if card.name == "translator":
            return "翻译结果（模拟远端模型）：Hello, Agent."
        if card.name == "weather":
            return "明日晴，22-28℃，东南风 3 级，适合户外运动（模拟远端数据）"
        return "（模拟远端响应）"

    def reset(self) -> None:
        self._tasks.clear()
        self._seq = 0

    def trace(self) -> list:
        return list(self._last_trace)
