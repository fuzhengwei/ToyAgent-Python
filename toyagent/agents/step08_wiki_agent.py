"""Step08 · LLM-Wiki 智能体 —— 知识编译与持久化。

RAG 解决「检索」：每次提问都从头翻文档。LLM-Wiki 解决「沉淀」：
项目知识（架构约定、代码规范、部署流程）被编译成结构化条目，
持久保存、随启动加载、增量维护 —— 知识编译一次、持续复用，而非每次从头检索。
思路源自 Andrej Karpathy 的 LLM-Wiki 提法（AGENTS.md / CLAUDE.md 即其工程形态）。

本场景：内置基础 Wiki（编译一次）；用户说「编译：主题：内容」即可新增条目，
持久化到 wiki/user-wiki.md，重启不丢；提问时按相关度取条目注入提示词作答。
教学版检索用字符 bigram 重合度模拟，换成 Embedding 即生产级实现。
"""
import os
import sys
from pathlib import Path

from ..agent import Agent, tr
from ..llm import Message
from .step07_rag_agent import score

# 基础 Wiki：项目知识，启动时编译一次（生产中来自仓库里的 AGENTS.md / CLAUDE.md）。
BASE_WIKI = {
    "架构约定": "ToyAgent 是零依赖教学项目：一个接口类就是一个智能体场景（step01-26），全部实现放在 Impl 内部类里，禁止引入第三方依赖；模型层统一走 ChatModel 接口。",
    "代码规范": "Java 17 语法；类与核心方法必须有中文 Javadoc 说明教学意图；trace 事件用 type/label/detail 三段式；JSON 一律走自研 Json 工具类。",
    "部署流程": "javac -d target/classes 编译后 java -cp target/classes cn.xiaofuge.ai.Application 启动，默认端口 8099，可用环境变量 TOY_AGENT_PORT 覆盖；前端静态资源在 web/ 目录由服务直接托管。",
    "业务规则": "配置只保存在各人浏览器 localStorage 随请求生效，服务端 config.properties 仅作默认兜底；未配置 API Key 时所有场景回落 Mock 演示模型。",
}

# 用户编译的增量条目（重启后从 wiki/user-wiki.md 恢复）。
USER_WIKI_FILE = Path("wiki") / "user-wiki.md"

SYSTEM_PROMPT = """你是项目 Wiki 问答智能体。回答规则：
1. 优先依据【项目 Wiki】的条目回答，并标注来源条目名，如（来源：架构约定）。
2. Wiki 条目不足以回答时，如实说"项目 Wiki 中暂无相关条目"，并提示可以用「编译：主题：内容」沉淀该知识。禁止编造。
"""


class WikiAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._user_wiki = {}
        self._last_trace = []
        self._load_user_wiki()

    def name(self) -> str:
        return "Step08 · LLM-Wiki 知识编译"

    def chat(self, input: str) -> str:
        self._last_trace.clear()

        # ---------------- 编译指令：把对话中的知识沉淀进 Wiki ----------------
        if input.startswith(("编译：", "编译:", "记住：", "记住:")):
            return self._compile(input)

        # ---------------- 加载 Wiki（启动编译一次 + 增量条目） ----------------
        wiki = self._current_wiki()
        self._last_trace.append(tr(
            "load", "加载项目 Wiki",
            "共 " + str(len(wiki)) + " 个条目（内置 " + str(len(BASE_WIKI))
            + " + 已编译 " + str(len(self._user_wiki)) + "），无需每次重新检索原始文档"))

        # ---------------- 相关条目选取（教学版 bigram 打分，生产换 Embedding） ----------------
        ranked = sorted(
            ((k, v, score(k + v, input)) for k, v in wiki.items()),
            key=lambda h: h[2], reverse=True)[:2]
        hits = [(k, v, s) for k, v, s in ranked if s > 0]
        self._last_trace.append(tr(
            "retrieve", "命中条目",
            "无相关条目" if not hits else
            " ｜ ".join(f"{k}（{s * 100:.0f} 分）" for k, v, s in hits)))

        # ---------------- 增强生成 ----------------
        messages = [Message.system(SYSTEM_PROMPT)]
        if hits:
            ctx = "【项目 Wiki】\n"
            for k, v, _s in hits:
                ctx += f"◆ {k}：{v}\n"
            messages.append(Message.user(ctx))
        messages.append(Message.user(input))

        answer = self._model.chat(messages)
        self._last_trace.append(tr(
            "final", "基于 Wiki 作答完成",
            "模型按约定提示可编译新条目" if not hits else "回答基于 " + str(len(hits)) + " 个 Wiki 条目"))
        return answer

    def _compile(self, input: str) -> str:
        """编译：解析「编译：主题：内容」，写入 Wiki 并持久化。"""
        body = input[3:].strip()  # 去掉前缀「编译：/记住：」（等宽 3 字符）
        sep = body.find("：")
        if sep < 0:
            sep = body.find(":")
        if not body.strip() or sep <= 0 or sep == len(body) - 1:
            self._last_trace.append(tr("compile", "编译失败", "格式：编译：主题：内容"))
            return "编译格式不对哦。请用：「编译：主题：内容」，例如「编译：发布流程：每周三灰度，周五全量上线」。"
        topic = body[:sep].strip()
        content = body[sep + 1:].strip()
        update = topic in self._user_wiki
        self._user_wiki[topic] = content
        self._persist_user_wiki()

        self._last_trace.append(tr(
            "compile", "更新条目" if update else "新增条目",
            "◆ " + topic + "（已持久化到 " + str(USER_WIKI_FILE) + "，重启不丢）"))
        self._last_trace.append(tr(
            "final", "知识编译完成",
            "Wiki 现有 " + str(len(self._current_wiki())) + " 个条目，后续提问直接复用"))
        return ("已编译进项目 Wiki：◆ " + topic + "\n\n" + content
                + "\n\n之后所有提问都会带上这条知识（已持久化，重启不丢）。试试问我相关内容。")

    def _current_wiki(self) -> dict:
        all_wiki = dict(BASE_WIKI)
        all_wiki.update(self._user_wiki)
        return all_wiki

    def _persist_user_wiki(self):
        """持久化用户编译条目（教学版写本地 markdown；生产中提交到代码仓库）。"""
        try:
            USER_WIKI_FILE.parent.mkdir(parents=True, exist_ok=True)
            sb = "# ToyAgent 用户 Wiki（由页面编译生成）\n\n"
            for k, v in self._user_wiki.items():
                sb += f"## {k}\n{v}\n\n"
            USER_WIKI_FILE.write_text(sb, encoding="utf-8")
        except OSError as e:
            print("[ToyAgent] 写入 user-wiki 失败: " + str(e), file=sys.stderr)

    def _load_user_wiki(self):
        if not USER_WIKI_FILE.exists():
            return
        try:
            lines = USER_WIKI_FILE.read_text(encoding="utf-8").splitlines()
            topic = None
            content = []
            for line in lines:
                if line.startswith("## "):
                    if topic is not None:
                        self._user_wiki[topic] = "\n".join(content).strip()
                    topic = line[3:].strip()
                    content = []
                elif topic is not None and not line.startswith("#"):
                    content.append(line)
            if topic is not None:
                self._user_wiki[topic] = "\n".join(content).strip()
            if self._user_wiki:
                print("[ToyAgent] 已从 " + str(USER_WIKI_FILE) + " 恢复 "
                      + str(len(self._user_wiki)) + " 个编译条目")
        except OSError as e:
            print("[ToyAgent] 读取 user-wiki 失败: " + str(e), file=sys.stderr)

    def reset(self) -> None:
        self._user_wiki.clear()
        try:
            os.remove(USER_WIKI_FILE)
        except OSError:
            # 删除失败不影响会话
            pass
        self._last_trace.clear()

    def trace(self) -> list:
        return list(self._last_trace)
