"""Step24 · Hooks 钩子体系 —— 挂在工具调用生命周期上的「插件化拦截点」。

工具执行前后并不是「裸奔」的：宿主在两个生命周期点暴露钩子位，
任何符合签名的钩子都能挂上去：
1. before：工具执行前 —— 可拦截（DENY）或改写参数；
2. after：工具执行后 —— 可改写输出（脱敏、过滤、追加告警）。

三个要素：Hook（钩子本体）+ 匹配规则（挂在哪类工具上，即 HookMatcher）+
输出合并（多个 after 钩子按注册顺序依次加工，即 HookOutputMerger）。

关键设计思想：横切逻辑与工具解耦 —— 审计、脱敏、风控都不写进工具代码，
挂钩即生效，摘钩即消失；工具作者对此无感知。
"""
import json
import re
from collections import namedtuple
from datetime import datetime

from ..agent import Agent, try_parse_json, tr
from ..llm import Message

# 钩子本体：name 标识、phase 挂载点（before/after）、match 匹配的工具名（* = 全部）。
# handler 返回 None = 不干预；返回 "DENY:原因" = 拦截；否则用返回值替换原内容。
Hook = namedtuple("Hook", ["name", "phase", "match", "handler"])

SYSTEM_PROMPT = """你是 ToyAgent 的钩子讲解员。用户可能询问钩子（Hooks）、拦截、脱敏、审计等话题。
只输出 JSON：{"final": "回答"}。请引导用户试用：「查询：张三」「发送：给 13812345678 发 密码是123456」「钩子：关」「审计日志」。"""

_PHONE_PTN = re.compile(r"(1[3-9])\d{9}", re.ASCII)
_EMAIL_PTN = re.compile(r"([\w.]+)@([\w.]+)", re.ASCII)


class HooksAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._last_trace = []
        # before 审计钩子的落点：生产中是审计库 / SIEM，教学中是内存列表。
        self._audit_log = []
        self._hooks = []
        self._enabled = True
        self._install_default_hooks()

    def _install_default_hooks(self):
        """出厂自带三条钩子：审计（before/全部）、风控（before/发短信）、脱敏（after/全部）。"""
        self._hooks.append(Hook("审计", "before", ["*"], self._audit_hook))
        self._hooks.append(Hook("风控-敏感外发", "before", ["send_sms"], self._risk_hook))
        self._hooks.append(Hook("脱敏", "after", ["*"], self._mask_hook))

    # ---- 三个钩子实现 ----

    def _audit_hook(self, tool, payload):
        self._audit_log.append(
            datetime.now().strftime("%H:%M:%S") + "  " + tool + "  args=" + payload)
        return None  # 审计只记录，不干预

    @staticmethod
    def _risk_hook(tool, payload):
        if "密码" in payload or "验证码" in payload:
            return "DENY:短信内容含「密码/验证码」等敏感信息，禁止外发"
        return None

    @staticmethod
    def _mask_hook(tool, payload):
        masked = _PHONE_PTN.sub(r"\1*********", payload)
        masked = _EMAIL_PTN.sub(
            lambda m: m.group(1)[0] + "***@" + m.group(2), masked)
        return None if masked == payload else masked

    def name(self) -> str:
        return "Step24 · Hooks 钩子体系（before 拦截/改写 · after 脱敏/审计，挂钩即生效）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()

        # —— 钩子管理指令（宿主能力，不经过模型） ——
        if input.startswith("钩子："):
            self._enabled = "开" in input
            self._last_trace.append(tr(
                "action", "钩子已全部挂载" if self._enabled else "钩子已全部摘除",
                "当前 " + str(len(self._hooks)) + " 条钩子（审计/风控/脱敏），enabled=" + str(self._enabled)))
            return ("✅ 钩子已挂回：" + self._hook_list() if self._enabled
                    else "⚠️ 钩子已全部摘除 —— 工具现在裸奔，注意对比接下来的输出。")
        if "审计日志" in input:
            self._last_trace.append(tr(
                "observation", "审计日志（before 钩子写入）",
                "暂无记录" if not self._audit_log else "\n".join(self._audit_log)))
            return ("📋 审计日志（before 钩子记录）：\n"
                    + ("（暂无）" if not self._audit_log else "\n".join(self._audit_log)))

        # —— 工具调用指令 → 走钩子管线 ——
        if input.startswith("查询："):
            return self._run_pipeline(
                "query_user",
                json.dumps({"name": input[3:].strip()}, ensure_ascii=False, separators=(",", ":")),
                '{"name": "张三", "phone": "13812345678", "email": "zhangsan@example.com", "level": "VIP"}')
        if input.startswith("发送："):
            payload = input[3:].strip()
            m = re.search(r"1[3-9]\d{9}", payload, re.ASCII)
            phone = m.group() if m else "13800000000"
            return self._run_pipeline(
                "send_sms",
                json.dumps({"phone": phone or "13800000000", "message": payload},
                           ensure_ascii=False, separators=(",", ":")),
                "短信已提交网关，回执 ok")

        # —— 普通对话 → 模型讲解 ——
        raw = self._model.chat([Message.system(SYSTEM_PROMPT), Message.user(input)]).strip()
        parsed = try_parse_json(raw)
        answer = (parsed.get("final") if isinstance(parsed, dict) and parsed.get("final") is not None
                  else raw)
        self._last_trace.append(tr(
            "final", "讲解回答",
            "试试「查询：张三」「发送：给 13812345678 发 密码是123456」「钩子：关」"))
        return answer

    def _run_pipeline(self, tool: str, args: str, tool_result: str) -> str:
        """钩子管线：before 链（审计/风控）→ 工具执行 → after 链（脱敏），任何 before 拦截即终止。"""
        self._last_trace.append(tr("action", "工具调用 " + tool, "参数 " + args))

        # ===== before 钩子链 =====
        payload = args
        for h in self._match_hooks("before", tool):
            out = h.handler(tool, payload)
            if out is not None and out.startswith("DENY:"):
                self._last_trace.append(tr(
                    "guard", "before · " + h.name + " → 拦截",
                    out[5:] + " —— 钩子在不改工具代码的前提下拦下调用"))
                return "🛑 工具 " + tool + " 被 before 钩子「" + h.name + "」拦截：" + out[5:]
            if out is not None:
                self._last_trace.append(tr(
                    "thought", "before · " + h.name + " → 改写参数",
                    payload + "  →  " + out))
                payload = out
            else:
                self._last_trace.append(tr(
                    "thought", "before · " + h.name + " → 放行",
                    "未干预（如审计钩子：只记录不改动）"))

        # ===== 工具执行（教学模拟） =====
        self._last_trace.append(tr("observation", "工具执行完成", tool_result))

        # ===== after 钩子链（输出合并：按注册顺序依次加工） =====
        output = tool_result
        for h in self._match_hooks("after", tool):
            out = h.handler(tool, output)
            if out is not None:
                self._last_trace.append(tr(
                    "thought", "after · " + h.name + " → 改写输出",
                    "输出已加工（" + h.name + "）"))
                output = out
        return ("🔗 钩子管线执行完毕：\n最终输出：" + output
                + ("\n（before 审计+风控 → 工具 → after 脱敏，按注册顺序依次生效）" if self._enabled else ""))

    def _match_hooks(self, phase: str, tool: str) -> list:
        """HookMatcher：按挂载点 + 工具名匹配；enabled=False 时整条链路直通。"""
        if not self._enabled:
            return []
        return [h for h in self._hooks
                if h.phase == phase and ("*" in h.match or tool in h.match)]

    def _hook_list(self) -> str:
        parts = [f"{h.phase}·{h.name}(match={h.match})" for h in self._hooks]
        return " ".join(parts)

    def reset(self) -> None:
        self._audit_log.clear()
        self._enabled = True

    def trace(self) -> list:
        return list(self._last_trace)
