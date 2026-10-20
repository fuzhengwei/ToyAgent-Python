"""Step22 · 事件溯源 —— 会话即日志，日志即状态。

三件套：
1. 事件追加：会话中发生的每件事（用户发言/Agent 回答/回合完成）
   都是一条不可变事件，append 到 JSONL 文件，永不改写历史；
2. 回放投影：每次对话前从日志重放全部事件，重建出会话上下文 ——
   Agent 的「记忆」不是内存里的列表，而是从日志投影出来的视图；
   服务重启后状态从日志完整恢复；
3. 审计能力：因为历史=事件流，天然获得审计、时间旅行、fork 会话等
   衍生能力（子代理 fork 就是复制事件前缀）。

事件日志在 events/step22-events.jsonl，可直接打开查看每一条。
"""
import json
import os
import time
from pathlib import Path

from ..agent import Agent, or_raw, try_parse_json, tr
from ..llm import Message

# 事件类型（教学子集）。
EV_SESSION = "SESSION_STARTED"
EV_USER = "USER_MESSAGE_APPENDED"
EV_AGENT = "AGENT_REPLY_APPENDED"
EV_TURN = "TURN_COMPLETED"

SYSTEM_PROMPT = """你是一个跑在事件溯源架构上的智能体：你的每次对话都会作为事件写入 JSONL 日志，
每轮开始前从日志回放重建记忆。像正常助手一样回答即可（记住用户告诉你的信息）。
只输出 JSON：{"final": "回答"}。"""


class EventSourcedAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._log_file = Path("events") / "step22-events.jsonl"
        self._last_trace = []

    def name(self) -> str:
        return "Step22 · 事件溯源（JSONL 事件日志 + 回放投影重建会话）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()
        try:
            # ===== 1. 回放：从日志重建会话状态 =====
            log = self._replay()
            self._last_trace.append(tr(
                "thought", "事件回放投影",
                "从 " + str(self._log_file) + " 重放 " + str(len(log)) + " 条事件，重建会话上下文"
                + ("（空日志 → 新会话）" if not log else "")))

            # ===== 2. 追加本轮事件 =====
            if not log:
                self._append(EV_SESSION, {"scenario": "step22"})
            self._append(EV_USER, {"content": input})

            # ===== 3. 上下文 = 投影结果（不是内存列表！）=====
            context = self._project(self._replay())
            req = [Message.system(
                SYSTEM_PROMPT + "（当前事件日志共 " + str(len(log)) + " 条）")]
            req.extend(context)

            raw = self._model.chat(req).strip()
            parsed = try_parse_json(raw)
            answer = or_raw(parsed.get("final") if isinstance(parsed, dict) else None, raw)

            self._append(EV_AGENT, {"content": answer})
            self._append(EV_TURN, {"events": len(log) + 3})
            self._last_trace.append(tr(
                "final", "本轮事件已落盘",
                "追加 4 条事件（含会话首条）：USER → AGENT_REPLY → TURN_COMPLETED。"
                "日志 append-only 永不改写，历史即事件流"))
            return answer
        except OSError as e:
            return "事件日志读写失败：" + str(e)

    def _replay(self) -> list:
        """回放：逐行读 JSONL（append-only，所以重放顺序 = 发生顺序）。"""
        self._ensure_file()
        log = []
        for line in self._log_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                log.append(json.loads(line))
            except Exception:
                # 损坏行跳过 —— append-only + 校验和是生产标配，教学从简
                pass
        return log

    def _project(self, log: list) -> list:
        """投影：事件流 → 会话上下文（只挑 USER / AGENT 两类，其余事件服务审计）。"""
        history = []
        for e in log:
            etype = str(e.get("type"))
            payload = e.get("payload") or {}
            content = str(payload.get("content"))
            if etype == EV_USER:
                history.append(Message.user(content))
            elif etype == EV_AGENT:
                history.append(Message.assistant(content))
            # SESSION_STARTED / TURN_COMPLETED 不进上下文，但留在日志里
        return history

    def _append(self, etype: str, payload: dict):
        """追加一条事件：seq 单调递增，一行一条 JSON。"""
        self._ensure_file()
        seq = len(self._log_file.read_text(encoding="utf-8").splitlines()) + 1
        event = {
            "seq": seq,
            "ts": int(time.time() * 1000),
            "type": etype,
            "payload": payload,
        }
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _ensure_file(self):
        if not self._log_file.exists():
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_file.touch()

    def reset(self) -> None:
        self._last_trace.clear()
        try:
            os.remove(self._log_file)
        except OSError:
            pass

    def trace(self) -> list:
        return list(self._last_trace)
