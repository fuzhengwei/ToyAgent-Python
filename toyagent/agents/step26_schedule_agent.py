"""Step26 · 定时工具 —— 给智能体装上「时间维度」的行动能力。

之前的工具都是即时的：调用 → 执行 → 返回。定时工具把「什么时候做」
也变成模型的决策对象：
- 一次性调度：n 秒后执行（threading.Timer，对应 Java schedule + delay）；
- 循环调度：每 n 秒执行一次（对应 scheduleAtFixedRate）；
- 任务管理：列出 / 取消，每个任务有 job_id。

关键工程点：HTTP 请求已经返回，任务却在未来执行 —— 所以触发结果不能塞进
本次响应，要落盘（events/step26-tasks.jsonl，append-only 事件溯源，呼应
Step22）并在后续对话中汇报。这也是生产提醒/巡检类 Agent 的最小雏形。
"""
import json
import re
import threading
from datetime import datetime
from itertools import count
from pathlib import Path

from ..agent import Agent, or_raw, try_parse_json, tr
from ..llm import Message

SYSTEM_PROMPT = """你是 ToyAgent 的定时工具讲解员。用户可能询问定时任务、调度、提醒、巡检等话题。
只输出 JSON：{"final": "回答"}。请引导用户试用：
「定时：5秒 提醒我喝水」「循环：每10秒 报一次时」「任务列表」「取消：job-1」「完成了什么」。"""

_ONCE_PTN = re.compile(r"定时：(\d+)\s*秒\s*(.*)", re.ASCII)
_LOOP_PTN = re.compile(r"循环：每(\d+)\s*秒\s*(.*)", re.ASCII)


class _OnceHandle:
    """一次性任务句柄（等价 Java ScheduledFuture 的 isDone/cancel 语义）。"""

    def __init__(self, timer: threading.Timer):
        self._timer = timer
        self.cancelled = False

    def cancel(self):
        self.cancelled = True
        self._timer.cancel()

    def is_done(self) -> bool:
        return self.cancelled or self._timer.finished.is_set()


class _RecurringHandle:
    """循环任务句柄：Event 控制停止，触发次数持续累计。"""

    def __init__(self, interval: float, fire_fn):
        self.cancelled = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(interval, fire_fn), daemon=True)
        self._thread.start()

    def _run(self, interval, fire_fn):
        # 首次延迟 = 周期（与 scheduleAtFixedRate 语义一致）
        while not self._stop.wait(interval):
            fire_fn()

    def cancel(self):
        self.cancelled = True
        self._stop.set()

    def is_done(self) -> bool:
        return self.cancelled


class _Job:
    def __init__(self, job_id, jtype, content, period, handle):
        self.id = job_id
        self.type = jtype
        self.content = content
        self.period = period
        self.handle = handle
        self.fired = 0
        self._lock = threading.Lock()

    def inc_fired(self):
        with self._lock:
            self.fired += 1

    def get_fired(self) -> int:
        with self._lock:
            return self.fired


class ScheduleAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._last_trace = []
        # 调度线程均为守护线程 —— 宿主退出即消亡，教学足够（生产用持久化调度器/延时队列）。
        self._jobs = {}  # job_id -> _Job（保持插入序）
        self._fired_log = []
        self._fired_lock = threading.Lock()
        self._event_file = Path("events") / "step26-tasks.jsonl"
        self._seq = count(1)

    def name(self) -> str:
        return "Step26 · 定时工具（一次性/循环调度 · 触发结果事件落盘）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()

        # —— 定时指令（宿主直连路由，不经过模型） ——
        once = _ONCE_PTN.fullmatch(input)
        if once:
            return self._schedule_once(int(once.group(1)), once.group(2).strip())
        loop = _LOOP_PTN.fullmatch(input)
        if loop:
            return self._schedule_recurring(int(loop.group(1)), loop.group(2).strip())
        if "任务列表" in input:
            return self._list_jobs()
        if input.startswith("取消："):
            return self._cancel(input[3:].strip())
        if "完成了什么" in input or "已触发" in input:
            return self._report_fired()

        # —— 普通对话 → 模型讲解 ——
        raw = self._model.chat([Message.system(SYSTEM_PROMPT), Message.user(input)]).strip()
        parsed = try_parse_json(raw)
        answer = or_raw(parsed.get("final") if isinstance(parsed, dict) and parsed.get("final") is not None
                        else None, raw)
        self._last_trace.append(tr(
            "final", "讲解回答",
            "试试「定时：5秒 提醒我喝水」「循环：每10秒 报一次时」「任务列表」「完成了什么」"))
        return answer

    def _schedule_once(self, delay_seconds: int, content: str) -> str:
        """一次性调度：schedule + delay，到点执行 → 落盘事件。"""
        job_id = "job-" + str(next(self._seq))
        timer = threading.Timer(delay_seconds, lambda: self._fire(job_id, content))
        timer.daemon = True
        timer.start()
        handle = _OnceHandle(timer)
        self._jobs[job_id] = _Job(job_id, "一次性", content, delay_seconds, handle)
        self._last_trace.append(tr(
            "action", "schedule_task · 一次性注册",
            job_id + "：" + str(delay_seconds) + " 秒后执行「" + content + "」"))
        self._last_trace.append(tr(
            "thought", "异步语义",
            "本次响应立即返回；任务在未来由调度线程执行，结果落盘 events/step26-tasks.jsonl"))
        return ("⏰ 已注册一次性任务 " + job_id + "：" + str(delay_seconds) + " 秒后「" + content + "」\n"
                + "本次请求已返回，任务到点后在后台执行并记录 —— 稍后问「完成了什么」看结果。")

    def _schedule_recurring(self, interval_seconds: int, content: str) -> str:
        """循环调度：scheduleAtFixedRate（首延迟 = 周期），触发次数持续累计。"""
        job_id = "job-" + str(next(self._seq))
        handle = _RecurringHandle(interval_seconds, lambda: self._fire(job_id, content))
        self._jobs[job_id] = _Job(job_id, "循环", content, interval_seconds, handle)
        self._last_trace.append(tr(
            "action", "schedule_recurring · 循环注册",
            job_id + "：每 " + str(interval_seconds) + " 秒执行「" + content + "」（首次延迟 = 周期）"))
        return ("🔁 已注册循环任务 " + job_id + "：每 " + str(interval_seconds) + " 秒「" + content + "」\n"
                + "问「任务列表」看触发次数，问「完成了什么」看已触发内容，不需要时「取消：" + job_id + "」。")

    def _fire(self, job_id: str, content: str):
        """调度线程回调：记录 + 落盘（append-only JSONL，事件溯源）。"""
        time_str = datetime.now().strftime("%H:%M:%S")
        with self._fired_lock:
            self._fired_log.append(time_str + "  [" + job_id + "] " + content)
        job = self._jobs.get(job_id)
        if job is not None:
            job.inc_fired()
        try:
            self._event_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._event_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {"time": time_str, "job": job_id, "content": content},
                    ensure_ascii=False, separators=(",", ":")) + "\n")
        except OSError:
            pass

    def _list_jobs(self) -> str:
        sb = "📋 定时任务：\n"
        for j in self._jobs.values():
            if j.handle.is_done():
                status = "done/取消"
            elif j.type == "循环":
                status = "循环中 · 已触发 " + str(j.get_fired()) + " 次"
            else:
                status = "pending"
            sb += (f"  {j.id} · {j.type} · 每/{j.period}秒 · 「{j.content}」 · {status}\n")
        self._last_trace.append(tr(
            "observation", "任务列表", str(len(self._jobs)) + " 个任务在册"))
        return sb.rstrip("\n")

    def _cancel(self, job_id: str) -> str:
        j = self._jobs.get(job_id)
        if j is None:
            return "没有任务 " + job_id + "。在册：[" + ", ".join(self._jobs.keys()) + "]"
        j.handle.cancel()
        self._last_trace.append(tr(
            "guard", "cancel_task · 取消",
            job_id + " 已停止（在册保留，状态 done/取消）"))
        return "🚫 已取消 " + job_id + "（「" + j.content + "」不再触发）。"

    def _report_fired(self) -> str:
        with self._fired_lock:
            log = list(self._fired_log)
        self._last_trace.append(tr(
            "observation", "已触发任务汇报",
            ("暂无触发记录" if not log else str(len(log)) + " 条，事件已 append 到 " + str(self._event_file))))
        return ("✅ 已触发的任务（事件日志 " + str(self._event_file) + "，append-only）：\n"
                + ("（暂无 —— 注册一个「定时：5秒 提醒我喝水」再来看看）" if not log
                   else "\n".join(log)))

    def reset(self) -> None:
        for j in self._jobs.values():
            j.handle.cancel()
        self._jobs.clear()
        with self._fired_lock:
            self._fired_log.clear()

    def trace(self) -> list:
        return list(self._last_trace)
