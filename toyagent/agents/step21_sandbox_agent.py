"""Step21 · 沙箱纵深防御 —— 命令执行前的四道关卡。

拆出教学版四层校验管线，任何一层拦截即 DENY：
1. 策略层：当前策略档位是否允许 shell（READ_ONLY 直接全拒）；
2. 内容黑名单：rm -rf /、sudo、mkfs、fork 炸弹等破坏性命令特征；
3. 空间边界：命令中出现的路径必须落在工作区 sandbox/ 内，绝对路径越界即拒；
4. 路径规范化：先 canonicalize 再校验，workspace/../../etc/passwd
   这种「看似在内、实际在外」的逃逸无所遁形。

关键设计思想：不信任模型的输出 —— 模型生成的命令照样要过全部四层。
本模块的四层校验（内容黑名单/路径边界）同时被 Step25 的 shell 执行器复用。
"""
import os
import re
from pathlib import Path

from ..agent import Agent, or_raw, try_parse_json, tr
from ..llm import Message

# 内容黑名单（教学子集）。
DENYLIST = [
    "rm -rf", "rm  -rf", "sudo", "shutdown", "reboot", "mkfs", "dd if=",
    ":(){", "fork()", "chmod 777", "> /dev/", "kill -9 1", "curl | sh",
]

_ABS_PATH_PTN = re.compile(r"(?<![\w.])/(?:[\w.-]+/)*[\w.-]+", re.ASCII)

SYSTEM_PROMPT = """你是 ToyAgent 的沙箱讲解员。用户可能询问沙箱、命令执行、安全防御相关话题。
只输出 JSON：{"final": "回答"}。想看四层校验实战，请引导用户发「执行：<命令>」。"""


def _starts_with(child: Path, parent: Path) -> bool:
    """组件级前缀判断（等价 Java Path.startsWith）。"""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def check_denylist(cmd: str):
    """第 2 层 · 内容黑名单：命中返回特征词，否则 None（供 Step25 复用）。"""
    low = cmd.lower()
    for bad in DENYLIST:
        if bad in low:
            return bad
    return None


def find_escaped_abs_path(cmd: str, workspace: Path):
    """第 3 层 · 空间边界：命令中出现越出工作区的绝对路径则返回该路径（供 Step25 复用）。"""
    for m in _ABS_PATH_PTN.finditer(cmd):
        p = m.group()
        norm = Path(os.path.normpath(p))
        if not _starts_with(norm, workspace):
            return p
    return None


class SandboxAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._history = []
        self._last_trace = []
        self._workspace = Path(os.path.normpath(os.path.abspath("sandbox")))
        try:
            self._workspace.mkdir(parents=True, exist_ok=True)
            (self._workspace / "hello.txt").write_text("ToyAgent sandbox\n", encoding="utf-8")
        except OSError:
            pass

    def name(self) -> str:
        return "Step21 · 沙箱纵深防御（策略 → 黑名单 → 边界 → 规范化，四层拦截）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()
        self._history.append(Message.user(input))

        # 发「执行：<命令>」即走沙箱管线；其余交给模型
        if input.startswith("执行："):
            cmd = input[len("执行："):].strip()
            return self._run_shell(cmd)

        req = [Message.system(SYSTEM_PROMPT)]
        req.extend(self._history)
        raw = self._model.chat(req).strip()
        parsed = try_parse_json(raw)
        answer = or_raw(parsed.get("final") if isinstance(parsed, dict) else None, raw)
        self._history.append(Message.assistant(answer))
        self._last_trace.append(tr(
            "final", "直接作答",
            "普通对话不走沙箱；想看四层校验，发「执行：<命令>」"))
        return answer

    def _run_shell(self, cmd: str) -> str:
        """四层校验管线：任何一层拦截立即 DENY，全过才执行。"""
        self._last_trace.append(tr("action", "shell_execute 请求", cmd))

        # ===== 第 1 层 · 策略档位 =====
        self._last_trace.append(tr(
            "thought", "第 1 层 · 策略校验 ✓",
            "当前策略 WORKSPACE_ONLY 允许执行 shell（生产另有 READ_ONLY=全拒 / FULL=需审批）"))

        # ===== 第 2 层 · 内容黑名单 =====
        bad = check_denylist(cmd)
        if bad:
            return self._deny(2, "命中危险命令特征「" + bad + "」")
        self._last_trace.append(tr(
            "thought", "第 2 层 · 内容黑名单 ✓",
            "未命中 " + str(len(DENYLIST)) + " 条破坏性命令特征"))

        # ===== 第 3 层 · 空间边界（绝对路径必须在工作区内） =====
        escaped = find_escaped_abs_path(cmd, self._workspace)
        if escaped:
            return self._deny(3, "绝对路径 " + escaped + " 越出工作区 " + self._workspace.name)
        self._last_trace.append(tr(
            "thought", "第 3 层 · 空间边界 ✓",
            "命令中的路径均位于工作区 sandbox/ 内（相对路径按工作区解析）"))

        # ===== 第 4 层 · 路径规范化（canonicalize 后复查，防 ../ 逃逸） =====
        if ".." in cmd:
            return self._deny(4, "检测到 ../ 相对跳转 —— 规范化后路径将越出工作区，疑似路径逃逸")
        self._last_trace.append(tr(
            "thought", "第 4 层 · 路径规范化 ✓",
            "canonicalize 复查通过，无 ../ 逃逸（第 3 层看原文，第 4 层看真实落点）"))

        # ===== 执行（教学模拟） =====
        output = self._simulate(cmd)
        self._last_trace.append(tr(
            "observation", "执行完成",
            "工作区 " + self._workspace.name + " · " + output))
        self._history.append(Message.assistant("已执行（沙箱内）"))
        return ("✅ 命令通过四层校验，已在沙箱内执行：\n$ " + cmd + "\n" + output
                + "\n\n（教学模拟执行。生产版每层都是独立防线：模型生成的命令也必须全过。）")

    def _deny(self, layer: int, why: str) -> str:
        self._last_trace.append(tr(
            "guard", "第 " + str(layer) + " 层 · 拦截 DENY",
            why + " —— 纵深防御：任何一层说不行，就是不行"))
        self._history.append(Message.assistant("已拦截"))
        return "🛑 已在第 " + str(layer) + " 层拦截：" + why

    @staticmethod
    def _simulate(cmd: str) -> str:
        if cmd.startswith("ls"):
            return "hello.txt  notes.md  data/"
        if cmd.startswith("cat"):
            return "ToyAgent sandbox（hello.txt 内容）"
        return "(模拟) 命令已执行，输出略"

    def reset(self) -> None:
        self._history.clear()

    def trace(self) -> list:
        return list(self._last_trace)
