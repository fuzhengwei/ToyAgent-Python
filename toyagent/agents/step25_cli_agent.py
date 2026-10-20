"""Step25 · CLI 智能体 —— 把智能体装进终端。

形态变了，骨架不变：仍然是 Agent.chat(input)，只是外面套一层终端 REPL。
1. 工具侧换执行器：run_command 直接执行 shell 命令，复用 Step21 的沙箱四层防御
   （跨场景导入 step21 的内容黑名单与路径边界校验，教学版再加命令白名单与受限工作区）；
2. 工作目录即上下文：受限工作区 sandbox/step25/ 进入系统提示词，像终端会话一样有 cwd；
3. 两种交互模式：主 CLI 的交互式 REPL 与 run 单命令。
"""
import subprocess
from pathlib import Path

from ..agent import Agent, jstr, try_parse_json, tr
from ..llm import Message
from .step21_sandbox_agent import check_denylist, find_escaped_abs_path

WORKSPACE = Path("sandbox") / "step25"
# 第①层：拒绝名单（无论是否在沙箱内，一律拒绝）。
DENYLIST = {"rm", "sudo", "shutdown", "reboot", "mkfs", "dd", "kill", "pkill"}
# 第③层：命令白名单 —— 只放行只读/无害命令。
ALLOWLIST = {"ls", "pwd", "echo", "cat", "date", "wc"}

SYSTEM_PROMPT_PREFIX = """你是一个运行在终端里的 CLI 智能体。当前受限工作区：{workspace}
可以调用 run_command 执行命令（沙箱受四层防御保护）。
"""


class CliAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._last_trace = []
        self._init_workspace()

    def _init_workspace(self):
        try:
            WORKSPACE.mkdir(parents=True, exist_ok=True)
            hello = WORKSPACE / "hello.txt"
            if not hello.exists():
                hello.write_text("你好，这里是 ToyAgent 的受限工作区。\n", encoding="utf-8")
        except OSError:
            pass

    def name(self) -> str:
        return "Step25 · CLI 智能体（终端 REPL + 沙箱执行器）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()
        self._last_trace.append(tr(
            "tool", "终端会话就绪",
            "工作区 " + str(WORKSPACE) + " 已进入系统提示词（cwd 上下文）"))

        # ---- 本地指令：运行：<cmd>（不经模型，直接看沙箱防御） ----
        if input.startswith(("运行：", "运行:")):
            return self._execute(input[3:].strip())

        # ---- 常规对话：模型自主决策是否调用 run_command ----
        messages = [
            Message.system(SYSTEM_PROMPT_PREFIX.format(workspace=WORKSPACE)
                           + '\n决策规则：\n- 需要执行命令时，只输出 JSON：{"tool": "run_command", "arguments": {"command": "命令"}}\n'
                           + "- 不需要命令时，直接用自然语言回答，不要输出 JSON。"),
            Message.user(input),
        ]

        for _round in range(1, 3):
            raw = self._model.chat(messages).strip()
            decision = try_parse_json(raw)

            if isinstance(decision, dict) and jstr(decision, "tool") is not None:
                args_obj = decision.get("arguments")
                cmd = ""
                if isinstance(args_obj, dict) and args_obj.get("command") is not None:
                    cmd = str(args_obj.get("command"))
                self._last_trace.append(tr("action", "模型发起命令执行", cmd))
                result = self._run_guarded(cmd)
                messages.append(Message.assistant(raw))
                messages.append(Message.user("工具 run_command 执行结果: " + result))
                continue
            self._last_trace.append(tr(
                "final", "最终回答", "CLI 形态：模型基于终端会话上下文完成回答"))
            return raw
        return "已达最大轮数，循环终止。"

    def _execute(self, cmd: str) -> str:
        """本地指令入口：同样走四层防御，但直接返回结果。"""
        result = self._run_guarded(cmd)
        return "命令：" + cmd + "\n结果：\n" + result

    def _run_guarded(self, cmd: str) -> str:
        """四层防御：拒绝名单 → 路径边界 → 命令白名单 → 受限工作区。

        其中内容黑名单与绝对路径边界直接复用 Step21 的校验函数（跨场景引用）。
        """
        if not cmd or not cmd.strip():
            return "（空命令）"

        # ① 拒绝名单：整个命令串里出现即拦截
        lower = cmd.lower()
        for bad in DENYLIST:
            if bad in lower:
                self._last_trace.append(tr(
                    "guard", "第①层 · 拒绝名单",
                    "命中 " + bad + " → 直接拒绝（高危命令无商量余地）"))
                return "已拦截：命令包含拒绝名单中的 " + bad + "。"
        # ①+ 复用 Step21 内容黑名单（破坏性命令特征，如 rm -rf / fork 炸弹）
        bad_feature = check_denylist(cmd)
        if bad_feature:
            self._last_trace.append(tr(
                "guard", "第①层 · 拒绝名单",
                "命中危险命令特征「" + bad_feature + "」（复用 Step21 内容黑名单） → 拒绝"))
            return "已拦截：命令包含危险特征 " + bad_feature + "。"

        # ② 路径边界：.. 或绝对路径一律拒绝
        for token in cmd.split():
            if ".." in token or token.startswith("/"):
                self._last_trace.append(tr(
                    "guard", "第②层 · 路径边界",
                    "token " + token + " 越出受限工作区 → 拒绝"))
                return "已拦截：路径 " + token + " 越出受限工作区 " + str(WORKSPACE) + "。"
        # ②+ 复用 Step21 空间边界（绝对路径规范化校验）
        escaped = find_escaped_abs_path(cmd, WORKSPACE)
        if escaped:
            self._last_trace.append(tr(
                "guard", "第②层 · 路径边界",
                "绝对路径 " + escaped + " 越出受限工作区（复用 Step21 边界校验） → 拒绝"))
            return "已拦截：路径 " + escaped + " 越出受限工作区 " + str(WORKSPACE) + "。"

        # ③ 命令白名单：只放行只读/无害命令
        head = cmd.split()[0]
        if head not in ALLOWLIST:
            self._last_trace.append(tr(
                "guard", "第③层 · 命令白名单",
                head + " 不在白名单 " + str(sorted(ALLOWLIST)) + " 中 → 拒绝"))
            return "已拦截：命令 " + head + " 不在白名单中（教学沙箱只放行 " + str(sorted(ALLOWLIST)) + "）。"

        # ④ 受限工作区：子进程指定工作目录，命令按空格拆分（不经 shell，无管道/重定向）
        try:
            proc = subprocess.run(
                cmd.split(),
                cwd=str(WORKSPACE),
                capture_output=True,
                text=True,
                timeout=5,
            )
            out = (proc.stdout or "").strip()
            self._last_trace.append(tr(
                "observation", "沙箱执行 " + cmd,
                "cwd=" + str(WORKSPACE) + "，退出码 " + str(proc.returncode)))
            return out if out.strip() else "（无输出）"
        except Exception as e:
            return "执行失败: " + str(e)

    def reset(self) -> None:
        self._init_workspace()

    def trace(self) -> list:
        return list(self._last_trace)
