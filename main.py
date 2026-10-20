#!/usr/bin/env python3
"""ToyAgent CLI 运行器 —— 替代 Java 版 web 页面的演示入口。

命令：
1. 无参数 / help —— 打印帮助 + 26 个场景清单；
2. list —— 场景清单；
3. run <stepId> "<message>" ["<message2>" ...] —— 单轮（或多轮）执行指定场景，
   打印场景名、回答、执行轨迹时间线、耗时 ms。同一命令内多条消息按顺序
   喂给同一个场景实例（用于 step05 记忆 / step19 挂起 / step20 审批等多轮验证）；
4. repl —— 交互模式：先选场景，再循环输入；支持 /reset、/exit。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from toyagent import agents
from toyagent.llm import models  # noqa: F401  （导入即完成配置装载与启动横幅输出）


def scenario_list_lines() -> list:
    lines = []
    for sid in agents.SCENARIO_ORDER:
        agent = agents.create_agent(sid)
        lines.append(f"  {sid}  {agent.name()}")
    return lines


def print_help() -> None:
    print("用法:")
    print("  python main.py                    # 帮助 + 场景清单，并进入交互 REPL")
    print("  python main.py list               # 场景清单")
    print('  python main.py run <stepId> "<message>" ["<message2>" ...]')
    print("                                    # 执行场景（多条消息按顺序喂给同一实例）")
    print("  python main.py repl               # 交互模式：选场景 → 循环对话")
    print("交互模式指令: /reset 重置场景状态  /exit 退出")
    print()
    print("26 个场景:")
    for line in scenario_list_lines():
        print(line)


def print_banner() -> None:
    print("==============================================")
    print("  ToyAgent v1.0 (Python)  启动成功")
    if models.is_real_model():
        print("  模型模式: 真实模型 " + models.model_name())
    else:
        print("  模型模式: Mock 模型（未配置 api-key）")
    print("==============================================")


def print_trace(agent) -> None:
    for t in agent.trace():
        print(f"  [{t['label']}] {t['detail']}")


def run_cmd(step_id: str, messages: list) -> int:
    if step_id not in agents.SCENARIO_ORDER:
        print("场景不存在: " + step_id + "。可用: " + ", ".join(agents.SCENARIO_ORDER))
        return 1
    if not messages:
        print('缺少消息参数。示例: python main.py run step03 "北京天气怎么样"')
        return 1
    agent = agents.create_agent(step_id)
    for msg in messages:
        start = time.perf_counter()
        answer = agent.chat(msg)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print("场景: " + agent.name())
        print("输入: " + msg)
        print("回答: " + answer)
        print("执行轨迹:")
        print_trace(agent)
        print(f"耗时: {elapsed_ms} ms")
        print("-" * 48)
    return 0


def repl() -> None:
    current = None
    agent = None
    print()
    print("交互 REPL：先选场景（如 step03，list 查看清单），再循环输入。/exit 退出。")
    while True:
        try:
            if agent is None:
                line = input("选择场景> ").strip()
            else:
                line = input("你> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line == "/exit":
            break
        if agent is None:
            if line == "list":
                for sid in agents.SCENARIO_ORDER:
                    print(f"  {sid}  {agents.create_agent(sid).name()}")
                continue
            if line == "/exit":
                break
            if line not in agents.SCENARIO_ORDER:
                print("场景不存在: " + line + "。请输入 step01~step26 之一（list 查看清单）。")
                continue
            current = line
            agent = agents.create_agent(current)
            print("已进入 " + agent.name() + "（/reset 清空状态，/exit 退出）")
            continue
        if line == "/reset":
            agent.reset()
            print("已重置场景状态。")
            continue
        start = time.perf_counter()
        answer = agent.chat(line)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print("Agent> " + answer)
        print("轨迹:")
        print_trace(agent)
        print(f"(耗时 {elapsed_ms} ms)")
    print("Bye.")


def main(argv: list) -> int:
    print_banner()
    if not argv or argv[0] == "help":
        print_help()
        if not argv:
            repl()
        return 0
    cmd = argv[0]
    if cmd == "list":
        print("26 个场景:")
        for line in scenario_list_lines():
            print(line)
        return 0
    if cmd == "repl":
        repl()
        return 0
    if cmd == "run":
        if len(argv) < 3:
            print('用法: python main.py run <stepId> "<message>" ["<message2>" ...]')
            return 1
        return run_cmd(argv[1], argv[2:])
    print("未知命令: " + cmd)
    print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
