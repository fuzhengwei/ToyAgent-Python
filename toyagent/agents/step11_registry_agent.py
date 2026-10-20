"""Step11 · 工具注册表 —— 用「协议 + 注册表」取代写死的分支。

与 Step04 的三点不同：
1. 工具实现 ToolDefinition 协议，注册进 ToolRegistry；
2. 系统提示词里的工具清单由注册表自动生成 —— 注册即生效、注销即消失；
3. 执行器只做 lookup + execute，不再认识任何具体工具。

本模块同时定义 ToolDefinition / ToolRegistry 两个基础设施，
后续 Step17（运行时）与 Step23（插件机制）跨模块复用。
"""
import json
import re
from abc import ABC, abstractmethod
from datetime import datetime
from threading import Lock

from ..agent import Agent, jstr, try_parse_json, tr
from ..llm import Message


def json_dumps(value) -> str:
    """紧凑 JSON 序列化。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# -------------------------------------------------------------- 工具定义协议

class ToolDefinition(ABC):
    """Step11 · 工具定义协议 —— 一个类 = 一个工具。

    - name() 工具名 —— 模型调用时使用的标识；
    - description() 用途描述 —— 模型据此判断"该不该用这个工具"；
    - parameters() 参数说明 —— 生产中是 JSON Schema，教学化简为「参数名 → 类型说明」；
    - execute(args) 执行逻辑 —— 工具的真正手脚。

    实现这个协议 + 注册进注册表 = 一个模型可见、可调用的工具。
    MCP 协议的 tools/list + tools/call，本质就是这套协议的标准化版本（见场景 09）。
    """

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def parameters(self) -> dict: ...

    @abstractmethod
    def execute(self, args: dict) -> str: ...

    def concurrency_safe(self) -> bool:
        """是否可并行执行（fs 读类 True，shell 写类 False）—— 预留并行调度语义。"""
        return False


class ToolRegistry:
    """Step11 · 工具注册表 —— 模型"看得见"的工具目录。

    三个职责：
    1. 注册：register() 把工具放进目录，并返回一个「注销器 disposer」——
       持有它就能把工具从目录摘除。插件停止/卸载时，宿主靠这个句柄回收注册项，
       这就是「注册-回收对称性」；
    2. 发现：lookup() 按名字取工具，执行器只认注册表、不认具体工具；
    3. 清单生成：prompt_catalog() 把目录渲染成系统提示词里的工具清单 ——
       模型能看见哪些工具，完全由注册表当前内容决定。注册即生效，注销即消失。
    """

    def __init__(self):
        self._tools = {}  # 保持插入序（LinkedHashMap 语义）
        self._lock = Lock()

    def register(self, tool: ToolDefinition):
        """注册工具，返回注销器（disposer）：调用即把该工具从目录摘除。"""
        with self._lock:
            self._tools[tool.name()] = tool

        def disposer():
            with self._lock:
                self._tools.pop(tool.name(), None)

        return disposer

    def lookup(self, name):
        with self._lock:
            if name is None:
                return None
            return self._tools.get(name)

    def size(self) -> int:
        with self._lock:
            return len(self._tools)

    def is_empty(self) -> bool:
        with self._lock:
            return not self._tools

    def clear(self):
        with self._lock:
            self._tools.clear()

    def prompt_catalog(self) -> str:
        """把当前目录渲染成系统提示词中的工具清单。"""
        sb = ""
        with self._lock:
            tools = list(self._tools.values())
        for t in tools:
            params = ", ".join(f"{k}: {v}" for k, v in t.parameters().items())
            sb += f"- {t.name()}({params}) — {t.description()}\n"
        return sb


# -------------------------------------------------------------- 内置工具

class WeatherTool(ToolDefinition):
    """天气查询工具：实现协议四件套，即插即用。"""

    DATA = {
        "北京": "晴，26℃", "上海": "多云，28℃", "广州": "雷阵雨，31℃",
        "深圳": "阵雨，30℃", "杭州": "晴，27℃", "成都": "阴，22℃",
    }

    def name(self):
        return "get_weather"

    def description(self):
        return "查询指定城市的实时天气"

    def parameters(self):
        return {"city": "string，城市名，如 北京"}

    def execute(self, args: dict) -> str:
        city = str(args.get("city", "")).strip()
        return WeatherTool.DATA.get(city, "未收录城市 " + city + " 的数据")


_CALC_PTN = re.compile(r"\s*(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*", re.ASCII)


class CalculatorTool(ToolDefinition):
    """四则运算工具。"""

    def name(self):
        return "calculator"

    def description(self):
        return "计算四则运算，如 12*8"

    def parameters(self):
        return {"expression": "string，算式，如 12*8"}

    def execute(self, args: dict) -> str:
        m = _CALC_PTN.fullmatch(str(args.get("expression", "")))
        if not m:
            return "无法解析表达式"
        a, b = float(m.group(1)), float(m.group(3))
        op = m.group(2)
        if op == "+":
            return str(a + b)
        if op == "-":
            return str(a - b)
        if op == "*":
            return str(a * b)
        if op == "/":
            return "除数不能为 0" if b == 0 else str(a / b)
        return "不支持的运算符"


class TimeTool(ToolDefinition):
    """时间工具。"""

    def name(self):
        return "get_time"

    def description(self):
        return "获取当前时间"

    def parameters(self):
        return {}

    def execute(self, args: dict) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# -------------------------------------------------------------- 场景

SYSTEM_PROMPT_PREFIX = """你是一个可以调用工具的智能体。可用工具清单（由注册表自动生成）：
"""


class RegistryAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._registry = ToolRegistry()
        # 每个工具的注销器：插件热卸载时宿主靠它回收注册项。
        self._disposers = {}
        self._last_trace = []
        self._register_defaults()

    def _register_defaults(self):
        """注册内置工具：注册这个动作本身，就是本场景的主角。"""
        self._disposers["get_weather"] = self._registry.register(WeatherTool())
        self._disposers["calculator"] = self._registry.register(CalculatorTool())
        self._disposers["get_time"] = self._registry.register(TimeTool())

    def name(self) -> str:
        return "Step11 · 工具注册表（ToolDefinition 协议）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()
        self._last_trace.append(tr(
            "tool", "注册表就绪",
            "当前 " + str(self._registry.size())
            + " 个工具，清单由注册表自动生成，无需改提示词代码"))

        # ---- 本地指令 1：查看工具清单（不经过模型，直接看注册表） ----
        if "查看工具" in input or "工具清单" in input:
            self._last_trace.append(tr(
                "prompt", "工具清单 promptCatalog()",
                self._registry.prompt_catalog().strip()))
            return ("当前注册表中共有 " + str(self._registry.size()) + " 个工具：\n"
                    + self._registry.prompt_catalog()
                    + "\n这份清单就是模型每次对话时在系统提示词里「看见」的内容 —— 注册即生效，卸载即消失。")

        # ---- 本地指令 2：卸载工具（演示注销器） ----
        if input.startswith(("卸载：", "卸载:")):
            tool_name = input[3:].strip()
            disposer = self._disposers.pop(tool_name, None)
            if disposer is None:
                return ("工具 " + tool_name + " 不在注册表中，无法卸载。当前清单：\n"
                        + self._registry.prompt_catalog())
            disposer()
            self._last_trace.append(tr(
                "guard", "注销器执行 disposer.run()",
                "工具 " + tool_name + " 已从注册表摘除，剩余 " + str(self._registry.size()) + " 个"))
            return ("已卸载工具 " + tool_name + "。注销器（disposer）就是注册时返回的回调 —— "
                    + "插件停止/卸载时，宿主正是靠它回收工具、系统提示词等注册项。\n当前清单：\n"
                    + self._registry.prompt_catalog()
                    + "\n现在再问相关问题，模型将「看不见」这个工具。")

        # ---- 常规对话：清单进提示词 → 模型决策 → 注册表执行 ----
        messages = [
            Message.system(SYSTEM_PROMPT_PREFIX + self._registry.prompt_catalog()
                           + '\n决策规则：\n- 需要工具时，只输出 JSON：{"tool": "工具名", "arguments": {"参数名": "值"}}\n'
                           + "- 不需要工具时，直接用自然语言回答，不要输出 JSON。"),
            Message.user(input),
        ]

        # 最多 2 轮：工具调用 → 结果回填 → 最终回答（与 Step04 同构，但执行走注册表）
        for _round in range(1, 3):
            raw = self._model.chat(messages).strip()
            decision = try_parse_json(raw)

            if isinstance(decision, dict) and jstr(decision, "tool") is not None:
                tool_name = jstr(decision, "tool")
                args_json = json_dumps(decision.get("arguments"))
                self._last_trace.append(tr(
                    "action", "模型发起工具调用", tool_name + " " + args_json))

                # 核心差异：执行器不认识任何具体工具，只查注册表
                tool = self._registry.lookup(tool_name)
                if tool is None:
                    result = "未知工具: " + tool_name + "。可用清单：\n" + self._registry.prompt_catalog()
                    self._last_trace.append(tr(
                        "guard", "注册表未命中",
                        "模型只能调用注册表里存在的工具 —— 卸载后的工具自然不可调用"))
                else:
                    try:
                        args = json.loads(json_dumps(decision.get("arguments")))
                        result = tool.execute(args)
                    except Exception as e:
                        result = "工具执行失败: " + str(e)
                    self._last_trace.append(tr(
                        "observation", "注册表执行 " + tool_name, result))

                messages.append(Message.assistant(raw))
                messages.append(Message.user("工具 " + tool_name + " 执行结果: " + result))
                continue

            # 模型直接给出自然语言回答
            self._last_trace.append(tr(
                "final", "最终回答", "模型基于注册表提供的工具完成决策"))
            return raw
        return "已达最大轮数，循环终止。"

    def reset(self) -> None:
        self._registry.clear()
        self._disposers.clear()
        self._register_defaults()

    def trace(self) -> list:
        return list(self._last_trace)
