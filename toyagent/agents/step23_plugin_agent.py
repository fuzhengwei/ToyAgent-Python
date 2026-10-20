"""Step23 · 插件机制 —— 把「一批工具」打包成可安装、可卸载的插件。

与 Step11 的关系：Step11 解决「工具怎么注册」，本步解决「工具从哪来」——
插件是工具的动态载体：安装插件 = 批量注册，卸载插件 = 批量调用 disposer。

四个关键设计（与 Java 版一一对应）：
1. 契约 SPI：AgentPlugin 接口（id/onStart/onStop/tools）；
2. 隔离加载：Java 用独立 URLClassLoader；Python 语言适配为 importlib
   动态加载 plugins/ 目录下的插件模块文件（模块独立命名空间，卸载时从
   sys.modules 摘除）—— 装载语义等价，特此注释说明语言差异；
3. 生命周期：install → onStart → 收集工具 → 批量注册 →（卸载）批量注销 → onStop；
4. 工具桥接：插件的 ToolDefinition 注册进 ToolRegistry，模型立即「看见」——
   注册项的 disposer 按 pluginId 归档。
"""
import importlib.util
import random
import sys
import time
from pathlib import Path

from ..agent import Agent, jstr, try_parse_json, tr
from ..llm import Message
from .step11_registry_agent import ToolDefinition, ToolRegistry, json_dumps

PLUGIN_DIR = Path("plugins") / "step23"
ENTRY_MODULE = "time_plugin"

# 演示插件源码：宿主安装时把它释放到 plugins/ 目录，模拟用户「放入插件包」。
PLUGIN_SOURCE = '''"""Step23 演示插件 TimePlugin —— 由宿主释放到 plugins/ 目录动态加载。

如果它有 main 函数的人生，这就是「插件作者」写的全部代码：
实现 AgentPlugin 契约，onStart 打个招呼，tools() 返回带来的工具。
"""
import time

import random

from toyagent.agents.step11_registry_agent import ToolDefinition


class TimePlugin:
    """插件契约实现（对应 Java 版 PluginAgent.TimePlugin）。"""

    def __init__(self):
        self.started = False

    def id(self):
        return "time-plugin"

    def version(self):
        return "1.0.0"

    def onStart(self):
        self.started = True

    def onStop(self):
        self.started = False

    def tools(self):
        return [UptimeTool(time.time()), CoinTool()]


class UptimeTool(ToolDefinition):
    """工具 1：宿主进程运行时长 —— 插件工具也能反查宿主状态。"""

    def __init__(self, started_at):
        self._started_at = started_at

    def name(self):
        return "uptime"

    def description(self):
        return "查询宿主进程已运行时长"

    def parameters(self):
        return {}

    def execute(self, args):
        sec = int(time.time() - self._started_at)
        h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
        return f"宿主已运行 {h} 小时 {m} 分 {s} 秒"


class CoinTool(ToolDefinition):
    """工具 2：掷硬币 —— 一个纯粹由插件带来的趣味能力。"""

    def name(self):
        return "coin_flip"

    def description(self):
        return "掷一次硬币，返回正面或反面"

    def parameters(self):
        return {}

    def execute(self, args):
        return "正面" if random.random() < 0.5 else "反面"
'''

SYSTEM_PROMPT_PREFIX = """你是一个可以调用工具的智能体。可用工具清单（由注册表自动生成，工具由插件提供）：
"""


class _Loaded:
    """已加载插件：实例、模块名、工具清单与回收句柄。"""

    def __init__(self, plugin, module_name, tool_names, disposers):
        self.plugin = plugin
        self.module_name = module_name
        self.tool_names = tool_names
        self.disposers = disposers


class PluginAgent(Agent):

    def __init__(self, model):
        self._model = model
        self._registry = ToolRegistry()
        self._installed = {}  # pluginId -> _Loaded（保持插入序）
        self._last_trace = []

    def name(self) -> str:
        return "Step23 · 插件机制（JavaPluginLoader 隔离加载）"

    def chat(self, input: str) -> str:
        self._last_trace.clear()
        self._last_trace.append(tr(
            "tool", "注册表就绪",
            "宿主内置 0 个工具 —— 能力全部由插件装进来（已安装 "
            + str(len(self._installed)) + " 个插件）"))

        # ---- 本地指令 1：安装插件 ----
        if "安装插件" in input or "装载插件" in input:
            return self._install()

        # ---- 本地指令 2：卸载插件 ----
        if "卸载插件" in input or "移除插件" in input:
            return self._uninstall()

        # ---- 本地指令 3：查看工具清单 ----
        if "查看工具" in input or "工具清单" in input:
            listing = "（无）" if not self._installed else "、".join(self._installed.keys())
            self._last_trace.append(tr(
                "prompt", "工具清单 promptCatalog()",
                "（注册表为空）" if self._registry.is_empty() else self._registry.prompt_catalog().strip()))
            return ("已安装插件：" + listing + "\n当前注册表共 " + str(self._registry.size()) + " 个工具：\n"
                    + ("（空）" if self._registry.is_empty() else self._registry.prompt_catalog())
                    + "\n插件注册的工具对模型完全透明 —— 它只看得到清单，不知道工具是内置的还是插件带来的。")

        # ---- 常规对话：清单进提示词 → 模型决策 → 注册表执行 ----
        messages = [
            Message.system(SYSTEM_PROMPT_PREFIX
                           + ("（当前没有任何工具）" if self._registry.is_empty() else self._registry.prompt_catalog())
                           + '\n决策规则：\n- 需要工具时，只输出 JSON：{"tool": "工具名", "arguments": {}}\n'
                           + "- 不需要工具时，直接用自然语言回答，不要输出 JSON。"),
            Message.user(input),
        ]

        for _round in range(1, 3):
            raw = self._model.chat(messages).strip()
            decision = try_parse_json(raw)

            if isinstance(decision, dict) and jstr(decision, "tool") is not None:
                tool_name = jstr(decision, "tool")
                self._last_trace.append(tr(
                    "action", "模型发起工具调用",
                    tool_name + " " + json_dumps(decision.get("arguments"))))

                tool = self._registry.lookup(tool_name)
                if tool is None:
                    result = "未知工具: " + tool_name + "（插件未安装，或已被卸载）"
                    self._last_trace.append(tr(
                        "guard", "注册表未命中",
                        "卸载插件后其工具随之消失 —— 模型看不见，也调不到"))
                else:
                    try:
                        result = tool.execute({})
                    except Exception as e:
                        result = "工具执行失败: " + str(e)
                    self._last_trace.append(tr(
                        "observation", "注册表执行 " + tool_name, result))

                messages.append(Message.assistant(raw))
                messages.append(Message.user("工具 " + tool_name + " 执行结果: " + result))
                continue

            self._last_trace.append(tr(
                "final", "最终回答", "模型基于插件注册的工具完成决策"))
            return raw
        return "已达最大轮数，循环终止。"

    def _install(self) -> str:
        """安装：释放插件包 → 隔离加载 → onStart → 收集工具 → 批量注册。"""
        if self._installed:
            return ("已有插件在运行：" + "、".join(self._installed.keys())
                    + "。先卸载再安装其他插件。")
        try:
            # 1. 释放插件包（真实场景：用户下载插件文件放入 plugins/ 目录）
            self._materialize_plugin_package()
            self._last_trace.append(tr(
                "tool", "插件包落盘 " + str(PLUGIN_DIR),
                "time_plugin.py + plugin.properties（entrypoint 声明入口模块，等价 dsh-java 的 META-INF/plugin.yaml）"))

            # 2. 独立命名空间隔离加载（Java 用 URLClassLoader；Python 用 importlib 独立模块名）
            module_name = "toyagent_step23_plugin_" + ENTRY_MODULE
            try:
                spec = importlib.util.spec_from_file_location(
                    module_name, PLUGIN_DIR / (ENTRY_MODULE + ".py"))
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                plugin = module.TimePlugin()
                plugin.onStart()
                self._last_trace.append(tr(
                    "action", "importlib 隔离加载",
                    "入口 TimePlugin 实例化成功，onStart() 已调用 —— 模块与宿主隔离，卸载时从 sys.modules 摘除"))

                # 3. 工具桥接：逐个注册，disposer 按 pluginId 归档
                tool_names = []
                disposers = []
                for tool in plugin.tools():
                    disposers.append(self._registry.register(tool))
                    tool_names.append(tool.name())
                self._last_trace.append(tr(
                    "observation", "工具桥接 batch register",
                    str(tool_names) + " 已注册进 ToolRegistry，disposers 按 pluginId 归档 —— 卸载即批量注销"))

                self._installed[plugin.id()] = _Loaded(plugin, module_name, tool_names, disposers)
                return ("插件 " + plugin.id() + " v" + plugin.version() + " 安装成功，带来 "
                        + str(len(tool_names)) + " 个工具：\n" + self._registry.prompt_catalog()
                        + "\n现在可以直接对话使用（如「掷一次硬币」「运行多久了」），模型自动看见新工具。")
            except Exception as e:
                self._unload_module(module_name)
                return "插件加载失败：" + str(e)
        except OSError as e:
            return "插件包落盘失败：" + str(e)

    def _uninstall(self) -> str:
        """卸载：批量注销工具 → onStop → 卸载模块。"""
        if not self._installed:
            return "当前没有已安装的插件。先试试「安装插件」。"
        sb = ""
        for plugin_id, loaded in self._installed.items():
            for disposer in loaded.disposers:
                disposer()
            loaded.plugin.onStop()
            self._unload_module(loaded.module_name)
            self._last_trace.append(tr(
                "guard", "卸载 " + plugin_id,
                str(loaded.tool_names) + " 已批量注销，onStop() 已调用，模块已卸载"))
            sb += str(loaded.tool_names) + " "
        self._installed.clear()
        return ("插件已卸载，工具 " + sb + "全部注销。注册表回到 " + str(self._registry.size())
                + " 个工具 —— 插件机制的三步回收：批量 disposer → onStop → 卸载模块。\n"
                + "现在再让模型掷硬币，它会说工具不存在 —— 卸载即消失。")

    def _materialize_plugin_package(self):
        """把演示插件源码释放到 plugins/ 目录，并写入迷你清单。"""
        PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
        (PLUGIN_DIR / (ENTRY_MODULE + ".py")).write_text(PLUGIN_SOURCE, encoding="utf-8")
        manifest = (
            "# mini plugin manifest (dsh-java 用 META-INF/plugin.yaml)\n"
            "id=time-plugin\n"
            "version=1.0.0\n"
            "entrypoint=time_plugin.TimePlugin\n")
        (PLUGIN_DIR / "plugin.properties").write_text(manifest, encoding="utf-8")

    @staticmethod
    def _unload_module(module_name: str):
        """等价 Java ClassLoader.close()：把插件模块从解释器里摘除。"""
        sys.modules.pop(module_name, None)

    def reset(self) -> None:
        for loaded in self._installed.values():
            for disposer in loaded.disposers:
                disposer()
            loaded.plugin.onStop()
            self._unload_module(loaded.module_name)
        self._installed.clear()
        self._registry.clear()

    def trace(self) -> list:
        return list(self._last_trace)
