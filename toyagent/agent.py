"""智能体契约 —— 整个 ToyAgent 的灵魂只有一个方法：

    chat(input)

输入一句话，输出一句话，中间发生了什么（思考、调工具、查记忆），
都是这个方法内部的策略 —— 26 个场景的演进，全是围绕这个方法的展开。

trace() 把中间过程暴露给调用方渲染（思考步骤、工具调用等），
reset() 供有状态场景（如记忆）清空上下文。

本模块同时承载各场景复用的少量工具函数（JSON 取值、代码围栏剥离等），
对应 Java 版里散落在各场景中重复出现的私有方法与 Json 工具类的
教学化简语义。
"""
import json
import re


def tr(entry_type: str, label: str, detail: str) -> dict:
    """构造一条轨迹事件：{type, label, detail} 三段式，与 Java 各场景一致。"""
    return {"type": entry_type, "label": label, "detail": detail}


class Agent:
    """所有场景的基类。子类必须实现 name() / chat(input)。"""

    def name(self) -> str:
        """场景名称，用于展示。"""
        raise NotImplementedError

    def chat(self, input: str) -> str:
        """智能体最小契约：输入用户消息，返回回答。"""
        raise NotImplementedError

    def trace(self) -> list:
        """最近一次 chat 的执行轨迹（供渲染时间线），无轨迹则返回空列表。"""
        return []

    def reset(self) -> None:
        """清空场景内部状态（如记忆），无状态场景无需实现。"""
        # 默认无状态


# ----------------------------------------------------------------- 工具函数

def jstr(obj, key):
    """等价 Java Json.str：从 dict 按 key 取值的字符串形式，缺失返回 None。

    与 Java String.valueOf 对齐：布尔 -> "true"/"false"，其余 str()。
    """
    if not isinstance(obj, dict):
        return None
    value = obj.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def strip_code_fence(s: str) -> str:
    """剥离模型输出里的 Markdown 代码围栏（```json ... ```）。"""
    if s.startswith("```"):
        s = re.sub(r"^```\w*\s*", "", s, count=1, flags=re.ASCII)
        s = re.sub(r"```\s*$", "", s, count=1, flags=re.ASCII).strip()
    return s


def try_parse_json(raw: str):
    """尝试把模型输出解析为 JSON 对象；失败时原样返回字符串（与 Java 各场景的 tryParse 一致）。"""
    try:
        return json.loads(strip_code_fence(raw).strip())
    except Exception:
        return raw


def or_raw(value, raw: str) -> str:
    """等价 Java orRaw：取值为空/None 时回退到原始输出。"""
    if value is None or (isinstance(value, str) and not value.strip()):
        return raw
    return value
