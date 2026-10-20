"""模型工厂：读取配置，产出 ChatModel。

配置优先级：项目根目录 config.properties > 环境变量。
未配置 api-key 时 open_ai_or_null() 返回 None，由上层为每个场景装配 MockChatModel。

Java 版 Models 的 update()/save()/useMock() 是 web 页面专用，本工程不含；
config.properties 为 Java properties 格式，此处手写解析（不引入 configparser
以保持语义一致：`=` 与 `:` 均为分隔符、`#`/`!` 注释）。
"""
import os
import sys

from .mock_chat_model import MockChatModel
from .openai_chat_model import OpenAiChatModel


def _read_properties(path: str) -> dict:
    """手写的 Java properties 解析（教学工程保持零依赖与行为一致）。"""
    props = {}
    if not os.path.exists(path):
        return props
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("!"):
                    continue
                i_eq, i_colon = s.find("="), s.find(":")
                if i_eq < 0 and i_colon < 0:
                    continue
                if i_eq < 0:
                    idx = i_colon
                elif i_colon < 0:
                    idx = i_eq
                else:
                    idx = min(i_eq, i_colon)
                props[s[:idx].strip()] = s[idx + 1:].strip()
    except OSError as e:
        print(f"[ToyAgent] 读取 config.properties 失败: {e}", file=sys.stderr)
    return props


def _env(key: str):
    v = os.environ.get(key)
    return v if (v is not None and v.strip()) else None


def _first_non_blank(*vals):
    """返回第一个非空值；全空则返回最后一个（内置默认）。"""
    for v in vals[:-1]:
        if v is not None and str(v).strip():
            return v
    return vals[-1]


def _load():
    props = _read_properties("config.properties")
    # 优先级：config.properties（非空值）> LLM_* 环境变量 > OPENAI_* 环境变量 > 内置默认
    # 文件里留空（api-key=）不遮蔽环境变量 —— 支持纯 docker run -e 方式配置默认模型
    base_url = _first_non_blank(
        props.get("base-url"),
        _env("LLM_BASE_URL"), _env("OPENAI_BASE_URL"),
        "https://api.deepseek.com/v1")
    api_key = _first_non_blank(
        props.get("api-key"),
        _env("LLM_API_KEY"), _env("OPENAI_API_KEY"),
        "")
    model_name = _first_non_blank(
        props.get("model"),
        _env("LLM_MODEL"), _env("OPENAI_MODEL"),
        "deepseek-chat")
    key_src = props.get("api-key")
    key_from_file = key_src is not None and bool(key_src.strip())
    if api_key.strip():
        print("[ToyAgent] 默认模型: " + model_name + " @ " + base_url
              + ("（Key 来自 config.properties）" if key_from_file else "（Key 来自环境变量）"))
    return base_url, api_key, model_name


_BASE_URL, _API_KEY, _MODEL_NAME = _load()


def open_ai_or_null():
    """有真实 Key 则返回 OpenAI 兼容模型，否则返回 None（使用 Mock）。"""
    if not _API_KEY or not _API_KEY.strip():
        return None
    return OpenAiChatModel(_BASE_URL, _API_KEY, _MODEL_NAME)


def mock(scenario: str) -> MockChatModel:
    """教学演示用 Mock 模型。"""
    return MockChatModel(scenario)


def is_real_model() -> bool:
    return bool(_API_KEY and _API_KEY.strip())


def model_name() -> str:
    return _MODEL_NAME if is_real_model() else "mock-model"


def base_url() -> str:
    return _BASE_URL


def masked_key() -> str:
    """脱敏后的 Key，用于展示回显。"""
    if not _API_KEY or not _API_KEY.strip():
        return ""
    if len(_API_KEY) <= 10:
        return "******"
    return _API_KEY[:5] + "****" + _API_KEY[-4:]
