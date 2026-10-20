"""OpenAI 兼容协议的模型实现（DeepSeek / 通义 / 智谱 / 火山方舟等均适用）。

核心就一次 HTTP 调用：POST {base-url}/chat/completions，
请求体是 messages 数组，响应里取 choices[0].message.content。
所有"大模型"对你而言都只是这一个端点。
HTTP 用 Python 标准库 urllib（零第三方依赖）。
"""
import json
import re
import urllib.error
import urllib.request

from .chat_model import ChatModel
from .message import Message


def _snippet(s: str) -> str:
    if s is None:
        return ""
    return s[:300] + "..." if len(s) > 300 else s


class OpenAiChatModel(ChatModel):

    def __init__(self, base_url: str, api_key: str, model: str):
        self._base_url = re.sub(r"/+$", "", base_url)
        self._api_key = api_key
        self._model = model

    def chat(self, messages) -> str:
        # 1. 组装请求体：{"model": "...", "messages": [{"role": "...", "content": "..."}]}
        body = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": 0.7,
        }
        request = urllib.request.Request(
            self._base_url + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self._api_key,
            },
            method="POST",
        )

        # 2. 调用模型
        try:
            with urllib.request.urlopen(request, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace") if e.fp else ""
            raise RuntimeError(f"模型调用失败 HTTP {e.code}: {_snippet(detail)}") from None
        except Exception as e:
            raise RuntimeError(f"模型调用异常: {e}") from None

        # 3. 解析响应：choices[0].message.content
        try:
            data = json.loads(raw)
            return str(data["choices"][0]["message"]["content"])
        except Exception as e:
            raise RuntimeError(f"模型调用异常: {e}") from None
