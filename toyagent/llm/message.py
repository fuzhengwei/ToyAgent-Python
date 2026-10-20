"""一条对话消息。role 取值：system / user / assistant。

用最纯的数据结构表达 —— 智能体与模型之间流动的
永远只是「角色 + 文本」，这就是一切复杂行为的原材料。
"""


class Message:
    __slots__ = ("role", "content")

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def __repr__(self):
        return f"Message(role={self.role!r}, content={self.content!r})"

    @staticmethod
    def system(content: str) -> "Message":
        return Message("system", content)

    @staticmethod
    def user(content: str) -> "Message":
        return Message("user", content)

    @staticmethod
    def assistant(content: str) -> "Message":
        return Message("assistant", content)
