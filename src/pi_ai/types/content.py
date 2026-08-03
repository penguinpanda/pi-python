"""pi_ai.types.content — 消息内容块（ContentBlock）。

ContentBlock 是消息内容的统一抽象：

    TextContent     文本
    ImageContent    图片
    ToolCall        工具调用
    ThinkingContent 推理过程
    CodeContent     代码块（插件化扩展示例）

扩展机制：所有内容块都继承 BaseContent（type 判别字段），
新增类型只需继承 BaseContent 并把 Literal 收窄为唯一 type。
"""

from typing import Any, Literal, NotRequired, TypedDict


class BaseContent(TypedDict):
    """内容块基础协议：所有 ContentBlock 共享 type 判别字段。"""

    type: str


class TextContent(BaseContent):
    """
    文本内容块

    示例：

    {
        "type": "text",
        "text": "Hello"
    }
    """

    type: Literal["text"]
    text: str  # 文本内容

    # OpenAI responses 的 message metadata（旧版 id 字符串或 TextSignatureV1 JSON）
    text_signature: NotRequired[str]


class ImageContent(BaseContent):
    """
    图片内容块

    可以使用：

    ① 图片 URL

    {
        "type":"image",
        "url":"https://..."
    }

    ② Base64 数据

    {
        "type":"image",
        "data":"xxxxx"
    }
    """

    type: Literal["image"]
    url: str | None        # 图片地址
    data: str | None       # Base64 编码图片
    mime_type: str | None  # MIME 类型，例如 image/png


class ToolCall(BaseContent):
    """
    Tool Calling 内容块

    AI 告诉客户端：

    "我要调用某个工具。"

    示例（已解析）：

    {
        "type":"toolCall",
        "id":"call_1",
        "name":"search",
        "raw_arguments":"{\"query\":\"...\"}",
        "arguments":{"query":"..."}
    }

    arguments 语义：

    - 流式累积期间：None（尚未解析）
    - 流式结束（toolcall_end / done）：已解析 dict；
      解析失败时为 {"_error": "Invalid JSON arguments"}
    """

    type: Literal["toolCall"]
    id: str                           # Tool Call 唯一 ID
    name: str                         # 工具名称
    raw_arguments: str                # 流式累积的原始 JSON 字符串
    arguments: dict[str, Any] | None  # 已解析对象；None 表示尚未解析/解析失败
    thought_signature: NotRequired[str]  # Google 专用：复用思考上下文的签名


class ThinkingContent(BaseContent):
    """
    推理内容块（Reasoning）

    部分模型（如 DeepSeek、Claude）
    会返回中间思考过程。
    """

    type: Literal["thinking"]
    thinking: str                         # 思考内容
    thinking_signature: NotRequired[str]  # 推理签名（某些 API 使用；如 OpenAI responses 的 reasoning item ID）

    # True 表示思考内容被安全过滤改写（加密占位存于 thinking_signature，以支持多轮续传）
    redacted: NotRequired[bool]


class CodeContent(BaseContent):
    """
    代码内容块（插件化扩展示例）

    {
        "type": "code",
        "language": "python",
        "code": "print('hi')"
    }
    """

    type: Literal["code"]
    language: str   # 编程语言（python / typescript / ...）
    code: str       # 代码内容


# ContentBlock 可以是任意一种内容
ContentBlock = TextContent | ImageContent | ToolCall | ThinkingContent | CodeContent


class TextSignatureV1(TypedDict):
    """OpenAI responses 的文本签名（v1 结构）"""

    v: Literal[1]
    id: str
    phase: NotRequired[Literal["commentary", "final_answer"]]


__all__ = [
    "BaseContent",
    "TextContent",
    "ImageContent",
    "ToolCall",
    "ThinkingContent",
    "CodeContent",
    "ContentBlock",
    "TextSignatureV1",
]
