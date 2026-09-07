"""问答请求/响应模型。

V1(从简)模型: 单条提问 + 会话标识; 答案通过 SSE 流式返回,
此处仅定义 HTTP 请求/响应所需的 Pydantic 模型。
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """提问请求。"""

    session_id: str = Field(default="", description="会话ID(为空则服务端新生成)")
    question: str = Field(..., min_length=1, description="用户问题")
    is_stream: bool = Field(default=True, description="是否流式输出(SSE)")


class QueryResponse(BaseModel):
    """提问受理响应(答案经 SSE /query/stream/{task_id} 推送)。"""

    session_id: str = Field(..., description="会话ID")
    task_id: str = Field(..., description="任务ID(SSE 订阅用)")
    message_id: str = Field(..., description="消息ID")
    message: str = Field(default="已受理,请订阅 /query/stream/{task_id}", description="提示信息")


class HistoryMessage(BaseModel):
    """会话历史中的一条消息。"""

    message_id: str = Field(default="", description="消息ID")
    role: str = Field(..., description="角色: user/assistant")
    text: str = Field(default="", description="内容")
    ts: Optional[float] = Field(default=None, description="时间戳")
