"""问答侧路由(V1 从简实现,SSE 流式)。

模板 knowledge/api/query_router.py 为空文件(问答 HTTP 层未完成),此处按模板
已有的 sse_util / task_util / query_process 能力补齐:

    GET  /query                 问答前端页(front/query.html)
    POST /query/ask             受理提问 → 后台运行问答图 → 返回 task_id(SSE 订阅)
    GET  /query/stream/{task_id} SSE 流: progress(节点进度) / delta(增量) / final(完整答案)
    GET  /query/history/{session_id}   读取会话历史(MongoDB, 新→旧)
    POST /query/history/clear          清空会话历史

说明(V1 从简):
    * 多轮: 提问时自动加载该 session 最近历史注入问答图,答案自动落库
    * 问答图延迟到首次提问才构建,应用启动不加载重型模型
"""
import os
import uuid

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from core.paths import get_front_page_dir
from schema.query_schema import HistoryMessage, QueryRequest, QueryResponse
from utils.mongo_history_util import clear_history, get_recent_messages
from utils.sse_util import create_sse_queue, sse_generator
from utils.task_util import get_task_result, update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED


def _build_history(session_id: str, limit: int = 10) -> list:
    """读取最近会话记录并按时间正序返回 [{role, text}] 供问答图组装上下文。"""
    records = get_recent_messages(session_id=session_id, limit=limit)
    # Mongo 按 ts 降序返回(新→旧);这里翻转成 旧→新
    records = sorted(records, key=lambda r: r.get("ts", 0))
    history = []
    for r in records:
        role = r.get("role")
        text = (r.get("text") or "").strip()
        if role in ("user", "assistant") and text:
            history.append({"role": role, "text": text})
    return history


def _run_query_graph(state: dict) -> None:
    """后台执行问答图(延迟导入,避免启动加载重型依赖)。"""
    from processor.query_process.main_graph import query_app  # noqa: PLC0415
    update_task_status(state["task_id"], TASK_STATUS_PROCESSING)
    try:
        query_app.invoke(state)
        update_task_status(state["task_id"], TASK_STATUS_COMPLETED)
    except Exception as exc:
        update_task_status(state["task_id"], TASK_STATUS_FAILED, str(exc))


def register_router(app: FastAPI) -> None:
    """注册问答侧路由。"""

    @app.get("/query", tags=["query"])
    def query_page():
        """问答前端页面。"""
        return FileResponse(path=os.path.join(get_front_page_dir(), "query.html"))

    @app.post("/query/ask", response_model=QueryResponse, tags=["query"])
    async def ask(
            payload: QueryRequest,
            background_tasks: BackgroundTasks):
        """受理提问: 创建 SSE 队列 → 组装状态(含历史) → 后台运行问答图。"""
        question = payload.question.strip()
        if not question:
            return JSONResponse(status_code=400, content={"detail": "question 不能为空"})

        session_id = payload.session_id.strip() or str(uuid.uuid4())
        task_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())

        # 1. 创建 SSE 事件队列(问答图各节点/答案输出向此队列推送)
        create_sse_queue(task_id)
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        # 2. 组装问答图初始状态
        from processor.query_process.state import create_default_state  # noqa: PLC0415
        state = create_default_state(
            session_id=session_id,
            task_id=task_id,
            message_id=message_id,
            original_query=question,
            is_stream=payload.is_stream,
            history=_build_history(session_id),
        )

        # 3. 后台运行
        background_tasks.add_task(_run_query_graph, state)

        return QueryResponse(session_id=session_id, task_id=task_id, message_id=message_id)

    @app.get("/query/stream/{task_id}", tags=["query"])
    async def stream(task_id: str, request: Request):
        """SSE 流式输出: progress / delta / final 事件。"""
        return StreamingResponse(
            sse_generator(task_id, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/query/result/{task_id}", tags=["query"])
    async def result(task_id: str):
        answer = get_task_result(task_id, "answer")
        error = get_task_result(task_id, "error")
        if not answer and not error:
            return JSONResponse(status_code=404, content={"detail": "结果不存在或尚未完成"})
        return {"task_id": task_id, "answer": answer, "error": error}

    @app.get("/query/history/{session_id}", response_model=list[HistoryMessage], tags=["query"])
    async def history(session_id: str, limit: int = 20):
        """读取会话历史(新→旧)。"""
        records = get_recent_messages(session_id=session_id, limit=max(1, min(limit, 100)))
        records = sorted(records, key=lambda r: r.get("ts", 0), reverse=True)
        return [
            HistoryMessage(
                message_id=str(r.get("_id", "")),
                role=r.get("role", ""),
                text=r.get("text", ""),
                ts=r.get("ts"),
            )
            for r in records
        ]

    @app.post("/query/history/clear", tags=["query"])
    async def history_clear(payload: dict):
        """清空会话历史。请求体: {"session_id": "..."}"""
        session_id = (payload or {}).get("session_id", "")
        if not session_id:
            return JSONResponse(status_code=400, content={"detail": "session_id 不能为空"})
        deleted = clear_history(session_id)
        return JSONResponse({"session_id": session_id, "deleted": deleted})
