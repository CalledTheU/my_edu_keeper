import os
import sys

import uvicorn
from fastapi import UploadFile, File, Depends, BackgroundTasks
from starlette.responses import FileResponse, JSONResponse

# 兜底把仓库根加入 sys.path：命令行 `python api/import_router.py` 时 sys.path[0]
# 是 api 目录，core / services / utils 等包会导入失败（PyCharm 默认勾选了
# Add content roots to PYTHONPATH，所以在 IDE 里运行时不暴露该问题）。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.deps import get_import_file_service  # noqa: E402
from core.paths import get_front_page_dir  # noqa: E402
from schema.upload_schema import UploadResponse, TaskStatusResponse  # noqa: E402
from services.file_import_service import ImportFileService  # noqa: E402
from utils.task_util import get_task_info, add_running_task  # noqa: E402


def register_router(app):
    @app.get("/import")
    def import_page():
        return FileResponse(path=os.path.join(get_front_page_dir(), "import.html"))

    """
        1.文件上传处理  由业务层来处理，需要创建业务层类，及对象。采用依赖注入的方式创建业务层对象，并且单例创建，并且缓存重复利用
        2.异步启动Langgraph流程
    """

    @app.post("/upload", response_model=UploadResponse)
    async def upload_file(
            background_tasks: BackgroundTasks,
            service: ImportFileService = Depends(get_import_file_service),  # 依赖注入
            file: UploadFile = File(...)):
        """POST 请求处理 —— 接收上传文件"""
        # 文件上传（双写）
        # 任务ID（每次请求创建唯一字符串值 - 用于查询任务状态）
        # 文件上传目录
        # 文件上传后的完整路径
        task_id, file_dir, import_file_path = service.upload_file(file)

        # 异步启动流程
        background_tasks.add_task(service.run_import_graph, task_id, file_dir, import_file_path)
        # 这里是同步启动流程,用户需要等待很长时间
        # service.run_import_graph(task_id, file_dir, import_file_path)

        # 返回结果数据模型： Pydantic 模型
        return UploadResponse(task_id=task_id, message="文件上传成功!")

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_status(task_id: str):
        """GET 获取任务状态
            1.在文件上传及启动langgraph时，都需要记录各个任务的状态
            2.前端根据task_id，查询任务状态
            3.状态维护容器：
        """
        task_info = get_task_info(task_id)
        if not task_info["status"] and not task_info["done_list"] and not task_info["running_list"]:
            return JSONResponse(status_code=404, content={"detail": "任务不存在"})
        return TaskStatusResponse(**task_info)


def create_app():
    """装配【完整】应用（知识库导入 + 智能问答）。

    注意：本函数历史实现只注册导入路由，如果用 `python api/import_router.py`
    启动服务，/import 与 /upload 正常，但 /query 页面和 /query/ask 都会 404
    （问答路由压根没注册，与前端无关）。这里统一委托根目录的 main.create_app()，
    保证无论从 main.py 还是 api/import_router.py 启动，装配出的应用完全一致。
    """
    from main import create_app as build_app  # 延迟导入：避免与 main.py 循环依赖
    return build_app()


if __name__ == '__main__':
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8000)
