"""掌柜智库 · 教育版 —— FastAPI 总入口。

与模板差异：模板在 knowledge/api/import_router.py 内自带 create_app 并直接
uvicorn 启动；教育版采用「代码平铺在仓库根」的布局，因此将应用装配统一收敛到
根目录的 main.py：
    * 注册 api.import_router 与 api.query_router 两组路由
    * 跨域中间件
    * 挂载 front 前端静态资源
    * 健康检查

运行:
    python main.py
    # 或 uvicorn main:app --reload --port 8000
"""
import os

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles

from api.import_router import register_router as register_import_router
from api.query_router import register_router as register_query_router
from core.paths import get_front_page_dir


def create_app() -> FastAPI:
    """装配 FastAPI 应用(占位骨架阶段即可启动)。"""
    app = FastAPI(
        title="掌柜智库 · 教育版",
        description="尚硅谷实战项目:掌柜智库教育版(RAG 知识问答),整体框架占位阶段",
        version="0.1.0",
    )

    # 路由(各端点当前为占位实现,按模块逐步补齐)
    register_import_router(app)
    register_query_router(app)

    # 跨域
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["system"])
    def health() -> JSONResponse:
        """健康检查。"""
        return JSONResponse({"status": "ok", "project": "atguigu_edu_keeper"})

    # 前端静态资源(front 目录不存在时跳过挂载)
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.isdir(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir), name="front")

    return app


app = create_app()


if __name__ == "__main__":
    # 开发期如需热重载可改用: uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    uvicorn.run(app=app, host="0.0.0.0", port=8000)
