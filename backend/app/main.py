from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.api.graph import router as graph_router
from app.api.map import router as map_router

app = FastAPI(title="EmpathicVisionEpoch")
app.include_router(graph_router)
app.include_router(map_router)

# 配置静态文件服务
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def index():
    """返回调试UI页面"""
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "调试UI未找到，请确保frontend/index.html存在"}

@app.get("/health")
async def health():
    return {"status": "ok"}
