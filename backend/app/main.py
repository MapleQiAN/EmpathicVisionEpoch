from fastapi import FastAPI

from app.api.graph import router as graph_router
from app.api.map import router as map_router

app = FastAPI(title="EmpathicVisionEpoch")
app.include_router(graph_router)
app.include_router(map_router)

@app.get("/health")
async def health():
    return {"status": "ok"}
