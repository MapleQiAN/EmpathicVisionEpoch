from fastapi import FastAPI

app = FastAPI(title="EmpathicVisionEpoch")

@app.get("/health")
async def health():
    return {"status": "ok"}