from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx, os
from fastapi.staticfiles import StaticFiles

INFER_URL = os.getenv("INFER_URL", "http://vllm:8000/v1/chat/completions")
MODEL_ID  = os.getenv("MODEL_ID",  "merged-llama3")
TIMEOUT   = 30

app = FastAPI(title="LLM Proxy")

class ChatReq(BaseModel):
    prompt: str
    max_tokens: int = 256

@app.post("/chat")
async def chat(req: ChatReq):
    payload = {
        "model": MODEL_ID,
        "messages":[{"role":"user","content":req.prompt}],
        "max_tokens": req.max_tokens,
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.post(INFER_URL, json=payload)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

app.mount("/", StaticFiles(directory="static", html=True), name="static")
