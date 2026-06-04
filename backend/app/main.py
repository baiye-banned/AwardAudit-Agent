from uuid import uuid4
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# 直接运行这个文件时，Python 不知道它属于 app 包。
# 先补上 backend 目录和包名，下面的相对导入才能正常工作。
if __name__ == "__main__" and __package__ is None:
    BACKEND_DIR = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(BACKEND_DIR))
    __package__ = "app"

from .services.agent import answer_question
from .services.data_tools import build_data_profile, read_table_file


app = FastAPI(title="智能报表分析 Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# v1 只支持单机演示，所以直接用内存保存上传后的 DataFrame。
# 如果服务重启，session 会消失；这符合当前“不上数据库”的边界。
DATASETS: dict[str, pd.DataFrame] = {}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    question: str
    history: list[ChatMessage] = []


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传 CSV/XLSX，并返回数据概览。"""

    try:
        file_bytes = await file.read()
        frame = read_table_file(file.filename or "", file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail="文件解析失败，请检查格式") from exc

    session_id = str(uuid4())
    DATASETS[session_id] = frame

    return {
        "session_id": session_id,
        "profile": build_data_profile(frame),
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    """根据用户问题，对已上传的数据进行分析。"""

    frame = DATASETS.get(request.session_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="数据会话不存在，请重新上传文件")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    history = [message.model_dump() for message in request.history]
    return answer_question(
        question=request.question.strip(),
        frame=frame,
        history=history,
    )


def run_server():
    """直接运行 `python backend/app/main.py` 时启动后端服务。"""

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8002,
        reload=False,
        app_dir="backend",
    )


if __name__ == "__main__":
    run_server()
