from typing import Optional
from fastapi import UploadFile
from pydantic import BaseModel
from config import LLMModel
from fastapi import UploadFile, File
from fastapi import Form

# 채팅 요청 모델
class ChatRequest(BaseModel):
    session_id: str = Form(None)
    message: str = Form(...)
    model: str = Form(LLMModel.CLAUDE_3_7_SONNET.value)
    temperature: float = Form(LLMModel.TEMPERATURE.value)
    max_tokens: int = Form(LLMModel.MAX_TOKENS.value)
