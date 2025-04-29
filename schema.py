from typing import Optional
from pydantic import BaseModel
from config import LLMModel


# 채팅 요청 모델
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    model: str = LLMModel.CLAUDE_3_7_SONNET.value
    temperature: float = LLMModel.TEMPERATURE.value
    max_tokens: int = LLMModel.MAX_TOKENS.value
