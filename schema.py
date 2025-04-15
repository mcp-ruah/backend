from typing import Optional
from pydantic import BaseModel


# 채팅 요청 모델
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
