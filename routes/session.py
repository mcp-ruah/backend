from fastapi import APIRouter
from typing import Dict, List
from utils.logger import logger

router = APIRouter(prefix="/api", tags=["session"])

# 세션 임시 저장
sessions: Dict[str, List[str]] = {}


@router.get("/sessions")
async def get_sessions():
    """
    디버깅용: 현재 활성화된 세션 목록 반환
    """
    global sessions

    session_info = {}
    for session_id, messages in sessions.items():
        session_info[session_id] = {
            "message_count": len(messages),
            "first_message": messages[0] if messages else None,
            "last_message": messages[-1] if messages else None,
        }
    return {"total_sessions": len(sessions), "sessions": session_info}
