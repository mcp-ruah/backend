from fastapi import APIRouter, Request, Header, Depends
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, List, Any
import uuid
from schema import ChatRequest
from chat import ChatSession
from utils import logger
from llms import get_llm_client, get_client
from fastapi import File, UploadFile, Form
from backend.utils.convert_img import image_variation
from config import LLMModel

router = APIRouter(prefix="/api", tags=["chatbot"])

# 세션 임시 저장
sessions: Dict[str, List[str]] = {}


def _get_session_id(session_id: Optional[str] = None) -> str:
    """
    세션 ID를 가져오거나 새로 생성하는 공통 함수

    Args:
        session_id (Optional[str]): 기존 세션 ID

    Returns:
        str: 유효한 세션 ID
    """
    return session_id or str(uuid.uuid4())


def _ensure_session_exists(session_id: str) -> None:
    """
    세션이 존재하지 않으면 새로 생성하는 공통 함수

    Args:
        session_id (str): 세션 ID
    """
    global sessions
    if session_id not in sessions:
        logger.info(f"세션 ID {session_id}에 대한 새 대화 기록 생성")
        sessions[session_id] = []


@router.post("/chat")
async def api_chat(
    request: Request,
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    model: str = Form(LLMModel.GPT_4O.value),
    temperature: float = Form(LLMModel.TEMPERATURE.value),
    max_tokens: int = Form(LLMModel.MAX_TOKENS.value),
    file: UploadFile = File(None),
):
    """
    채팅 API 엔드포인트
    """
    global sessions
    session_id = _get_session_id(session_id)

    config = request.app.state.config
    _ensure_session_exists(session_id)

    # 사용자가 선택한 모델명에 따라 LLM 클라이언트 생성
    llm_client = get_llm_client(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        config=config,
    )
    logger.info(f"\n\nFILE : {file}\n\n")

    # 이미지 존재하면 -> IMgBB 이미지 변형 API 호출
    user_content = await llm_client.build_user_message(message, file)

    # 기존 chat_session 인스턴스를 새로 만듦 (서버 리스트는 기존대로)
    chat_session = ChatSession(
        servers=request.app.state.chat_session.servers, llm_client=llm_client
    )
    if not chat_session:
        return {
            "response": "시스템이 초기화되지 않았습니다. 나중에 다시 시도해 주세요.",
            "session_id": session_id,
            "confidence": 0.0,
        }

    async def generate():
        async for chunk in chat_session.chat(user_content, session_id, sessions):
            # 줄바꿈 문자 변환 없이 그대로 전달 (프론트엔드에서 처리)
            yield chunk

    return StreamingResponse(generate(), media_type="text/markdown")


@router.post("/reset-chat")
async def reset_chat(session_id: Optional[str] = Header(None)):
    """
    채팅 세션 초기화 API 엔드포인트
    """
    global sessions

    # 새 세션 생성
    new_session_id = str(uuid.uuid4())
    sessions[new_session_id] = []

    # 기존 세션 정리 (선택적)
    if session_id and session_id in sessions:
        del sessions[session_id]

    return {"message": "대화 기록이 초기화되었습니다.", "session_id": new_session_id}
