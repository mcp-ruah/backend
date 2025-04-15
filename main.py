import sys
import uvicorn
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Body, Cookie, Header, Query
from typing import List, Optional, AsyncGenerator, Dict
import uuid
from pydantic import BaseModel
from schema import ChatRequest
from server import initialize_mcp_servers, ChatSession
from contextlib import asynccontextmanager
from utils.logger import logger


# Windows에서 ProactorEventLoop 설정
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# 전역 변수 (lifespan 외부에서 접근 가능하도록)
chat_session: Optional[ChatSession] = None
sessions: Dict[str, List[str]] = {}


# lifespan context manager
@asynccontextmanager
async def lifspan(app: FastAPI):
    """
    Application Lifespan Event Handler
    시작시 실행되는 코드는 yield 전에 실행
    종료시 실행되는 코드는 yield 후에 배치
    """
    global chat_session
    try:
        # MCP 서버 및 채팅 세션 초기화
        chat_session = await initialize_mcp_servers()
        if not chat_session:
            logger.error("채팅 세션 초기화 실패")
        else:
            logger.info("mcp 서버 초기화 완료")
    except Exception as e:
        logger.error(f"시작 이벤트 중 오류 : {e}")

    yield  # 이 지점에서 FastAPI 어플리케이션 실행

    # 종료시
    if chat_session:
        try:
            await chat_session.cleanup_servers()
            logger.info("MCP 서버 정리 완료")
        except Exception as e:
            logger.error(f"서버 정리 중 오류 : {e}")


app = FastAPI(lifespan=lifspan)


# CORS 설정
origins = [
    "http://localhost:5173",  # 프론트엔드 개발 서버 주소 (예: React)
    "http://127.0.0.1:5173",  # localhost 대체 주소
    # 필요한 경우 추가 도메인 설정
]


app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,  # 접근 허용할 출처 목록
    allow_origins=["*"],  # 접근 허용할 출처 목록
    allow_credentials=True,  # 쿠키 포함 요청 허용 여부
    allow_methods=["*"],  # 허용할 HTTP 메서드 목록
    allow_headers=["*"],  # 허용할 HTTP 헤더 목록
)


@app.post("/api/chat")
async def api_chat(request_data: ChatRequest):
    """
    채팅 API 엔드포인트
    """
    global chat_session, sessions

    # 세션 ID 확인/생성
    session_id = request_data.session_id
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())

    # 채팅 세션 확인
    if not chat_session:
        return {
            "response": "시스템이 초기화되지 않았습니다. 나중에 다시 시도해 주세요.",
            "session_id": session_id,
            "confidence": 0.0,
        }

    try:
        # 채팅 응답 처리
        response = await chat_session.chat(request_data.message, session_id, sessions)

        return {"response": response, "session_id": session_id, "confidence": 0.95}
    except Exception as e:
        logger.error(f"채팅 처리 중 오류: {str(e)}")
        return {
            "response": f"죄송합니다, 오류가 발생했습니다: {str(e)}",
            "session_id": session_id,
            "confidence": 0.1,
        }


@app.post("/api/reset-chat")
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


@app.get("/")
async def root():
    return {"message": "Hello World"}
