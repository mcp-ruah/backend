import sys
import uvicorn
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request, Body, Cookie, Header, Query
from fastapi.responses import StreamingResponse
from typing import List, Optional, AsyncGenerator, Dict
import uuid
from pydantic import BaseModel
from schema import ChatRequest
from chat import ChatSession
from utils.logger import logger
import json
import subprocess
from lifecycle import lifespan
from llms import get_llm_client


# Windows에서 ProactorEventLoop 설정
if sys.platform == "win32":
    from asyncio import WindowsSelectorEventLoopPolicy, WindowsProactorEventLoopPolicy

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
else:
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# 세션 임시 저장
sessions: Dict[str, List[str]] = {}

app = FastAPI(lifespan=lifespan)


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
async def api_chat(request: Request, chat_req: ChatRequest):
    """
    채팅 API 엔드포인트
    """
    global sessions
    session_id = chat_req.session_id or str(uuid.uuid4())
    message = chat_req.message

    config = request.app.state.config
    if session_id not in sessions:
        logger.info(f"세션 ID {session_id}에 대한 새 대화 기록 생성")
        sessions[session_id] = []

    # 사용자가 선택한 모델명에 따라 LLM 클라이언트 생성
    llm_client = get_llm_client(
        model=chat_req.model,
        temperature=chat_req.temperature,
        max_tokens=chat_req.max_tokens,
        config=config,
    )

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
        async for chunk in chat_session.chat(message, session_id, sessions):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/api/sessions")
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


@app.get("/api/mcp-status")
async def get_mcp_status(request: Request):
    """
    MCP 서버 상태 확인 API 엔드포인트

    Returns:
        Dict: MCP 서버들의 상태 정보
    """
    chat_session = request.app.state.chat_session

    if not chat_session:
        raise HTTPException(
            status_code=503,
            detail="MCP 서버가 초기화되지 않았습니다. 서버를 다시 시작해주세요.",
        )

    try:
        servers_status = await chat_session.get_servers_status()
        return {
            "status": "ok",
            "servers_count": len(servers_status),
            "servers": servers_status,
        }
    except Exception as e:
        logger.error(f"MCP 상태 확인 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"MCP 상태 확인 중 오류가 발생했습니다: {str(e)}"
        )


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
