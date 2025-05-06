import sys
import uvicorn
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from typing import Dict, List
from utils.logger import logger
from chat import ChatSession
from contextlib import asynccontextmanager
from config import Configuration


# 라우터 임포트
from routes import chat_router, server_router, session_router

# Windows에서만 ProactorEventLoop 설정
if sys.platform == "win32":
    from asyncio import WindowsSelectorEventLoopPolicy, WindowsProactorEventLoopPolicy

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Event Handler
    시작시 실행되는 코드는 yield 전에 실행
    종료시 실행되는 코드는 yield 후에 배치
    """
    config = Configuration()
    app.state.config = config

    try:
        logger.info(f"config file load start: mcp_servers.json")
        server_config = config.load_config("mcp_servers.json")
        logger.info("config file load success")

        # 빈 서버 리스트로 ChatSession 생성 (초기에는 어떤 서버도 시작하지 않음)
        chat_session = ChatSession([])
        app.state.chat_session = chat_session
        logger.info(
            "빈 MCP ChatSession 생성 완료 - 서버는 API를 통해 개별적으로 시작해야 합니다"
        )

        yield
    except Exception as e:
        logger.error(f"start event failed: {e}")
        app.state.chat_session = None
        yield

    # 종료시 실행 중인 서버들 정리
    if app.state.chat_session and app.state.chat_session.servers:
        try:
            await app.state.chat_session.cleanup_servers()
            logger.info("MCP server cleanup success")
        except Exception as e:
            logger.error(f"server cleanup failed: {e}")


app = FastAPI(lifespan=lifespan)

# 라우터 등록
app.include_router(chat_router)
app.include_router(server_router)
app.include_router(session_router)

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


@app.get("/")
async def root():
    return {"message": "Hello World"}
