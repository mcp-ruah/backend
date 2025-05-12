import sys
import uvicorn
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from typing import Dict, List
from utils import logger
from chat import ChatSession
from contextlib import asynccontextmanager
from config import Configuration
from mcp_server.mcp_server import Server

# 라우터 임포트
from routes import chat_router, server_router, session_router, test_router

# Windows에서 이벤트 루프 설정
if sys.platform == "win32":
    # ProactorEventLoop를 기본으로 사용
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = Configuration()
    app.state.config = config

    try:
        logger.info(f"config file load start: mcp_servers.json")
        server_config = config.load_config("mcp_servers.json")
        logger.info("config file load success")

        # 빈 서버 리스트로 ChatSession 생성
        chat_session = ChatSession([])
        app.state.chat_session = chat_session

        yield
    except Exception as e:
        logger.error(f"start event failed: {e}")
        raise

    # # 종료
    # try:
    #     # await app.state.chat_session.cleanup_servers()
    #     logger.info("MCP server cleanup success")
    # except asyncio.CancelledError as ce:
    #     logger.info(f"작업이 취소되었습니다")
    # except Exception as e:
    #     logger.error(f"server cleanup failed: {e}")


app = FastAPI(lifespan=lifespan)

# 라우터 등록
app.include_router(chat_router)
app.include_router(server_router)
app.include_router(session_router)
app.include_router(test_router)

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
