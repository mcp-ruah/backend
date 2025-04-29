from chat import ChatSession
from utils.logger import logger
from contextlib import asynccontextmanager
from fastapi import FastAPI
from mcp_server import Server
from config import Configuration
from typing import Optional

# chat_session = None
# sessions = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Event Handler
    시작시 실행되는 코드는 yield 전에 실행
    종료시 실행되는 코드는 yield 후에 배치
    """
    config = Configuration()
    app.state.config = config
    initialized_servers = []
    try:
        logger.info(f"config file load start: mcp_servers.json")
        server_config = config.load_config("mcp_servers.json")
        logger.info("config file load success")

        servers = [
            Server(name, srv_config)
            for name, srv_config in server_config["mcpServers"].items()
        ]
        logger.info(f"{len(servers)} servers instance create success")

        logger.info("LLM client create success")

        # initialize servers
        for server in servers:
            try:
                logger.info(f"{server.name} server initialize......")
                await server.initialize()
                initialized_servers.append(server)  # 초기화 성공한 서버 추가
                logger.info(f"{server.name} server initialize success")
            except Exception as e:
                logger.error(
                    f"{server.name} server initialize failed: {str(e)}", exc_info=True
                )

        if not initialized_servers:
            logger.error("all servers initialize failed")
            app.state.chat_session = None
        else:
            # create chat session
            chat_session = ChatSession(initialized_servers)
            app.state.chat_session = chat_session
            logger.info("mcp server initialize success")
        yield
    except Exception as e:
        logger.error(f"start event failed: {e}")
        app.state.chat_session = None
        yield
    # 종료시
    if app.state.chat_session:
        try:
            await app.state.chat_session.cleanup_servers()
            logger.info("MCP server cleanup success")
        except Exception as e:
            logger.error(f"server cleanup failed: {e}")
