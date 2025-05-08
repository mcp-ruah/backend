from fastapi import APIRouter, Request, HTTPException
from typing import Dict, Tuple, Any, Optional
from utils.logger import logger
from mcp_server import Server
from chat import ChatSession

router = APIRouter(prefix="/api", tags=["server"])

@router.get("/mcp-status")
async def get_mcp_status(request: Request):
    try:
        # 1. 설정 로드 확인
        logger.info("MCP 상태 조회 시작")
        server_config, chat_session = await _get_server_config(request)
        logger.info(f"서버 설정 로드됨: {server_config}")

        # 2. chat_session 상태 확인
        if not chat_session:
            logger.error("ChatSession이 None입니다")
            return {"status": "error", "message": "ChatSession이 초기화되지 않았습니다"}

        # 3. 실행 중인 서버 상태 확인
        try:
            running_servers_status = await chat_session.get_servers_status()
            logger.info(f"실행 중인 서버 상태: {running_servers_status}")

            running_server_names = [server["name"] for server in running_servers_status]
            logger.info(f"실행 중인 서버 이름: {running_server_names}")

            # 4. 도커 컨테이너 상태 확인
            running_containers = await Server.get_running_containers()
            logger.info(f"실행 중인 도커 컨테이너: {running_containers}")

            # 5. 전체 서버 상태 목록 생성
            all_servers_status = await Server.get_server_status_list(
                server_config["mcpServers"],
                running_servers_status,
                running_server_names,
            )
            logger.info(f"전체 서버 상태: {all_servers_status}")

            return {
                "status": "ok",
                "servers_count": len(all_servers_status),
                "servers": all_servers_status,
            }
        except Exception as e:
            logger.error(f"서버 상태 조회 중 오류: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "servers_count": 0,
                "servers": [],
                "message": f"서버 상태 조회 중 오류가 발생했습니다: {str(e)}",
            }
    except Exception as e:
        logger.error(f"MCP 상태 조회 중 최상위 오류: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "message": "MCP 상태 조회 중 오류가 발생했습니다",
        }

@router.post("/server/{server_name}/start")
async def start_server(server_name: str, request: Request):
    return await _manage_server("start", server_name, request)


@router.post("/server/{server_name}/stop")
async def stop_server(server_name: str, request: Request):
    return await _manage_server("stop", server_name, request)


@router.post("/server/{server_name}/restart")
async def restart_server(server_name: str, request: Request):
    return await _manage_server("restart", server_name, request)


####################################################################################


class ServerError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=500, detail=f"서버 오류: {detail}")


async def _get_server_config(
    request: Request, server_name: Optional[str] = None
) -> Tuple[Dict, ChatSession]:  # type: ignore
    chat_session = request.app.state.chat_session
    if not chat_session:
        logger.error("MCP 서버가 초기화되지 않았습니다.")
        raise HTTPException(status_code=503, detail="MCP 서버가 초기화되지 않았습니다.")

    try:
        config = request.app.state.config.load_config("mcp_servers.json")
        if server_name and server_name not in config["mcpServers"]:
            logger.error(f"서버 '{server_name}'을(를) 찾을 수 없습니다.")
            raise HTTPException(
                status_code=404, detail=f"서버 '{server_name}'을(를) 찾을 수 없습니다."
            )
        return config, chat_session
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {str(e)}")
        raise ServerError(f"설정 파일 로드 실패: {str(e)}")


async def _manage_server(action: str, server_name: str, request: Request) -> Dict:
    server_config, chat_session = await _get_server_config(request, server_name)

    try:
        target_server = next(
            (s for s in chat_session.servers if s.name == server_name), None
        )
        container_name = Server.get_container_name_from_config(
            server_name, server_config["mcpServers"][server_name]
        )

        if action == "start":
            if target_server and target_server.session:
                logger.info(f"서버 '{server_name}'이(가) 이미 실행 중입니다.")
                return {
                    "status": "already_running",
                    "name": server_name,
                    "message": f"서버 '{server_name}'은(는) 이미 실행 중입니다.",
                }

            new_server = Server(server_name, server_config["mcpServers"][server_name])
            await new_server.initialize()
            chat_session.servers.append(new_server)
            logger.info(f"서버 '{server_name}'이(가) 성공적으로 시작되었습니다.")
            return {
                "status": "success",
                "name": server_name,
                "message": f"서버 '{server_name}'이(가) 성공적으로 시작되었습니다.",
            }

        elif action in ["stop", "restart"]:
            if target_server:
                await target_server.cleanup()
                chat_session.servers.remove(target_server)
            elif container_name:
                running_containers = await Server.get_running_containers()
                if container_name in running_containers:
                    if not await Server.stop_container(container_name):
                        logger.error(f"도커 컨테이너 '{container_name}' 종료 실패")
                        raise ServerError(f"도커 컨테이너 '{container_name}' 종료 실패")
                elif action == "stop":
                    logger.info(f"서버 '{server_name}'이(가) 실행 중이 아닙니다.")
                    return {
                        "status": "not_running",
                        "name": server_name,
                        "message": f"서버 '{server_name}'이(가) 실행 중이 아닙니다.",
                    }

            if action == "restart":
                new_server = Server(
                    server_name, server_config["mcpServers"][server_name]
                )
                await new_server.initialize()
                chat_session.servers.append(new_server)
                logger.info(f"서버 '{server_name}'이(가) 성공적으로 재시작되었습니다.")
                return {
                    "status": "success",
                    "name": server_name,
                    "message": f"서버 '{server_name}'이(가) 성공적으로 재시작되었습니다.",
                }
            logger.info(f"서버 '{server_name}'이(가) 성공적으로 종료되었습니다.")
            return {
                "status": "success",
                "name": server_name,
                "message": f"서버 '{server_name}'이(가) 성공적으로 종료되었습니다.",
            }

    except Exception as e:
        logger.error(f"서버 '{server_name}' {action} 중 오류: {str(e)}", exc_info=True)
        raise ServerError(str(e))
