from fastapi import APIRouter, Request, HTTPException
from typing import Dict, Tuple, Any, Optional
from utils.logger import logger
from mcp_server import Server
from chat import ChatSession

router = APIRouter(prefix="/api", tags=["server"])


async def _get_server_config(
    request: Request, server_name: Optional[str] = None
) -> Tuple[Dict, Any]:
    """
    서버 설정을 로드하고 chat_session을 확인하는 공통 함수

    Args:
        request (Request): FastAPI 요청 객체
        server_name (str, optional): 서버 이름

    Returns:
        Tuple[Dict, Any]: 서버 설정과 chat_session 객체

    Raises:
        HTTPException: 설정 로드 실패 또는 서버를 찾을 수 없는 경우
    """
    chat_session = request.app.state.chat_session
    config = request.app.state.config

    if not chat_session:
        raise HTTPException(
            status_code=503,
            detail="MCP 서버가 초기화되지 않았습니다. 서버를 다시 시작해주세요.",
        )

    # 서버 설정 파일 로드
    try:
        server_config = config.load_config("mcp_servers.json")
    except Exception as e:
        logger.error(f"설정 파일 로드 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"서버 설정 파일 로드 중 오류가 발생했습니다: {str(e)}",
        )

    # 서버 이름 검증 (지정된 경우)
    if server_name and server_name not in server_config["mcpServers"]:
        raise HTTPException(
            status_code=404, detail=f"서버 '{server_name}'을(를) 찾을 수 없습니다."
        )

    return server_config, chat_session


async def _find_server_in_session(
    chat_session, server_name: str
) -> Tuple[Optional[Server], Optional[int]]:
    """
    chat_session에서 서버 인스턴스를 찾는 공통 함수

    Args:
        chat_session: ChatSession 인스턴스
        server_name (str): 찾을 서버 이름

    Returns:
        Tuple[Optional[Server], Optional[int]]: 찾은 서버와 인덱스, 없으면 (None, None)
    """
    for idx, server in enumerate(chat_session.servers):
        if server.name == server_name:
            return server, idx
    return None, None


@router.get("/mcp-status")
async def get_mcp_status(request: Request):
    """
    MCP 서버 상태 확인 API 엔드포인트

    mcp_servers.json에 정의된 모든 서버와 그 실행 상태를 반환

    Returns:
        Dict: MCP 서버들의 상태 정보
    """
    server_config, chat_session = await _get_server_config(request)

    try:
        # 실행 중인 서버들의 상태 가져오기
        running_servers_status = await chat_session.get_servers_status()
        running_server_names = [server["name"] for server in running_servers_status]

        # 서버 상태 목록 가져오기
        all_servers_status = await Server.get_server_status_list(
            server_config["mcpServers"], running_servers_status, running_server_names
        )

        return {
            "status": "ok",
            "servers_count": len(all_servers_status),
            "servers": all_servers_status,
        }
    except Exception as e:
        logger.error(f"MCP 상태 확인 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"MCP 상태 확인 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/server/{server_name}/start")
async def start_server(server_name: str, request: Request):
    """
    특정 MCP 서버를 시작하는 API 엔드포인트

    Args:
        server_name (str): 시작할 서버 이름

    Returns:
        Dict: 서버 시작 결과 정보
    """
    server_config, chat_session = await _get_server_config(request, server_name)

    # 서버가 이미 시작되었는지 확인
    target_server, _ = await _find_server_in_session(chat_session, server_name)
    if target_server and target_server.session:
        return {
            "status": "already_running",
            "name": server_name,
            "message": f"서버 '{server_name}'은(는) 이미 실행 중입니다.",
        }

    # 새 서버 인스턴스 생성 및 초기화
    try:
        new_server = Server(server_name, server_config["mcpServers"][server_name])
        await new_server.initialize()

        # 새 서버 추가
        chat_session.servers.append(new_server)

        return {
            "status": "success",
            "name": server_name,
            "message": f"서버 '{server_name}'이(가) 성공적으로 시작되었습니다.",
        }
    except Exception as e:
        logger.error(f"서버 '{server_name}' 시작 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"서버 시작 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/server/{server_name}/stop")
async def stop_server(server_name: str, request: Request):
    """
    특정 MCP 서버를 종료하는 API 엔드포인트

    Args:
        server_name (str): 종료할 서버 이름

    Returns:
        Dict: 서버 종료 결과 정보
    """
    server_config, chat_session = await _get_server_config(request, server_name)

    # 해당 서버 찾기
    target_server, target_index = await _find_server_in_session(
        chat_session, server_name
    )

    # ChatSession에 서버가 있으면 정상 종료
    if target_server:
        try:
            await target_server.cleanup()
            # 서버 목록에서 제거
            chat_session.servers.pop(target_index)

            return {
                "status": "success",
                "name": server_name,
                "message": f"서버 '{server_name}'이(가) 성공적으로 종료되었습니다.",
            }
        except Exception as e:
            logger.error(f"서버 '{server_name}' 종료 중 오류: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"서버 종료 중 오류가 발생했습니다: {str(e)}"
            )
    # ChatSession에 서버가 없지만 외부 도커 컨테이너가 있을 수 있음
    else:
        # 컨테이너 이름 확인
        srv_config = server_config["mcpServers"][server_name]
        container_name = Server.get_container_name_from_config(server_name, srv_config)

        if container_name:
            try:
                # 실행 중인 컨테이너 확인
                running_containers = await Server.get_running_containers()

                if container_name in running_containers:
                    # 컨테이너 중지
                    success = await Server.stop_container(container_name)
                    if success:
                        return {
                            "status": "success",
                            "name": server_name,
                            "message": f"외부 도커 컨테이너 '{container_name}'이(가) 성공적으로 종료되었습니다.",
                        }
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail=f"도커 컨테이너 '{container_name}' 종료 중 오류가 발생했습니다.",
                        )
                else:
                    return {
                        "status": "not_running",
                        "name": server_name,
                        "message": f"서버 '{server_name}'이(가) 실행 중이 아닙니다.",
                    }
            except Exception as e:
                logger.error(
                    f"서버 '{server_name}' 종료 중 오류: {str(e)}", exc_info=True
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"서버 종료 중 오류가 발생했습니다: {str(e)}",
                )
        else:
            return {
                "status": "not_running",
                "name": server_name,
                "message": f"서버 '{server_name}'이(가) 실행 중이 아닙니다.",
            }


@router.post("/server/{server_name}/restart")
async def restart_server(server_name: str, request: Request):
    """
    특정 MCP 서버를 재시작하는 API 엔드포인트

    Args:
        server_name (str): 재시작할 서버 이름

    Returns:
        Dict: 서버 재시작 결과 정보
    """
    server_config, chat_session = await _get_server_config(request, server_name)

    # 실행 중인 도커 컨테이너 확인
    container_name = Server.get_container_name_from_config(
        server_name, server_config["mcpServers"][server_name]
    )
    running_containers = await Server.get_running_containers()

    # 기존 서버 찾기 및 종료
    target_server, target_index = await _find_server_in_session(
        chat_session, server_name
    )

    # ChatSession에 있는 서버 정리
    if target_server:
        try:
            await target_server.cleanup()
            chat_session.servers.pop(target_index)
        except Exception as e:
            logger.error(f"서버 '{server_name}' 종료 중 오류: {str(e)}", exc_info=True)
            # 종료 오류는 기록만 하고 계속 진행 (새로 시작 시도)
    # 외부 도커 컨테이너 정리
    elif container_name in running_containers:
        try:
            await Server.stop_container(container_name)
        except Exception as e:
            logger.error(
                f"외부 컨테이너 '{container_name}' 종료 중 오류: {str(e)}",
                exc_info=True,
            )
            # 종료 오류는 기록만 하고 계속 진행

    # 새 서버 인스턴스 생성 및 초기화
    try:
        new_server = Server(server_name, server_config["mcpServers"][server_name])
        await new_server.initialize()
        chat_session.servers.append(new_server)

        return {
            "status": "success",
            "name": server_name,
            "message": f"서버 '{server_name}'이(가) 성공적으로 재시작되었습니다.",
        }
    except Exception as e:
        logger.error(f"서버 '{server_name}' 재시작 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"서버 재시작 중 오류가 발생했습니다: {str(e)}"
        )
