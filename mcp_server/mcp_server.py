import shutil, uuid, httpx, json, os, asyncio, subprocess
from dataclasses import dataclass, field
from contextlib import AsyncExitStack
from typing import Optional, Dict, Any, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from utils.logger import logger


@dataclass
class Server:
    """MCP 서버 연결 및 도구 실행 관리"""

    name: str
    config: Dict[str, Any]
    stdio_context: Optional[Any] = None
    session: Optional[ClientSession] = None
    _cleanup_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    exit_stack: AsyncExitStack = field(default_factory=AsyncExitStack)

    @staticmethod
    async def get_running_containers() -> List[str]:
        """실행 중인 도커 컨테이너 이름 목록 반환"""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                "ps",
                "--format",
                "{{.Names}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"도커 명령어 실행 실패: {stderr.decode()}")
                raise Exception(f"도커 명령어 실행 실패: {stderr.decode()}")

            output = stdout.decode().strip()
            return output.split("\n") if output else []
        except Exception as e:
            logger.error(f"도커 컨테이너 상태 확인 중 오류: {str(e)}")
            raise  # 에러를 상위로 전파

    @staticmethod
    async def stop_container(container_name: str) -> bool:
        """도커 컨테이너 종료"""
        try:
            result = subprocess.run(
                ["docker", "stop", container_name],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(f"도커 컨테이너 '{container_name}' 종료 성공")
            return True
        except Exception as e:
            logger.error(f"도커 컨테이너 '{container_name}' 종료 중 오류: {str(e)}")
            return False

    @staticmethod
    def get_container_name_from_config(server_name: str, config: Dict[str, Any]) -> str:
        """서버 설정에서 컨테이너 이름 추출"""
        if config.get("command") == "docker" and config.get("args"):
            args = config.get("args", [])
            if "--name" in args:
                idx = args.index("--name")
                if idx + 1 < len(args):
                    return args[idx + 1]
        # 기본 이름 형식 반환
        return f"mcp-{server_name}"

    @staticmethod
    async def get_server_status_list(
        server_configs: Dict[str, Dict[str, Any]],
        running_servers_status: List[Dict[str, Any]],
        running_server_names: List[str],
    ) -> List[Dict[str, Any]]:
        """모든 서버의 상태 목록 생성"""
        all_servers_status = []
        running_containers = await Server.get_running_containers()

        for server_name, srv_config in server_configs.items():
            # 컨테이너 이름 확인
            container_name = Server.get_container_name_from_config(
                server_name, srv_config
            )

            # ChatSession에 있는 실행 중인 서버라면 상세 정보 포함
            if server_name in running_server_names:
                for running_server in running_servers_status:
                    if running_server["name"] == server_name:
                        all_servers_status.append(running_server)
                        break
            # 도커에는 실행 중이지만 ChatSession에 없는 경우
            elif container_name in running_containers:
                all_servers_status.append(
                    {
                        "name": server_name,
                        "initialized": False,
                        "status": "external_running",  # 외부에서 실행 중이지만 ChatSession에 연결되지 않음
                        "container_name": container_name,
                        "config": {
                            "command": srv_config.get("command", ""),
                        },
                        "message": "도커에서 실행 중이지만 백엔드에 연결되지 않았습니다. 재시작하세요.",
                    }
                )
            # 실행 중이 아니라면 기본 정보만 포함
            else:
                all_servers_status.append(
                    {
                        "name": server_name,
                        "initialized": False,
                        "status": "stopped",
                        "config": {
                            "command": srv_config.get("command", ""),
                        },
                    }
                )

        return all_servers_status

    async def initialize(self) -> None:
        """서버 초기화 및 세션 설정"""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError(f"Command {self.config['command']} not found")

        # Docker 실행 시 환경 변수를 명시적으로 추가
        if self.config["command"] == "docker" and self.config.get("env"):
            modified_args = self.config["args"].copy()
            env_args = []
            for key, value in self.config["env"].items():
                env_args.extend(["-e", f"{key}={value}"])

            # Docker run 명령 다음에 환경 변수 옵션 삽입
            run_index = modified_args.index("run")
            modified_args[run_index + 1 : run_index + 1] = env_args

            server_params = StdioServerParameters(
                command=command,
                args=modified_args,
                env=None,  # Docker에 직접 전달하므로 여기서는 필요 없음
            )
        else:
            server_params = StdioServerParameters(
                command=command,
                args=self.config["args"],
                env=self.config.get("env"),
            )

        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logger.error(f"Error initializing server {self.name}: {e}", exc_info=True)
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """서버로부터 사용가능한 도구 리스트 반환

        Returns :
            도구 리스트

        Raises:
            RuntimeError : 서버 초기화 실패 또는 도구 목록 가져오기 실패
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":

                for tool in item[1]:
                    tools.append(
                        Tool(
                            tool.name,
                            tool.description,
                            tool.inputSchema,
                        )
                    )

        return tools

    async def get_status(self) -> Dict[str, Any]:
        """서버 상태 정보 반환

        Returns:
            Dict[str, Any]: 서버 상태 정보를 포함한 딕셔너리
        """
        status = {
            "name": self.name,
            "initialized": self.session is not None,
            "status": "running" if self.session is not None else "stopped",
            "config": {
                "command": self.config.get("command", ""),
            },
        }

        # 초기화된 경우 추가 정보
        if self.session:
            try:
                tools = await self.list_tools()
                status["tools_count"] = len(tools)
                status["tools"] = [
                    {"name": tool.name, "description": tool.description}
                    for tool in tools
                ]
            except Exception as e:
                status["error"] = str(e)

        return status

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """재시도 매커니즘을 포함한 도구 실행

        Args:
            tool_name : 실행할 도구 이름
            arguments : 도구 인자
            retries : 실행 재시도 횟수
            delay : 재시도 간격(초)

        Returns:
            도구 실행 결과

        Raises:
            RuntimeError : 서버 초기화 실패
            Exception : 모든 재시도 도구 실행 오류
        """
        if not self.session:
            raise RuntimeError(f"{self.name} 서버 초기화 실패")

        attempt = 0
        while attempt < retries:
            try:
                logger.info(f"도구 실행 중... {tool_name} ")
                result = await self.session.call_tool(tool_name, arguments)

                return result
            except Exception as e:
                attempt += 1
                logger.warning(
                    f"도구 실행 오류 : {e}. 재시도 {attempt} 회 / {retries} 회"
                )
                if attempt < retries:
                    logger.info(f"재시도 중... {delay} 초 대기")
                    await asyncio.sleep(delay)
                else:
                    logger.error("재시도 횟수 초과. 실패")
                    raise

    async def cleanup(self) -> None:
        """서버 세션 정리"""
        async with self._cleanup_lock:
            try:
                # MCP 서버 종료
                if self.config.get("command") == "docker":
                    args = self.config.get("args", [])
                    container_name = self.name  # 기본 이름은 서버 이름 그대로 사용
                    if "--name" in args:
                        idx = args.index("--name")
                        if idx + 1 < len(args):
                            container_name = args[idx + 1]
                    logger.info(f"도커 컨테이너 종료 시도: {container_name}")
                    result = subprocess.run(
                        ["docker", "stop", container_name],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        logger.error(
                            f"도커 컨테이너 종료 실패: {container_name} - {result.stderr.strip()}"
                        )
                    else:
                        logger.info(f"도커 컨테이너 종료 성공: {container_name}")
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logger.error(f"{self.name} 서버 Clean up 중 오류 발생 : {e}")

                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logger.error(f"{self.name} 서버 Clean up 중 오류 발생 : {e}")


@dataclass
class Tool:
    """도구 정보를 나타내고 LLM에 대한 포맷을 관리"""

    name: str
    description: str
    input_schema: Dict[str, Any]

    def format_for_llm(self) -> str:
        """LLM을 위한 도구 정보 포맷"""
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"{param_name} : {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""
