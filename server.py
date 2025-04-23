import os
import json
import asyncio
import shutil
import uuid
import httpx
from dotenv import load_dotenv
from utils.logger import logger
from typing import Any, Optional, Dict, List, AsyncGenerator
from dataclasses import dataclass, field
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
from contextlib import AsyncExitStack


@dataclass
class Configuration:
    """MCP client 용 환경 변수와 설정 관리"""

    api_key: Optional[str] = None

    def __post_init__(self):
        """dataclass 초기화 후 추가 작업을 수행합니다."""
        self.load_env()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    @staticmethod
    def load_env() -> None:
        """환경변수를 .env 파일에서 로드"""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Json 파일에서 서버 설정 로드

        Args:
            file_path : Json 설정 파일 경로

        Raises:
            FileNotFoundError : 설정 파일이 존재하지 않는 경우
            JSONDecodeError : 설정 파일이 유효한 JSON이 아닌 경우
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def llm_api_key(self) -> str:
        """LLM API 키 반환

        Returns:
            문자열 API 키

        Raises:
            ValueError : API 키가 설정되지 않은 경우
        """
        if not self.api_key:
            raise ValueError("API 키가 설정되지 않았습니다.")
        return self.api_key


@dataclass
class Server:
    """MCP 서버 연결 및 도구 실행 관리"""

    name: str
    config: Dict[str, Any]
    stdio_context: Optional[Any] = None
    session: Optional[ClientSession] = None
    _cleanup_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    exit_stack: AsyncExitStack = field(default_factory=AsyncExitStack)

    async def initialize(self) -> None:
        """서버 초기화 및 세션 설정"""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError(f"Command {self.config['command']} not found")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env=(
                {**os.environ, **self.config["env"]} if self.config.get("env") else None
            ),
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
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))

        return tools

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


@dataclass
class LLMClient:
    """LLM 클라이언트 연결 및 요청 관리"""

    api_key: str

    async def get_response(
        self, messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """LLM에서 응답을 가져옴

        Args:
            messages : 메시지 리스트

        Returns:
            문자열 LLM 응답

        Raises:
            httpx.RequestError: LLM 요청이 실패하는 경우
        """
        from anthropic import AsyncAnthropic
        from anthropic import APIError

        client = AsyncAnthropic(api_key=str(self.api_key))
        logger.debug(f"API key : {type(self.api_key)}")
        model = "claude-3-7-sonnet-20250219"
        temperature = 0.7

        try:
            formatted_messages = []
            system_content = None

            # 시스템 메시지 추출
            for msg in messages:
                if msg.get("role") == "system" and msg.get("content"):
                    content = msg.get("content")
                    if isinstance(content, (tuple, list)):
                        system_content = "".join(content)
                        logger.debug(f"시스템 메시지 발견: {system_content}")
                    break

            # 나머지 메시지 형식 변환
            for msg in messages:
                if (
                    msg.get("role")
                    and msg.get("content")
                    and msg.get("role") != "system"
                ):
                    formatted_messages.append(
                        {"role": msg["role"], "content": msg["content"]}
                    )

            async with client.messages.stream(
                max_tokens=4096,
                model=model,
                system=system_content,
                temperature=temperature,
                messages=formatted_messages,
            ) as stream:
                async for event in stream:
                    if event.type == "text":
                        yield event.text

        except APIError as e:
            logger.error(f"Anthropic API 오류 : {e}")
            yield f"오류가 발생했습니다. 다시 시도해주세요. {str(e)}"
        except Exception as e:
            logger.error(f"LLM 응답 가져오기 실패 : {str(e)}")
            yield f"오류가 발생했습니다. 다시 시도해주세요. {str(e)}"


@dataclass
class ChatSession:
    """user, LLM, tools 간의 상호작용 관리"""

    servers: list[Server]
    llm_client: LLMClient

    async def cleanup_servers(self) -> None:
        """모든 서버 세션 정리"""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))

        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logger.warning(f"Warning during final cleanup: {e}")

    async def process_llm_response(self, llm_response: str) -> str:
        """LLM 응답을 처리하고 필요한 경우 도구 호출

        Args:
            llm_response: LLM 응답 문자열

        Returns:
            도구 실행 결과 또는 응답
        """
        try:
            # 전체 응답에서 JSON 부분 추출 시도
            json_obj = None

            # 1. 먼저 전체 텍스트를 JSON으로 파싱 시도
            try:
                json_obj = json.loads(llm_response)
                logger.info("전체 텍스트가 유효한 JSON입니다.")
            except json.JSONDecodeError:
                # 2. 실패하면 중괄호 기준으로 JSON 부분 추출 시도
                start_idx = llm_response.find("{")
                end_idx = llm_response.rfind("}") + 1

                if start_idx != -1 and end_idx > start_idx:
                    try:
                        json_str = llm_response[start_idx:end_idx]
                        json_obj = json.loads(json_str)
                        logger.info(
                            f"텍스트에서 JSON 부분을 추출했습니다: {json_str[:50]}..."
                        )
                    except json.JSONDecodeError as e:
                        logger.error(f"추출된 JSON 부분 파싱 실패: {e}")
                else:
                    logger.warning("JSON 객체를 찾을 수 없습니다.")
            # 도구 실행
            if json_obj and "tool" in json_obj and "arguments" in json_obj:
                logger.info(f"Executing tool: {json_obj['tool']}")
                logger.info(f"With arguments: {json_obj['arguments']}")

                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == json_obj["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(
                                json_obj["tool"], json_obj["arguments"]
                            )
                            if isinstance(result, dict) and "progress" in result:
                                progress = result["progress"]
                                total = result["total"]
                                percentage = (progress / total) * 100
                                logger.info(
                                    f"Progress: {progress}/{total}"
                                    f"({percentage:.1f}%)"
                                )

                            return f"Tool execution result: {result}"
                        except Exception as e:
                            error_msg = f"Error executing tool: {str(e)}"
                            logger.error(error_msg)
                            return error_msg
                return f"No server found with tool: {json_obj['tool']}"

            # JSON 객체가 없거나, tool이나 arguments가 없는 경우
            logger.info(
                "도구 실행 조건을 만족하지 않음(JSON 형식이 아니거나, tool이나 arguments 필드가 없음)"
            )
            return llm_response
        except json.JSONDecodeError:
            return llm_response

    async def chat(
        self, message: str, session_id: str, sessions: Dict[str, List[str]]
    ) -> AsyncGenerator[str, None]:
        """메인 채팅 세션 handler"""
        try:

            # 세션 메시지 관리
            if session_id not in sessions:
                sessions[session_id] = []

            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)

            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])

            system_message = (
                "You are a helpful assistant with access to these tools: \n\n"
                f"{tools_description}\n"
                "Choose the appropriate tool based on the user's question. "
                "If no tool is needed, replay directly. \n\n"
                "IMPORTANT: When you need to use a tool, you must ONLY respond with "
                "the exact JSON object format below, nothing else:\n"
                "{\n"
                '  "tool" : "tool-name"\n'
                '  "arguments" : {\n'
                '       "argument-name" : "value",\n'
                "   }\n"
                "}\n\n"
                "After receiving a tool's response:\n"
                "1. Transform the raw data into a natural, conversational response\n"
                "2. Keep responses concise but informative\n"
                "3. Focus on the most relevant information\n"
                "4. Use appropriate context from the user's question\n"
                "5. Avoid simply repeating the raw data\n\n"
                "Please use only the tools that are explicitly defined above. ",
            )
            conversation = sessions[session_id]

            messages = [{"role": "system", "content": system_message}]

            for msg in conversation:
                messages.append(
                    {
                        "role": "user" if msg.startswith("user: ") else "assistant",
                        "content": msg.replace("user: ", "").replace("assistant: ", ""),
                    }
                )

            # add new user message
            messages.append({"role": "user", "content": message})
            conversation.append(f"user: {message}")

            # LLM 응답을 직접 yield
            full_response = ""
            async for chunk in self.llm_client.get_response(messages):
                full_response += chunk
                # logger.debug(f"첫 응답 청크: {chunk}")
                yield chunk
            logger.debug(f"전체 LLM 응답: {full_response}")

            conversation.append(f"assistant: {full_response}")
            try:
                logger.info("도구 실행 로직 시작...")
                # 도구 실행 결과 가져오기
                tool_result = await self.process_llm_response(full_response)

                # 도구 실행 결과가 원본 응답과 다른 경우 (도구가 실행된 경우)
                if tool_result != full_response:
                    logger.info("도구가 성공적으로 실행되었습니다.")
                    yield "\n\n도구 실행 결과를 분석 중...\n\n"

                    # 도구 실행 결과로 새 메시지 생성
                    messages.append({"role": "assistant", "content": full_response})
                    messages.append({"role": "assistant", "content": tool_result})
                    logger.debug(f"도구 실행 결과: {tool_result}")

                    # 최종 응답 생성
                    final_response = ""
                    async for chunk in self.llm_client.get_response(messages):
                        final_response += chunk
                        # logger.debug(f"최종 응답 청크: {chunk}")
                        yield chunk

                    # 최종 응답 저장
                    conversation.append(f"assistant: {final_response}")
                    logger.info("도구 실행 결과를 바탕으로 최종 응답 생성 완료")
                else:
                    logger.info("도구 실행이 필요 없거나, JSON 파싱에 실패했습니다.")
            except Exception as e:
                logger.error(f"도구 처리 중 오류: {e}", exc_info=True)
                yield f"\n\n도구 처리 중 오류가 발생했습니다: {str(e)}"

        except Exception as e:
            logger.error(f"채팅 처리 중 오류: {e}", exc_info=True)
            yield f"오류가 발생했습니다. 다시 시도해주세요. {str(e)}"


# main.py 에서 사용하기 위한 서버초기화 및 세션 관리 함수
async def initialize_mcp_servers(
    config_path: str = "mcp_servers.json",
) -> Optional[ChatSession]:
    """MCP 서버, LLM 클라이언트 및 채팅 세션을 초기화합니다."""

    config = Configuration()
    initialized_servers = []

    try:
        logger.info(f"설정 파일 로드 시작: {config_path}")
        server_config = config.load_config(config_path)
        logger.info("설정 파일 로드 완료")

        servers = [
            Server(name, srv_config)
            for name, srv_config in server_config["mcpServers"].items()
        ]
        logger.info(f"서버 인스턴스 생성 완료: {len(servers)}개")

        # create LLM client
        llm_client = LLMClient(config.api_key)
        logger.info("LLM 클라이언트 생성 완료")

        # initialize servers
        for server in servers:
            try:
                logger.info(f"서버 {server.name} 초기화 시작")
                await server.initialize()
                initialized_servers.append(server)  # 초기화 성공한 서버 추가
                logger.info(f"서버 {server.name} 초기화 완료")
            except Exception as e:
                logger.error(f"서버 {server.name} 초기화 실패: {str(e)}", exc_info=True)

        if not initialized_servers:
            logger.error("모든 서버 초기화 실패")
            return None

        # create chat session
        logger.info("채팅 세션 생성 시작")
        chat_session = ChatSession(initialized_servers, llm_client)
        logger.info("채팅 세션 생성 완료")

        return chat_session

    except Exception as e:
        logger.error(f"MCP 서버 초기화 중 오류: {str(e)}", exc_info=True)
        for server in initialized_servers:
            try:
                await server.cleanup()
            except Exception as cleanup_error:
                logger.error(f"서버 정리 중 오류: {str(cleanup_error)}")
        return None
