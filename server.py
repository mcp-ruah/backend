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

        if self.config.get("env"):
            logger.info("=" * 40)
            logger.info(f"서버 {self.name}의 환경 변수:")
            logger.info(self.config["env"])
            logger.info("명령어 인자:")
            logger.info(server_params.args)
            logger.info("=" * 40)

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

    async def get_status(self) -> Dict[str, Any]:
        """서버 상태 정보 반환

        Returns:
            Dict[str, Any]: 서버 상태 정보를 포함한 딕셔너리
        """
        status = {
            "name": self.name,
            "initialized": self.session is not None,
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

    async def get_servers_status(self) -> List[Dict[str, Any]]:
        """모든 MCP 서버의 상태 정보 반환

        Returns:
            List[Dict[str, Any]]: 각 서버의 상태 정보 목록
        """
        status_list = []
        for server in self.servers:
            try:
                server_status = await server.get_status()
                status_list.append(server_status)
            except Exception as e:
                logger.error(f"{server.name} 서버 상태 확인 중 오류: {str(e)}")
                status_list.append(
                    {"name": server.name, "initialized": False, "error": str(e)}
                )

        return status_list

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

            logger.debug(f"LLM 응답 처리 시작: {llm_response[:100]}...")

            # 1. JSON 블록 찾기 시도
            json_start = llm_response.find("{")
            json_end = llm_response.rfind("}")

            if json_start != -1 and json_end > json_start:
                potential_json = llm_response[json_start : json_end + 1]
                logger.debug(f"잠재적 JSON 발견: {potential_json[:100]}...")

                try:
                    json_obj = json.loads(potential_json)
                    logger.info("유효한 JSON 형식 발견")
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 파싱 실패: {e}")

                    # 중첩된 JSON이 있는 경우 처리 시도
                    try:
                        # 가장 바깥쪽 중괄호 쌍 찾기
                        depth = 0
                        start = -1
                        end = -1

                        for i, char in enumerate(llm_response):
                            if char == "{":
                                if depth == 0:
                                    start = i
                                depth += 1
                            elif char == "}":
                                depth -= 1
                                if depth == 0 and start != -1:
                                    end = i
                                    break

                        if start != -1 and end != -1:
                            clean_json = llm_response[start : end + 1]
                            logger.debug(f"중첩 처리된 JSON: {clean_json[:100]}...")
                            json_obj = json.loads(clean_json)
                    except Exception as nested_error:
                        logger.warning(f"중첩 JSON 파싱 시도 실패: {nested_error}")

            # 도구 실행
            if json_obj and "tool" in json_obj and "arguments" in json_obj:
                tool_name = json_obj["tool"]
                arguments = json_obj["arguments"]
                logger.info(f"도구 실행: {tool_name}")
                logger.info(f"인자: {arguments}")

                # 적절한 서버 찾기
                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_name for tool in tools):
                        try:
                            result = await server.execute_tool(tool_name, arguments)

                            # 결과가 딕셔너리인 경우 포맷팅
                            if isinstance(result, dict):
                                if "progress" in result and "total" in result:
                                    progress = result["progress"]
                                    total = result["total"]
                                    percentage = (progress / total) * 100
                                    logger.info(
                                        f"진행 상황: {progress}/{total} ({percentage:.1f}%)"
                                    )

                                # 결과를 JSON 문자열로 변환
                                result_str = json.dumps(
                                    result, ensure_ascii=False, indent=2
                                )
                                logger.info(
                                    f"도구 실행 결과(dict): {result_str[:200]}..."
                                )
                                return f"Tool result: {result_str}"
                            else:
                                logger.info(f"도구 실행 결과: {str(result)[:200]}...")
                                return f"Tool result: {result}"
                        except Exception as e:
                            error_msg = f"도구 실행 오류: {str(e)}"
                            logger.error(error_msg)
                            return error_msg

                return f"도구를 실행할 수 있는 서버를 찾을 수 없습니다: {tool_name}"

            # JSON 객체가 없거나, tool이나 arguments가 없는 경우
            logger.info("도구 실행 조건을 만족하지 않음")
            return llm_response

        except json.JSONDecodeError as json_error:
            logger.warning(f"JSON 디코딩 오류: {json_error}")
            return llm_response
        except Exception as e:
            logger.error(f"도구 실행 처리 중 오류: {str(e)}", exc_info=True)
            return f"오류 발생: {str(e)}"

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
                "5. Avoid simply repeating the raw data\n"
                "6. You can use multiple tools in sequence if needed\n\n"
                "Multi-tool execution strategy:\n"
                "- If you need to run multiple tools in sequence, respond ONLY with the next tool JSON format after seeing a tool result\n"
                "- Do NOT provide explanation text between tool calls; ONLY output the JSON format for the next tool\n"
                "- Only after completing ALL necessary tool calls, provide your final conversational response\n"
                "- You can use up to 5 tools in sequence if needed\n\n"
                "For example, to use multiple tools:\n"
                '1. First tool call (only output this JSON): {"tool":"tool1","arguments":{"param":"value"}}\n'
                '2. After seeing tool1\'s result, if needed: {"tool":"tool2","arguments":{"param":"value"}}\n'
                "3. After all tool results, respond conversationally\n\n"
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
                yield chunk
            logger.debug(f"전체 LLM 응답: {full_response}")

            # 도구 실행 결과와 메시지 처리
            current_messages = messages.copy()
            current_response = full_response
            max_tool_calls = 5  # 최대 도구 호출 횟수 제한
            tool_call_count = 0

            # 응답 저장
            conversation.append(f"assistant: {full_response}")

            while tool_call_count < max_tool_calls:
                try:
                    # 도구 실행 로직 시작
                    logger.info(
                        f"도구 실행 로직 시작... (호출 {tool_call_count+1}/{max_tool_calls})"
                    )

                    # 도구 실행 시도
                    tool_result = await self.process_llm_response(current_response)

                    # 도구 실행 결과가 원본 응답과 다른 경우 (도구가 실행된 경우)
                    if tool_result != current_response:
                        tool_call_count += 1
                        logger.info(f"도구 {tool_call_count}번째 실행 완료")
                        yield f"\n\n도구 실행 결과를 분석 중...\n\n"

                        # 도구 실행 결과로 새 메시지 추가
                        current_messages.append(
                            {"role": "assistant", "content": current_response}
                        )
                        current_messages.append(
                            {"role": "user", "content": f"Tool result: {tool_result}"}
                        )
                        logger.debug(f"도구 실행 결과: {tool_result}")

                        # 사용자에게 툴 실행 중임을 알림
                        try:
                            # JSON 파싱 시도
                            json_start = current_response.find("{")
                            json_end = current_response.rfind("}")
                            if json_start != -1 and json_end > json_start:
                                json_str = current_response[json_start : json_end + 1]
                                json_obj = json.loads(json_str)
                                tool_name = json_obj.get("tool", "unknown")
                            else:
                                tool_name = "unknown"
                        except:
                            tool_name = "unknown"

                        yield f"\n\n[도구 실행 {tool_call_count}/{max_tool_calls}] {tool_name} 도구 실행 결과:\n"

                        # 결과 미리보기 표시 (너무 길면 잘라서)
                        result_preview = tool_result
                        if len(result_preview) > 500:
                            result_preview = result_preview[:500] + "... (결과 계속)"
                        yield f"{result_preview}\n\n"

                        # 다음 응답 생성
                        next_response = ""
                        yield "다음 단계 분석 중...\n"

                        async for chunk in self.llm_client.get_response(
                            current_messages
                        ):
                            next_response += chunk
                            yield chunk

                        # 응답 업데이트
                        current_response = next_response
                        conversation.append(f"assistant: {next_response}")

                        # 응답에 도구 호출이 없으면 종료
                        if "{" not in next_response or "}" not in next_response:
                            logger.info("더 이상의 도구 호출이 없어 종료합니다.")
                            break
                    else:
                        # 도구 실행이 필요 없으면 종료
                        logger.info(
                            "도구 실행이 필요 없거나, JSON 파싱에 실패했습니다."
                        )
                        break

                except Exception as e:
                    logger.error(f"도구 처리 중 오류: {e}", exc_info=True)
                    yield f"\n\n도구 처리 중 오류가 발생했습니다: {str(e)}"
                    break

            # 도구 실행이 완료되었음을 사용자에게 알림
            if tool_call_count > 0:
                logger.info(f"총 {tool_call_count}개의 도구가 실행되었습니다.")
                yield f"\n\n[완료] 총 {tool_call_count}개의 도구를 실행했습니다.\n\n"

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
        logger.info("=" * 40)
        logger.info(servers)
        logger.info("=" * 40)
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
