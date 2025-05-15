import json
from core.prompt import SystemPrompt, PROMPT_TEXT
from core.utils import logger
from typing import List, Dict, Any, AsyncGenerator, Optional
from contextlib import AsyncExitStack
from dataclasses import dataclass
from core.mcp_server import MCPServer
from core.llms import LLMClientBase
import asyncio
from fastapi import File


@dataclass
class ChatSession:
    """user, LLM, tools 간의 상호작용 관리"""

    servers: list[MCPServer]
    llm_client: Optional[LLMClientBase] = None

    # async def cleanup_servers(self) -> None:
    #     """모든 서버 세션 정리 (순차적으로)"""
    #     for server in self.servers:
    #         try:
    #             await server.cleanup()
    #         except asyncio.CancelledError as ce:
    #             logger.info(f"서버 정리 중 작업이 취소되었습니다: {server.name}")
    #             # CancelledError는 무시하고 다음 서버 정리 진행
    #         except Exception as e:
    #             logger.warning(f"Warning during final cleanup: {e}")

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
                logger.info(f"서버 상태 확인 완료: {server_status}")
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
            # <tool_call> 태그 내부의 JSON 추출
            tool_call_start = llm_response.find("<tool_call>")
            tool_call_end = llm_response.find("</tool_call>")

            if tool_call_start != -1 and tool_call_end > tool_call_start:
                # <tool_call> 태그 사이의 내용 추출
                tool_call_content = llm_response[
                    tool_call_start + len("<tool_call>") : tool_call_end
                ].strip()
                # logger.debug(f"도구 호출 내용: {tool_call_content[:100]}...")

                # JSON 파싱
                try:
                    json_obj = json.loads(tool_call_content)
                    logger.info("유효한 JSON 형식 발견")

                    if "tool" in json_obj and "arguments" in json_obj:
                        tool_name = json_obj["tool"]
                        arguments = json_obj["arguments"]
                        logger.info(f"도구 실행: {tool_name}")
                        logger.info(f"인자: {arguments}")

                        # 적절한 서버 찾기
                        for server in self.servers:
                            tools = await server.list_tools()
                            if any(tool.name == tool_name for tool in tools):
                                try:
                                    result = await server.execute_tool(
                                        tool_name, arguments
                                    )

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
                                        return (
                                            f"<tool_result>{result_str}</tool_result>"
                                        )
                                    else:
                                        logger.info(
                                            f"도구 실행 결과: {str(result)[:200]}..."
                                        )
                                        return f"<tool_result>{result}</tool_result>"
                                except Exception as e:
                                    error_msg = f"도구 실행 오류: {str(e)}"
                                    logger.error(error_msg)
                                    return error_msg

                        return f"도구를 실행할 수 있는 서버를 찾을 수 없습니다: {tool_name}"
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 파싱 실패: {e}")
                    return llm_response

            # <tool_call> 태그가 없거나 파싱 실패시 원본 응답 반환
            logger.info("도구 실행 조건을 만족하지 않음")
            return llm_response

        except Exception as e:
            logger.error(f"도구 실행 처리 중 오류: {str(e)}", exc_info=True)
            return f"오류 발생: {str(e)}"

    async def chat(
        self,
        user_content: Any,
        session_id: str,
        sessions: Dict[str, List[str]],
        sys_prompt_id: SystemPrompt = SystemPrompt.WITH_TOOLS,
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

            system_message = PROMPT_TEXT[sys_prompt_id].format(
                tools_description=tools_description
            )
            conversation = sessions[session_id]

            # messages = [{"role": "system", "content": system_message}]
            messages = []
            for msg in conversation:
                messages.append(
                    {
                        "role": "user" if msg.startswith("user: ") else "assistant",
                        "content": msg.replace("user: ", "").replace("assistant: ", ""),
                    }
                )

            # add new user message
            messages.append({"role": "user", "content": user_content})
            conversation.append(f"user: {user_content}")

            # 도구 호출 여부를 검사하는 함수
            def has_tool_call(resp: str) -> bool:
                tool_call_start = resp.find("<tool_call>")
                tool_call_end = resp.find("</tool_call>")

                if tool_call_start != -1 and tool_call_end > tool_call_start:
                    try:
                        # <tool_call> 태그 사이의 내용 추출
                        tool_call_content = resp[
                            tool_call_start + len("<tool_call>") : tool_call_end
                        ].strip()
                        json_obj = json.loads(tool_call_content)
                        return "tool" in json_obj and "arguments" in json_obj
                    except Exception as e:
                        logger.error(f"JSON 파싱 오류: {str(e)}")
                        return False
                return False

            # LLM 첫 응답 스트리밍
            response_chunks = []
            print("\n\n LLM 첫 응답 스트리밍  : \n")
            async for chunk in self.llm_client.stream_chat(system_message, messages):
                print(chunk, end="", flush=True)
                yield chunk
                response_chunks.append(chunk)

            current_response = "".join(response_chunks)
            current_messages = messages.copy()
            current_messages.append({"role": "assistant", "content": current_response})
            conversation.append(f"assistant: {current_response}")

            # 도구 호출이 필요한지 확인
            if not has_tool_call(current_response):
                logger.debug("도구 호출 없음")
                return

            # 도구 호출 처리
            max_tool_calls = 5  # 최대 도구 호출 횟수 제한
            tool_call_count = 0

            while tool_call_count < max_tool_calls:
                try:
                    logger.info(
                        f"도구 실행 로직 시작... (호출 {tool_call_count+1}/{max_tool_calls})"
                    )
                    tool_result = await self.process_llm_response(current_response)

                    # 도구 실행 결과가 원본 응답과 다른 경우 (도구가 실행된 경우)
                    if tool_result != current_response:
                        tool_call_count += 1
                        # 도구 실행 결과를 사용자에게 전달
                        yield f"{tool_result}"

                        current_messages.append(
                            {
                                "role": "assistant",
                                "content": f"<tool_result> {tool_result}</tool_result>",
                            }
                        )
                        # logger.debug(f"도구 실행 결과: {tool_result}")

                        # 다음 응답 생성 및 직접 스트리밍
                        response_chunks = []
                        print("\n\n LLM 다음 응답 스트리밍  : \n")
                        async for chunk in self.llm_client.stream_chat(
                            system_message, current_messages
                        ):
                            print(chunk, end="", flush=True)
                            yield chunk
                            response_chunks.append(chunk)

                        next_response = "".join(response_chunks)
                        current_response = next_response
                        conversation.append(f"assistant: {current_response}")

                        # 다음 응답에 도구 호출이 없으면 종료
                        if not has_tool_call(next_response):
                            break
                    else:
                        # 도구 실행이 필요 없으면 종료
                        logger.info(
                            "도구 실행이 필요 없거나, JSON 파싱에 실패했습니다."
                        )
                        break

                except Exception as e:
                    logger.error(f"도구 처리 중 오류: {e}", exc_info=True)
                    yield f"오류가 발생했습니다. 다시 시도해주세요. {str(e)}"
                    break

        except Exception as e:
            logger.error(f"채팅 처리 중 오류: {e}", exc_info=True)
            yield f"오류가 발생했습니다. 다시 시도해주세요. {str(e)}"
