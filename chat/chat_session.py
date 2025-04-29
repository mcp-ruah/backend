import json
from utils.logger import logger
from typing import List, Dict, Any, AsyncGenerator, Optional
from contextlib import AsyncExitStack
from dataclasses import dataclass
from mcp_server import Server
from llms import LLMClientBase


@dataclass
class ChatSession:
    """user, LLM, tools 간의 상호작용 관리"""

    servers: list[Server]
    llm_client: Optional[LLMClientBase] = None

    async def cleanup_servers(self) -> None:
        """모든 서버 세션 정리 (순차적으로)"""
        for server in self.servers:
            try:
                await server.cleanup()
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
                '  "tool" : "tool-name",\n'
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
                "Please use only the tools that are explicitly defined above. Answer should be in Korean."
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

            # LLM 응답을 우선 변수에 저장 (바로 yield하지 않음)
            full_response = ""
            async for chunk in self.llm_client.get_response(messages):
                yield chunk
                full_response += chunk
            logger.debug(f"전체 LLM 응답: {full_response}")

            current_messages = messages.copy()
            current_response = full_response
            max_tool_calls = 5  # 최대 도구 호출 횟수 제한
            tool_call_count = 0

            # 응답 저장
            conversation.append(f"assistant: {full_response}")

            # 첫 응답에 도구 호출이 있는지 검사
            def has_tool_call(resp: str) -> bool:
                logger.debug(f"첫 응답에 도구 호출이 있는지 검사: {resp}")
                resp = resp.replace("```json", "").replace("```", "").strip()
                json_start = resp.find("{")
                json_end = resp.rfind("}")
                if json_start != -1 and json_end > json_start:
                    try:
                        json_obj = json.loads(resp[json_start : json_end + 1])
                        return "tool" in json_obj and "arguments" in json_obj
                    except Exception as e:
                        logger.error(f"JSON 파싱 오류: {str(e)}")
                        return False
                return False

            # 도구 호출이 없으면 바로 사용자에게 응답
            if not has_tool_call(current_response):
                logger.debug(f"도구 없음")
                yield current_response
                return

            # 도구 호출이 있으면, 도구 실행 및 후속 LLM 응답만 사용자에게 보여줌
            while tool_call_count < max_tool_calls:
                try:
                    logger.info(
                        f"도구 실행 로직 시작... (호출 {tool_call_count+1}/{max_tool_calls})"
                    )
                    tool_result = await self.process_llm_response(current_response)

                    # 도구 실행 결과가 원본 응답과 다른 경우 (도구가 실행된 경우)
                    if tool_result != current_response:
                        tool_call_count += 1
                        # 안내 메시지 yield 제거, 내부 로그만 남김
                        current_messages.append(
                            {"role": "assistant", "content": current_response}
                        )
                        current_messages.append(
                            {"role": "user", "content": f"Tool result: {tool_result}"}
                        )
                        logger.debug(f"도구 실행 결과: {tool_result}")

                        # 다음 응답 생성
                        next_response = ""
                        async for chunk in self.llm_client.get_response(
                            current_messages
                        ):
                            next_response += chunk
                        # 사용자에게는 오직 최종 자연어 응답만 yield
                        current_response = next_response
                        conversation.append(f"assistant: {next_response}")

                        # 다음 응답에 도구 호출이 없으면 종료
                        if not has_tool_call(next_response):
                            yield next_response
                            break
                    else:
                        # 도구 실행이 필요 없으면 종료
                        logger.info(
                            "도구 실행이 필요 없거나, JSON 파싱에 실패했습니다."
                        )
                        yield current_response
                        break

                except Exception as e:
                    logger.error(f"도구 처리 중 오류: {e}", exc_info=True)
                    yield f"오류가 발생했습니다. 다시 시도해주세요. {str(e)}"
                    break

        except Exception as e:
            logger.error(f"채팅 처리 중 오류: {e}", exc_info=True)
            yield f"오류가 발생했습니다. 다시 시도해주세요. {str(e)}"
