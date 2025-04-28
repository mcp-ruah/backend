from typing import Any, Optional, Dict, List, AsyncGenerator
from dataclasses import dataclass, field

from utils.logger import logger
from llms.base import LLMClientBase

# from base import LLMClientBase
from anthropic import AsyncAnthropic
from anthropic import APIError
import asyncio
from config import LLMModel


@dataclass
class AnthropicLLM(LLMClientBase):
    """Anthropic LLM 클라이언트 구현"""

    api_key: str
    model: str
    temperature: float
    max_tokens: int
    client: AsyncAnthropic = field(init=False)

    def __post_init__(self):
        self.client = AsyncAnthropic(api_key=self.api_key)

    async def get_response(
        self, messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """LLM에서 응답을 가져옴 (Anthropic API)"""

        try:
            formatted_messages = []
            system_content = None

            # 시스템 메시지 추출
            for msg in messages:
                if msg.get("role") == "system" and msg.get("content"):
                    content = msg.get("content")
                    print(content)
                    if isinstance(content, (tuple, list)):
                        system_content = "".join(content)
                        logger.debug(f"시스템 메시지 발견: {system_content}")
                    else:
                        system_content = content
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

            async with self.client.messages.stream(
                system=system_content,
                max_tokens=4096,
                model=self.model,
                temperature=self.temperature,
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


# # 테스트 코드 (실행 예시)
# async def main():
#     llm = AnthropicLLM(
#         api_key="sk-ant-api.....",
#         model=LLMModel.CLAUDE_3_7_SONNET,
#     )
#     messages = [
#         {
#             "role": "system",
#             "content": "You are a helpful assistant.",
#         },
#         {"role": "user", "content": "한국의 수도는 어디야?"},
#     ]
#     print("Anthropic LLM 응답:")
#     async for chunk in llm.get_response(messages):
#         print(chunk, end="", flush=True)


# if __name__ == "__main__":
#     asyncio.run(main())
