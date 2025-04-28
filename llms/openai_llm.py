from typing import Any, Optional, Dict, List, AsyncGenerator
from dataclasses import dataclass, field

from utils.logger import logger
from llms.base import LLMClientBase

# from base import LLMClientBase
from openai import AsyncOpenAI
from openai import APIError
import asyncio
from config import LLMModel


@dataclass
class OpenAILLM(LLMClientBase):
    """Ollama LLM 클라이언트 구현 (비동기, 스트리밍 지원)"""

    api_key: str
    model: str
    temperature: float
    max_tokens: int
    client: AsyncOpenAI = field(init=False)

    def __post_init__(self):
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def get_response(
        self, messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """OPENAI LLM에서 스트리밍 응답을 비동기로 가져옴"""
        try:

            completions = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )
            # OpenAI AsyncOpenAI 스트리밍 호출
            async for chunk in completions:
                # OpenAI 공식 문서: chunk.choices[0].delta.content
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except APIError as e:
            logger.error(f"OpenAI API 오류 : {e}")
            yield f"오류가 발생했습니다. 다시 시도해주세요. {str(e)}"
        except Exception as e:
            yield f"오류가 발생했습니다. 다시 시도해주세요. {str(e)}"


# # 테스트 코드 (실행 예시)
# async def main():
#     llm = OpenAILLM(
#         api_key="sk-pro....",
#         model=LLMModel.GPT_4O,
#     )
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "한국의 수도는 어디야?"},
#     ]
#     print("OpenAI LLM 응답:")
#     async for chunk in llm.get_response(messages):
#         print(chunk, end="", flush=True)


# if __name__ == "__main__":
#     asyncio.run(main())
