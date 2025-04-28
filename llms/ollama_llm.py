from typing import Any, Optional, Dict, List, AsyncGenerator
from dataclasses import dataclass, field

from utils.logger import logger
from llms.base import LLMClientBase
from ollama import AsyncClient
import asyncio
from config import LLMModel


@dataclass
class OllamaLLM(LLMClientBase):
    """Ollama LLM 클라이언트 구현 (비동기, 스트리밍 지원)"""

    model: str
    temperature: float
    max_tokens: int
    client: AsyncClient = field(default_factory=AsyncClient)

    async def get_response(
        self, messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """Ollama LLM에서 스트리밍 응답을 비동기로 가져옴"""
        try:

            # Ollama AsyncClient로 스트리밍 호출
            async for chunk in await self.client.chat(
                model=self.model,
                messages=messages,
                # max_tokens=self.max_tokens,
                stream=True,
            ):
                # Ollama stream은 각 chunk에 'message' 키가 있음
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]

        except Exception as e:
            # logger.error(f"Ollama LLM 응답 가져오기 실패 : {str(e)}")
            yield f"오류가 발생했습니다. 다시 시도해주세요. {str(e)}"


async def main():
    # OllamaLLM 인스턴스 생성 (필요시 model명 변경)
    llm = OllamaLLM(model=LLMModel.GEMMA3_4B.value)

    # 테스트 메시지
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "한국의 수도는 어디야?"},
    ]

    print("Ollama LLM 응답:")
    async for chunk in llm.get_response(messages):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
