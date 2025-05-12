from typing import Any, Optional, Dict, List, AsyncGenerator, Type, TypeVar
from pydantic import BaseModel
from dataclasses import dataclass, field

from utils import logger, image_variation
from llms.base import LLMClientBase

# from base import LLMClientBase
from openai import AsyncOpenAI, OpenAI
from openai import APIError
import asyncio
from config import LLMModel
from fastapi import UploadFile


@dataclass
class OpenAILLM(LLMClientBase):
    """Ollama LLM 클라이언트 구현 (비동기, 스트리밍 지원)"""

    api_key: str
    model: str = None
    temperature: float = None
    max_tokens: int = None
    async_client: AsyncOpenAI = field(init=False)
    client: OpenAI = field(init=False)

    def __post_init__(self):
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.client = OpenAI(api_key=self.api_key)

    async def stream_chat(self, system_prompt, messages):
        """OPENAI LLM에서 스트리밍 응답을 비동기로 가져옴"""
        try:
            full_messages = []
            logger.info(f"messages : {messages}")
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)
            completions = await self.async_client.chat.completions.create(
                model=self.model,
                messages=full_messages,
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

    async def build_user_message(
        self, text: str | None, file: UploadFile | None
    ) -> Any:
        if not file:
            # return text
            return text
        try:
            img_bytes = await file.read()

            logger.info(
                f"파일명: {file.filename}, 타입: {file.content_type}, 크기: {len(img_bytes)}"
            )
            img_url = await image_variation(file.filename, img_bytes, file.content_type)

            text_with_image = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": img_url}},
            ]
            logger.info(f"text_with_image : {text_with_image}")
            return text_with_image
        except Exception as e:
            logger.error(f"이미지 처리 중 오류: {str(e)}")
            return [{"type": "text", "text": text}]


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
#     async for chunk in llm.stream_chat(messages):
#         print(chunk, end="", flush=True)


# if __name__ == "__main__":
#     asyncio.run(main())
