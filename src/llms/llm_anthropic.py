from typing import Any, Optional, Dict, List, AsyncGenerator
from dataclasses import dataclass, field

from utils import logger, image_variation
from llms.base import LLMClientBase

# from base import LLMClientBase
from anthropic import AsyncAnthropic
from anthropic import APIError
import asyncio
from config import LLMModel
import base64
from fastapi import UploadFile


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

    async def stream_chat(self, system_prompt, messages):
        """LLM에서 응답을 가져옴 (Anthropic API)"""
        try:
            # print(f"\n\nsystem_prompt: {system_prompt}\n\n")
            # print(f"\n\nmessages: {messages}\n\n")
            async with self.client.messages.stream(
                system=system_prompt,
                max_tokens=4096,
                model=self.model,
                temperature=self.temperature,
                messages=messages,
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

    async def build_user_message(self, text: str, file: UploadFile | None) -> Any:
        if not file:
            return text
        try:
            img_bytes = await file.read()
            img_url = await image_variation(file.filename, img_bytes, file.content_type)
            logger.info(
                f"파일명: {file.filename}, 타입: {file.content_type}, 크기: {len(img_bytes)}"
            )
            return [
                {"type": "image", "source": {"type": "url", "url": img_url}},
                {"type": "text", "text": text},
            ]
        except Exception as e:
            logger.error(f"이미지 처리 중 오류: {str(e)}")
            return [{"type": "text", "text": text}]


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
#     async for chunk in llm.stream_chat(messages):
#         print(chunk, end="", flush=True)


# if __name__ == "__main__":
#     asyncio.run(main())
