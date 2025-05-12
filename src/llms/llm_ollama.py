from typing import Any, Optional, Dict, List, Type, TypeVar
import uuid
from pydantic import BaseModel
from dataclasses import dataclass, field

from utils import logger, convert_img
from llms.base import LLMClientBase
from ollama import AsyncClient
import asyncio
from config import LLMModel
from fastapi import UploadFile


@dataclass
class OllamaLLM(LLMClientBase):
    """Ollama LLM 클라이언트 구현 (비동기, 스트리밍 지원)"""

    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    client: AsyncClient = field(default_factory=AsyncClient)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

    async def stream_chat(self, system_prompt, messages):
        """Ollama LLM에서 스트리밍 응답을 비동기로 가져옴"""
        try:
            print(f"messages : \n\n{messages}\n\n")
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            # 메시지 형식을 확인하고 변환
            # for msg in messages :
                # if msg['content']가 딕셔너리라면, 텍스트 부분으로 변환

            full_messages.extend(messages)
            logger.info(f"full_messages : {full_messages}")

            # Ollama AsyncClient로 스트리밍 호출
            async for chunk in await self.client.chat(
                model=self.model,
                messages=full_messages,
                stream=True,
                # temperature=self.temperature,
                # num_predict=self.max_tokens,  # max_tokens 대신 num_predict 사용
            ):
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    yield content

        except Exception as e:
            logger.error(f"Ollama LLM 응답 가져오기 실패: {str(e)}")
            yield f"오류가 발생했습니다. 다시 시도해주세요. {str(e)}"

    async def build_user_message(
        self, text: str | None, file: UploadFile | None
    ) -> dict:
        """텍스트와 이미지를 Ollama API 형식(base64)으로 변환"""

        user_message = {"role": "user", "content": text or ""}

        if not file:
            logger.debug(f"text type : {type(text)}")
            logger.info(f"user_message : {text}")
            return text
        try:
            logger.info(f"file : {file}")
            img_bytes = await file.read()

            import tempfile
            import os

            # 임시파일 생성
            temp_dir = tempfile.gettempdir()
            temp_img_path = os.path.join(
                temp_dir, f"{uuid.uuid4()}.{file.content_type.split('/')[-1]}"
            )

            # 이미지 바이트를 파일로 저장
            with open(temp_img_path, "wb") as f:
                f.write(img_bytes)

            # content는 그대로 유지하고 images 필드 추가
            user_message["images"] = [temp_img_path]
            logger.info(f"user_message : {user_message}")
            return user_message
        except Exception as e:
            logger.error(f"이미지 처리 중 오류: {str(e)}")
            # 오류 발생 시 기본 메시지 반환
            return user_message


async def main():
    # OllamaLLM 인스턴스 생성 (필요시 model명 변경)
    llm = OllamaLLM(model=LLMModel.GEMMA3_12B.value)

    # 테스트 메시지
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "한국의 수도는 어디야?"},
    ]

    print("Ollama LLM 응답:")
    async for chunk in llm.stream_chat(messages):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
