from typing import Dict, List
import uuid
from dataclasses import dataclass, field
from utils import logger
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
    image_path: str = None


    async def stream_chat(self, system_prompt, messages):
        """Ollama LLM에서 스트리밍 응답을 비동기로 가져옴"""
        try:
            print(f"messages : \n\n{messages}\n\n")
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            # 메시지 형식을 확인하고 변환
            for msg in messages :
                if msg["role"] == "user" and hasattr(self, 'image_path'):
                    full_messages.append({
                        "role":"user",
                        "content": msg["content"],
                        "images": [self.image_path]
                    })
                    # 이미지 경로 초기화 
                    delattr(self, 'image_path')
                else : 
                    full_messages.append(msg)
            # full_messages.extend(messages)
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


        if not file:
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

            # 이미지 경로를 클래스에 저장
            self.image_path = temp_img_path
            print(f"self.image_path : {temp_img_path}")

            return text or ""
        
        except Exception as e:
            logger.error(f"이미지 처리 중 오류: {str(e)}")
            # 오류 발생 시 기본 메시지 반환
            return text or ""


async def main():
    # OllamaLLM 인스턴스 생성 (필요시 model명 변경)
    
    llm = OllamaLLM(model=LLMModel.GEMMA3_12B.value)
    system_prompt = "You are a helpful assistant."

    # 테스트 메시지
    messages = [
        {
            "role": "user", 
            "content": "한국의 수도는 어디야?"
            # "images": ["/home/ruah0807/Desktop/mcp-agent/backend/tmp/sample.jpg"]
            },
    ]

    print("Ollama LLM 응답:")
    async for chunk in llm.stream_chat(system_prompt,messages):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
