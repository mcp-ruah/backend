from typing import List, Dict, AsyncGenerator, Any
from abc import ABC, abstractmethod
from fastapi import File, UploadFile

# from asyncio import Protocol


class LLMClientBase(ABC):
    """
    모든 LLM 클라이언트가 따라야 하는 추상 베이스 클래스.
    """

    @abstractmethod
    async def stream_chat(
        self, system_prompt: str | None, messages: List[dict] = None
    ) -> AsyncGenerator[str, None]:
        """
        LLM에서 응답을 비동기로 스트리밍 방식으로 가져옴.
        """
        pass

    @abstractmethod
    def build_user_message(self, text: str | None, file: UploadFile | None) -> Any:
        """
        텍스트와 이미지 url을 LLM 포멧으로 변환
        """
        pass

    # @abstractmethod
    # async def create_image_variation(
    #     self, n: int, size: str, file: File
    # ) -> AsyncGenerator[str, None]:
    #     """
    #     이미지 변형 생성
    #     """
    #     pass
