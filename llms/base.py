from typing import List, Dict, AsyncGenerator
from abc import ABC, abstractmethod


class LLMClientBase(ABC):
    """
    모든 LLM 클라이언트가 따라야 하는 추상 베이스 클래스.
    """

    @abstractmethod
    async def get_response(
        self, messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """
        LLM에서 응답을 비동기로 스트리밍 방식으로 가져옴.
        """
        pass
