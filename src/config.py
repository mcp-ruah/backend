import os, json
from dotenv import load_dotenv
from typing import Optional, Any
from dataclasses import dataclass
from enum import Enum


class LLMModel(Enum):
    GEMMA3_4B = "gemma3:4b"
    GEMMA3_12B = "gemma3:12b"
    LLAMA_3_2 = "llama3.2:3b"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"
    QWEN3_14B = "qwen3:14b"
    GPT_4O = "gpt-4o"
    TEMPERATURE = 0.76
    MAX_TOKENS = 4096


@dataclass
class Configuration:
    """MCP client 용 환경 변수와 설정 관리"""

    api_key: Optional[str] = None

    def __post_init__(self):
        """dataclass 초기화 후 추가 작업을 수행합니다."""
        self.load_env()
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    @staticmethod
    def load_env() -> None:
        """환경변수를 .env 파일에서 로드"""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Json 파일에서 서버 설정 로드

        Args:
            file_path : Json 설정 파일 경로

        Raises:
            FileNotFoundError : 설정 파일이 존재하지 않는 경우
            JSONDecodeError : 설정 파일이 유효한 JSON이 아닌 경우
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def llm_api_key(self) -> str:
        """LLM API 키 반환

        Returns:
            문자열 API 키

        Raises:
            ValueError : API 키가 설정되지 않은 경우
        """
        if not self.api_key:
            raise ValueError("API 키가 설정되지 않았습니다.")
        return self.api_key
