from typing import Any, Optional, Dict, List, AsyncGenerator, Type, TypeVar
from pydantic import BaseModel
from dataclasses import dataclass, field

from utils.logger import logger
from llms.base import LLMClientBase

# from base import LLMClientBase
from openai import AsyncOpenAI
from openai import APIError
import asyncio
from config import LLMModel

# Pydantic 모델 타입 변수 정의
T = TypeVar("T", bound=BaseModel)

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

    def structured_output(self, schema_model: Type[T]) -> "OpenAILLM":
        """Pydantic 모델을 사용하여 구조화된 출력을 반환하는 새 인스턴스 생성

        'output_parser': <function OpenAILLM.structured_output.<locals>.parse_response at 0x000001FA495731A0>,
        'output_schema': {
            'properties': {
                'next': {
                    'enum': ['FINISH',
                            'WebSearcher',
                            'WeatherSearcher'],
                    'title': 'Next',
                    'type': 'string'
                    }
                },
            'required': ['next'],
            'title': 'RouteResponse',
            'type': 'object'
            },
        'tools': [
            {
                'function': {
                    'description': '특정 위치의 날씨 정보를 조회합니다.',
                    'name': 'get_weather',
                    'parameters': {
                        'properties': {
                            'location': {
                                'description': 'location 매개변수',
                                'type': 'string'
                            }
                        },
                        'required': ['location'],
                        'type': 'object'
                        }
                    },
                'type': 'function'
                }
            ]
        }

        """
        # 새 인스턴스 생성 - 인자 없이 생성 후 속성 직접 설정
        new_instance = OpenAILLM()

        # 속성 복사
        new_instance.client = self.client
        new_instance.async_client = self.async_client
        new_instance.model = self.model
        new_instance.tools = self.tools.copy() if self.tools else []
        new_instance.available_functions = (
            self.available_functions.copy() if self.available_functions else {}
        )

        # 스키마 설정 - OpenAI의 새로운 구조화된 출력 형식 사용
        new_instance.output_schema = schema_model.model_json_schema()

        # 파서 함수 설정 (응답을 Pydantic 모델로 변환)
        def parse_response(response_content: str) -> T:
            try:
                # JSON 문자열을 파싱하여 Pydantic 모델로 변환
                return schema_model.model_validate_json(response_content)
            except Exception as e:
                print(f"응답 파싱 오류: {e}")
                print(f"원본 응답: {response_content}")
                raise ValueError(
                    f"응답을 {schema_model.__name__} 모델로 파싱할 수 없습니다: {e}"
                )

        new_instance.output_parser = parse_response

        return new_instance


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
