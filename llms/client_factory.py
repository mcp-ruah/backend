from .llm_anthropic import AnthropicLLM
from .llm_openai import OpenAILLM
from .llm_ollama import OllamaLLM
from config import Configuration
from utils import logger
from fastapi import File


def get_llm_client(
    config: Configuration,
    model: str,
    temperature: float,
    max_tokens: int,
    **kwargs,
):
    """class에 저장할 LLM 모델별 클라이언트 반환"""
    # 모델명으로 LLM 타입 자동 판별
    model_lower = model.lower()
    if model_lower.startswith("claude"):
        logger.info(f"\n\nUsing Anthropic Model: {model_lower}\n\n")
        return AnthropicLLM(
            api_key=config.anthropic_api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif model_lower.startswith("gpt"):
        logger.info(f"\n\nUsing OpenAI Model: {model_lower}\n\n")
        return OpenAILLM(
            api_key=config.openai_api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        logger.info(f"\n\nUsing Ollama Model: {model_lower}\n\n")
        return OllamaLLM(model=model, temperature=temperature, max_tokens=max_tokens)


def get_client(config: Configuration):
    """오직 OpenAI 클라이언트 반환"""
    return OpenAILLM(
        api_key=config.openai_api_key,
    )
