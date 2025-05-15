from config import Configuration
from core.utils import logger
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
        from .llm_anthropic import AnthropicLLM

        logger.info(f"\n\nUsing Anthropic Model: {model_lower}\n\n")
        return AnthropicLLM(
            api_key=config.anthropic_api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif model_lower.startswith("gpt"):
        from .llm_openai import OpenAILLM

        logger.info(f"\n\nUsing OpenAI Model: {model_lower}\n\n")
        return OpenAILLM(
            api_key=config.openai_api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif model_lower.startswith("/") and "local" in kwargs:
        from .llm_gemma3 import HuggingFaceGemma3

        # HuggingFace 로컬 모델 경로가 제공된 경우
        logger.info(f"\n\nUsing HuggingFace Model: {model_lower}\n\n")
        return HuggingFaceGemma3(
            model_path=kwargs["local"],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        from .llm_ollama import OllamaLLM

        logger.info(f"\n\nUsing Ollama Model: {model_lower}\n\n")
        return OllamaLLM(model=model, temperature=temperature, max_tokens=max_tokens)


def get_client(config: Configuration):
    from .llm_openai import OpenAILLM

    """오직 OpenAI 클라이언트 반환"""
    return OpenAILLM(
        api_key=config.openai_api_key,
    )
