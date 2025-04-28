from .anthropic_llm import AnthropicLLM
from .openai_llm import OpenAILLM
from .ollama_llm import OllamaLLM
from config import Configuration


def get_llm_client(
    config: Configuration,
    model: str,
    temperature: float,
    max_tokens: int,
    **kwargs,
):
    # 모델명으로 LLM 타입 자동 판별
    model_lower = model.lower()
    if model_lower.startswith("claude"):
        return AnthropicLLM(
            api_key=config.anthropic_api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif model_lower.startswith("gpt"):
        return OpenAILLM(
            api_key=config.openai_api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        return OllamaLLM(model=model, temperature=temperature, max_tokens=max_tokens)
