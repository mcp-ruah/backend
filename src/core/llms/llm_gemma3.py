import uuid, os, asyncio, torch, re, tempfile
from typing import Dict, List, AsyncGenerator, Any
from dataclasses import dataclass, field
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, Gemma3Config
from fastapi import UploadFile


# if hasattr(torch, "_dynamo"):
#     if hasattr(torch._dynamo, "config"):
#         torch._dynamo.config.disable = True
#     torch._dynamo.disable()

# # torch dynamo 컴파일 비활성화
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# os.environ["TORCH_COMPILE_DEBUG"] = "0"
# os.environ["TORCH_COMPILE_DISABLE"] = "1"


@dataclass
class HuggingFaceGemma3:
    """HuggingFace Gemma3 모델 클라이언트구현 (비동기, 스트리밍 지원)"""

    model_path: str  # 로컬 모델 경로
    model: Any = None
    processor: Any = None
    temperature: float = 0.7
    max_tokens: int = 2000
    image_path: str = None

    def __post_init__(self):
        """모델과 프로세서 초기화"""
        try:
            # 모델 로드 전에 설정 수정
            config = Gemma3Config.from_pretrained(self.model_path)

            # 수정된 설정으로 모델 로드
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                # attn_implementation="eager",
            ).eval()

            # 명시적으로 attention 구현 방식 설정
            # if hasattr(self.model.config, "text_config"):
            #     self.model.config.text_config._attn_implementation = "eager"

            # processor 로드
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_path)
            except Exception as e:
                print(f"AutoProcessor 로드 실패: {e}, fallback 방식으로 시도합니다")
                # processor가 로드되지 않을 경우 직접 모듈에서 로드
                from transformers.models.gemma3 import Gemma3Processor

                self.processor = Gemma3Processor.from_pretrained(self.model_path)

            # print(f"model : {self.model}")
            # print(f"processor : {self.processor}")
            print("Gemma3 model loaded successfully")

        except Exception as e:
            print(f"Gemma3 model loading failed: {e}")
            raise

    async def stream_chat(self, system_prompt, messages) -> AsyncGenerator[str, None]:
        """Gemma3 응답을 비동기로 생성 및 스트리밍"""

        try:
            print(f"messages : \n\n{messages}\n\n")
            # 대화 형식 구성
            conversation = []
            # 시스템 프롬프트
            if system_prompt:
                conversation.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    }
                )
            # 메시지 형식 변환
            for msg in messages:
                role = msg["role"]
                content = []

                # 텍스트 컨텐츠 추가
                if "content" in msg:
                    content.append({"type": "text", "text": msg["content"]})

                # 이미지가 있는 경우 추가
                if role == "user" and hasattr(self, "image_path") and self.image_path:
                    content.append({"type": "image", "image": self.image_path})
                    # 이미지 경로 초기화
                    delattr(self, "image_path")

                conversation.append({"role": role, "content": content})

            print(f"conversation: {conversation}")

            # 입력 포멧팅 및 모델 추론 실행
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_dict=True,
                tokenize=True,
                return_tensors="pt",
            ).to(self.model.device, dtype=torch.bfloat16)

            # inference_mode로 감싸서 실행
            with torch.inference_mode():
                # 한번에 토큰 생성, 스트리밍 시뮬레이션
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

            # 출력 디코딩
            output = self.processor.decode(outputs[0], skip_special_tokens=True)

            # 응답 추출
            response = output.split("model")[-1]

            # 청크 단위로 분할하여 스트리밍 시뮬레이션
            chunk_size = 10  # 단어 단위로 청크 크기 결정
            words = response.split()

            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                yield chunk
                await asyncio.sleep(0.05)

        except Exception as e:
            print(f"Error in stream_chat: {e}")
            yield f"오류가 발생했습니다. 다시 시도해주세요. -  {e}"

    async def build_user_message(
        self, text: str | None, file: UploadFile | None
    ) -> Any:
        """텍스트와 이미지를 HuggingFace 형식으로 변환"""
        if not file:
            print(f"user_message: {text}")
            return text

        try:
            print(f"file: {file}")
            img_bytes = await file.read()

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
            print(f"self.image_path: {temp_img_path}")

            return text or ""

        except Exception as e:
            print(f"이미지 처리 중 오류: {str(e)}")
            # 오류 발생 시 기본 메시지 반환
            return text or ""


async def main():
    # 테스트용 로컬 모델 경로

    model_path = "/home/work/for_train/local_gemma3_4b"

    # HuggingFaceLLM 인스턴스 생성
    llm = HuggingFaceGemma3(model_path=model_path)
    torch._dynamo.disable()

    system_prompt = "You are a helpful assistant."

    # 테스트 메시지
    messages = [
        {"role": "user", "content": "한국의 수도는 어디야?"},
    ]

    print("HuggingFace LLM 응답:")
    async for chunk in llm.stream_chat(system_prompt, messages):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
