from fastapi import UploadFile
import os, httpx
from dotenv import load_dotenv
from core.utils import logger
import base64

load_dotenv()

api_key = os.getenv("IMGBB_API_KEY")


async def image_variation(file_name: str, data: bytes, mime: str) -> str:
    assert api_key, "IMGBB_API_KEY가 설정되지 않았습니다."
    if not data:
        logger.error(
            "이미지 데이터가 비어 있습니다. 파일이 올바르게 읽혔는지 확인하세요."
        )
        raise ValueError(
            "이미지 데이터가 비어 있습니다. 파일이 올바르게 읽혔는지 확인하세요."
        )
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://api.imgbb.com/1/upload?key={api_key}",
            files={"image": (file_name, data, mime or "application/octet-stream")},
        )
        logger.info(f"ImgBB 응답: {resp.text}")

    try:
        resp.raise_for_status()
    except Exception as e:
        print("ImgBB 응답:", resp.text)
        raise
    return resp.json()["data"]["url"]


async def image_to_base64(file: UploadFile) -> str:
    # 이미지가 있을대 - base64로 변환
    img_bytes = await file.read()

    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    return base64_image
