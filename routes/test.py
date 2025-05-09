from fastapi import APIRouter, File, UploadFile, HTTPException
from utils import logger, image_variation

router = APIRouter(prefix="/api/test", tags=["test"])


@router.post("/upload")
async def test_imgbb_upload(file: UploadFile = File(...)):
    """ImgBB 이미지 업로드 API 기능을 테스트하는 엔드포인트"""
    try:
        # 파일 읽기
        img_bytes = await file.read()

        # 파일 정보 로깅
        logger.info(
            f"파일명: {file.filename}, 타입: {file.content_type}, 크기: {len(img_bytes)}"
        )

        # 빈 파일 체크
        if not img_bytes:
            logger.error("업로드된 파일이 비어 있습니다.")
            raise HTTPException(400, "업로드된 파일이 비어 있습니다.")

        # ImgBB로 업로드
        img_url = await image_variation(file.filename, img_bytes, file.content_type)

        # 결과 반환
        return {
            "success": True,
            "file_info": {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(img_bytes),
            },
            "image_url": img_url,
        }
    except Exception as e:
        logger.error(f"이미지 업로드 중 오류: {str(e)}")
        raise HTTPException(500, f"이미지 업로드 실패: {str(e)}")
