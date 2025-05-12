import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("./src"))


# src 폴더의 app 객체를 가져옴
from src.main import app

# 이 파일을 직접 실행할 경우 서버 시작
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
