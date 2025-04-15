# back
mcp backend

## quick start
```
uv init --python 3.13 // 원하는 버전		

uv venv 	// .venv 가상환경 설치

// Window
./.venv/Scripts/activate

// MacOS
source .venv/bin/activate 


// 라이브러리 설치
uv pip install -e .
```

```
// 백엔드 서버 시작
uvicorn main:app --host 127.0.0.1 --port 8000
```