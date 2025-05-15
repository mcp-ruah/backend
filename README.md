# back
mcp backend

## quick start
```
uv init --python 3.13 // 원하는 버전		


//pyproject.toml이 존재할 경우
uv venv --python 3.13 	// .venv 가상환경 설치

// Window
./.venv/Scripts/activate

// MacOS
source .venv/bin/activate 


// 라이브러리 설치
uv pip install -e .
```

```
// 백엔드 서버 시작(루트폴더에서 실행 )
uvicorn app:app --host 127.0.0.1 --port 8000
```



## Linux Docker 서비스 관리 명령어
```
# Docker 서비스 상태 확인
sudo systemctl status docker

# Docker 서비스 중지
sudo systemctl stop docker

# Docker 서비스 시작
sudo systemctl start docker

# 시스템 부팅 시 자동 시작 설정
sudo systemctl enable docker

# 시스템 부팅 시 자동 시작 해제
sudo systemctl disable docker
```