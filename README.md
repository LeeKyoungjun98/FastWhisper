# Faster-Whisper STT Server

Faster-Whisper 기반 실시간 음성 인식(STT) 서버입니다.
WebSocket 실시간 스트리밍과 REST API 파일 전사를 모두 지원합니다.

## 주요 기능

- **실시간 음성 인식** - WebSocket으로 마이크 음성을 실시간 텍스트 변환
- **파일 전사** - 오디오 파일(mp3, wav, m4a 등)을 텍스트로 변환
- **다중 접속** - 동시에 여러 사용자 접속 지원 (세마포어 기반 GPU 과부하 방지)
- **API 키 인증** - API 키를 통한 접근 제어
- **WS / WSS** - HTTP, HTTPS 모두 지원
- **자동 언어 감지** - 언어를 지정하지 않으면 자동 감지
- **오디오 형식 자동 감지** - float32, PCM16 자동 판별
- **침묵 감지** - 말이 끝나면 자동으로 최종 텍스트 전송
- **디버그 모드** - 수신한 오디오를 파일로 저장하여 확인 가능

---

## 요구사항

- Python 3.10 이상
- CUDA 11.x 또는 12.x (GPU 사용 시)
- 8GB+ RAM
- ffmpeg

---

## 설치

### Linux (Ubuntu/Debian)

```bash
# 1. 시스템 패키지 설치
sudo apt update && sudo apt install -y python3 python3-pip python3-venv ffmpeg

# 2. 프로젝트 클론
git clone https://github.com/LeeKyoungjun98/FastWhisper.git
cd FastWhisper

# 3. 스크립트 줄바꿈 변환 및 권한 부여
sed -i 's/\r$//' *.sh && chmod +x *.sh

# 4. 설치 스크립트 실행 (가상환경 생성 + 패키지 설치)
./install.sh

# 5. 서버 실행
./start_server.sh
```

### Windows

```powershell
# 1. 프로젝트 클론
git clone https://github.com/your-repo/FastWhisper.git
cd FastWhisper

# 2. 가상환경 생성
python -m venv venv
.\venv\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 서버 실행
start_server.bat
```

---

## 서버 실행

### 기본 실행

```bash
python whisper_server.py --mode ws --ws-port 8000
```

### 실행 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--mode` | `both` | `ws`, `wss`, `both` 중 선택 |
| `--ws-port` | `8765` | WS(HTTP) 포트 |
| `--wss-port` | `8766` | WSS(HTTPS) 포트 |
| `--ssl-key` | `./key.pem` | SSL 개인키 경로 |
| `--ssl-cert` | `./cert.pem` | SSL 인증서 경로 |
| `--debug` | off | 수신한 오디오를 파일로 저장 |
| `--debug-dir` | `./debug_audio` | 디버그 오디오 저장 폴더 |
| `--no-auth` | off | API 키 인증 비활성화 (테스트용) |
| `--admin-key` | 자동생성 | 관리자 키 직접 지정 |
| `--generate-key` | - | 서버 시작 시 API 키 생성 |

### 실행 예시

```bash
# 인증 없이 테스트
python whisper_server.py --mode ws --ws-port 8000 --no-auth

# WSS + 인증
python whisper_server.py --mode wss --wss-port 8001 --admin-key "my-secret"

# WS + WSS 동시 실행
python whisper_server.py --mode both --ws-port 8000 --wss-port 8001

# 디버그 모드
python whisper_server.py --mode ws --ws-port 8000 --debug
```

---

## 백그라운드 실행 (Linux)

MobaXterm 등 SSH 클라이언트를 종료해도 서버가 유지되도록 하려면:

```bash
# screen 설치 (최초 1회)
sudo apt install -y screen

# screen 세션 생성 + 자동 재시작 스크립트 실행
screen -S whisper
./run_forever.sh

# 분리 (서버 유지한 채 터미널 빠져나오기)
# Ctrl+A 누르고 D

# 다시 접속
screen -r whisper

# 세션 목록 확인
screen -ls
```

---

## API 키 인증

서버를 처음 시작하면 **관리자 키**와 **기본 API 키**가 자동 생성되어 로그에 출력됩니다.

### API 키 관리 (관리자)

```bash
# 새 API 키 생성
curl -X POST "http://서버:포트/admin/keys/generate?name=유저이름" \
  -H "X-Admin-Key: 관리자키"

# API 키 목록 조회
curl "http://서버:포트/admin/keys" \
  -H "X-Admin-Key: 관리자키"

# API 키 삭제
curl -X DELETE "http://서버:포트/admin/keys/sk-삭제할키" \
  -H "X-Admin-Key: 관리자키"
```

---

## API 사용법

### 1. WebSocket - 실시간 음성 인식

```
엔드포인트: ws://서버:포트/ws/transcribe
```

#### 접속 흐름

```
1. WebSocket 연결

2. API 키 인증 (인증 활성화 시)
   → { "api_key": "sk-your-api-key" }
   ← { "type": "auth", "status": "success" }

3. 연결 확인 메시지 수신
   ← { "type": "connected", "model": "large-v3", ... }

4. 녹음 시작
   → { "command": "start", "language": "ko", "audio_format": "auto" }

5. 오디오 데이터 전송 (바이너리)
   → [audio bytes]

6. 실시간 결과 수신
   ← { "type": "partial", "text": "안녕하세요" }
   ← { "type": "final", "text": "안녕하세요 반갑습니다" }

7. 녹음 중지
   → { "command": "stop" }
```

#### start 명령어 옵션

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `language` | `""` (자동감지) | 언어 코드 (ko, en, ja 등) |
| `audio_format` | `"auto"` | `auto`, `float32`, `pcm16` |
| `silence_threshold` | `2.0` | 침묵 감지 시간 (초) |

### 2. WebSocket - 파일 전사

```
1. 파일 전사 모드 시작
   → { "command": "transcribe_file", "language": "ko" }

2. 파일 바이트 전송 (바이너리)
   → [file bytes]

3. 전송 완료
   → { "command": "transcribe_file_end" }

4. 결과 수신
   ← { "type": "file_result", "text": "...", "duration": 30.5, "segments": [...] }
```

### 3. REST API - 파일 업로드

```
POST /transcribe
```

```bash
curl -X POST http://서버:포트/transcribe \
  -H "X-API-Key: sk-your-api-key" \
  -F "file=@audio.mp3" \
  -F "language=ko"
```

#### 응답

```json
{
  "success": true,
  "text": "전사된 텍스트",
  "language": "ko",
  "language_probability": 0.98,
  "duration": 30.52,
  "segments": [
    { "start": 0.0, "end": 2.5, "text": "안녕하세요", "confidence": -0.25 }
  ]
}
```

### 4. REST API - Base64

```
POST /transcribe/base64
```

```bash
curl -X POST http://서버:포트/transcribe/base64 \
  -H "X-API-Key: sk-your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"audio": "BASE64_DATA", "language": "ko"}'
```

---

## 엔드포인트 정리

| 엔드포인트 | 메서드 | 인증 | 설명 |
|-----------|--------|------|------|
| `/` | GET | X | 서버 상태 |
| `/health` | GET | X | 헬스 체크 (GPU 정보 포함) |
| `/stats` | GET | O | 상세 통계 |
| `/transcribe` | POST | O | 파일 업로드 전사 |
| `/transcribe/base64` | POST | O | Base64 전사 |
| `/ws/transcribe` | WebSocket | O | 실시간 / 파일 전사 |
| `/admin/keys/generate` | POST | 관리자 | API 키 생성 |
| `/admin/keys` | GET | 관리자 | API 키 목록 |
| `/admin/keys/{key}` | DELETE | 관리자 | API 키 삭제 |

---

## 프로덕션 설정

`whisper_server.py` 상단에서 조정:

```python
MAX_CONNECTIONS = 200               # 최대 동시 접속 수
MAX_CONCURRENT_TRANSCRIPTIONS = 15  # 동시 음성 인식 처리 수
MAX_AUDIO_BUFFER_SIZE = 16000 * 300 # 최대 오디오 버퍼 (300초)
```

### GPU별 권장 설정

| GPU | MAX_CONCURRENT | MAX_CONNECTIONS |
|-----|----------------|-----------------|
| RTX 3060 (12GB) | 2 | 50 |
| RTX 3090 (24GB) | 4~5 | 100 |
| RTX 4090 (24GB) | 5~6 | 150 |
| H100 (80GB) | 15 | 200 |

---

## 파일 구조

```
FasterWhisper/
├── whisper_server.py         # 메인 서버 코드
├── requirements.txt          # Python 패키지 목록
├── api_keys.json             # API 키 저장 (자동 생성)
│
├── install.sh                # Linux 설치 스크립트
├── start_server.sh           # Linux 실행 메뉴
├── run_forever.sh            # 자동 재시작 스크립트
├── setup_service.sh          # systemd 서비스 설정 (systemd 환경용)
├── whisper-stt.service       # systemd 서비스 파일
│
├── install_dependencies.bat  # Windows 패키지 설치
├── start_server.bat          # Windows 실행 메뉴
├── start_server_ws.bat       # Windows WS 실행
├── start_server_wss.bat      # Windows WSS 실행
├── start_server_both.bat     # Windows WS+WSS 실행
├── start_server_debug.bat    # Windows 디버그 모드
│
├── test_transcribe.py        # 파일 전사 테스트
├── test_transcribe.bat       # 테스트 실행 (Windows)
│
├── generate_ssl_cert.py      # SSL 인증서 생성
├── generate_ssl_cert.bat     # SSL 인증서 생성 (Windows)
├── cert.pem                  # SSL 인증서
├── key.pem                   # SSL 개인키
│
├── models/                   # Whisper 모델 (자동 다운로드)
└── debug_audio/              # 디버그 오디오 저장 (--debug 시)
```

---

## 문제 해결

### NumPy 버전 오류

```bash
pip install "numpy<2"
```

### python-multipart 오류

```bash
pip install python-multipart
```

### CUDA Out of Memory

```python
# whisper_server.py에서 동시 처리 수 줄이기
MAX_CONCURRENT_TRANSCRIPTIONS = 2

# 또는 더 작은 모델 사용
MODEL_SIZE = "medium"
```

### 포트 충돌

```bash
# Linux: 포트 사용 중인 프로세스 찾기
sudo lsof -i :8000

# 프로세스 종료
kill -9 프로세스번호

# 또는 한번에
pkill -f whisper_server
```

### Windows asyncio 에러 (WinError 10054)

정상적인 동작입니다. 서버 코드에서 자동으로 숨김 처리됩니다.

---

## 라이선스

MIT License
