"""
Faster-Whisper 실시간 음성 인식 WebSocket 서버
Unity 클라이언트와 연동하여 실시간으로 음성을 텍스트로 변환

[프로덕션 기능]
- API 키 인증
- 동시 처리 제한 (세마포어)
- 최대 연결 수 제한
- 연결 상태 모니터링
- 대기열 시스템
"""

# OpenMP 라이브러리 충돌 방지
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import asyncio
import json
import logging
import io
import wave
import time
import tempfile
import hashlib
import secrets
import numpy as np
from typing import Optional, Dict, Set, List
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form, Header, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import torch

# 오디오 파일 처리용 (선택적)
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # INFO 레벨로 변경 (필요시 DEBUG로)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Windows asyncio 연결 종료 관련 에러 로그 숨기기 (정상적인 노이즈)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# FastAPI 앱 생성
app = FastAPI(title="Faster-Whisper Real-time STT Server")

# CORS 설정 (Unity WebGL 빌드 대응)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Whisper 모델 전역 변수
whisper_model: Optional[WhisperModel] = None
MODEL_SIZE = "large-v3"  # tiny, base, small, medium, large-v2, large-v3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# ============================================
# 🔑 API 키 인증 설정
# ============================================
API_KEYS_FILE = "./api_keys.json"  # API 키 저장 파일
AUTH_ENABLED = True                # 인증 활성화 여부 (--no-auth로 비활성화 가능)

class APIKeyManager:
    """API 키 관리자"""
    
    def __init__(self, keys_file: str):
        self.keys_file = keys_file
        self.api_keys: Dict[str, dict] = {}  # key -> {name, created_at, last_used, request_count}
        self._load_keys()
    
    def _load_keys(self):
        """파일에서 API 키 로드"""
        if os.path.exists(self.keys_file):
            try:
                with open(self.keys_file, "r", encoding="utf-8") as f:
                    self.api_keys = json.load(f)
                logger.info(f"🔑 API 키 {len(self.api_keys)}개 로드됨")
            except Exception as e:
                logger.error(f"❌ API 키 파일 로드 실패: {e}")
                self.api_keys = {}
        else:
            logger.info("🔑 API 키 파일 없음. 새로 생성합니다.")
            self.api_keys = {}
            self._save_keys()
    
    def _save_keys(self):
        """API 키를 파일로 저장"""
        try:
            with open(self.keys_file, "w", encoding="utf-8") as f:
                json.dump(self.api_keys, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ API 키 파일 저장 실패: {e}")
    
    def generate_key(self, name: str = "") -> str:
        """새 API 키 생성"""
        # sk- 접두사 + 48자 랜덤 문자열
        key = f"sk-{secrets.token_hex(24)}"
        
        self.api_keys[key] = {
            "name": name or f"key_{len(self.api_keys) + 1}",
            "created_at": datetime.now().isoformat(),
            "last_used": None,
            "request_count": 0
        }
        self._save_keys()
        logger.info(f"🔑 새 API 키 생성: {name} ({key[:12]}...)")
        return key
    
    def validate_key(self, key: str) -> bool:
        """API 키 검증"""
        if not key or key not in self.api_keys:
            return False
        
        # 사용 기록 업데이트
        self.api_keys[key]["last_used"] = datetime.now().isoformat()
        self.api_keys[key]["request_count"] += 1
        
        # 100번 요청마다 파일 저장 (성능 위해)
        if self.api_keys[key]["request_count"] % 100 == 0:
            self._save_keys()
        
        return True
    
    def revoke_key(self, key: str) -> bool:
        """API 키 삭제"""
        if key in self.api_keys:
            name = self.api_keys[key]["name"]
            del self.api_keys[key]
            self._save_keys()
            logger.info(f"🔑 API 키 삭제됨: {name}")
            return True
        return False
    
    def list_keys(self) -> list:
        """모든 API 키 목록 (키 값은 마스킹)"""
        result = []
        for key, info in self.api_keys.items():
            result.append({
                "key_preview": f"{key[:8]}...{key[-4:]}",
                "name": info["name"],
                "created_at": info["created_at"],
                "last_used": info["last_used"],
                "request_count": info["request_count"]
            })
        return result
    
    def save_all(self):
        """모든 데이터 저장"""
        self._save_keys()

# API 키 관리자 인스턴스
api_key_manager = APIKeyManager(API_KEYS_FILE)

def verify_api_key(api_key: str) -> bool:
    """API 키 검증 (전역 함수)"""
    if not AUTH_ENABLED:
        return True
    return api_key_manager.validate_key(api_key)

async def get_api_key_from_header(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """REST API용 API 키 검증 (Header에서 추출)"""
    if not AUTH_ENABLED:
        return "no-auth"
    
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API 키가 필요합니다. 'X-API-Key' 헤더를 포함해주세요."
        )
    
    if not api_key_manager.validate_key(x_api_key):
        raise HTTPException(
            status_code=403,
            detail="유효하지 않은 API 키입니다."
        )
    
    return x_api_key


# ============================================
# 🚀 프로덕션 설정 (H100 80GB 최적화)
# ============================================
MAX_CONNECTIONS = 200              # 최대 동시 WebSocket 연결 수
MAX_CONCURRENT_TRANSCRIPTIONS = 15  # 동시 음성 인식 처리 수 (H100 80GB 기준)
MAX_AUDIO_BUFFER_SIZE = 16000 * 300  # 최대 오디오 버퍼 크기 (300초)
CONNECTION_TIMEOUT = 600          # 연결 타임아웃 (초) - 5분간 활동 없으면 종료

# ============================================
# 🐛 디버그 설정
# ============================================
DEBUG_MODE = False                 # 디버그 모드 (명령줄에서 --debug로 활성화)
DEBUG_AUDIO_DIR = "./debug_audio"  # 디버그 오디오 저장 폴더

def save_debug_audio(audio_data: bytes, client_id: int, audio_type: str = "realtime", audio_format: str = "float32"):
    """디버그 모드에서 수신한 오디오를 파일로 저장
    
    Args:
        audio_data: 오디오 바이트 데이터
        client_id: 클라이언트 ID
        audio_type: 오디오 타입 (realtime, file)
        audio_format: 오디오 형식 (float32, pcm16)
    """
    if not DEBUG_MODE:
        return None
    
    try:
        # 디버그 폴더 생성
        os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)
        
        # 파일명 생성 (타임스탬프 + 클라이언트ID + 타입)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        if audio_type == "file":
            # 파일 전사: 원본 그대로 저장
            filename = f"{DEBUG_AUDIO_DIR}/{timestamp}_{client_id}_file.audio"
            with open(filename, "wb") as f:
                f.write(audio_data)
        else:
            # 실시간 스트리밍: WAV로 저장
            filename = f"{DEBUG_AUDIO_DIR}/{timestamp}_{client_id}_{audio_format}_realtime.wav"
            
            if audio_format == "pcm16":
                # PCM16: 이미 int16이므로 그대로 저장
                audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
            else:
                # float32: int16으로 변환
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                audio_int16 = (audio_array * 32767).astype(np.int16)
            
            with wave.open(filename, "wb") as wav_file:
                wav_file.setnchannels(1)  # 모노
                wav_file.setsampwidth(2)  # 16비트
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(audio_int16.tobytes())
        
        logger.info(f"🐛 [DEBUG] 오디오 저장: {filename} ({len(audio_data)} bytes, {audio_format})")
        return filename
        
    except Exception as e:
        logger.error(f"🐛 [DEBUG] 오디오 저장 실패: {e}")
        return None

def save_debug_audio_buffer(audio_buffer: list, client_id: int):
    """디버그 모드에서 누적된 오디오 버퍼를 파일로 저장"""
    if not DEBUG_MODE or not audio_buffer:
        return None
    
    try:
        os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{DEBUG_AUDIO_DIR}/{timestamp}_{client_id}_buffer.wav"
        
        # 버퍼를 numpy array로 변환
        audio_array = np.array(audio_buffer, dtype=np.float32)
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        with wave.open(filename, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_int16.tobytes())
        
        duration = len(audio_array) / 16000
        logger.info(f"🐛 [DEBUG] 버퍼 저장: {filename} ({duration:.2f}초)")
        return filename
        
    except Exception as e:
        logger.error(f"🐛 [DEBUG] 버퍼 저장 실패: {e}")
        return None

# 동시 처리 제한 세마포어
transcription_semaphore: Optional[asyncio.Semaphore] = None

# 연결 관리
class ConnectionManager:
    """WebSocket 연결 관리자"""
    
    def __init__(self):
        self.active_connections: Dict[int, dict] = {}  # client_id -> connection info
        self.total_connections_served = 0
        self.total_transcriptions = 0
        self.server_start_time = datetime.now()
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, client_id: int) -> bool:
        """새 연결 등록 (최대 연결 수 체크)"""
        async with self._lock:
            if len(self.active_connections) >= MAX_CONNECTIONS:
                return False
            
            self.active_connections[client_id] = {
                "websocket": websocket,
                "connected_at": datetime.now(),
                "last_activity": datetime.now(),
                "transcription_count": 0,
                "client_info": str(websocket.client)
            }
            self.total_connections_served += 1
            return True
    
    async def disconnect(self, client_id: int):
        """연결 해제"""
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
    
    async def update_activity(self, client_id: int):
        """활동 시간 업데이트"""
        if client_id in self.active_connections:
            self.active_connections[client_id]["last_activity"] = datetime.now()
    
    async def increment_transcription(self, client_id: int):
        """음성 인식 카운트 증가"""
        async with self._lock:
            if client_id in self.active_connections:
                self.active_connections[client_id]["transcription_count"] += 1
            self.total_transcriptions += 1
    
    def get_stats(self) -> dict:
        """서버 통계 반환"""
        uptime = datetime.now() - self.server_start_time
        return {
            "active_connections": len(self.active_connections),
            "max_connections": MAX_CONNECTIONS,
            "total_connections_served": self.total_connections_served,
            "total_transcriptions": self.total_transcriptions,
            "uptime_seconds": int(uptime.total_seconds()),
            "uptime_formatted": str(uptime).split('.')[0]
        }
    
    def get_connection_details(self) -> list:
        """모든 연결 상세 정보"""
        details = []
        for client_id, info in self.active_connections.items():
            connected_duration = datetime.now() - info["connected_at"]
            details.append({
                "client_id": client_id,
                "client_info": info["client_info"],
                "connected_duration": str(connected_duration).split('.')[0],
                "transcription_count": info["transcription_count"]
            })
        return details

# 연결 관리자 인스턴스
connection_manager = ConnectionManager()

logger.info(f"디바이스: {DEVICE}, 연산 타입: {COMPUTE_TYPE}")
logger.info(f"🔧 프로덕션 설정: 최대 연결={MAX_CONNECTIONS}, 동시 처리={MAX_CONCURRENT_TRANSCRIPTIONS}")


class AudioBuffer:
    """실시간 오디오 버퍼 관리 클래스"""
    
    # 지원하는 오디오 형식
    SUPPORTED_FORMATS = ["float32", "pcm16", "int16"]  # int16은 pcm16의 별칭
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.buffer = []
        self.min_audio_length = sample_rate * 1  # 최소 1초
        self.max_buffer_size = MAX_AUDIO_BUFFER_SIZE  # 최대 버퍼 크기 (메모리 보호)
        self.language = None  # None = 자동 감지, 그 외 = 지정된 언어
        self.audio_format = "float32"  # 오디오 형식: float32, pcm16
        self.last_partial_text = ""  # ⭐ 마지막 Partial 텍스트 (변화 감지용)
        self.last_partial_change_time = 0  # ⭐ 마지막으로 Partial이 변한 시간
        self.last_sent_text = ""  # 마지막으로 전송한 텍스트 (중복 방지)
        self.accumulated_text = []  # 누적된 텍스트 세그먼트
        self.silence_threshold = 2.0  # 침묵 감지 시간 (초) - 클라이언트에서 설정 가능
    
    def set_audio_format(self, audio_format: str):
        """오디오 형식 설정
        
        Args:
            audio_format: "float32", "pcm16", "int16", 또는 "auto" (자동 감지)
        """
        fmt = audio_format.lower() if audio_format else "auto"
        if fmt in ["pcm16", "int16"]:
            self.audio_format = "pcm16"
        elif fmt == "float32":
            self.audio_format = "float32"
        else:
            self.audio_format = "auto"  # 자동 감지
        logger.info(f"🎵 오디오 형식 설정: {self.audio_format}")
    
    def detect_audio_format(self, audio_data: bytes) -> str:
        """오디오 형식 자동 감지
        
        휴리스틱:
        1. float32로 파싱했을 때 값이 대부분 -1.0 ~ 1.0 범위 내 → float32
        2. 그렇지 않으면 → pcm16
        
        Returns:
            "float32" 또는 "pcm16"
        """
        try:
            # float32로 시도
            if len(audio_data) % 4 == 0:  # float32는 4바이트
                audio_float = np.frombuffer(audio_data, dtype=np.float32)
                
                # 값 범위 체크 (float32는 보통 -1.0 ~ 1.0)
                # 95% 이상의 값이 -1.5 ~ 1.5 범위 내에 있으면 float32로 판단
                in_range = np.abs(audio_float) <= 1.5
                ratio = np.sum(in_range) / len(audio_float) if len(audio_float) > 0 else 0
                
                if ratio > 0.95:
                    return "float32"
            
            # pcm16로 판단
            return "pcm16"
            
        except Exception:
            return "float32"  # 기본값
        
    def add_chunk(self, audio_data: bytes) -> bool:
        """오디오 청크 추가
        
        Returns:
            bool: 추가 성공 여부 (버퍼 초과 시 False)
        """
        # 자동 감지 모드일 때
        actual_format = self.audio_format
        if self.audio_format == "auto":
            actual_format = self.detect_audio_format(audio_data)
            # 첫 번째 청크에서 감지한 형식을 저장 (이후 청크는 같은 형식 사용)
            if len(self.buffer) == 0:
                logger.info(f"🎵 오디오 형식 자동 감지: {actual_format}")
        
        # 형식에 따라 변환
        if actual_format == "pcm16":
            # PCM16 (int16) → float32 변환
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_int16.astype(np.float32) / 32768.0  # -1.0 ~ 1.0 범위로 정규화
        else:
            # float32 그대로 사용
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
        
        # 버퍼 크기 제한 체크
        if len(self.buffer) + len(audio_array) > self.max_buffer_size:
            logger.warning(f"⚠️ 오디오 버퍼 최대 크기 초과! 오래된 데이터 삭제")
            # 오래된 데이터 삭제 (새 데이터 크기만큼)
            self.buffer = self.buffer[len(audio_array):]
        
        self.buffer.extend(audio_array)
        return True
        
    def get_audio(self) -> Optional[np.ndarray]:
        """버퍼에서 오디오 데이터 가져오기"""
        if len(self.buffer) < self.min_audio_length:
            return None
        
        audio = np.array(self.buffer, dtype=np.float32)
        return audio
    
    def clear(self):
        """버퍼 초기화"""
        self.buffer = []
        self.last_partial_text = ""
        self.last_partial_change_time = 0
        self.last_sent_text = ""
        self.accumulated_text = []
        
    def has_enough_audio(self) -> bool:
        """처리 가능한 충분한 오디오가 있는지 확인"""
        return len(self.buffer) >= self.min_audio_length
    
    def update_partial_text(self, text: str):
        """Partial 텍스트 업데이트 (변화 감지)"""
        # 텍스트가 변경되었으면 시간 갱신
        if text != self.last_partial_text:
            self.last_partial_text = text
            self.last_partial_change_time = time.time()
            logger.info(f"📝 Partial 변화 감지: '{text}' (타이머 리셋)")
            return True
        return False
    
    def is_silent(self) -> bool:
        """침묵 상태인지 확인
        
        조건:
        - Partial 변화가 silence_threshold 동안 없을 때
        """
        if self.last_partial_change_time == 0:
            return False
        
        current_time = time.time()
        elapsed = current_time - self.last_partial_change_time
        
        return elapsed > self.silence_threshold
    
    def get_silence_duration(self) -> float:
        """마지막 Partial 변화 이후 경과 시간 반환"""
        if self.last_partial_change_time == 0:
            return 0.0
        return time.time() - self.last_partial_change_time


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 Whisper 모델 로드"""
    global whisper_model, transcription_semaphore
    
    # 세마포어 초기화
    transcription_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIPTIONS)
    logger.info(f"🔒 동시 처리 세마포어 초기화: {MAX_CONCURRENT_TRANSCRIPTIONS}개")
    
    logger.info(f"Whisper 모델 로딩 중... (모델: {MODEL_SIZE})")
    try:
        whisper_model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root="./models",  # 모델 저장 경로
            cpu_threads=os.cpu_count() or 4,  # CPU 스레드 최대 활용
            num_workers=4  # 전처리 워커 수
        )
        logger.info("✅ Whisper 모델 로드 완료!")
    except Exception as e:
        logger.error(f"❌ 모델 로드 실패: {e}")
        raise


@app.get("/")
async def root():
    """서버 상태 확인 (인증 불필요)"""
    stats = connection_manager.get_stats()
    return {
        "status": "running",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "auth_enabled": AUTH_ENABLED,
        "message": "Faster-Whisper STT Server is ready!",
        "connections": f"{stats['active_connections']}/{MAX_CONNECTIONS}",
        "uptime": stats["uptime_formatted"]
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트 (인증 불필요)"""
    stats = connection_manager.get_stats()
    
    # GPU 메모리 상태 (CUDA인 경우)
    gpu_info = {}
    if DEVICE == "cuda":
        try:
            gpu_info = {
                "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
                "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
                "gpu_name": torch.cuda.get_device_name(0)
            }
        except:
            pass
    
    return {
        "status": "healthy",
        "model_loaded": whisper_model is not None,
        "model": MODEL_SIZE,
        "device": DEVICE,
        **stats,
        **gpu_info
    }


@app.get("/stats")
async def get_stats(api_key: str = Depends(get_api_key_from_header)):
    """상세 서버 통계 (인증 필요)"""
    stats = connection_manager.get_stats()
    connections = connection_manager.get_connection_details()
    
    # 세마포어 상태 확인
    pending_transcriptions = MAX_CONCURRENT_TRANSCRIPTIONS - transcription_semaphore._value if transcription_semaphore else 0
    
    return {
        **stats,
        "max_concurrent_transcriptions": MAX_CONCURRENT_TRANSCRIPTIONS,
        "active_transcriptions": pending_transcriptions,
        "connections_detail": connections
    }


# ============================================
# 🔑 API 키 관리 엔드포인트
# ============================================
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")  # 환경변수 또는 --admin-key로 설정

def verify_admin(x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")):
    """관리자 키 검증"""
    if not ADMIN_KEY:
        raise HTTPException(status_code=403, detail="관리자 키가 설정되지 않았습니다. --admin-key 옵션으로 설정하세요.")
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="관리자 키가 유효하지 않습니다.")
    return True

@app.post("/admin/keys/generate")
async def generate_api_key(
    name: str = Query("", description="API 키 이름/설명"),
    admin: bool = Depends(verify_admin)
):
    """새 API 키 생성 (관리자 전용)"""
    key = api_key_manager.generate_key(name)
    return {"api_key": key, "name": name, "message": "API 키가 생성되었습니다. 안전하게 보관하세요."}

@app.get("/admin/keys")
async def list_api_keys(admin: bool = Depends(verify_admin)):
    """API 키 목록 조회 (관리자 전용)"""
    return {"keys": api_key_manager.list_keys(), "total": len(api_key_manager.api_keys)}

@app.delete("/admin/keys/{key}")
async def revoke_api_key(key: str, admin: bool = Depends(verify_admin)):
    """API 키 삭제 (관리자 전용)"""
    if api_key_manager.revoke_key(key):
        return {"message": "API 키가 삭제되었습니다."}
    raise HTTPException(status_code=404, detail="해당 API 키를 찾을 수 없습니다.")


# ============================================
# 📤 REST API - 파일 업로드 전사
# ============================================
@app.post("/transcribe")
async def transcribe_file_upload(
    file: UploadFile = File(..., description="오디오 파일 (mp3, wav, m4a, ogg, flac 등)"),
    language: Optional[str] = Form(None, description="언어 코드 (예: ko, en, ja). 비워두면 자동 감지"),
    api_key: str = Depends(get_api_key_from_header)
):
    """
    오디오 파일을 업로드하여 텍스트로 변환 (REST API, 인증 필요)
    
    - **file**: 오디오 파일 (필수)
    - **language**: 언어 코드 (선택, 비워두면 자동 감지)
    
    지원 형식: mp3, wav, m4a, ogg, flac, webm 등
    """
    # 파일 크기 제한 (100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    try:
        # 파일 읽기
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")
        
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"파일이 너무 큽니다. 최대 {MAX_FILE_SIZE // 1024 // 1024}MB")
        
        # 언어 설정
        lang = None if (language == "" or language is None) else language
        lang_display = lang if lang else "자동 감지"
        
        logger.info(f"📤 REST API 파일 전사 요청: {file.filename} ({len(file_bytes)} bytes, 언어: {lang_display})")
        
        # 전사 실행
        result = await transcribe_file_bytes(file_bytes, lang, client_id=None)
        
        if "error" in result and result["error"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        logger.info(f"📤 REST API 전사 완료: {result['text'][:100]}...")
        
        return JSONResponse(content={
            "success": True,
            "text": result["text"],
            "language": result["language"],
            "language_probability": result["language_probability"],
            "duration": result["duration"],
            "segments": result["segments"]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ REST API 전사 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe/base64")
async def transcribe_base64(
    data: dict,
    api_key: str = Depends(get_api_key_from_header)
):
    """
    Base64 인코딩된 오디오 데이터를 텍스트로 변환 (인증 필요)
    
    요청 본문:
    {
        "audio": "base64 인코딩된 오디오 데이터",
        "language": "ko"  // 선택사항
    }
    """
    import base64
    
    try:
        audio_base64 = data.get("audio")
        language = data.get("language")
        
        if not audio_base64:
            raise HTTPException(status_code=400, detail="audio 필드가 필요합니다.")
        
        # Base64 디코딩
        try:
            file_bytes = base64.b64decode(audio_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="잘못된 Base64 형식입니다.")
        
        # 언어 설정
        lang = None if (language == "" or language is None) else language
        
        logger.info(f"📤 REST API (Base64) 전사 요청: {len(file_bytes)} bytes")
        
        # 전사 실행
        result = await transcribe_file_bytes(file_bytes, lang, client_id=None)
        
        if "error" in result and result["error"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return JSONResponse(content={
            "success": True,
            "text": result["text"],
            "language": result["language"],
            "language_probability": result["language_probability"],
            "duration": result["duration"],
            "segments": result["segments"]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ REST API (Base64) 전사 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def silence_detector(websocket: WebSocket, audio_buffer: AudioBuffer, client_id: int, stop_event: asyncio.Event):
    """침묵 감지 및 자동 final 전송 백그라운드 태스크"""
    logger.info(f"[{client_id}] 🎯 침묵 감지 태스크 시작! (Partial 변화 없이 {audio_buffer.silence_threshold}초 경과 시 Final 전송)")
    try:
        while not stop_event.is_set():
            await asyncio.sleep(0.5)  # 0.5초마다 체크
            
            silence_duration = audio_buffer.get_silence_duration()
            has_text = len(audio_buffer.accumulated_text) > 0
            
            # 상태 로그 (누적 텍스트가 있고 침묵이 절반 이상일 때만)
            threshold_half = audio_buffer.silence_threshold / 2
            if has_text and silence_duration > threshold_half:
                logger.info(f"[{client_id}] ⏳ Partial 변화 없음 {silence_duration:.1f}초 / 누적: {len(audio_buffer.accumulated_text)}개")
            
            # 침묵 감지 (Partial 변화 없음)
            if audio_buffer.is_silent() and audio_buffer.accumulated_text:
                # 누적된 텍스트가 있으면 final로 전송
                final_text = " ".join(audio_buffer.accumulated_text).strip()
                
                if final_text and final_text != audio_buffer.last_sent_text:
                    logger.info(f"[{client_id}] 🔇🔇🔇 침묵 {silence_duration:.1f}초 감지 → Final 전송!")
                    logger.info(f"[{client_id}] 📤 Final 텍스트: {final_text}")
                    
                    await websocket.send_json({
                        "type": "final",
                        "text": final_text,
                        "language": audio_buffer.language or "auto"
                    })
                    
                    # 버퍼 초기화
                    audio_buffer.accumulated_text = []
                    audio_buffer.last_sent_text = final_text
                    audio_buffer.last_partial_text = ""
                    audio_buffer.last_partial_change_time = 0  # 타이머 리셋
                    audio_buffer.buffer = []
                    logger.info(f"[{client_id}] ✅ Final 전송 완료, 버퍼 초기화, 다음 문장 대기 중...")
                    
    except Exception as e:
        logger.error(f"[{client_id}] 침묵 감지기 오류: {e}", exc_info=True)
    finally:
        logger.info(f"[{client_id}] 🛑 침묵 감지 태스크 종료")


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """실시간 음성 인식 WebSocket 엔드포인트 (API 키 인증 필요)"""
    client_id = id(websocket)
    
    # 연결 수 제한 체크
    await websocket.accept()
    
    # API 키 인증 체크
    if AUTH_ENABLED:
        # 첫 번째 메시지로 API 키를 받음
        try:
            auth_data = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            auth_message = json.loads(auth_data)
            api_key = auth_message.get("api_key", "")
            
            if not api_key_manager.validate_key(api_key):
                logger.warning(f"🔒 인증 실패: {websocket.client}")
                await websocket.send_json({
                    "type": "error",
                    "code": "AUTH_FAILED",
                    "message": "유효하지 않은 API 키입니다."
                })
                await websocket.close(code=4001)
                return
            
            logger.info(f"🔑 인증 성공: {websocket.client} (키: {api_key[:8]}...)")
            await websocket.send_json({
                "type": "auth",
                "status": "success",
                "message": "인증 성공"
            })
            
        except asyncio.TimeoutError:
            logger.warning(f"🔒 인증 타임아웃: {websocket.client}")
            await websocket.send_json({
                "type": "error",
                "code": "AUTH_TIMEOUT",
                "message": "인증 타임아웃. 연결 후 10초 내에 API 키를 전송해주세요."
            })
            await websocket.close(code=4002)
            return
        except Exception as e:
            logger.warning(f"🔒 인증 오류: {websocket.client} - {e}")
            await websocket.send_json({
                "type": "error",
                "code": "AUTH_ERROR",
                "message": "인증 처리 중 오류가 발생했습니다."
            })
            await websocket.close(code=4003)
            return
    
    if not await connection_manager.connect(websocket, client_id):
        # 연결 수 초과 - 거부 메시지 보내고 종료
        logger.warning(f"⚠️ 연결 거부 (최대 {MAX_CONNECTIONS}개 초과): {websocket.client}")
        await websocket.send_json({
            "type": "error",
            "code": "MAX_CONNECTIONS",
            "message": f"서버가 혼잡합니다. 잠시 후 다시 시도해주세요. (현재 {MAX_CONNECTIONS}명 접속 중)"
        })
        await websocket.close(code=1013)  # 1013 = Try Again Later
        return
    
    stats = connection_manager.get_stats()
    logger.info(f"🔌 클라이언트 연결됨: {websocket.client} (현재 {stats['active_connections']}/{MAX_CONNECTIONS})")
    
    audio_buffer = AudioBuffer()
    stop_event = asyncio.Event()
    silence_task = None
    
    # 파일 전사 모드 상태
    file_mode = False
    file_language = None
    file_buffer = bytearray()
    
    try:
        # 연결 확인 메시지 전송
        await websocket.send_json({
            "type": "connected",
            "message": "서버에 연결되었습니다",
            "model": MODEL_SIZE,
            "device": DEVICE,
            "server_load": f"{stats['active_connections']}/{MAX_CONNECTIONS}"
        })
        
        while True:
            # 클라이언트로부터 데이터 수신
            try:
                data = await websocket.receive()
            except RuntimeError as e:
                # 연결이 이미 종료된 경우 (정상 종료)
                if "disconnect" in str(e).lower():
                    logger.info(f"[{client_id}] 클라이언트 정상 종료")
                    break
                raise
            
            # 텍스트 메시지 처리 (제어 명령)
            if "text" in data:
                message = json.loads(data["text"])
                command = message.get("command")
                language = message.get("language")  # 언어 정보 받기
                silence_threshold = message.get("silence_threshold")  # 침묵 감지 시간 받기
                audio_format = message.get("audio_format")  # 오디오 형식 (float32, pcm16)
                
                if command == "start":
                    # 빈 문자열("")이나 None이면 자동 감지로 처리
                    if language == "" or language is None:
                        audio_buffer.language = None  # 자동 감지
                    else:
                        audio_buffer.language = language  # 지정된 언어
                    
                    # 오디오 형식 설정 (auto, float32, pcm16)
                    audio_buffer.set_audio_format(audio_format or "auto")
                    
                    # 침묵 감지 시간 설정 (클라이언트에서 전송한 값 사용)
                    if silence_threshold is not None and silence_threshold > 0:
                        audio_buffer.silence_threshold = float(silence_threshold)
                        logger.info(f"[{client_id}] 침묵 감지 시간 설정: {audio_buffer.silence_threshold}초")
                    
                    lang_display = audio_buffer.language if audio_buffer.language else "자동 감지"
                    logger.info(f"[{client_id}] 녹음 시작 (언어: {lang_display}, 형식: {audio_buffer.audio_format}, 침묵: {audio_buffer.silence_threshold}초)")
                    audio_buffer.clear()
                    
                    # 침묵 감지 태스크 시작
                    if silence_task is None or silence_task.done():
                        stop_event.clear()
                        silence_task = asyncio.create_task(
                            silence_detector(websocket, audio_buffer, client_id, stop_event)
                        )
                        logger.info(f"[{client_id}] 침묵 감지 태스크 시작")
                    
                    await websocket.send_json({
                        "type": "status",
                        "message": f"녹음 시작됨 (언어: {lang_display}, 형식: {audio_buffer.audio_format}, 침묵: {audio_buffer.silence_threshold}초)"
                    })
                    
                elif command == "stop":
                    logger.info(f"[{client_id}] 녹음 중지, 최종 처리 중...")
                    
                    # 침묵 감지 태스크 중지
                    stop_event.set()
                    if silence_task and not silence_task.done():
                        await silence_task
                        logger.info(f"[{client_id}] 침묵 감지 태스크 중지")
                    
                    # 버퍼에 남은 오디오 처리
                    if audio_buffer.buffer:
                        audio = audio_buffer.get_audio()
                        if audio is not None:
                            result = await transcribe_audio(audio, audio_buffer.language, client_id)
                            if result["text"].strip():
                                audio_buffer.accumulated_text.append(result["text"])
                    
                    # 누적된 텍스트가 있으면 final 전송
                    if audio_buffer.accumulated_text:
                        final_text = " ".join(audio_buffer.accumulated_text).strip()
                        await websocket.send_json({
                            "type": "final",
                            "text": final_text,
                            "language": audio_buffer.language or "auto"
                        })
                        logger.info(f"[{client_id}] ⏹️ 수동 중지 → Final 전송: {final_text}")
                    
                    audio_buffer.clear()
                    await websocket.send_json({
                        "type": "status",
                        "message": "녹음 중지됨"
                    })
                    
                elif command == "clear":
                    audio_buffer.clear()
                    file_buffer.clear()
                    file_mode = False
                    logger.info(f"[{client_id}] 버퍼 초기화")
                
                elif command == "transcribe_file":
                    # 파일 전사 모드 시작
                    file_mode = True
                    file_buffer.clear()
                    
                    # 언어 설정
                    if language == "" or language is None:
                        file_language = None
                    else:
                        file_language = language
                    
                    lang_display = file_language if file_language else "자동 감지"
                    logger.info(f"[{client_id}] 📁 파일 전사 모드 시작 (언어: {lang_display})")
                    
                    await websocket.send_json({
                        "type": "status",
                        "message": f"파일 전사 모드 시작 (언어: {lang_display}). 오디오 파일을 전송하세요."
                    })
                
                elif command == "transcribe_file_end":
                    # 파일 전송 완료, 전사 시작
                    if not file_mode or len(file_buffer) == 0:
                        await websocket.send_json({
                            "type": "error",
                            "message": "전송된 파일이 없습니다. transcribe_file 명령 후 파일을 전송하세요."
                        })
                        continue
                    
                    logger.info(f"[{client_id}] 📁 파일 수신 완료 ({len(file_buffer)} bytes), 전사 시작...")
                    
                    # 🐛 디버그: 파일 저장
                    save_debug_audio(bytes(file_buffer), client_id, "file")
                    
                    await websocket.send_json({
                        "type": "status",
                        "message": f"파일 수신 완료 ({len(file_buffer)} bytes). 전사 중..."
                    })
                    
                    try:
                        # 파일 전사 실행
                        result = await transcribe_file_bytes(bytes(file_buffer), file_language, client_id)
                        
                        await websocket.send_json({
                            "type": "file_result",
                            "text": result["text"],
                            "language": result["language"],
                            "duration": result.get("duration", 0),
                            "segments": result.get("segments", [])
                        })
                        logger.info(f"[{client_id}] 📁 파일 전사 완료: {result['text'][:100]}...")
                        
                    except Exception as e:
                        logger.error(f"[{client_id}] 파일 전사 오류: {e}", exc_info=True)
                        await websocket.send_json({
                            "type": "error",
                            "message": f"파일 전사 오류: {str(e)}"
                        })
                    
                    # 파일 모드 종료
                    file_mode = False
                    file_buffer.clear()
                    
            # 바이너리 데이터 처리 (오디오)
            elif "bytes" in data:
                audio_data = data["bytes"]
                
                # 🐛 디버그: 수신한 오디오 청크 저장
                save_debug_audio(audio_data, client_id, "file" if file_mode else "realtime", audio_buffer.audio_format)
                
                # 파일 모드일 때는 파일 버퍼에 저장
                if file_mode:
                    file_buffer.extend(audio_data)
                    # 진행 상황 로그 (1MB마다)
                    if len(file_buffer) % (1024 * 1024) < len(audio_data):
                        logger.info(f"[{client_id}] 📁 파일 수신 중... {len(file_buffer) / 1024 / 1024:.1f} MB")
                    continue
                
                audio_buffer.add_chunk(audio_data)
                
                # 충분한 오디오가 쌓이면 실시간 전사
                if audio_buffer.has_enough_audio():
                    audio = audio_buffer.get_audio()
                    
                    if audio is not None:
                        # 🐛 디버그: 전사 직전 버퍼 저장
                        save_debug_audio_buffer(audio_buffer.buffer, client_id)
                        
                        lang_display = audio_buffer.language if audio_buffer.language else "자동 감지"
                        logger.info(f"[{client_id}] 오디오 처리 중... (길이: {len(audio)/16000:.2f}초, 언어: {lang_display})")
                        
                        # 활동 시간 업데이트
                        await connection_manager.update_activity(client_id)
                        
                        result = await transcribe_audio(audio, audio_buffer.language, client_id)
                        
                        if result["text"].strip():  # 빈 텍스트가 아닌 경우만 전송
                            # ⭐ Partial 텍스트 변화 감지 (침묵 타이머 리셋)
                            audio_buffer.update_partial_text(result["text"])
                            
                            # Partial 결과 전송
                            await websocket.send_json({
                                "type": "partial",
                                "text": result["text"],
                                "language": result["language"],
                                "duration": len(audio) / 16000
                            })
                            detected_lang = result.get("language", "unknown")
                            logger.info(f"[{client_id}] ✅ 인식 결과 (감지된 언어: {detected_lang}): {result['text']}")
                            
                            # 누적 텍스트에 추가
                            audio_buffer.accumulated_text.append(result["text"])
                            logger.info(f"[{client_id}] 📝 누적 텍스트 추가 (총 {len(audio_buffer.accumulated_text)}개)")
                        
                        # 버퍼 완전 초기화 (중복 방지)
                        # 오버랩을 제거하여 중복 인식 방지
                        audio_buffer.buffer = []
                        
    except WebSocketDisconnect:
        logger.info(f"🔌 클라이언트 연결 해제: {websocket.client}")
    except RuntimeError as e:
        # 연결 종료 관련 에러는 무시 (정상 종료)
        if "disconnect" in str(e).lower() or "receive" in str(e).lower():
            logger.info(f"🔌 클라이언트 정상 종료: {websocket.client}")
        else:
            logger.error(f"❌ WebSocket 런타임 오류: {e}")
    except Exception as e:
        logger.error(f"❌ WebSocket 오류: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        # 연결 관리자에서 제거
        await connection_manager.disconnect(client_id)
        stats = connection_manager.get_stats()
        logger.info(f"🔌 연결 종료 (남은 연결: {stats['active_connections']}/{MAX_CONNECTIONS})")
        
        # 침묵 감지 태스크 종료
        stop_event.set()
        if silence_task and not silence_task.done():
            try:
                await silence_task
            except:
                pass
        
        try:
            await websocket.close()
        except:
            pass


async def transcribe_audio(audio: np.ndarray, language: str = None, client_id: int = None) -> dict:
    """오디오를 텍스트로 변환
    
    Args:
        audio: 오디오 데이터
        language: 언어 코드 (None이면 자동 감지)
        client_id: 클라이언트 ID (통계용)
    """
    try:
        # 세마포어로 동시 처리 제한 (GPU 과부하 방지)
        async with transcription_semaphore:
            # 통계 업데이트
            if client_id:
                await connection_manager.increment_transcription(client_id)
            
            # Whisper 모델 실행 (동기 함수를 비동기로 실행)
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                lambda: whisper_model.transcribe(
                    audio,
                    language=language,  # None이면 자동 감지, 값이 있으면 해당 언어 사용
                    beam_size=5,  # 빔 서치 크기 (기본 5, 크면 정확도↑ 속도↓)
                    best_of=5,  # 후보 개수 (기본 5)
                    temperature=0.0,  # 0.0 = 가장 확실한 결과만
                    condition_on_previous_text=False,  # 환각 방지
                    initial_prompt=None,  # 환각 방지
                    no_speech_threshold=0.8,  # 무음 감지 강화
                    log_prob_threshold=-0.5,  # 낮은 확률 세그먼트 제거
                    compression_ratio_threshold=2.4,  # 반복 감지
                    repetition_penalty=1.2,  # 반복 억제
                    vad_filter=True,  # VAD 필터 사용
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        threshold=0.6,
                        min_speech_duration_ms=250
                    )
                )
            )
            
            # 세그먼트 리스트로 변환
            segments_list = list(segments)
            
            # 전체 텍스트 조합
            full_text = " ".join([segment.text for segment in segments_list])
            
            # 세그먼트 정보
            segments_info = [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "confidence": segment.avg_logprob
                }
                for segment in segments_list
            ]
            
            return {
                "text": full_text.strip(),
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": segments_info
            }
        
    except Exception as e:
        logger.error(f"❌ 전사 오류: {e}", exc_info=True)
        return {
            "text": "",
            "language": "unknown",
            "language_probability": 0.0,
            "segments": []
        }


async def transcribe_file_bytes(file_bytes: bytes, language: str = None, client_id: int = None) -> dict:
    """오디오 파일 바이트를 텍스트로 변환
    
    Args:
        file_bytes: 오디오 파일 바이트 (mp3, wav, m4a, ogg, flac 등)
        language: 언어 코드 (None이면 자동 감지)
        client_id: 클라이언트 ID (통계용)
    
    Returns:
        전사 결과 딕셔너리
    """
    temp_file = None
    try:
        # 세마포어로 동시 처리 제한
        async with transcription_semaphore:
            # 통계 업데이트
            if client_id:
                await connection_manager.increment_transcription(client_id)
            
            # 임시 파일로 저장 (faster-whisper가 파일 경로를 받을 수 있음)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".audio")
            temp_file.write(file_bytes)
            temp_file.close()
            
            logger.info(f"📁 임시 파일 생성: {temp_file.name} ({len(file_bytes)} bytes)")
            
            # Whisper 모델 실행
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                lambda: whisper_model.transcribe(
                    temp_file.name,  # 파일 경로 전달
                    language=language,
                    beam_size=5,
                    best_of=5,
                    temperature=0.0,
                    condition_on_previous_text=True,  # 파일 전사에서는 맥락 사용
                    initial_prompt=None,
                    no_speech_threshold=0.6,
                    log_prob_threshold=-1.0,
                    compression_ratio_threshold=2.4,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        threshold=0.5,
                        min_speech_duration_ms=250
                    )
                )
            )
            
            # 세그먼트 리스트로 변환
            segments_list = list(segments)
            
            # 전체 텍스트 조합
            full_text = " ".join([segment.text for segment in segments_list])
            
            # 세그먼트 정보 (타임스탬프 포함)
            segments_info = [
                {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip(),
                    "confidence": round(segment.avg_logprob, 3)
                }
                for segment in segments_list
            ]
            
            # 총 오디오 길이 계산
            duration = segments_list[-1].end if segments_list else 0
            
            return {
                "text": full_text.strip(),
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
                "duration": round(duration, 2),
                "segments": segments_info
            }
    
    except Exception as e:
        logger.error(f"❌ 파일 전사 오류: {e}", exc_info=True)
        return {
            "text": "",
            "language": "unknown",
            "language_probability": 0.0,
            "duration": 0,
            "segments": [],
            "error": str(e)
        }
    
    finally:
        # 임시 파일 삭제
        if temp_file:
            try:
                os.unlink(temp_file.name)
                logger.info(f"📁 임시 파일 삭제: {temp_file.name}")
            except:
                pass


if __name__ == "__main__":
    import uvicorn
    import argparse
    import threading
    
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="Faster-Whisper STT Server")
    parser.add_argument("--mode", choices=["ws", "wss", "both"], default="both",
                        help="실행 모드: ws(HTTP만), wss(HTTPS만), both(둘 다) - 기본값: both")
    parser.add_argument("--ws-port", type=int, default=9880, help="WS 포트 (기본값: 9880)")
    parser.add_argument("--wss-port", type=int, default=9880, help="WSS 포트 (기본값: 9880)")
    parser.add_argument("--ssl-key", default="./key.pem", help="SSL 개인키 경로")
    parser.add_argument("--ssl-cert", default="./cert.pem", help="SSL 인증서 경로")
    parser.add_argument("--debug", action="store_true", 
                        help="디버그 모드: 수신한 오디오를 파일로 저장 (./debug_audio/)")
    parser.add_argument("--debug-dir", default="./debug_audio", 
                        help="디버그 오디오 저장 폴더 (기본값: ./debug_audio)")
    parser.add_argument("--no-auth", action="store_true",
                        help="API 키 인증 비활성화 (개발/테스트용)")
    parser.add_argument("--admin-key", default="",
                        help="관리자 키 (API 키 생성/삭제에 필요)")
    parser.add_argument("--generate-key", default="",
                        help="서버 시작 시 API 키 자동 생성 (이름 지정)")
    args = parser.parse_args()
    
    # 디버그 모드 설정
    if args.debug:
        DEBUG_MODE = True
        DEBUG_AUDIO_DIR = args.debug_dir
        os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)
        logger.info(f"🐛 디버그 모드 활성화! 오디오 저장 폴더: {os.path.abspath(DEBUG_AUDIO_DIR)}")
    
    # 인증 설정
    if args.no_auth:
        AUTH_ENABLED = False
        logger.warning("⚠️ API 키 인증이 비활성화되었습니다! (--no-auth)")
    else:
        AUTH_ENABLED = True
        logger.info("🔑 API 키 인증 활성화됨")
    
    # 관리자 키 설정
    if args.admin_key:
        ADMIN_KEY = args.admin_key
        logger.info(f"🔑 관리자 키 설정됨: {ADMIN_KEY[:4]}...")
    else:
        # 관리자 키 자동 생성
        ADMIN_KEY = secrets.token_hex(16)
        logger.info(f"🔑 관리자 키 자동 생성됨: {ADMIN_KEY}")
        logger.info(f"   → 이 키를 사용하여 API 키를 관리하세요")
    
    # API 키 자동 생성
    if args.generate_key:
        key = api_key_manager.generate_key(args.generate_key)
        logger.info(f"🔑 API 키 자동 생성: {key}")
    
    # API 키가 없으면 하나 자동 생성
    if AUTH_ENABLED and len(api_key_manager.api_keys) == 0:
        key = api_key_manager.generate_key("default")
        logger.info(f"🔑 기본 API 키 생성됨: {key}")
        logger.info(f"   → 이 키를 클라이언트에 제공하세요")
    
    # SSL 인증서 경로
    SSL_KEYFILE = args.ssl_key
    SSL_CERTFILE = args.ssl_cert
    
    # SSL 인증서 존재 여부 확인
    ssl_available = os.path.exists(SSL_KEYFILE) and os.path.exists(SSL_CERTFILE)
    
    def run_ws_server():
        """WS (HTTP) 서버 실행"""
        logger.info(f"🔓 WS 서버 시작: ws://0.0.0.0:{args.ws_port}")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=args.ws_port,
            log_level="info",
            ws_ping_interval=30,
            ws_ping_timeout=30
        )
    
    def run_wss_server():
        """WSS (HTTPS) 서버 실행"""
        logger.info(f"🔒 WSS 서버 시작: wss://0.0.0.0:{args.wss_port}")
        logger.info(f"   - 인증서: {os.path.abspath(SSL_CERTFILE)}")
        logger.info(f"   - 개인키: {os.path.abspath(SSL_KEYFILE)}")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=args.wss_port,
            log_level="info",
            ws_ping_interval=30,
            ws_ping_timeout=30,
            ssl_keyfile=SSL_KEYFILE,
            ssl_certfile=SSL_CERTFILE
        )
    
    # 서버 실행
    logger.info("🚀 Faster-Whisper STT 서버 시작...")
    logger.info(f"📋 설정: 최대 연결={MAX_CONNECTIONS}, 동시 처리={MAX_CONCURRENT_TRANSCRIPTIONS}")
    
    if args.mode == "ws":
        # WS만 실행
        run_ws_server()
        
    elif args.mode == "wss":
        # WSS만 실행
        if not ssl_available:
            logger.error("❌ SSL 인증서를 찾을 수 없습니다!")
            logger.error(f"   - 필요: {SSL_KEYFILE}, {SSL_CERTFILE}")
            logger.error("   → generate_ssl_cert.bat을 실행하거나 --mode ws 로 시작하세요")
            exit(1)
        run_wss_server()
        
    else:  # both
        # WS와 WSS 둘 다 실행
        if not ssl_available:
            logger.warning("⚠️ SSL 인증서가 없어서 WS 모드로만 실행합니다")
            logger.warning(f"   → WSS도 사용하려면 {SSL_KEYFILE}, {SSL_CERTFILE} 파일을 생성하세요")
            run_ws_server()
        else:
            logger.info("🌐 WS + WSS 동시 실행 모드")
            
            # WSS를 별도 스레드에서 실행
            wss_thread = threading.Thread(target=run_wss_server, daemon=True)
            wss_thread.start()
            
            # WS는 메인 스레드에서 실행
            run_ws_server()

