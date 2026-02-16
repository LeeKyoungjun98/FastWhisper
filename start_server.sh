#!/bin/bash
# ========================================
# Faster-Whisper STT Server 실행 스크립트
# ========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 가상환경 활성화
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "[가상환경 활성화됨: venv]"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "[가상환경 활성화됨: .venv]"
fi

echo
echo "========================================"
echo "  Whisper STT Server 실행 메뉴"
echo "========================================"
echo
echo "[1] WS만 실행 (HTTP, 인증서 불필요)"
echo "[2] WSS만 실행 (HTTPS, 인증서 필요)"
echo "[3] WS + WSS 둘 다 실행 (권장)"
echo "[4] 디버그 모드 (오디오 저장)"
echo "[5] 종료"
echo
read -p "선택하세요 (1-5): " choice

case $choice in
    1)
        echo
        echo "WS 모드로 시작합니다..."
        python whisper_server.py --mode ws
        ;;
    2)
        echo
        echo "WSS 모드로 시작합니다..."
        python whisper_server.py --mode wss
        ;;
    3)
        echo
        echo "WS + WSS 동시 모드로 시작합니다..."
        python whisper_server.py --mode both
        ;;
    4)
        echo
        echo "디버그 모드로 시작합니다..."
        python whisper_server.py --mode ws --debug
        ;;
    5)
        echo "종료합니다."
        exit 0
        ;;
    *)
        echo "잘못된 선택입니다."
        exit 1
        ;;
esac
