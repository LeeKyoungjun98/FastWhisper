#!/bin/bash
# ========================================
# Faster-Whisper STT Server 설치 스크립트
# Ubuntu/Debian 기준
# ========================================

set -e  # 오류 발생 시 중단

echo "========================================"
echo "  Faster-Whisper STT Server 설치"
echo "========================================"
echo

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 현재 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Python 버전 확인
echo -e "${YELLOW}[1/5] Python 버전 확인...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}Python $PYTHON_VERSION 발견${NC}"
else
    echo -e "${RED}Python3가 설치되어 있지 않습니다.${NC}"
    echo "설치: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

# 가상환경 생성
echo
echo -e "${YELLOW}[2/5] 가상환경 생성...${NC}"
if [ -d "venv" ]; then
    echo "기존 가상환경 발견, 건너뜀"
else
    python3 -m venv venv
    echo -e "${GREEN}가상환경 생성 완료${NC}"
fi

# 가상환경 활성화
echo
echo -e "${YELLOW}[3/5] 가상환경 활성화...${NC}"
source venv/bin/activate
echo -e "${GREEN}가상환경 활성화됨${NC}"

# pip 업그레이드
echo
echo -e "${YELLOW}[4/5] pip 업그레이드...${NC}"
pip install --upgrade pip

# 의존성 설치
echo
echo -e "${YELLOW}[5/5] 패키지 설치...${NC}"
pip install -r requirements.txt

# CUDA 확인
echo
echo -e "${YELLOW}CUDA 확인...${NC}"
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo -e "${GREEN}✓ CUDA 사용 가능${NC}"
    python3 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
else
    echo -e "${YELLOW}⚠ CUDA 사용 불가 (CPU 모드로 실행됩니다)${NC}"
fi

echo
echo "========================================"
echo -e "${GREEN}  설치 완료!${NC}"
echo "========================================"
echo
echo "실행 방법:"
echo "  ./start_server.sh"
echo
echo "또는 수동 실행:"
echo "  source venv/bin/activate"
echo "  python whisper_server.py --mode ws"
echo
