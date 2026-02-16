#!/bin/bash
# ========================================
# systemd 서비스 자동 설정 스크립트
# ========================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 현재 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_USER=$(whoami)
SERVICE_FILE="/etc/systemd/system/whisper-stt.service"

echo "========================================"
echo "  Whisper STT systemd 서비스 설정"
echo "========================================"
echo

# root 권한 확인
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}sudo 권한이 필요합니다.${NC}"
    echo "다시 실행: sudo ./setup_service.sh"
    exit 1
fi

# 실제 사용자 확인 (sudo 실행 시 원래 사용자)
if [ -n "$SUDO_USER" ]; then
    ACTUAL_USER=$SUDO_USER
else
    ACTUAL_USER=$CURRENT_USER
fi

echo -e "${YELLOW}[1/4] 서비스 파일 생성 중...${NC}"

# 서비스 파일 생성
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Faster-Whisper STT Server
After=network.target

[Service]
User=$ACTUAL_USER
Group=$ACTUAL_USER
WorkingDirectory=$SCRIPT_DIR
Environment="PATH=$SCRIPT_DIR/venv/bin"
Environment="CUDA_VISIBLE_DEVICES=0"

ExecStart=$SCRIPT_DIR/venv/bin/python whisper_server.py --mode ws

Restart=always
RestartSec=10

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}서비스 파일 생성됨: $SERVICE_FILE${NC}"

echo
echo -e "${YELLOW}[2/4] systemd 데몬 리로드...${NC}"
systemctl daemon-reload
echo -e "${GREEN}완료${NC}"

echo
echo -e "${YELLOW}[3/4] 서비스 활성화 (부팅 시 자동 시작)...${NC}"
systemctl enable whisper-stt
echo -e "${GREEN}완료${NC}"

echo
echo -e "${YELLOW}[4/4] 서비스 시작...${NC}"
systemctl start whisper-stt
sleep 2

# 상태 확인
if systemctl is-active --quiet whisper-stt; then
    echo -e "${GREEN}✓ 서비스가 실행 중입니다!${NC}"
else
    echo -e "${RED}✗ 서비스 시작 실패${NC}"
    echo "로그 확인: sudo journalctl -u whisper-stt -n 50"
    exit 1
fi

echo
echo "========================================"
echo -e "${GREEN}  설정 완료!${NC}"
echo "========================================"
echo
echo "유용한 명령어:"
echo "  상태 확인:    sudo systemctl status whisper-stt"
echo "  로그 확인:    sudo journalctl -u whisper-stt -f"
echo "  재시작:       sudo systemctl restart whisper-stt"
echo "  중지:         sudo systemctl stop whisper-stt"
echo "  비활성화:     sudo systemctl disable whisper-stt"
echo
