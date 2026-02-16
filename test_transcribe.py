"""
Whisper STT 서버 파일 전사 테스트
- WebSocket 방식
- REST API 방식
"""

import asyncio
import json
import sys

# ============================================
# 설정
# ============================================
SERVER_HOST = "112.147.51.230"  # 서버 주소
SERVER_PORT = 8765         # 서버 포트
TEST_AUDIO_FILE = "test.mp3"  # 테스트할 오디오 파일 경로


# ============================================
# 1. REST API 방식 테스트 (파일 업로드)
# ============================================
def test_rest_api(audio_file: str, language: str = None):
    """REST API로 파일 전사 테스트"""
    import requests
    
    print("=" * 50)
    print("📤 REST API 방식 테스트")
    print("=" * 50)
    
    url = f"http://{SERVER_HOST}:{SERVER_PORT}/transcribe"
    
    try:
        with open(audio_file, "rb") as f:
            files = {"file": (audio_file, f)}
            data = {"language": language} if language else {}
            
            print(f"파일: {audio_file}")
            print(f"언어: {language or '자동 감지'}")
            print("전송 중...")
            
            response = requests.post(url, files=files, data=data, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                print("\n✅ 전사 성공!")
                print(f"텍스트: {result['text']}")
                print(f"언어: {result['language']} (확률: {result['language_probability']})")
                print(f"길이: {result['duration']}초")
                print(f"세그먼트 수: {len(result['segments'])}")
                
                if result['segments']:
                    print("\n📝 세그먼트:")
                    for seg in result['segments'][:5]:  # 처음 5개만 출력
                        print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
                    if len(result['segments']) > 5:
                        print(f"  ... 외 {len(result['segments']) - 5}개")
            else:
                print(f"❌ 오류: {response.status_code}")
                print(response.text)
                
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {audio_file}")
    except requests.exceptions.ConnectionError:
        print(f"❌ 서버에 연결할 수 없습니다: {url}")
    except Exception as e:
        print(f"❌ 오류: {e}")


# ============================================
# 2. WebSocket 방식 테스트
# ============================================
async def test_websocket(audio_file: str, language: str = None):
    """WebSocket으로 파일 전사 테스트"""
    import websockets
    
    print("=" * 50)
    print("🔌 WebSocket 방식 테스트")
    print("=" * 50)
    
    uri = f"ws://{SERVER_HOST}:{SERVER_PORT}/ws/transcribe"
    
    try:
        # 오디오 파일 읽기
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        
        print(f"파일: {audio_file} ({len(audio_bytes)} bytes)")
        print(f"언어: {language or '자동 감지'}")
        print(f"연결 중: {uri}")
        
        async with websockets.connect(uri) as ws:
            # 연결 확인 메시지 수신
            response = await ws.recv()
            data = json.loads(response)
            print(f"연결됨: {data.get('message', '')}")
            
            # 1. 파일 전사 모드 시작
            await ws.send(json.dumps({
                "command": "transcribe_file",
                "language": language or ""
            }))
            
            response = await ws.recv()
            data = json.loads(response)
            print(f"상태: {data.get('message', '')}")
            
            # 2. 파일 바이트 전송
            print("파일 전송 중...")
            await ws.send(audio_bytes)
            
            # 3. 전송 완료 알림
            await ws.send(json.dumps({
                "command": "transcribe_file_end"
            }))
            
            # 상태 메시지 수신
            response = await ws.recv()
            data = json.loads(response)
            print(f"상태: {data.get('message', '')}")
            
            # 4. 결과 수신
            print("전사 대기 중...")
            response = await ws.recv()
            result = json.loads(response)
            
            if result.get("type") == "file_result":
                print("\n✅ 전사 성공!")
                print(f"텍스트: {result['text']}")
                print(f"언어: {result['language']}")
                print(f"길이: {result.get('duration', 0)}초")
                print(f"세그먼트 수: {len(result.get('segments', []))}")
                
                segments = result.get('segments', [])
                if segments:
                    print("\n📝 세그먼트:")
                    for seg in segments[:5]:
                        print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
                    if len(segments) > 5:
                        print(f"  ... 외 {len(segments) - 5}개")
            else:
                print(f"❌ 오류: {result}")
                
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {audio_file}")
    except Exception as e:
        print(f"❌ 오류: {e}")


# ============================================
# 메인
# ============================================
def main():
    print("=" * 50)
    print("  Whisper STT 파일 전사 테스트")
    print("=" * 50)
    print()
    
    # 오디오 파일 경로 (명령줄 인자 또는 기본값)
    audio_file = sys.argv[1] if len(sys.argv) > 1 else TEST_AUDIO_FILE
    language = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"서버: {SERVER_HOST}:{SERVER_PORT}")
    print(f"파일: {audio_file}")
    print(f"언어: {language or '자동 감지'}")
    print()
    
    # 테스트 방식 선택
    print("[1] REST API 방식")
    print("[2] WebSocket 방식")
    print("[3] 둘 다 테스트")
    print()
    
    choice = input("선택하세요 (1-3): ").strip()
    print()
    
    if choice == "1":
        test_rest_api(audio_file, language)
    elif choice == "2":
        asyncio.run(test_websocket(audio_file, language))
    elif choice == "3":
        test_rest_api(audio_file, language)
        print()
        asyncio.run(test_websocket(audio_file, language))
    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()
