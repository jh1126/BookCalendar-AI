import os
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "error_log.txt")
MAX_LINES = 50  # 에러 수 최대 50개까지만 유지

def log_error(message: str):
    # 시간 정보 포함
    timestamped_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"

    # 기존 로그 읽기
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = []

    # 에러 추가 후 오래된 로그 제거
    lines.append(timestamped_message + "\n")
    lines = lines[-MAX_LINES:]

    # 로그 저장
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
