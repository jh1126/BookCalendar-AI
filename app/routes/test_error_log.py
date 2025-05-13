import sys
import os

# 🔧 BookCalendar-AI 루트를 path에 추가
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, BASE_DIR)

from main import app

from fastapi.testclient import TestClient

client = TestClient(app)

response = client.get("/errorRequest")

print("응답 상태 코드:", response.status_code)
print("응답 본문:", response.json())
