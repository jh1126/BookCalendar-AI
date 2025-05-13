# test_error_log.py
from fastapi.testclient import TestClient
from your_app_module import app  # FastAPI app 객체 import

client = TestClient(app)

def test_get_recent_errors():
    response = client.get("/errorRequest")
    assert response.status_code == 200
    print("응답 시작 :", response.json())
