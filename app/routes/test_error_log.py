from fastapi.testclient import TestClient
from app.main import app 
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app.main import app

client = TestClient(app)

def test_get_error_log():
    response = client.get("/errorRequest")
    print("상태 코드:", response.status_code)
    print("응답 내용:", response.json())

test_get_error_log()
