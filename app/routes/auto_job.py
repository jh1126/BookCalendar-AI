import os
import json
from datetime import datetime
import requests

AUTO_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "auto_model.json")
API_BASE = "http://localhost:3004"  # FastAPI 서버 주소 맞는지 확인 필요

def run_auto_training():
    today = datetime.today().strftime("%Y%m%d")

    with open(AUTO_PATH, encoding="utf-8") as f:
        auto_flags = json.load(f)

    def trigger_question():
        payload = {
            "newModelName": f"auto_question_model_v{today}",
            "epoch": 20,
            "batchSize": 8
        }
        requests.post(f"{API_BASE}/train_question", json=payload)
        try:
            res = requests.post(f"{API_BASE}/train_question", json=payload, timeout=10)
            print(f"[question] status: {res.status_code} | response: {res.text}")
        except Exception as e:
            print(f"[question] 요청 실패: {e}")

    def trigger_intent():
        payload = {
            "newModelName": f"auto_intent_model_v{today}",
            "epoch": 10,
            "dropOut": 0.3
        }
        requests.post(f"{API_BASE}/train_intent", json=payload)
        try:
            res = requests.post(f"{API_BASE}/train_intent", json=payload, timeout=10)
            print(f"[intent] status: {res.status_code} | response: {res.text}")
        except Exception as e:
            print(f"[intent] 요청 실패: {e}")

    def trigger_emotion():
        payload = {
            "newModelName": f"auto_emotion_model_v{today}",
            "epoch": 10,
            "dropOut": 0.3
        }
        requests.post(f"{API_BASE}/train_emotion", json=payload)
        try:
            res = requests.post(f"{API_BASE}/train_emotion", json=payload, timeout=10)
            print(f"[emotion] status: {res.status_code} | response: {res.text}")
        except Exception as e:
            print(f"[emotion] 요청 실패: {e}")

    if auto_flags.get("questionAuto") == 1:
        trigger_question()

    if auto_flags.get("intentAuto") == 1:
        trigger_intent()

    if auto_flags.get("emotionAuto") == 1:
        trigger_emotion()
