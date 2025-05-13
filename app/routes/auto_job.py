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
        requests.post(f"{API_BASE}/questionModelTrain", json=payload)

    def trigger_intent():
        payload = {
            "newModelName": f"auto_intent_model_v{today}",
            "epoch": 10,
            "dropOut": 0.3
        }
        requests.post(f"{API_BASE}/intentionModelTrain", json=payload)

    def trigger_emotion():
        payload = {
            "newModelName": f"auto_emotion_model_v{today}",
            "epoch": 10,
            "dropOut": 0.3
        }
        requests.post(f"{API_BASE}/emotionModelTrain", json=payload)

    if auto_flags.get("questionModelAuto") == 1:
        trigger_question()

    if auto_flags.get("intentModelAuto") == 1:
        trigger_intent()

    if auto_flags.get("emotionModelAuto") == 1:
        trigger_emotion()
