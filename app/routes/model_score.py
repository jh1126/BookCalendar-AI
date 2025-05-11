from fastapi import APIRouter
import os
import json

router = APIRouter()

MODEL_TYPES = ["question", "intent", "emotion"]

@router.get("/testModel")
def get_current_models_status():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    result = {}

    for model_type in MODEL_TYPES:
        # 경로 설정
        run_path = os.path.join(BASE_DIR, "..","models",model_type,f"{model_type}_model_run.json")
        metrics_path = os.path.join(BASE_DIR, "..","..", "data", model_type, f"{model_type}_model_metrics.json")
      
        # run.json 로드
        try:
            with open(run_path, "r") as f:
                run_data = json.load(f)
            model_name = run_data.get("model_name", "")
        except:
            model_name = ""

        # metrics.json 로드
        score_key = "ROUGE Score" if model_type == "question" else "f1_score"
        try:
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
            score = next(
                (entry.get(score_key, 0.0) for entry in metrics_data if entry.get("model_name") == model_name),
                0.0
            )
        except:
            f1_score = 0.0

        # 결과에 추가
        result[f"{model_type}Model"] = model_name
        result[f"{model_type}Score"] = round(f1_score, 4)

    return result
