from fastapi import APIRouter
import os
import json
from pydantic import BaseModel
from fastapi.responses import JSONResponse

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
            model_name = run_data[0]["model_name"]
        except:
            model_name = ""

        # metrics.json 로드
        # 모델 타입에 따른 score_key 설정
        if model_type == "emotion":
            score_key = "f1_score"
        elif model_type == "intent":
            score_key = "accuracy"
        elif model_type == "question":
            score_key = "ROUGE Score"
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        try:
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
            score = next(
                (entry.get(score_key, 0.0) for entry in metrics_data if entry.get("model_name") == model_name),
                0.0
            )
        except:
            score = 0.0

        # 결과에 추가
        result[f"{model_type}Model"] = model_name
        print(model_name)
        result[f"{model_type}Score"] = round(score, 4)
        print(score)

    return JSONResponse(content=result)

