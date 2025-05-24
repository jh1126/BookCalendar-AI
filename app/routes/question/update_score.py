from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os, json, torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer

router = APIRouter()

class ScoreUpdateRequest(BaseModel):
    model_name: str

def evaluate_rouge_for_model(model, tokenizer, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"평가 데이터 로드 실패: {e}")

    inputs = [item["paragraph"] for item in data]
    references = [item["target_text"] for item in data]
    predictions = []

    for text in inputs:
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model.generate(input_ids, max_length=128)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        predictions.append(decoded)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for ref, pred in zip(references, predictions)
    ]

    return sum(scores) / len(scores)

@router.post("/question/update_score")
def update_model_score(data: ScoreUpdateRequest):
    model_name = data.model_name
    model_path = os.path.join("models", "question", model_name)
    data_path = os.path.join("data", "question", "processed", "question_data.json")
    metrics_path = os.path.join("data", "question", "question_model_metrics.json")

    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 로딩 실패: {e}")

    try:
        rouge_score = evaluate_rouge_for_model(model, tokenizer, data_path)
        rouge_score = round(rouge_score, 4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ROUGE 평가 실패: {e}")

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)

        updated = False
        for m in model_data:
            if m["model_name"] == model_name:
                m["ROUGE Score"] = rouge_score
                updated = True
                break

        if not updated:
            raise HTTPException(status_code=404, detail=f"모델 '{model_name}'을 metrics.json에서 찾을 수 없습니다.")

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"결과 저장 실패: {e}")

    return {
        "message": f"모델 '{model_name}'의 ROUGE Score가 {rouge_score}로 갱신되었습니다.",
        "ROUGE Score": rouge_score
    }

