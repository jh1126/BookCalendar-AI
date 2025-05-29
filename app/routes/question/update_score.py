from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os, json, torch, random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

router = APIRouter()

class ScoreUpdateRequest(BaseModel):
    model_name: str

def evaluate_bleu_for_model(model, tokenizer, data_path, sample_size=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"평가 데이터 로드 실패: {e}")

    if len(data) > sample_size:
        data = random.sample(data, sample_size)

    inputs = [item["paragraph"] for item in data]
    references = [item["summary"] for item in data]
    predictions = []

    for text in inputs:
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model.generate(input_ids, max_length=128)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        predictions.append(decoded)

    smoothie = SmoothingFunction().method4
    scores = [
        sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
        for ref, pred in zip(references, predictions)
    ]

    return round(sum(scores) / len(scores), 4)

@router.post("/question/update_score")
def update_model_score(data: ScoreUpdateRequest):
    model_name = data.model_name
    CURRENT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
    model_path = os.path.join(PROJECT_ROOT, "models", "question", model_name)
    data_path = os.path.join(PROJECT_ROOT, "data", "question", "processed", "question_all_data.json")
    metrics_path = os.path.join(PROJECT_ROOT, "data", "question", "question_model_metrics.json")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 로딩 실패: {e}")

    try:
        bleu_score = evaluate_bleu_for_model(model, tokenizer, data_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BLEU 평가 실패: {e}")

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)

        updated = False
        for m in model_data:
            if m["model_name"] == model_name:
                m["BLEU Score"] = bleu_score
                updated = True
                break

        if not updated:
            raise HTTPException(status_code=404, detail=f"모델 '{model_name}'을 metrics.json에서 찾을 수 없습니다.")

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"결과 저장 실패: {e}")

    return {
        "message": f"모델 '{model_name}'의 ROUGE Score가 {bleu_score}로 갱신되었습니다.",
        "ROUGE Score": bleu_score  # 이름은 그대로 두되 실제 값은 BLEU
    }


    return {
        "message": f"모델 '{model_name}'의 ROUGE Score가 {rouge_score}로 갱신되었습니다.",
        "ROUGE Score": rouge_score
    }
