from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os, json, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

router = APIRouter()

class PreviewRequest(BaseModel):
    model_name: str
    sample_size: int = 3

@router.post("/question/preview_score")
def preview_model_output(data: PreviewRequest):
    model_name = data.model_name
    sample_size = data.sample_size

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    model_path = os.path.join(PROJECT_ROOT, "models", "question", model_name)
    data_path = os.path.join(PROJECT_ROOT, "data", "question", "processed", "question_all_data.json")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cpu")  # GPU 대신 CPU 사용
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 로딩 실패: {e}")

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = data[:sample_size]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 로딩 실패: {e}")

    results = []
    smoothie = SmoothingFunction().method4
    for item in data:
        paragraph = item.get("paragraph", "")
        reference = item.get("summary", "")
        inputs = tokenizer.encode(paragraph, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        outputs = model.generate(inputs, max_length=128)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        bleu_score = sentence_bleu(
            [reference.split()], prediction.split(), smoothing_function=smoothie
        )

        results.append({
            "paragraph": paragraph[:100] + ("..." if len(paragraph) > 100 else ""),
            "prediction": prediction,
            "reference": reference,
            "BLEU": round(bleu_score, 4)
        })

    return {"samples": results}

