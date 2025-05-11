from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from transformers import (
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset, load_metric
import os, json, torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

router = APIRouter()

# 요청 데이터 포맷
class QuestionModelConfig(BaseModel):
    newModelName: str
    epoch: int
    batchSize: int

# 경로 설정
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))

DATA_PATH = os.path.join(ROOT_DIR, 'data', 'question', 'processed', 'question_all_data.json')
MODEL_DIR = os.path.join(ROOT_DIR, 'models', 'question')
ACTIVE_MODEL_PATH = os.path.join(ROOT_DIR, 'app', 'models', 'question', 'question_model_run.json')
METRICS_PATH = os.path.join(ROOT_DIR, 'data', 'question', 'question_model_metrics.json')

# ===================== 내부 함수 =====================

def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)

def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_model_metrics(model_name: str, rouge_l: float):
    try:
        metrics = load_metrics()
        updated = False
        for entry in metrics:
            if entry["model_name"] == model_name:
                entry["ROUGE Score"] = round(rouge_l, 3)
                updated = True
                break
        if not updated:
            metrics.append({
                "model_name": model_name,
                "ROUGE Score": round(rouge_l, 3)
            })

        print("METRICS 저장 시작")
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print("METRICS 저장 완료")
    except Exception as e:
        print(f"METRICS 저장 실패: {e}")

def save_active_model(model_name):
    try:
        print("ACTIVE MODEL 저장 시작")
        with open(ACTIVE_MODEL_PATH, "w", encoding="utf-8") as f:
            json.dump([{"model_name": model_name}], f, indent=4, ensure_ascii=False)
        print("ACTIVE MODEL 저장 완료")
    except Exception as e:
        print(f"ACTIVE MODEL 저장 실패: {e}")

# ===================== 메인 학습 함수 =====================

def train_question_model(config: QuestionModelConfig):
    print("질문 생성 모델 학습 시작")
    raw_data = load_data()
    dataset = Dataset.from_list(raw_data).train_test_split(test_size=0.1)

    model_name_or_path = "digit82/kobart-summarization"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path)
    model = BartForConditionalGeneration.from_pretrained(model_name_or_path)

    def preprocess(example):
        input_text = example["paragraph"]
        target_text = example["summary"]
        model_inputs = tokenizer(
            input_text,
            max_length=512,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            target_text,
            max_length=64,
            padding="max_length",
            truncation=True
        )["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs

    tokenized = dataset.map(preprocess, batched=True)
    collator = DataCollatorForSeq2Seq(tokenizer, model)

    save_path = os.path.join(MODEL_DIR, config.newModelName)
    os.makedirs(save_path, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=save_path,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=config.batchSize,
        per_device_eval_batch_size=config.batchSize,
        num_train_epochs=config.epoch,
        weight_decay=0.01,
        save_total_limit=1,
        eval_accumulation_steps=8,
        fp16=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    print("학습 완료, 모델 저장 중...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("모델 저장 완료")

    # ROUGE 평가
    print("ROUGE 평가 시작")
    metric = load_metric("rouge")
    preds = trainer.predict(tokenized["test"])

    decoded_preds = tokenizer.batch_decode(
        [pred.tolist() if hasattr(pred, "tolist") else pred for pred in preds.predictions],
        skip_special_tokens=True
    )
    decoded_labels = tokenizer.batch_decode(
        [label.tolist() if hasattr(label, "tolist") else label for label in preds.label_ids],
        skip_special_tokens=True
    )

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    rouge_l = result["rougeL"].mid.fmeasure
    print(f"ROUGE-L F1: {rouge_l:.4f}")

    save_model_metrics(config.newModelName, rouge_l)
    save_active_model(config.newModelName)

    print("전체 학습 및 평가 완료")

# ===================== FastAPI 엔드포인트 =====================

@router.post("/train_question")
def train_question_api(config: QuestionModelConfig, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(train_question_model, config)
        return {"message": f"질문 생성 모델 '{config.newModelName}' 학습이 시작되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

