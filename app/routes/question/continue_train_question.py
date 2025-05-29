from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from transformers import EarlyStoppingCallback

from transformers import (
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import os, json, torch, random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

router = APIRouter()

class QuestionModelConfig(BaseModel):
    newModelName: str
    epoch: int
    batchSize: int

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))

DATA_PATH = os.path.join(ROOT_DIR, 'data', 'question', 'processed', 'question_all_data.json')
MODEL_DIR = os.path.join(ROOT_DIR, 'models', 'question')
ACTIVE_MODEL_PATH = os.path.join(ROOT_DIR, 'app', 'models', 'question', 'question_model_run.json')
METRICS_PATH = os.path.join(ROOT_DIR, 'data', 'question', 'question_model_metrics.json')

def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)

def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def get_current_model_name():
    try:
        with open(ACTIVE_MODEL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data[0]["model_name"] if data else None
    except:
        return None

import shutil

def continue_training_question_model(config: QuestionModelConfig):
    raw_data = load_data()
    dataset = Dataset.from_list(raw_data).train_test_split(test_size=0.1)

    base_model_name = get_current_model_name()
    if base_model_name is None:
        raise RuntimeError("현재 사용 중인 question 모델을 찾을 수 없습니다.")

    # 새로운 모델 이름과 경로 설정
    new_model_name = config.newModelName
    base_model_path = os.path.join(MODEL_DIR, base_model_name)
    new_model_path = os.path.join(MODEL_DIR, new_model_name)

    # 모델 디렉토리 복사
    if os.path.exists(new_model_path):
        raise RuntimeError(f"이미 존재하는 모델 이름입니다: {new_model_name}")
    shutil.copytree(base_model_path, new_model_path)

    # 모델/토크나이저 로드 (복사된 위치에서)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(new_model_path)
    model = BartForConditionalGeneration.from_pretrained(new_model_path)

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

    batchsiz = 16
    training_args = TrainingArguments(
        output_dir=new_model_path,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batchsize,
        per_device_eval_batch_size=batchsiz,
        num_train_epochs=config.epoch,
        weight_decay=0.01,
        save_total_limit=1,
        eval_accumulation_steps=1,
        fp16=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    model.save_pretrained(new_model_path)
    tokenizer.save_pretrained(new_model_path)

    # ===== BLEU 평가 및 저장 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    try:
        eval_data = raw_data
        if len(eval_data) > 100:
            eval_data = random.sample(eval_data, 100)

        inputs = [item["paragraph"] for item in eval_data]
        references = [item["summary"] for item in eval_data]
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

        bleu_score = round(sum(scores) / len(scores), 4)

        metrics = load_metrics()
        metrics.append({
            "model_name": new_model_name,
            "BLEU Score": bleu_score,
            "q_num": config.batchSize
        })

        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        # 현재 모델 교체 (선택 사항)
        with open(ACTIVE_MODEL_PATH, "w", encoding="utf-8") as f:
            json.dump([{"model_name": new_model_name}], f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"[BLEU 평가 실패] {e}")


@router.post("/continue_train_question")
def continue_train_question_api(config: QuestionModelConfig, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(continue_training_question_model, config)
        return JSONResponse(content={"detail": "추가 학습이 백그라운드에서 시작되었습니다."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
