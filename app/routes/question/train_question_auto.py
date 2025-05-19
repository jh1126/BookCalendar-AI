#자동화 라우터, 현재는 train 라우터 그대로 옮겨옴. 후에 데이터 파트만 변곁하면 
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
from datasets import Dataset
from rouge_score import rouge_scorer
import os, json, torch
from database import get_connection

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

MONTH_PATH = os.path.join(ROOT_DIR, 'data', 'question', 'train_data_month.json')
MODEL_DIR = os.path.join(ROOT_DIR, 'models', 'question')
ACTIVE_MODEL_PATH = os.path.join(ROOT_DIR, 'app', 'models', 'question', 'question_model_run.json')
METRICS_PATH = os.path.join(ROOT_DIR, 'data', 'question', 'question_model_metrics.json')


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_model_metrics(model_name: str, rouge_l: float, q_num: int):
    try:
        metrics = load_metrics()
        updated = False
        for entry in metrics:
            if entry["model_name"] == model_name:
                entry["ROUGE Score"] = round(rouge_l, 3)
                entry["q_num"] = q_num
                updated = True
                break
        if not updated:
            metrics.append({
                "model_name": model_name,
                "ROUGE Score": round(rouge_l, 3),
                "q_num": q_num
            })
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
    except:
        pass

def save_active_model(model_name):
    try:
        with open(ACTIVE_MODEL_PATH, "w", encoding="utf-8") as f:
            json.dump([{"model_name": model_name}], f, indent=4, ensure_ascii=False)
    except:
        pass


#db에서 해당 달 데이터 불러오기
def fetch_question_data_by_month(month: str):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                SELECT paragraph, summary
                FROM questionData
                WHERE date LIKE %s
            """
            cursor.execute(sql, (f"{month}%",))
            return cursor.fetchall()
    finally:
        conn.close()

def train_question_model_auto(config: QuestionModelConfig):
    # 자동화 할 데이터 달 로
    with open(MONTH_PATH, encoding="utf-8") as f:
        month_info = json.load(f)
    target_month = month_info.get("questionDataLoad")
    if not target_month:
        raise RuntimeError("자동화 학습용 달 정보가 없습니다.")
    
    #  해당 월의 데이터 로드(DB에서)
    raw_data = fetch_question_data_by_month(target_month)
    if not raw_data:
        raise RuntimeError(f"{target_month}에 해당하는 DB 데이터가 없습니다.")

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
        save_strategy="no",
        learning_rate=5e-5,
        per_device_train_batch_size=config.batchSize,
        per_device_eval_batch_size=config.batchSize,
        num_train_epochs=config.epoch,
        weight_decay=0.01,
        save_total_limit=1,
        eval_accumulation_steps=1,
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

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # ROUGE 평가
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    preds = trainer.predict(tokenized["test"])

    decoded_preds = tokenizer.batch_decode(preds.predictions[:20], skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(preds.label_ids[:20], skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    rouge_l_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(decoded_labels, decoded_preds)]
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    save_model_metrics(config.newModelName, avg_rouge_l, q_num=config.batchSize)
    save_active_model(config.newModelName)



@router.post("/train_question_auto")
def train_question_auto_api(config: QuestionModelConfig, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(train_question_model_auto, config)
        return {message : "질문생성자동화시작" }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
