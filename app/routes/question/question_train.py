from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from transformers import EarlyStoppingCallback
from rouge_score import rouge_scorer

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
    batchSize: int  # 고정 사용하므로 무시됨

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

def save_active_model(model_name):
    try:
        with open(ACTIVE_MODEL_PATH, "w", encoding="utf-8") as f:
            json.dump([{"model_name": model_name}], f, indent=4, ensure_ascii=False)
    except:
        pass

def train_question_model(config: QuestionModelConfig):
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

    fixed_batch_size = 16  # 여기서 고정됨

    training_args = TrainingArguments(
        output_dir=save_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=fixed_batch_size,
        per_device_eval_batch_size=fixed_batch_size,
        num_train_epochs=config.epoch,
        weight_decay=0.01,
        save_total_limit=1,
        eval_accumulation_steps=1,
        fp16=torch.cuda.is_available(),
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
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    save_active_model(config.newModelName)

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

        # ROUGE-L 단독 평가
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rougeL_scores = []

        for ref, pred in zip(references, predictions):
            score = scorer.score(ref, pred)
            rougeL_scores.append(score['rougeL'].fmeasure)

        rougeL_avg = round(sum(rougeL_scores) / len(rougeL_scores), 4)

        metrics = load_metrics()
        updated = False
        for entry in metrics:
            if entry["model_name"] == config.newModelName:
                entry["ROUGE Score"] = rougeL_avg
                entry["q_num"] = fixed_batch_size
                updated = True
                break
        if not updated:
            metrics.append({
                "model_name": config.newModelName,
                "ROUGE-L": rougeL_avg,
                "q_num": fixed_batch_size
            })

        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

    except Exception as e:
        print(f"[ROUGE 평가 실패] {e}")

@router.post("/train_question")
def train_question_api(config: QuestionModelConfig, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(train_question_model, config)
        return JSONResponse(content={"detail": f"{config.newModelName} 훈련이 백그라운드에서 시작되었습니다."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
