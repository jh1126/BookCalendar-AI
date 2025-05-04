from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import evaluate
import pandas as pd
import os
import json
from pydantic import BaseModel

class QuestionModelConfig(BaseModel):
    modelName: str
    epoch: int
    batchSize: int
    dropOut: float  # 현재는 사용하지 않지만, 나중에 config에 반영할 수 있음

def train_question_model(data: QuestionModelConfig):
    # 1. 데이터 로드
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'question', 'processed', 'question_train_data.csv')
    df = pd.read_csv(data_path)
    dataset = Dataset.from_pandas(df)

    # 2. 토크나이저 및 모델 로딩
    model_name = "KETI-AIR/ke-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 3. 전처리 함수 정의
    def preprocess(example):
        inputs = tokenizer(
            example["text"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        targets = tokenizer(
            example["label"],
            max_length=64,
            truncation=True,
            padding="max_length"
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized_dataset = dataset.map(preprocess, remove_columns=["text", "label"])

    # 4. 데이터 분리
    split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # 5. 평가 함수 설정
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"]
        }

    # 6. 훈련 인자 설정
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'question', data.modelName)
    os.makedirs(output_dir, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=data.epoch,
        per_device_train_batch_size=data.batchSize,
        per_device_eval_batch_size=data.batchSize,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        predict_with_generate=True,
        save_total_limit=1,
        learning_rate=5e-5,
        weight_decay=0.01,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 7. 훈련
    trainer.train()

    # 8. 평가 및 ROUGE 점수 추출
    eval_result = trainer.evaluate()
    rouge_score = eval_result["eval_rougeL"]

    # 9. 모델 저장
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 10. 메트릭 저장 (모델 이름 + ROUGE 점수만)
    metrics_file = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'question', 'question_model_metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = []

    metrics.append({
        "model_name": data.modelName,
        "rouge_score": round(rouge_score, 4)
    })

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return rouge_score

        dropout=data.dropOut,
        rouge_score=rouge_score
    )
