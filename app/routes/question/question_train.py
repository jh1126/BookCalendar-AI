from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import os, json, torch

router = APIRouter()

# 요청 데이터 모델 정의
class TrainRequest(BaseModel):
    model_name: str
    epoch: int
    batch_size: int

# 경로 설정
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'question', 'processed', 'summary_data.json')
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, 'models', 'question')
CONFIG_PATH = os.path.join(ROOT_DIR, 'data', 'question', 'question_model_run.json')

# 전처리 함수
def preprocess(example, tokenizer):
    input_text = example["paragraph"]
    target_text = example["question"]
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

# 라우터
@router.post("/train_question_model")
def train_question_model(request: TrainRequest):
    try:
        # 1. 데이터 로드
        with open(DATA_PATH, encoding="utf-8") as f:
            raw_data = json.load(f)
        dataset = Dataset.from_list(raw_data).train_test_split(test_size=0.1)

        # 2. 모델 및 토크나이저 로딩
        model_name_or_path = "digit82/kobart-summarization"
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path)
        model = BartForConditionalGeneration.from_pretrained(model_name_or_path)

        # 3. 전처리
        tokenized = dataset.map(lambda x: preprocess(x, tokenizer), batched=True)
        collator = DataCollatorForSeq2Seq(tokenizer, model)

        # 4. 학습 인자 설정
        output_path = os.path.join(MODEL_SAVE_DIR, request.model_name)
        args = TrainingArguments(
            output_dir=output_path,
            evaluation_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=request.batch_size,
            per_device_eval_batch_size=request.batch_size,
            num_train_epochs=request.epoch,
            weight_decay=0.01,
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )

        # 5. Trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["test"],
            tokenizer=tokenizer,
            data_collator=collator,
        )

        trainer.train()

        # 6. 모델 저장
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        # 7. question_model_run.json 갱신
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump([{"model_name": request.model_name}], f, indent=4, ensure_ascii=False)

        return {
            "status": "success",
            "message": f"모델 '{request.model_name}' 학습 완료 및 저장됨.",
            "model_path": output_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 학습 중 오류: {e}")

