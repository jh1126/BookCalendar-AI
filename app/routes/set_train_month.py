from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import json

router = APIRouter()


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
MONTH_CONFIG_PATH = os.path.join(PROJECT_ROOT, "data", "train_data_month.json")

class MonthConfig(BaseModel):
    questionDataLoad: str
    intentDataLoad: str
    emotionDataLoad: str


@router.post("/set_train_month")
def set_train_month(config: MonthConfig):
    try:
