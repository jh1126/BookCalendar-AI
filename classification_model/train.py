# koBERT Library import
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

# Transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

#setting Library
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import pandas as pd

#GPU 활용
#device = torch.device("cuda:0") 
#CPU 활용
#device = torch.device("cpu")

# BERT 모델, Vocabulary 불러오기
bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")

# [AI Hub] 감정 분류를 위한 감정 라벨링 대화 데이터셋 불러오기
data = pd.read_csv(".csv", encoding='cp949')

#7개의 감정 class
data['상황'].unique()
>>> array(['happiness', 'neutral', 'sadness', 'angry', 'surprise', 'disgust','fear'], dtype=object)

#7개의 감정을 숫자로 표현
# 7개의 감정 class → 숫자
data.loc[(data['상황'] == "fear"), '상황'] = 0  # 두려움 → 0
data.loc[(data['상황'] == "surprise"), '상황'] = 1  # 놀람 → 1
data.loc[(data['상황'] == "angry"), '상황'] = 2  # 분노 → 2
data.loc[(data['상황'] == "sadness"), '상황'] = 3  # 슬픔 → 3
data.loc[(data['상황'] == "neutral"), '상황'] = 4  # 중간 → 4
data.loc[(data['상황'] == "happiness"), '상황'] = 5  # 행복 → 5
data.loc[(data['상황'] == "disgust"), '상황'] = 6  # 혐오 → 6
