"""
共用模块:配置、数据加载、BERT 特征提取、MLP 模型。
被 train.py 和 infer.py 同时引用,避免重复代码。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# ============================================================
# 配置:路径与超参,要改只改这里
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = r"E:/3/COLDataset/COLDataset"
TRAIN_CSV   = os.path.join(DATA_DIR, "train.csv")
DEV_CSV     = os.path.join(DATA_DIR, "dev.csv")
TEST_CSV    = os.path.join(DATA_DIR, "test.csv")

# 预训练 BERT:优先用仓库里已下载的本地权重,没有再回退联网。
_LOCAL_BERT = os.path.normpath(os.path.join(PROJECT_DIR, "..", "models", "bert-base-chinese"))
PRETRAINED = _LOCAL_BERT if os.path.exists(os.path.join(_LOCAL_BERT, "config.json")) else "bert-base-chinese"

MAX_LEN  = 128
BERT_DIM = 768

ARTIFACT_DIR = os.path.join(PROJECT_DIR, "artifacts")
FEAT_DIR     = os.path.join(ARTIFACT_DIR, "features")
os.makedirs(FEAT_DIR, exist_ok=True)
MLP_PATH    = os.path.join(ARTIFACT_DIR, "mlp.pt")
REPORT_PATH = os.path.join(ARTIFACT_DIR, "test_report.txt")

SEED         = 42
HIDDEN       = 256
DROPOUT      = 0.3
BATCH_SIZE   = 256
EPOCHS       = 30
LR           = 5e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP   = 5

LABEL2NAME = {0: "正常", 1: "违规"}


def set_seed(s=SEED):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ============================================================
# 数据加载
# COLDataset CSV 是 UTF-8-BOM + CRLF,首列是无名行索引:
#   utf-8-sig 剥 BOM,index_col=0 吃掉索引列,只留 split/topic/label/TEXT。
# ============================================================
_COLS = ["split", "topic", "label", "TEXT"]

def _clean(df):
    df = df[_COLS].copy()
    df["TEXT"]  = df["TEXT"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    df["topic"] = df["topic"].astype(str)
    return df

def load_train(): return _clean(pd.read_csv(TRAIN_CSV, encoding="utf-8-sig", index_col=0))
def load_dev():   return _clean(pd.read_csv(DEV_CSV,   encoding="utf-8-sig", index_col=0))
def load_test():  return _clean(pd.read_csv(TEST_CSV,  encoding="utf-8-sig", index_col=0))


# ============================================================
# BERT 特征提取 + 缓存
# ============================================================
@torch.no_grad()
def extract_features(texts, batch_size=32, desc="extract", tokenizer=None, model=None,
                     pretrained=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    own = model is None
    if own:
        name = pretrained or PRETRAINED
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(device).eval()
    feats, texts, n = [], list(texts), len(texts)
    try:
        for i in range(0, n, batch_size):
            enc = tokenizer(texts[i:i+batch_size], padding=True, truncation=True,
                            max_length=MAX_LEN, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            feats.append(out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32))
            if (i // batch_size) % 25 == 0 or i + batch_size >= n:
                print(f"  [{desc}] {min(i+batch_size, n)}/{n}", flush=True)
    finally:
        if own:
            del model
            try: torch.cuda.empty_cache()
            except Exception: pass
    return np.concatenate(feats, axis=0)


def get_or_extract(split, texts, force=False, pretrained=None):
    path = os.path.join(FEAT_DIR, f"{split}.npy")
    if os.path.exists(path) and not force:
        feats = np.load(path)
        print(f"加载缓存特征 {path}  {feats.shape}")
        return feats
    print(f"提取特征 {split} ({len(texts)} 条,首次较慢,CPU 全量约 20-40 分钟)...")
    feats = extract_features(texts, desc=split, pretrained=pretrained)
    np.save(path, feats)
    print(f"已缓存 -> {path}")
    return feats


# ============================================================
# MLP 分类头:768 -> hidden -> hidden//2 -> 2
# ============================================================
class MLPClassifier(nn.Module):
    def __init__(self, in_dim=BERT_DIM, hidden=HIDDEN, dropout=DROPOUT, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)
