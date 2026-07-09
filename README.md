# 中文违规语言检测

基于 **COLDataset**(COLD: A Benchmark for Chinese Offensive Language Detection, EMNLP 2022)
训练的二分类模型,判定中文文本是否为违规/冒犯性语言。

## 方案:feature-based BERT + MLP

1. 用预训练中文 BERT(`bert-base-chinese`)对每条文本做一次前向,提取 `[CLS]` 句向量(768 维),
   缓存为 `.npy`(`artifacts/features/`)——一次性成本(全量约 20–40 分钟,之后全部复用)。
2. 在缓存特征上训练轻量 **MLP 二分类头**(秒级~分钟级,可反复调参)。
3. 推理时复用同一 BERT + 已训练 MLP。
4. 使用数据集为中文冒犯语言检测数据集(https://huggingface.co/thu-coai/roberta-base-cold?text=%E4%BD%A0%E6%98%AF%E4%B8%8D%E6%98%AF%E5%82%BB)


## 代码结构(三个 .py)

```
offensive_detect/
├── common.py    # 共用:配置/超参 + 数据加载 + BERT 特征提取 + MLP 模型
├── train.py     # 训练 + 评估入口(只训练,不含推理)
├── infer.py     # 推理入口(只预测,不含训练)
├── README.md
└── artifacts/
    ├── features/      # *.npy 特征缓存
    ├── mlp.pt         # 训练好的 MLP 权重(train.py 生成)
    └── test_report.txt# 测试集评估报告
```

- **common.py** 是 train.py 和 infer.py 共用的部分:配置、读 CSV、BERT 特征提取、MLP 结构。只写一份,不重复。
- **train.py** 只负责训练循环、早停、在 test 上评估、保存权重。**没有任何预测用户句子的代码。**
- **infer.py** 只负责加载已训练的模型做预测。**没有任何训练代码。**

## 环境依赖

版本:`Python 3.10.9`、`torch 1.12.1 (CPU)`、`transformers 4.24.0`、`pandas`、`scikit-learn`、`numpy`。

```bash
pip install -r requirements.txt
```

## 使用

### 1. 准备 BERT(一次性)
老版 `huggingface_hub 0.10.1` 直接联网下载会报 `missing commit header`,
改用镜像手动下载到 `E:/3/models/bert-base-chinese/`(config.py 会自动用本地路径):

```bash
# 用镜像下载 config.json / vocab.txt / tokenizer_config.json / pytorch_model.bin(~411MB)
# 例:
mkdir -p E:/3/models/bert-base-chinese
cd E:/3/models/bert-base-chinese
for f in config.json vocab.txt tokenizer_config.json pytorch_model.bin; do
  curl -L -o "$f" "https://hf-mirror.com/bert-base-chinese/resolve/main/$f"
done
```

### 2. 训练
```bash
# 完整训练(首次会提取并缓存特征,约 20-40 分钟;之后秒级)
python -m offensive_detect.train

# 快速验证(仅用 2000 条训练样本,8 个 epoch)
python -m offensive_detect.train --max_train 2000 --epochs 8

# 重新提取特征(换模型后需要)
python -m offensive_detect.train --rebuild_features
```
训练按 dev F1 早停,最优权重存 `artifacts/mlp.pt`,测试集报告存 `artifacts/test_report.txt`
(含 Accuracy/F1/Precision/Recall、混淆矩阵、按 race/gender/region 分组性能)。

### 3. 推理(须先训练生成 mlp.pt)
```bash
# 单句
python -m offensive_detect.infer infer "你这条评论的内容"

# 批量(--batch 文件,每行一句)
python -m offensive_detect.infer predict --batch sentences.txt
```
编程接口:
```python
from offensive_detect.infer import Predictor
p = Predictor()
print(p.predict("某条评论"))
# {'text': ..., 'label': '违规', 'is_offensive': True, 'confidence': 0.98, 'probs': {...}}
```

## 标签
- `0` = 正常 (safe)
- `1` = 违规 (offensive)

test.csv 的 `fine-grained-label`(攻击个人/群体、反偏见)本任务不使用,仅用二分类 `label`。

## 关于内容
数据集中 `TEXT` 含故意构造的种族/性别/地域歧视言论,目的是让模型学会**检测**它们,
属于研究对象而非表达。模型仅用于违规语言识别这一防御性用途。
