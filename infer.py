"""
推理入口:输入中文句子 -> {正常/违规, 置信度, 概率}。

用法(在 E:/3 目录下,需先跑过 train.py 生成 artifacts/mlp.pt):
    python -m offensive_detect.infer "要检测的句子"
    python -m offensive_detect.predict --batch 句子文件.txt   # 每行一句

编程接口:
    from offensive_detect.infer import Predictor
    p = Predictor()
    print(p.predict("某条评论"))
"""
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModel

from .common import (extract_features, MLPClassifier, MLP_PATH, PRETRAINED)


class Predictor:
    """加载 BERT + 已训练的 MLP,提供单句/批量预测。"""

    def __init__(self, ckpt=MLP_PATH, pretrained=PRETRAINED):
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"未找到模型权重 {ckpt},请先运行 python -m offensive_detect.train")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok  = AutoTokenizer.from_pretrained(pretrained)
        self.bert = AutoModel.from_pretrained(pretrained).to(self.device).eval()
        self.mlp = MLPClassifier().to(self.device)
        self.mlp.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.mlp.eval()
        self.names = ["正常", "违规"]

    @torch.no_grad()
    def predict(self, text):
        """单句预测,返回 dict。"""
        feat = extract_features([text], tokenizer=self.tok, model=self.bert)
        prob = torch.softmax(self.mlp(torch.from_numpy(feat).to(self.device)), dim=1)[0].cpu().numpy()
        i = int(prob.argmax())
        return {"text": text, "label": self.names[i], "is_offensive": bool(i),
                "confidence": float(prob[i]),
                "probs": {"正常": float(prob[0]), "违规": float(prob[1])}}

    @torch.no_grad()
    def predict_batch(self, texts):
        """批量预测,返回 list[dict]。"""
        feats = extract_features(list(texts), tokenizer=self.tok, model=self.bert)
        probs = torch.softmax(self.mlp(torch.from_numpy(feats).to(self.device)), dim=1).cpu().numpy()
        out = []
        for t, p in zip(texts, probs):
            i = int(p.argmax())
            out.append({"text": t, "label": self.names[i], "is_offensive": bool(i), "confidence": float(p[i])})
        return out


def main():
    ap = argparse.ArgumentParser(description="违规语言检测推理")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("infer", help="检测单句")          # 兼容旧写法
    p1.add_argument("text", help="待检测句子")
    p1.add_argument("--pretrained", default=None)
    p1.set_defaults(_kind="infer")

    p2 = sub.add_parser("predict", help="批量检测(--batch 文件,每行一句)")
    p2.add_argument("--batch", required=True, help="句子文件路径")
    p2.add_argument("--pretrained", default=None)
    p2.set_defaults(_kind="batch")

    args = ap.parse_args()
    pt = args.pretrained or PRETRAINED
    pred = Predictor(pretrained=pt)

    if args._kind == "infer":
        r = pred.predict(args.text)
        print(f"文本: {r['text']}")
        print(f"判定: {r['label']}  置信度 {r['confidence']:.4f}")
        print(f"概率: {r['probs']}")
    else:
        with open(args.batch, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        for r in pred.predict_batch(lines):
            print(f"[{r['label']}] {r['confidence']:.3f}  {r['text']}")


if __name__ == "__main__":
    main()
