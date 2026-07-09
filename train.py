"""
训练 + 评估入口。

用法(在 E:/3 目录下):
    python -m offensive_detect.train                          # 完整训练(首次提特征约20-40分钟)
    python -m offensive_detect.train --max_train 2000 --epochs 8   # 快速验证
    python -m offensive_detect.train --rebuild_features       # 重新提取特征(换模型后用)
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)

from .common import (set_seed, load_train, load_dev, load_test,
                     get_or_extract, MLPClassifier, MLP_PATH, REPORT_PATH,
                     EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, EARLY_STOP, SEED,
                     PRETRAINED)


def _dataset(feats, labels):
    return TensorDataset(torch.from_numpy(feats), torch.from_numpy(labels.astype(np.int64)))


def _predict(model, loader, device):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            ys.append(yb.numpy()); ps.append(out.argmax(1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser(description="训练违规语言检测模型 (feature-based BERT + MLP)")
    ap.add_argument("--max_train", type=int, default=0,  help=">0 时只用这么多训练样本(快速验证)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--rebuild_features", action="store_true", help="强制重新提取特征")
    ap.add_argument("--pretrained", default=None, help="覆盖预训练模型名/路径")
    args = ap.parse_args()

    pretrained = args.pretrained or PRETRAINED
    if args.pretrained:
        print(f"覆盖预训练模型: {pretrained}")

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}  预训练: {pretrained}")

    # --- 数据 ---
    tr, dv, te = load_train(), load_dev(), load_test()
    print(f"数据: train {len(tr)} / dev {len(dv)} / test {len(te)}")

    # --- 特征(一次性提取并缓存)---
    if args.max_train and 0 < args.max_train < len(tr):    # 快速验证:训练集采样
        idx = np.random.RandomState(SEED).choice(len(tr), args.max_train, replace=False)
        sub = tr.iloc[idx].reset_index(drop=True)
        Xtr = get_or_extract(f"train_sub{args.max_train}", sub["TEXT"].values,
                             force=args.rebuild_features, pretrained=pretrained)
        ytr = sub["label"].values
        print(f"快速验证:训练采样至 {len(sub)} 条")
    else:
        Xtr = get_or_extract("train_full", tr["TEXT"].values,
                             force=args.rebuild_features, pretrained=pretrained)
        ytr = tr["label"].values
    Xdv = get_or_extract("dev",  dv["TEXT"].values, force=args.rebuild_features, pretrained=pretrained)
    Xte = get_or_extract("test", te["TEXT"].values, force=args.rebuild_features, pretrained=pretrained)
    ydv, yte = dv["label"].values, te["label"].values
    print(f"特征: train {Xtr.shape} dev {Xdv.shape} test {Xte.shape}")

    # --- 训练 ---
    tr_loader = DataLoader(_dataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
    dv_loader = DataLoader(_dataset(Xdv, ydv), batch_size=args.batch_size)
    te_loader = DataLoader(_dataset(Xte, yte), batch_size=args.batch_size)

    model = MLPClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    best_f1, no_improve, best_state, last_ep = 0.0, 0, None, 0
    for ep in range(1, args.epochs + 1):
        last_ep = ep
        model.train(); running = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb); loss.backward(); opt.step()
            running += loss.item() * len(xb)
        ys, ps = _predict(model, dv_loader, device)
        f1, acc = f1_score(ys, ps, pos_label=1), accuracy_score(ys, ps)
        print(f"epoch {ep:02d}  loss {running/len(tr_loader.dataset):.4f}  "
              f"dev acc {acc:.4f}  dev f1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                print(f"早停(epoch {ep},已 {no_improve} 轮无提升)"); break

    # --- 在 test 上评估最优模型 ---
    model.load_state_dict(best_state)
    torch.save(best_state, MLP_PATH)
    ys, ps = _predict(model, te_loader, device)
    rep = classification_report(ys, ps, target_names=["safe(0)", "offensive(1)"], digits=4)
    cm = confusion_matrix(ys, ps)
    print("\n===== Test 评估 ====="); print(rep)
    print("混淆矩阵 (行=真实, 列=预测):"); print(cm)

    te_eval = te.copy(); te_eval["pred"] = ps
    print("===== 按 topic ====="); topic_lines = []
    for t, g in te_eval.groupby("topic"):
        a = accuracy_score(g["label"], g["pred"]); f = f1_score(g["label"], g["pred"], pos_label=1)
        line = f"  {t:8s}  acc {a:.4f}  f1 {f:.4f}  n {len(g)}"
        print(line); topic_lines.append(line)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"预训练: {pretrained}\nmax_train: {args.max_train}\n"
                f"epochs: {last_ep}\nbest dev f1: {best_f1:.4f}\n\n")
        f.write(rep)
        f.write("\n混淆矩阵 (行=真实,列=预测):\n" + str(cm) + "\n\n按 topic:\n" + "\n".join(topic_lines) + "\n")
    print(f"\n模型已保存: {MLP_PATH}\n报告已保存: {REPORT_PATH}")


if __name__ == "__main__":
    main()
