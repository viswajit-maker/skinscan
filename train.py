"""Step 2: train (transfer learning with EfficientNet-B0) and evaluate on the held-out test set."""
import copy, json, random
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import config as C

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def get_transforms(train):
    if train:  # simulate real phone photos: different framing, rotation, lighting, colour cast
        return T.Compose([
            T.RandomResizedCrop(C.IMG_SIZE, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
            T.ToTensor(), T.Normalize(MEAN, STD)])
    return T.Compose([T.Resize((C.IMG_SIZE, C.IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


class SkinDS(Dataset):
    def __init__(self, df, class_to_idx, train):
        self.df, self.c2i, self.tf = df.reset_index(drop=True), class_to_idx, get_transforms(train)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(Image.open(r["path"]).convert("RGB")), self.c2i[r["label"]]


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                out = model(x); loss = loss_fn(out, y)
            if train:
                optimizer.zero_grad()
                if scaler: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else: loss.backward(); optimizer.step()
            total += loss.item() * len(y)
            preds += out.argmax(1).tolist(); labels += y.tolist()
    return total / len(loader.dataset), f1_score(labels, preds, average="macro"), np.mean(np.array(preds) == np.array(labels))


def main():
    seed_everything(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv(C.SPLITS_CSV)
    classes = sorted(df["label"].unique()); c2i = {c: i for i, c in enumerate(classes)}
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    mk = lambda d, t, sh: DataLoader(SkinDS(d, c2i, t), batch_size=C.BATCH_SIZE, shuffle=sh,
                                     num_workers=C.NUM_WORKERS, pin_memory=device.type == "cuda")
    tr_dl, va_dl, te_dl = mk(tr, True, True), mk(va, False, False), mk(te, False, False)

    # class imbalance (Atopic=70 vs Contact=477): weight rare classes more in the loss
    counts = tr["label"].value_counts().reindex(classes).values.astype(float)
    w = (counts.sum() / counts) ** 0.5; w = w / w.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(device), label_smoothing=0.1)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[1].in_features, len(classes)))
    model.to(device)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    best_f1, best_state, bad, hist = -1, None, 0, []

    def fit(epochs, optimizer, scheduler, tag):
        nonlocal best_f1, best_state, bad
        for ep in range(1, epochs + 1):
            trl, trf, tra = run_epoch(model, tr_dl, loss_fn, device, optimizer, scaler)
            val, vaf, vaa = run_epoch(model, va_dl, loss_fn, device)
            if scheduler: scheduler.step()
            hist.append(dict(phase=tag, train_loss=trl, val_loss=val, train_acc=tra, val_acc=vaa, val_f1=vaf))
            print(f"[{tag} {ep:02d}/{epochs}] train loss {trl:.3f} acc {tra:.3f} | val loss {val:.3f} acc {vaa:.3f} macroF1 {vaf:.3f}")
            if vaf > best_f1:
                best_f1, best_state, bad = vaf, copy.deepcopy(model.state_dict()), 0
                torch.save({"state_dict": best_state, "classes": classes, "img_size": C.IMG_SIZE}, C.MODEL_PATH)
            else:
                bad += 1
                if tag == "finetune" and bad >= C.PATIENCE:
                    print("Early stopping."); return

    # Phase 1: only the new classifier head
    for p in model.features.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=C.LR_HEAD, weight_decay=1e-4)
    fit(C.HEAD_EPOCHS, opt, None, "head")

    # Phase 2: unfreeze everything, small learning rate
    for p in model.features.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR_FINETUNE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.FINETUNE_EPOCHS)
    bad = 0
    fit(C.FINETUNE_EPOCHS, opt, sched, "finetune")

    # ---- final evaluation on the untouched TEST set --------------------------------
    model.load_state_dict(best_state); model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in te_dl:
            ps += model(x.to(device)).argmax(1).cpu().tolist(); ys += y.tolist()
    report = classification_report(ys, ps, target_names=classes, digits=3, zero_division=0)
    print("\n=== TEST RESULTS ===\n" + report)
    (C.OUTPUT_DIR / "test_report.txt").write_text(report)

    cm = confusion_matrix(ys, ps)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(C.OUTPUT_DIR / "confusion_matrix.png", dpi=150); plt.close()

    h = pd.DataFrame(hist); h.to_csv(C.OUTPUT_DIR / "history.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    h[["train_loss", "val_loss"]].plot(ax=ax[0], title="Loss"); h[["train_acc", "val_acc", "val_f1"]].plot(ax=ax[1], title="Accuracy / F1")
    plt.tight_layout(); plt.savefig(C.OUTPUT_DIR / "training_curves.png", dpi=150)
    json.dump(classes, open(C.OUTPUT_DIR / "classes.json", "w"))
    print(f"\nModel saved -> {C.MODEL_PATH}")


if __name__ == "__main__":   # required on Windows (DataLoader workers)
    main()
