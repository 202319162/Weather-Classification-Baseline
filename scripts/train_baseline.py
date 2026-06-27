
---

# 十一、`scripts/train_baseline.py`

```python
import argparse
import random
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm


CLASS_NAMES = ["cloudy", "rainy", "snowy", "sunny"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WeatherFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for label, cls in enumerate(CLASS_NAMES):
            cls_dir = self.root / cls
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {cls_dir}")

            for p in cls_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in EXTS:
                    self.samples.append((p, label))

        if not self.samples:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")

        if self.transform is not None:
            im = self.transform(im)

        return im, label


def build_transforms(img_size):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.75, 1.0),
                ratio=(0.8, 1.25),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.15,
                hue=0.03,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_tf, val_tf


def get_class_counts(dataset):
    labels = [label for _, label in dataset.samples]
    counter = Counter(labels)
    return [counter[i] for i in range(len(CLASS_NAMES))]


def build_class_weights(counts, device):
    total = sum(counts)
    weights = []

    for c in counts:
        weights.append(total / max(c, 1))

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.mean()
    return weights.to(device)


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="Eval", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    macro = f1_score(all_targets, all_preds, average="macro")

    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets,
        all_preds,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )

    cm = confusion_matrix(
        all_targets,
        all_preds,
        labels=list(range(len(CLASS_NAMES))),
    )

    return {
        "macro_f1": macro,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "cm": cm,
    }


def print_metrics(metrics):
    print(f"\nMacro F1 = {metrics['macro_f1']:.6f}")
    print("Per-class metrics:")

    for i, cls in enumerate(CLASS_NAMES):
        print(
            f"{cls:7s} "
            f"P={metrics['precision'][i]:.4f} "
            f"R={metrics['recall'][i]:.4f} "
            f"F1={metrics['f1'][i]:.4f} "
            f"N={int(metrics['support'][i])}"
        )

    print("\nConfusion matrix: rows=true, cols=pred")
    print("labels:", CLASS_NAMES)
    for i, cls in enumerate(CLASS_NAMES):
        print(cls, metrics["cm"][i].tolist())


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_tf, val_tf = build_transforms(args.img_size)

    train_set = WeatherFolderDataset(args.train_dir, transform=train_tf)
    val_set = WeatherFolderDataset(args.val_dir, transform=val_tf)

    print("train images:", len(train_set))
    print("val images:", len(val_set))

    counts = get_class_counts(train_set)
    print("class counts:", dict(zip(CLASS_NAMES, counts)))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = timm.create_model(
        args.model_name,
        pretrained=not args.no_pretrained,
        num_classes=len(CLASS_NAMES),
    )

    if args.resume_ckpt:
        print("Loading checkpoint:", args.resume_ckpt)
        ckpt = safe_torch_load(args.resume_ckpt, map_location="cpu")

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict, strict=True)

    model.to(device)

    if args.eval_only:
        metrics = evaluate(model, val_loader, device)
        print_metrics(metrics)
        return

    class_weights = build_class_weights(counts, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs * len(train_loader)),
        eta_min=args.min_lr,
    )

    best_macro = -1.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_num = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_num += batch_size

            pbar.set_postfix(loss=total_loss / max(total_num, 1))

        train_loss = total_loss / max(total_num, 1)

        metrics = evaluate(model, val_loader, device)
        macro = metrics["macro_f1"]

        print(f"\nEpoch {epoch}: train_loss={train_loss:.5f}, val_macro_f1={macro:.6f}")
        print_metrics(metrics)

        if macro > best_macro:
            best_macro = macro
            best_epoch = epoch
            patience_counter = 0

            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_name": args.model_name,
                "class_names": CLASS_NAMES,
                "macro_f1": float(best_macro),
                "epoch": int(epoch),
                "img_size": int(args.img_size),
            }

            torch.save(checkpoint, out_path)
            print("Saved best checkpoint:", out_path)
        else:
            patience_counter += 1
            print(f"No improvement. patience={patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print("Early stopping.")
            break

    print("\nTraining finished.")
    print("best_epoch:", best_epoch)
    print("best_macro_f1:", best_macro)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="efficientnet_b0")
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--eval_only", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pretrained", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())