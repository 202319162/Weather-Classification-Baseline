import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


CLASSES = ["cloudy", "rainy", "snowy", "sunny"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_md5(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def average_hash(path, size=16):
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("L").resize((size, size), resample)
        arr = np.asarray(im, dtype=np.float32)

    bits = arr > arr.mean()
    return bits.flatten().astype(np.uint8)


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def collect_images(root, split_name):
    root = Path(root)
    items = []

    for cls in CLASSES:
        cls_dir = root / cls
        if not cls_dir.exists():
            continue

        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in EXTS:
                items.append(
                    {
                        "path": p.resolve(),
                        "split": split_name,
                        "class": cls,
                    }
                )

    return items


def safe_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./duplicate_review")
    parser.add_argument("--threshold", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_items = collect_images(args.train_dir, "train")
    val_items = collect_images(args.val_dir, "val")
    all_items = train_items + val_items

    print("train images:", len(train_items))
    print("val images:", len(val_items))
    print("total images:", len(all_items))

    md5_map = {}

    for item in all_items:
        m = file_md5(item["path"])
        md5_map.setdefault(m, []).append(item)

    exact_groups = [v for v in md5_map.values() if len(v) > 1]
    exact_cross_groups = [
        group for group in exact_groups
        if len({x["split"] for x in group}) > 1
    ]

    exact_report = out_dir / "exact_duplicates.txt"
    with open(exact_report, "w", encoding="utf-8") as f:
        for i, group in enumerate(exact_groups, 1):
            f.write(f"\n[Exact Group {i}]\n")
            for x in group:
                f.write(f"{x['split']}\t{x['class']}\t{x['path']}\n")

    print("exact duplicate groups:", len(exact_groups))
    print("train-val exact duplicate groups:", len(exact_cross_groups))

    print("computing perceptual hashes...")

    for item in all_items:
        try:
            item["ahash"] = average_hash(item["path"])
        except Exception as e:
            print("hash failed:", item["path"], e)
            item["ahash"] = None

    near_pairs = []

    for i, val_item in enumerate(val_items, 1):
        if val_item["ahash"] is None:
            continue

        best_train = None
        best_dist = 999

        for train_item in train_items:
            if train_item["ahash"] is None:
                continue

            d = hamming(val_item["ahash"], train_item["ahash"])

            if d < best_dist:
                best_dist = d
                best_train = train_item

        if best_train is not None and best_dist <= args.threshold:
            near_pairs.append((val_item, best_train, best_dist))

        if i % 50 == 0:
            print(f"checked val {i}/{len(val_items)}")

    near_report = out_dir / "near_duplicates_train_val.txt"
    with open(near_report, "w", encoding="utf-8") as f:
        for i, (val_item, train_item, dist) in enumerate(near_pairs, 1):
            f.write(f"\n[Near Pair {i}] distance={dist}\n")
            f.write(f"VAL\t{val_item['class']}\t{val_item['path']}\n")
            f.write(f"TRAIN\t{train_item['class']}\t{train_item['path']}\n")

            pair_dir = out_dir / "near_train_val" / f"pair_{i:04d}_d{dist}"
            safe_copy(
                val_item["path"],
                pair_dir / f"VAL_{val_item['class']}_{val_item['path'].name}",
            )
            safe_copy(
                train_item["path"],
                pair_dir / f"TRAIN_{train_item['class']}_{train_item['path'].name}",
            )

    print("near duplicate train-val pairs:", len(near_pairs))
    print("reports saved to:", out_dir)


if __name__ == "__main__":
    main()