import argparse
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path


CLASSES = ["cloudy", "rainy", "snowy", "sunny"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_md5(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def collect_images(input_dirs):
    items = []

    for root in input_dirs:
        root = Path(root)

        if not root.exists():
            continue

        for cls in CLASSES:
            cls_dir = root / cls

            if not cls_dir.exists():
                continue

            for p in cls_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in EXTS:
                    items.append(
                        {
                            "path": p.resolve(),
                            "class": cls,
                        }
                    )

    return items


def safe_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official_train", type=str, required=True)
    parser.add_argument("--official_val", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./official_data_clean")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--val_cloudy", type=int, default=150)
    parser.add_argument("--val_rainy", type=int, default=100)
    parser.add_argument("--val_snowy", type=int, default=100)
    parser.add_argument("--val_sunny", type=int, default=150)

    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    clean_train = out_dir / "clean_train"
    clean_val = out_dir / "clean_val"
    report_dir = out_dir / "reports"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    report_dir.mkdir(parents=True, exist_ok=True)

    raw_items = collect_images([args.official_train, args.official_val])
    print("raw images:", len(raw_items))

    md5_seen = {}
    unique_items = []
    duplicate_items = []
    bad_items = []

    for item in raw_items:
        try:
            m = file_md5(item["path"])
        except Exception as e:
            bad_items.append((item["path"], str(e)))
            continue

        if m in md5_seen:
            duplicate_items.append((item["path"], md5_seen[m]))
        else:
            md5_seen[m] = item["path"]
            unique_items.append(item)

    by_class = defaultdict(list)

    for item in unique_items:
        by_class[item["class"]].append(item)

    val_nums = {
        "cloudy": args.val_cloudy,
        "rainy": args.val_rainy,
        "snowy": args.val_snowy,
        "sunny": args.val_sunny,
    }

    for cls in CLASSES:
        items = by_class[cls]
        random.shuffle(items)

        need_val = val_nums[cls]

        if len(items) <= need_val:
            raise RuntimeError(
                f"Class {cls} has only {len(items)} images, "
                f"not enough for validation size {need_val}."
            )

        val_items = items[:need_val]
        train_items = items[need_val:]

        for i, item in enumerate(train_items):
            src = item["path"]
            dst = clean_train / cls / f"{i:05d}_{src.name}"
            safe_copy(src, dst)

        for i, item in enumerate(val_items):
            src = item["path"]
            dst = clean_val / cls / f"{i:05d}_{src.name}"
            safe_copy(src, dst)

    with open(report_dir / "duplicates_by_md5.txt", "w", encoding="utf-8") as f:
        for dup, original in duplicate_items:
            f.write(f"DUP\t{dup}\n")
            f.write(f"ORG\t{original}\n\n")

    with open(report_dir / "bad_images.txt", "w", encoding="utf-8") as f:
        for p, err in bad_items:
            f.write(f"{p}\t{err}\n")

    with open(report_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"raw_images\t{len(raw_items)}\n")
        f.write(f"unique_images\t{len(unique_items)}\n")
        f.write(f"md5_duplicates\t{len(duplicate_items)}\n")
        f.write(f"bad_images\t{len(bad_items)}\n\n")

        for split_name, root in [("train", clean_train), ("val", clean_val)]:
            f.write(f"[{split_name}]\n")

            for cls in CLASSES:
                cls_dir = root / cls
                n = 0
                if cls_dir.exists():
                    n = sum(
                        1 for p in cls_dir.rglob("*")
                        if p.is_file() and p.suffix.lower() in EXTS
                    )

                f.write(f"{cls}\t{n}\n")

            f.write("\n")

    print("clean split created:", out_dir)
    print("reports saved to:", report_dir)


if __name__ == "__main__":
    main()