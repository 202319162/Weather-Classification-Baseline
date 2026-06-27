import argparse
import gc
import sys
import time
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import main as submit_main


CLASSES = ["cloudy", "rainy", "snowy", "sunny"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(root):
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")

    paths = []

    for cls in CLASSES:
        cls_dir = root / cls

        if not cls_dir.exists():
            continue

        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in EXTS:
                paths.append(p)

    return sorted(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    paths = collect_images(args.image_dir)

    if args.num_images > 0:
        paths = paths[: args.num_images]

    if not paths:
        raise RuntimeError(f"No images found in {args.image_dir}")

    print("number of images:", len(paths))

    print("warmup...")
    for p in paths[: args.warmup]:
        img = cv2.imread(str(p))
        pred = submit_main.predict(img)

        if pred not in CLASSES:
            raise RuntimeError(f"Invalid prediction: {pred}")

        del img

    gc.collect()

    start = time.perf_counter()

    for i, p in enumerate(paths, 1):
        img = cv2.imread(str(p))
        pred = submit_main.predict(img)

        if pred not in CLASSES:
            raise RuntimeError(f"Invalid prediction: {pred}")

        del img

        if i % 50 == 0:
            gc.collect()
            print(f"processed {i}/{len(paths)}")

    elapsed = time.perf_counter() - start
    seconds_per_image = elapsed / len(paths)
    estimated_5000_seconds = seconds_per_image * 5000

    print("\nBenchmark Result")
    print("images:", len(paths))
    print("total seconds:", elapsed)
    print("seconds per image:", seconds_per_image)
    print("estimated 5000 images seconds:", estimated_5000_seconds)
    print("estimated 5000 images minutes:", estimated_5000_seconds / 60)

    if estimated_5000_seconds < 70 * 60:
        print("Conclusion: within 70 minutes.")
    else:
        print("Conclusion: may exceed 70 minutes.")


if __name__ == "__main__":
    main()