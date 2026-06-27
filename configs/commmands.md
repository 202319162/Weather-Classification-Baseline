# 实验命令记录

本文件记录公开 baseline 版本中常用的数据处理、训练、验证和 CPU 推理测试命令。

注意：

- 本仓库是 baseline 公开版，不包含最终比赛模型和完整调参细节。
- 所有路径均为示例路径，请根据自己的实际目录修改。
- 不要在本文件中写入 SSH、密码、服务器 IP、平台账号等敏感信息。

---

## 1. 安装训练依赖

```bash
pip install -r requirements_train.txt
```

如果出现 NumPy 版本兼容问题，可以固定为：

```bash
pip uninstall -y numpy
pip install numpy==1.26.4
```

---

## 2. 数据目录格式

原始数据建议整理为：

```text
official_data/
  official_train/
    cloudy/
    rainy/
    snowy/
    sunny/
  official_val/
    cloudy/
    rainy/
    snowy/
    sunny/
```

---

## 3. 构建 clean train / clean val

```bash
python scripts/make_clean_split.py \
  --official_train ./official_data/official_train \
  --official_val ./official_data/official_val \
  --out_dir ./official_data_clean \
  --seed 42
```

输出目录：

```text
official_data_clean/
  clean_train/
  clean_val/
  reports/
```

---

## 4. 统计数据集数量

```bash
python - <<'PY'
from pathlib import Path

CLASSES = ["cloudy", "rainy", "snowy", "sunny"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ["clean_train", "clean_val"]:
    print("\\n", split)
    root = Path("./official_data_clean") / split
    total = 0

    for cls in CLASSES:
        d = root / cls
        n = 0
        if d.exists():
            n = sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
        total += n
        print(cls, n)

    print("total", total)
PY
```

---

## 5. 检查 train-val 重复图

```bash
python scripts/check_duplicates.py \
  --train_dir ./official_data_clean/clean_train \
  --val_dir ./official_data_clean/clean_val \
  --out_dir ./duplicate_review \
  --threshold 8
```

输出目录：

```text
duplicate_review/
  exact_duplicates.txt
  near_duplicates_train_val.txt
  near_train_val/
```

---

## 6. 训练 EfficientNet-B0 baseline

```bash
python scripts/train_baseline.py \
  --train_dir ./official_data_clean/clean_train \
  --val_dir ./official_data_clean/clean_val \
  --out ./results/efficientnet_b0_baseline.pth \
  --model_name efficientnet_b0 \
  --epochs 20 \
  --batch_size 32 \
  --lr 3e-4 \
  --num_workers 0
```

如果显存不足，可以减小 batch size：

```bash
python scripts/train_baseline.py \
  --train_dir ./official_data_clean/clean_train \
  --val_dir ./official_data_clean/clean_val \
  --out ./results/efficientnet_b0_baseline.pth \
  --model_name efficientnet_b0 \
  --epochs 20 \
  --batch_size 16 \
  --lr 3e-4 \
  --num_workers 0
```

---

## 7. 只验证已有模型

```bash
python scripts/train_baseline.py \
  --train_dir ./official_data_clean/clean_train \
  --val_dir ./official_data_clean/clean_val \
  --out ./results/efficientnet_b0_baseline.pth \
  --model_name efficientnet_b0 \
  --resume_ckpt ./results/efficientnet_b0_baseline.pth \
  --eval_only \
  --num_workers 0
```

---

## 8. 准备平台推理权重

`main.py` 默认读取：

```text
results/weather_best.pth
```

可以复制 baseline 权重：

```bash
mkdir -p results
cp ./results/efficientnet_b0_baseline.pth ./results/weather_best.pth
```

---

## 9. 本地测试 main.py

```bash
python main.py
```

如果输出类似下面内容，说明模型加载和 `predict(X)` 接口正常：

```text
model_name = efficientnet_b0
class_names = ['cloudy', 'rainy', 'snowy', 'sunny']
predict(dummy) = cloudy
```

---

## 10. CPU 推理时间测试

```bash
python scripts/weather_cls_cpu_benchmark.py \
  --image_dir ./official_data_clean/clean_val \
  --num_images 500
```

该脚本会估算 5000 张图片的 CPU 推理时间。