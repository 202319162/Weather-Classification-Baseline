# Weather Classification Baseline：天气图像四分类从训练到 CPU 推理

这是一个面向图像分类比赛的 **天气图像四分类 Baseline 项目**。

项目目标是将输入图片分类为四种天气类型：

* `cloudy`：多云
* `rainy`：雨天
* `snowy`：雪天
* `sunny`：晴天

本仓库提供一个完整、轻量、容易复现的 baseline 流程，覆盖：

* 数据目录整理
* 训练集 / 验证集划分
* 重复图检测
* EfficientNet-B0 baseline 训练
* Macro F1 验证
* CPU 推理脚本
* 比赛平台 `predict(X)` 接口适配
* CPU 推理耗时估算

这个仓库适合用于学习图像分类比赛的基本流程，也可以作为天气分类、场景分类、小型视觉分类任务的起点。

---

## 项目特点

本项目不是只给一个模型文件，而是尽量整理出一套完整的比赛级入门流程：

* 使用 PyTorch + timm 训练 EfficientNet-B0 baseline
* 使用 Macro F1 作为主要评价指标
* 支持四分类文件夹格式数据集
* 支持类别不均衡场景下的 class weight
* 支持 train-val 重复图检查
* 支持 CPU-only 推理
* 适配平台常见的 `predict(X)` 提交接口
* 提供 CPU 推理耗时测试脚本

---

## 任务类别

模型输出必须是以下四个字符串之一：

```text
cloudy
rainy
snowy
sunny
```

类别含义：

| 类别       | 含义                       |
| -------- | ------------------------ |
| `cloudy` | 多云 / 阴天 / 云层明显           |
| `rainy`  | 雨天 / 雨滴 / 湿路 / 雨伞等明显雨天特征 |
| `snowy`  | 雪天 / 积雪 / 下雪场景           |
| `sunny`  | 晴天 / 蓝天 / 阳光明显           |

---

## 仓库结构

```text
Weather-Classification-Baseline/
  README.md
  .gitignore
  requirements.txt
  requirements_train.txt
  main.py

  configs/
    commands.md

  docs/
    methodology.md
    results.md
    screenshots/
      README.md

  scripts/
    train_baseline.py
    check_duplicates.py
    make_clean_split.py
    weather_cls_cpu_benchmark.py
```

---

## 数据说明

本仓库不包含任何比赛官方数据、测试数据、外部图片或训练权重。

请自行准备如下格式的数据集：

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

整理后推荐使用：

```text
official_data_clean/
  clean_train/
    cloudy/
    rainy/
    snowy/
    sunny/
  clean_val/
    cloudy/
    rainy/
    snowy/
    sunny/
```

---

## 快速开始

### 1. 安装训练依赖

```bash
pip install -r requirements_train.txt
```

### 2. 构建 clean 数据集

```bash
python scripts/make_clean_split.py \
  --official_train ./official_data/official_train \
  --official_val ./official_data/official_val \
  --out_dir ./official_data_clean
```

### 3. 检查 train-val 重复图

```bash
python scripts/check_duplicates.py \
  --train_dir ./official_data_clean/clean_train \
  --val_dir ./official_data_clean/clean_val \
  --out_dir ./duplicate_review
```

### 4. 训练 EfficientNet-B0 baseline

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

### 5. 准备推理权重

平台推理脚本默认读取：

```text
results/weather_best.pth
```

因此可以将训练得到的模型复制为：

```bash
mkdir -p results
cp ./results/efficientnet_b0_baseline.pth ./results/weather_best.pth
```

### 6. 测试 CPU 推理速度

```bash
python scripts/weather_cls_cpu_benchmark.py \
  --image_dir ./official_data_clean/clean_val \
  --num_images 500
```

---

## 平台推理接口

`main.py` 中提供了：

```python
def predict(X):
    ...
```

其中：

* `X` 是 OpenCV 读取的 BGR 图片
* 返回值是 `cloudy`、`rainy`、`snowy`、`sunny` 之一

默认使用 CPU 推理，适合 CPU-only 评测环境。

---

## 公开版说明

本仓库是公开 baseline 版本，主要用于展示天气图像分类任务的完整工程流程。

由于相关比赛可能存在后续赛段，本仓库暂不公开最终比赛模型、最终调参细节、外部数据策略和训练权重。

本仓库重点关注：

* baseline 训练
* 数据清洗思路
* 推理接口适配
* CPU 时间评估
* 项目结构整理

---

## Disclaimer

This repository is for learning and technical documentation only.

Official datasets, external images, trained weights, private competition solutions, and platform credentials are not included.
