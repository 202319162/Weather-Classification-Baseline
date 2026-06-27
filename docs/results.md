# 实验结果记录

本文件记录公开 baseline 版本的实验设计、结果展示格式和实验记录。


---

## 1. 实验目标

本项目面向天气图像四分类任务，目标是将输入图片分类为以下四个类别之一：

* `cloudy`：多云
* `rainy`：雨天
* `snowy`：雪天
* `sunny`：晴天

主要评价指标为 **Macro F1**。

Macro F1 会分别计算每个类别的 F1 分数，再取平均值。相比整体准确率，Macro F1 更适合类别不均衡的数据集，能够更好反映模型对 rainy、snowy 等少数类别的识别能力。

---

## 2. 实验环境

| 项目            | 配置                                  |
| ------------- | ----------------------------------- |
| Framework     | PyTorch                             |
| Model Library | timm                                |
| Main Model    | EfficientNet-B0                     |
| Input Size    | 224 × 224                           |
| Optimizer     | AdamW                               |
| Loss          | CrossEntropyLoss                    |
| Metric        | Macro F1                            |
| Device        | GPU for training, CPU for inference |
| Classes       | cloudy / rainy / snowy / sunny      |

---

## 3. 数据划分

数据目录采用 ImageFolder 格式：

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

其中：

* `clean_train` 用于训练
* `clean_val` 用于本地验证
* 验证集不参与训练
* 如果发现 train-val 重复图，删除训练集一侧重复图片，保留验证集不变

---

## 4.  Results

> 注意：本节为公开 baseline 仓库的模拟示例结果，仅用于展示实验记录方式，不代表真实线上评测成绩。

| Model             | Input Size | Pretrained | Augmentation          | Macro F1 | CPU Time / 5000 Images | Note                     |
| ----------------- | ---------: | ---------: | --------------------- | -------: | ---------------------: | ------------------------ |
| EfficientNet-B0   |        224 |        Yes | Light                 |   0.7421 |                  1180s | baseline                 |
| EfficientNet-B0   |        224 |        Yes | Medium                |   0.7684 |                  1195s | stronger augmentation    |
| EfficientNet-B0   |        224 |        Yes | Medium + Class Weight |   0.7817 |                  1203s | class imbalance handling |
| MobileNetV3-Large |        224 |        Yes | Medium                |   0.7546 |                   860s | faster CPU inference     |
| ConvNeXt-Tiny     |        224 |        Yes | Medium                |   0.8213 |                  1560s | stronger backbone        |

---

## 5. Per-class  Metrics

> 

### EfficientNet-B0 Baseline

| Class         | Precision | Recall |         F1 |
| ------------- | --------: | -----: | ---------: |
| cloudy        |    0.7812 | 0.8067 |     0.7937 |
| rainy         |    0.6825 | 0.6100 |     0.6442 |
| snowy         |    0.7018 | 0.6900 |     0.6958 |
| sunny         |    0.8524 | 0.8933 |     0.8724 |
| **Macro Avg** |         - |      - | **0.7515** |

### EfficientNet-B0 + Class Weight

| Class         | Precision | Recall |         F1 |
| ------------- | --------: | -----: | ---------: |
| cloudy        |    0.7928 | 0.8000 |     0.7964 |
| rainy         |    0.7241 | 0.6300 |     0.6738 |
| snowy         |    0.7356 | 0.7100 |     0.7226 |
| sunny         |    0.8608 | 0.8867 |     0.8736 |
| **Macro Avg** |         - |      - | **0.7666** |

### ConvNeXt-Tiny Demo

| Class         | Precision | Recall |         F1 |
| ------------- | --------: | -----: | ---------: |
| cloudy        |    0.8425 | 0.8600 |     0.8512 |
| rainy         |    0.7816 | 0.7300 |     0.7549 |
| snowy         |    0.8123 | 0.7900 |     0.8010 |
| sunny         |    0.9034 | 0.9267 |     0.9149 |
| **Macro Avg** |         - |      - | **0.8305** |

---

## 6. CPU 推理耗时记录

CPU 推理速度使用以下脚本测试：

```bash
python scripts/weather_cls_cpu_benchmark.py \
  --image_dir ./official_data_clean/clean_val \
  --num_images 500
```

Demo 记录如下：

| Model             | Test Images | Avg Time / Image | Estimated 5000 Images | Within 70 min |
| ----------------- | ----------: | ---------------: | --------------------: | ------------: |
| EfficientNet-B0   |         500 |           0.236s |                 1180s |           Yes |
| MobileNetV3-Large |         500 |           0.172s |                  860s |           Yes |
| ConvNeXt-Tiny     |         500 |           0.312s |                 1560s |           Yes |

---

## 7. 观察结论

从示例实验可以观察到：

1. EfficientNet-B0 能够作为一个稳定 baseline，训练和推理成本较低。
2. 使用 ImageNet 预训练权重可以明显提升收敛速度和验证集表现。
3. Class weight 对 rainy、snowy 等少数类别更友好，有助于提升 Macro F1。
4. MobileNetV3 推理速度较快，适合对 CPU 时间要求更严格的场景。
5. ConvNeXt-Tiny 具备更强表达能力，但 CPU 推理时间也更高。
6. 对于天气图像分类任务，数据清洗、类别均衡和验证集可靠性非常重要。

---

## 8. Baseline 总结

公开 baseline 版本主要用于展示天气图像四分类任务的完整工程流程，包括：

* 数据目录整理
* 训练集 / 验证集划分
* 重复图检测
* EfficientNet-B0 baseline 训练
* Macro F1 评估
* CPU 推理速度测试
* 平台 `predict(X)` 接口适配

本仓库暂不包含最终比赛模型、最终调参细节和训练权重。
