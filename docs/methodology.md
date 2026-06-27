# 方法说明

## 任务介绍

本项目面向天气图像四分类任务，需要将输入图片分类为以下四个类别之一：

- `cloudy`：多云
- `rainy`：雨天
- `snowy`：雪天
- `sunny`：晴天

在比赛类任务中，评价指标通常不仅关注整体准确率，也会关注类别均衡表现。因此，本项目使用 Macro F1 作为主要验证指标。

Macro F1 会分别计算每个类别的 F1 分数，再取平均值。相比普通 accuracy，Macro F1 对小样本类别更加敏感。

---

## 整体流程

公开 baseline 版本采用以下流程：

1. 整理数据目录
2. 构建 clean train / clean val
3. 检查 train-val 重复图
4. 训练 EfficientNet-B0 baseline
5. 在验证集上计算 Macro F1
6. 保存最佳模型权重
7. 使用 `main.py` 适配平台推理接口
8. 使用 CPU benchmark 脚本估算推理时间

---

## 数据组织

数据采用常见的 ImageFolder 目录结构：

```text
train/
  cloudy/
  rainy/
  snowy/
  sunny/
每个类别一个文件夹，文件夹名即类别名。
该结构简单直观，适合快速训练分类模型，也方便后续切换到其他主干网络。
￼
数据清洗
天气图片中经常存在一些噪声样本，例如：
• 天气主体不明显
• 图标或天气预报截图
• 非真实拍摄图片
• 标签错误图片
• 多云和晴天边界模糊
• 阴天和雨天边界模糊
• 白云、雾天和雪天混淆
公开 baseline 版本主要提供流程脚本，实际清洗仍建议人工参与。
人工清洗时可以遵循以下原则：
• 明显错类图片删除
• 非真实天气图片删除
• 看不清天气状态的图片删除
• 标签争议很大的图片谨慎保留
• 类别边界样本应保持规则一致
￼
重复图检测
如果训练集中存在和验证集重复或近似重复的图片，本地验证分数可能虚高。
因此本项目提供 check_duplicates.py，用于检测：
1. MD5 完全重复图片
2. 平均哈希近重复图片
如果发现 train-val 重复，建议删除训练集一侧的重复图片，保留验证集不变。
￼
Baseline 模型
公开版本使用 EfficientNet-B0 作为 baseline 模型。
选择 EfficientNet-B0 的原因：
• 模型较轻量
• 推理速度较快
• timm 支持完善
• 适合作为图像分类任务的入门 baseline
• CPU 推理压力相对可控
该 baseline 的目的不是追求极限分数，而是提供一个完整、清晰、可复现的训练与推理模板。
￼
训练策略
训练脚本支持：
• ImageNet 预训练权重
• 随机裁剪
• 水平翻转
• 颜色扰动
• class weight
• Macro F1 验证
• 保存最佳 checkpoint
• eval-only 模式
训练过程中以验证集 Macro F1 作为模型选择标准。
￼
推理设计
平台推理文件为 main.py，其中必须实现：
Python
Copy
￼
Run
￼
def predict(X):
    ...
输入 X 是 OpenCV 读取的 BGR 图片，输出为：
Copy
cloudy / rainy / snowy / sunny
推理流程包括：
1. BGR 转 RGB
2. resize 到 224×224
3. ImageNet normalize
4. EfficientNet-B0 前向推理
5. 返回预测类别字符串
￼
CPU 推理
部分比赛评测环境只提供 CPU，因此本项目提供：
Copy
scripts/weather_cls_cpu_benchmark.py
用于估算模型在 CPU 上处理指定数量图片的耗时。
这有助于判断模型是否适合提交到 CPU-only 平台。
￼
公开版边界
本仓库是 baseline 公开版，主要用于学习和展示工程流程。
本仓库不包含：
• 最终比赛模型
• 最终调参细节
• 训练权重
• 官方数据集
• 外部图片数据
• 平台账号或服务器信息
这样可以在公开展示项目的同时，避免泄露后续比赛可能仍会用到的核心策略。