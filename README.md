# Skin Cancer Classification Using Deep Learning Models

**Author:** Pranay Sai Beemanapalli  
**Course:** Data Mining — COMP 7118  
**Dataset:** ISIC Dermoscopic Image Dataset

---

A comparative deep learning study for binary classification of skin cancer images (benign vs. malignant) using three architectures: a Custom CNN, ResNet50, and MobileNetV2.

---

## Table of Contents

- [Abstract](#abstract)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Report](#report)

---

## Abstract

This project presents an automated skin lesion classification system using dermoscopic images from the ISIC dataset. Three deep learning models were developed and evaluated: a custom CNN, ResNet50 (transfer learning), and MobileNetV2 (lightweight transfer learning), using accuracy, precision, recall, F1-score, and ROC-AUC as evaluation metrics.

MobileNetV2 achieved the best overall performance (~85% accuracy) with the lowest false negatives and strong malignant recall. The custom CNN achieved ~84% accuracy with balanced performance. ResNet50 significantly underperformed at ~61% accuracy due to underfitting and critically poor malignant recall (0.16). Results highlight that in medical image classification, **recall matters more than accuracy** — missing a malignant case carries severe real-world consequences.

---

## Dataset

**[Skin Cancer ISIC Images](https://www.kaggle.com/datasets/rm1000/skin-cancer-isic-images)** — available on Kaggle.

| Property | Value |
|---|---|
| Image size | 224 x 224 pixels |
| Classes | Benign, Malignant |
| Training images | 2,638 |
| Validation images | 659 |
| Train / Val split | 80% / 20% |
| Batch size | 32 |
| Augmentation | Rotation (20°), zoom (0.2), horizontal flip |

> **Note:** The dataset is not included in this repository. See [Setup and Installation](#setup-and-installation) for download instructions.

---

## Models

### 1. Custom CNN (built from scratch)
- 3 convolutional blocks: Conv2D(32) → Conv2D(64) → Conv2D(128), each followed by MaxPooling
- Dense(128) + Dropout(0.5) + Sigmoid output
- Learns hierarchical features: edges → shapes → lesion-specific patterns

### 2. ResNet50 (Transfer Learning)
- Pre-trained on ImageNet, all base layers frozen
- Custom head: Flatten → Dense(128) → Dropout(0.5) → Sigmoid
- Full layer freezing caused underfitting on this dataset

### 3. MobileNetV2 (Transfer Learning)
- Pre-trained on ImageNet, backbone frozen; depthwise separable convolutions
- Custom head: Flatten → Dense(128) → Dropout(0.5) → Sigmoid
- Best overall model — accurate, efficient, lowest false negatives

**All models shared:**

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | Binary Crossentropy |
| Epochs | 10 |
| Batch size | 32 |
| GPU | NVIDIA Tesla T4 |

---

## Results

### Overall Accuracy Comparison

| Model | Validation Accuracy | AUC |
|---|---|---|
| Custom CNN | ~84% | ~0.93 |
| ResNet50 | ~61% | Low |
| **MobileNetV2** | **~85%** | **~0.93** |

---

### Custom CNN

| Class | Precision | Recall |
|---|---|---|
| Benign | ~0.88 | ~0.82 |
| Malignant | ~0.80 | ~0.86 |

- Overall accuracy: **84%**
- True Negatives (benign correctly classified): **297**
- True Positives (malignant correctly classified): **258**
- AUC: **~0.93**
- Steady training convergence, no significant overfitting
- Higher malignant recall than benign — desirable for medical diagnosis

---

### ResNet50

| Metric | Value |
|---|---|
| Validation Accuracy | ~61% |
| Malignant Recall | **0.16 (critically low)** |
| AUC | Low |

- Severe underfitting — model predicted benign for nearly all samples
- Root cause: all base layers frozen with no feature adaptation to dermoscopic images
- **Not suitable for medical use**

---

### MobileNetV2

| Metric | Value |
|---|---|
| Validation Accuracy | **~85%** |
| Precision | Balanced across both classes |
| Recall | Strong and balanced |
| False Negatives | **Lowest of all three models** |
| AUC | **~0.93** |

- Best and most stable learning curves
- Lowest false negative rate — most reliable for catching malignant cases
- Lightweight architecture suitable for mobile and edge healthcare deployment

---

### False Negative & Medical Suitability Summary

| Model | False Negatives | Medical Suitability |
|---|---|---|
| MobileNetV2 | Lowest | Best |
| Custom CNN | Moderate | Acceptable |
| ResNet50 | Extremely high | Not suitable |

> In medical diagnosis, a false negative = a missed cancer case, which can be life-threatening. MobileNetV2 minimizes this risk the most and is the recommended model.



## Project Structure
