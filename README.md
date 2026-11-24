# A Study of Lightweight Convolutional Neural Network Architectures for Glutinous Rice Variety Classification

**การศึกษาสถาปัตยกรรมโครงข่ายประสาทเทียมเชิงคอนโวลูชันแบบน้ำหนักเบาสำหรับการจำแนกสายพันธุ์ข้าวเหนียว**

Bachelor of Engineering Thesis (Computer Engineering), Faculty of Sciences and Engineering,
Kasetsart University, Chalermphrakiat Sakon Nakhon Province Campus, 2026.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Flutter](https://img.shields.io/badge/Flutter-Android-02569B)
![TFLite](https://img.shields.io/badge/TFLite-on--device-green)

## Abstract

This thesis aims to develop a lightweight Convolutional Neural Network (CNN) for classifying six varieties of glutinous rice and compare its performance with three established models: VGG16, MobileNetV2, and EfficientNet-B0. A dataset of 1,800 glutinous rice seed images was divided into training, validation, and testing sets at a ratio of 80:10:10, respectively. Different model architectures and parameters were evaluated to improve classification performance.

The selected proposed CNN architecture (`Flatten_32-64-128_Drop0.1`) consists of three convolutional layers with 32, 64, and 128 filters (3×3 kernels, ReLU activation), each followed by 2×2 max pooling, a Flatten layer, a 256-unit Dense layer with a dropout rate of 0.1, and a 6-unit softmax output layer. The results showed that the proposed CNN achieved a test accuracy of 96.15% and a weighted F1-score of 0.9614, while EfficientNet-B0 achieved 91.76% accuracy and a weighted F1-score of 0.9174. VGG16 and MobileNetV2 performed less effectively. After implementation in a mobile application, the proposed CNN trained for 60 epochs correctly classified all 60 test images, with an average confidence score of 98.30%. The application allows users to select images from a camera or gallery, view classification results, access usage history, and obtain information about glutinous rice cultivation and care.

In conclusion, the proposed CNN demonstrated strong classification performance and suitability for mobile application deployment. However, further work should expand the diversity of the dataset and test images captured under real-world conditions to evaluate the model's reliability in different environments.

## Dataset

- **1,800 images**, 6 glutinous rice varieties (300 images per variety)
- Split: **80% train / 10% validation / 10% test**
- Input size: 224 × 224 × 3

| # | Variety (Thai) |
|---|----------------|
| 1 | ข้าวเหนียวขี้ตมหางนาค |
| 2 | ข้าวเหนียวพันธุ์นางสวน |
| 3 | ข้าวเหนียวอีขาวน้อย |
| 4 | ข้าวเหนียวสันป่าตอง |
| 5 | ข้าวเหนียว กข6 (RD6) |
| 6 | ข้าวเหนียวพันธุ์อีขาวใหญ่ |

## Proposed CNN Architecture

```
Input (224×224×3)
→ Conv2D(32, 3×3, ReLU)  → MaxPooling2D(2×2)
→ Conv2D(64, 3×3, ReLU)  → MaxPooling2D(2×2)
→ Conv2D(128, 3×3, ReLU) → MaxPooling2D(2×2)
→ Flatten
→ Dropout(0.1)
→ Dense(256)
→ Dense(6, Softmax)
```

## Results

### Proposed CNN vs. pre-trained models (frozen base layers)

| Model | Params | Train Acc (%) | Val Acc (%) | Test Acc (%) | Test Loss |
|-------|-------:|--------------:|------------:|-------------:|----------:|
| **Proposed CNN** | 8,483,654 | 98.33 | 96.63 | **96.15** | 0.8735 |
| VGG16 (Frozen) | 16,813,638 | 15.10 | 16.85 | 16.48 | 1.7917 |
| MobileNetV2 (Frozen) | 7,502,662 | 94.15 | 29.78 | 28.02 | 6.6236 |
| EfficientNet-B0 (Frozen) | 9,294,249 | 97.84 | 93.82 | 91.76 | 0.3100 |

### Weighted metrics and efficiency

| Model | Precision | Recall | F1-Score | Training Time (s) | RAM (MB) | Inference (ms/img) |
|-------|----------:|-------:|---------:|------------------:|---------:|-------------------:|
| **Proposed CNN** | 0.9643 | 0.9615 | **0.9614** | 18.28 | 66.61 | **3.96** |
| VGG16 (Frozen) | 0.0272 | 0.1648 | 0.0467 | 81.65 | 42.69 | 25.12 |
| MobileNetV2 (Frozen) | 0.0971 | 0.2802 | 0.1435 | 81.17 | 775.32 | 100.21 |
| EfficientNet-B0 (Frozen) | 0.9268 | 0.9176 | 0.9174 | 164.91 | 1260.83 | 191.60 |

### Best checkpoints used for the mobile app (Custom_CNN vs. EfficientNet-B0)

| Model | Epoch | Total Params | Trainable Params | Non-trainable Params | Train Acc (%) | Val Acc (%) | Test Acc (%) | Test Loss | Precision (Weighted) |
|-------|------:|-------------:|-----------------:|---------------------:|--------------:|------------:|-------------:|----------:|---------------------:|
| Custom_CNN | 60 | 25,785,158 | 25,785,158 | 0 | 96.10 | 93.82 | 95.60 | 1.0022 | 0.9597 |
| EfficientNet-B0 | 20 | 20,107,689 | 16,058,118 | 4,049,571 | 91.72 | 90.45 | 91.21 | 0.2154 | 0.9154 |

> **Note:** The tables above come from two separate experiment runs in the thesis (Tables 4-10/4-11 and Table 4-15), so the Proposed CNN figures differ slightly between them (96.15% vs. 95.60% test accuracy; 8.48M vs. 25.79M parameters). See the thesis chapter 4 for the exact configuration of each run.

### Model size (Proposed CNN)

| Format | Size (MB) |
|--------|----------:|
| Keras (.keras) | 97.14 |
| TFLite FP32 | 32.37 |
| TFLite FP16 | 16.19 |
| TFLite INT8 | 8.11 |

### Mobile application test

- Test set: 60 images (10 per variety)
- Proposed CNN (60 epochs): **60/60 correct**, average confidence **98.30%**

## Mobile Application

Built with Flutter (Android) and a TensorFlow Lite model. Features:

- Pick an image from the camera or gallery
- View the classification result with confidence score
- Usage history
- Glutinous rice cultivation and care information

## Tech Stack

Python · TensorFlow / Keras · Google Colab · TensorFlow Lite · Flutter

## Repository Contents

```
.
├── thesis/        # Full thesis PDF
├── README.md
└── ...            # (add notebooks, model files, app source as available)
```

## Citation

```bibtex
@thesis{phummee2026glutinous,
  title  = {A Study of Lightweight Convolutional Neural Network Architectures for Glutinous Rice Variety Classification},
  author = {Phummee, Tharathep and Gatruang, Apirak},
  year   = {2026},
  school = {Kasetsart University, Chalermphrakiat Sakon Nakhon Province Campus},
  type   = {Bachelor of Engineering Thesis}
}
```

**Advisor:** Assoc. Prof. Susama Chokphoemphun

## Future Work

- Expand the dataset's diversity
- Test with images captured under real-world conditions to evaluate reliability in different environments
