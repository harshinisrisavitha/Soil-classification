
#  YOLO (You Only Look Once)

##  What is YOLO?

YOLO stands for **You Only Look Once**, a real-time object detection system that predicts bounding boxes and class probabilities directly from full images in a single evaluation. It's fast, accurate, and has evolved into multiple versions (YOLOv1 → YOLOv8+).

---

## How YOLO Works

### 1. Image Input
- YOLO receives an image.

### 2. Grid Division
- The image is divided into an SxS grid.
- Each cell is responsible for detecting objects whose **center** falls inside that cell.

###  3. Predictions Per Cell
Each grid cell predicts:
- **Bounding box coordinates** (x, y, width, height)
- **Confidence score** (probability of object presence)
- **Class probabilities** (which object?)

###  4. One-Pass Prediction
- YOLO uses a single convolutional neural network (CNN) pass to make all predictions.

###  5. Non-Maximum Suppression (NMS)
- Removes overlapping boxes and keeps only the most confident prediction.

---

##  YOLOv8 Classification Mode

When used for classification, YOLOv8 treats the image as a whole and predicts a **single label** — perfect for cases like **soil classification**.

---

##  How to Build a YOLO Model

### 1.  Install Ultralytics

```bash
pip install ultralytics
```

### 2. Prepare Dataset

For classification:
```
dataset/
├── train/
│   ├── sand/
│   ├── gravel/
├── val/
├── test/
```

### 3.  Load YOLOv8 Model

```python
from ultralytics import YOLO
model = YOLO('yolov8n-cls.pt')  # Classification model
```

### 4. 🏋️‍♂ Train the Model

```python
model.train(data='path/to/dataset', epochs=30, imgsz=224)
```

### 5.  Validate the Model

```python
metrics = model.val()
print(metrics)
```

### 6.  Predict New Images

```python
results = model.predict(source='image.jpg', save=True, save_txt=True)
print(results)
```

---

##  YOLOv8 Modes

| Mode            | Use Case                     | Example                        |
|------------------|-------------------------------|--------------------------------|
| Detection        | Object localization           | People, vehicles               |
| Classification   | Whole-image classification    | Rock type, dog breed           |
| Segmentation     | Pixel-level object masks      | Medical imaging, AR filters    |
| Pose             | Keypoint detection            | Human pose, hand tracking      |

---

##  Pro Tips

- Use **yolov8s-cls.pt** or **yolov8m-cls.pt** for more accuracy.
- Use **data augmentation** (blur, noise, grayscale) for robustness.
- Visualize confusion matrix for better understanding of misclassifications.

---


# Soil Classification using YOLOv8 
This project leverages the YOLOv8 classification model (yolov8n-cls.pt) to automatically categorize Martian soil textures into four distinct classes:
- Sand
- Sedimentary
- Gravel
- Cracked Rocks
The dataset utilized is derived from **Mars surface image (Curiosity rover) labeled data set**.
The dataset was imported via **Roboflow**.

- Model used: YOLOv8n-cls (Nano version — optimal for fast training/inference and limited hardware)
- Framework: Ultralytics YOLOv8
- Pretrained Weights: yolov8n-cls.pt (ImageNet pretrained)
- Epochs: 20
- Image size: 224x224
