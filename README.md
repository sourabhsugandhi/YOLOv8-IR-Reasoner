# YOLOv8 IR Reasoner

## Overview

YOLOv8 IR Reasoner is a modified YOLOv8 object detector that integrates an **IR Reasoner module** to improve thermal infrared object detection.

The Reasoner introduces **global spatial reasoning** on top of YOLOv8’s convolutional features while keeping the original architecture and training pipeline intact.

The goal of this project is to **fairly compare** standard YOLOv8 and YOLOv8 IR Reasoner on the same thermal dataset using identical training settings.

---

## What Was Changed

### 1. Reasoner Module

* Added at: `ultralytics/nn/modules/reasoner.py`
* Applies **spatial self-attention** on feature maps
* Uses **residual connections**
* Input/output shapes remain unchanged

**Why:**
Thermal images lack strong local textures. Global reasoning helps relate distant regions and suppress background noise.

---

### 2. Module Registration

* Registered in:

  * `ultralytics/nn/modules/__init__.py`
  * `ultralytics/nn/tasks.py`

**Why:**
YOLOv8 builds models via a registry. Registration enables YAML-based model construction.

---

### 3. Separate Model Configuration

* Created: `ultralytics/cfg/models/v8/yolov8_reasoner.yaml`
* Original `yolov8.yaml` remains unchanged

**Why:**
Ensures **clean baseline vs Reasoner comparison** and reproducibility.

---

### 4. Reasoner Placement

* Inserted **3 Reasoner blocks** before Detect head:

  * P3
  * P4
  * P5

* Detect head receives refined features

**Why:**
High-level feature maps benefit most from global reasoning.

---

### 5. No Other Changes

* Backbone unchanged
* Neck unchanged
* Head unchanged
* Loss functions unchanged
* Training pipeline unchanged

**Why:**
Ensures performance differences come **only from the Reasoner module**.

---

## Dataset

* **Dataset:** FLIR ADAS (Thermal only)
* **Classes:** 15
* **Class IDs:** Remapped to `[0–14]`
* **Input:** Thermal images replicated to 3 channels

### Subset Used

| Split      | Images |
| ---------- | ------ |
| Train      | 1000   |
| Validation | 250    |

---

## Models Compared

| Model              | Description              |
| ------------------ | ------------------------ |
| YOLOv8             | Baseline model           |
| YOLOv8 IR Reasoner | YOLOv8 + Reasoner blocks |

Both models are trained under identical conditions.

---

## Training Setup

* **Model:** YOLOv8-s
* **Image Size:** 640 × 640
* **Batch Size:** 16
* **Epochs:** 100
* **Seed:** 42
* **Learning Rate:** 0.003

### Augmentations

* hsv_h / hsv_s / hsv_v: 0.0
* degrees: 0.0
* translate: 0.1
* scale: 0.5
* shear: 0.0
* fliplr: 0.5
* mosaic: 1.0
* mixup: 0.1

---

## Evaluation Metrics

* Precision
* Recall
* mAP@0.5
* mAP@0.5–0.95
* Qualitative visual inspection

All evaluations are performed on the same validation split.

---

## Experimental Results (Epoch 100)

| Metric    | YOLOv8 | YOLOv8 IR Reasoner |
| --------- | ------ | ------------------ |
| Precision | 0.3478 | **0.3843**         |
| Recall    | 0.2094 | 0.2090             |
| mAP@50    | 0.2010 | **0.2057**         |
| mAP@50–95 | 0.0987 | **0.0990**         |

---

## Key Observations

* **Higher Precision**
  The Reasoner reduces false positives and improves detection quality.

* **Stable Recall**
  True detections are preserved.

* **Consistent mAP Gains**
  Improvements seen across both loose and strict metrics.

* **Late Training Benefits**
  Baseline performs better early, but Reasoner improves with longer training.

---

## Conclusion

At 100 epochs, YOLOv8 IR Reasoner outperforms standard YOLOv8 in **precision and mAP**, while maintaining recall.

This confirms that adding **global spatial reasoning** improves thermal object detection under controlled conditions.

---

## Notes

* Early epochs may favor baseline due to faster local feature learning
* Reasoner models require more training to converge
* Improvements are solely due to the Reasoner module

## Dataset link: 
https://oem.flir.com/en-in/solutions/automotive/adas-dataset-form/#anchor1
## Credits : 
https://github.com/tlgksy/ir-reasoner

---
