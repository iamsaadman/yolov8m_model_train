# BTF YOLOv8 Object Detection Pipeline (Bangladesh Traffic Dataset)

A full-scale multi-phase object detection system built using Ultralytics YOLOv8m trained on a custom Bangladesh traffic dataset.  
The project focuses on progressive model improvement through dataset optimization, augmentation, balancing, and real-world evaluation.

---

#  Dataset

The dataset used in this project is hosted on Google Drive:

🔗 https://drive.google.com/file/d/1fBh6BQcgMIsdTKCzBExLX5OxKHZdKP2A/view?usp=drive_link

### Dataset Features:
Annotated traffic images (XML → YOLO format conversion)<br>
Classes:<br>
- Vehicle (car, truck, cng, etc.)<br>
- Bus<br>
- Bike (motorcycle, rickshaw, bicycle)<br>
- Person (pedestrian)<br>

Large-scale real-world road scenes<br>
Oversampled minority classes for balance<br>

---

# Project Pipeline

This project is divided into 3 structured phases:

---

#  Phase 1 — Baseline Model Training

###  Goals:
Upgrade model to YOLOv8m<br>
Increase input resolution<br>
Apply initial augmentations<br>

###  Steps:
Model: yolov8m.pt<br>
Image size: 832<br>
Basic augmentation enabled:<br>
- mosaic<br>
- mixup<br>
- geometric transforms<br>

XML → YOLO format conversion<br>
Dataset splitting (train/valid)<br>
Initial oversampling (bus & bike)<br>

### Output:
Baseline trained model<br>
Initial mAP scores<br>
Class distribution analysis<br>

---

#  Phase 2 — Optimization & Improvement

### Goals:
Improve accuracy and stability<br>
Fix class imbalance<br>
Tune hyperparameters<br>

### Steps:
Class balancing refinement<br>
Remove noisy labels / duplicates<br>

Hyperparameter tuning:<br>
- learning rate<br>
- batch size<br>
- optimizer adjustments<br>

Error analysis:<br>
- misclassification review<br>
- confusion patterns<br>

###  Output:
Improved mAP<br>
Reduced class bias<br>
Better generalization<br>

---

#  Phase 3 — Advanced Evaluation & Deployment Readiness

###  Goals:
Compare models<br>
Clean dataset annotations<br>
Test in real-world scenarios<br>

###  Steps:
Model comparison:<br>
- YOLOv8n vs YOLOv8s vs YOLOv8m<br>

Annotation refinement<br>
Real-world inference testing<br>
Performance benchmarking<br>

### Output:
Final best model selection<br>
Deployment-ready weights<br>
Real-world performance report<br>
