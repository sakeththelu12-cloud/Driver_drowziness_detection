# 🚗 Driver Drowsiness Detection System

## 📌 Project Overview

The **Driver Drowsiness Detection System** is a deep learning-based computer vision project designed to classify a driver's facial image into one of two categories:

- 🟢 **Alert / Active**
- 🔴 **Drowsy / Fatigued**

The project uses **EfficientNetV2S**, a powerful pre-trained convolutional neural network, along with **transfer learning and fine-tuning** to achieve effective image classification.

The trained model is saved in `.keras` format and can later be converted for deployment, including use in web applications with **TensorFlow.js**.

---

## 🎯 Objective

The main objective of this project is to build an intelligent model that can detect signs of driver fatigue from facial images.

Early detection of drowsiness can help improve road safety by enabling systems to generate warnings or alerts when a driver appears fatigued.

---

## ✨ Features

- Deep learning-based image classification
- Uses **EfficientNetV2S** with ImageNet weights
- Transfer learning approach
- Two-phase training:
  - Frozen backbone warm-up
  - Partial fine-tuning
- Real-time deployment compatibility
- Data augmentation to improve generalization
- Stratified dataset splitting
- Multiple evaluation metrics
- Automatic learning-rate reduction
- Early stopping to reduce overfitting
- Best model checkpoint saving
- Training performance visualization
- Final model saved in `.keras` format

---

# 🧠 Model Architecture

The project uses **EfficientNetV2S** as the base model.

```text
Input Image
     │
     ▼
EfficientNetV2S
(Pre-trained on ImageNet)
     │
     ▼
Global Average Pooling
     │
     ▼
Batch Normalization
     │
     ▼
Dense Layer (256, ReLU)
     │
     ▼
Dropout (0.4)
     │
     ▼
Dense Layer (64, ReLU)
     │
     ▼
Dropout (0.3)
     │
     ▼
Dense Layer (1, Sigmoid)
     │
     ▼
Prediction:
Alert or Drowsy
```

---

# 📂 Dataset Structure

The project expects the dataset to have the following structure:

```text
drowsiness-prediction-dataset/
│
└── 0 FaceImages/
    │
    ├── Active Subjects/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    │
    └── Fatigue Subjects/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

### Class Labels

| Class | Label |
|---|---:|
| Active / Alert | `0` |
| Fatigue / Drowsy | `1` |

---

# ⚙️ Technologies Used

- **Python**
- **TensorFlow**
- **Keras**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **EfficientNetV2S**
- **CUDA / GPU** for accelerated training

---

# 🖥️ Recommended Environment

This project is designed to run on a GPU-enabled environment such as:

- Kaggle Notebooks
- Google Colab
- Local machine with NVIDIA GPU and CUDA support

GPU acceleration is recommended because EfficientNetV2S is a relatively large deep learning model.

---

# 📦 Installation

Install the required libraries:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

---

# 📁 Dataset Configuration

Update the dataset path according to your environment.

```python
DATASET_ROOT = "/kaggle/input/datasets/rakibuleceruet/drowsiness-prediction-dataset"

ACTIVE_PATH = os.path.join(
    DATASET_ROOT,
    "0 FaceImages/Active Subjects"
)

FATIGUE_PATH = os.path.join(
    DATASET_ROOT,
    "0 FaceImages/Fatigue Subjects"
)
```

For Kaggle, make sure the dataset is attached to your notebook.

---

# 🖼️ Image Configuration

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
```

### Image Size

All images are resized to:

```text
224 × 224 × 3
```

This matches the input requirements of EfficientNetV2S.

---

# 🔄 Dataset Analysis

Before training, the program analyzes the dataset and counts the number of images available in each class.

Example output:

```text
📂 Dataset Analysis:

Class: Active Subjects
Number of images: XXXX

Class: Fatigue Subjects
Number of images: XXXX

=================================
Total number of images: XXXX
```

This helps verify that the dataset paths and image files are correct before training begins.

---

# 🔀 Dataset Splitting

The dataset is divided using **stratified splitting** to preserve the class distribution.

```text
80% → Training + Validation/Test preparation

Final split:

80% → Training
10% → Validation
10% → Testing
```

The split is performed using:

```python
train_test_split(..., stratify=y)
```

Stratification helps ensure that both classes are properly represented in the training, validation, and testing datasets.

---

# 🔄 Data Augmentation

To improve the model's ability to generalize to unseen images, data augmentation is applied to the training dataset.

The following augmentations are used:

- Horizontal flipping
- Random brightness adjustment
- Random contrast adjustment
- Random saturation adjustment

Example:

```python
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    return image, label
```

Data augmentation is applied only to the training dataset.

---

# 🧠 Transfer Learning

The project uses:

```python
EfficientNetV2S
```

with pre-trained ImageNet weights.

Initially, the EfficientNetV2S backbone is frozen:

```python
base_model.trainable = False
```

This allows the newly added classification layers to learn first without significantly modifying the pre-trained feature extractor.

---

# 🚀 Training Process

The training process consists of two phases.

## Phase 1: Warm-Up Training

During the first phase:

- EfficientNetV2S backbone is frozen
- Only the custom classification layers are trained
- Adam optimizer is used
- Learning rate: `0.001`

```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-3
)
```

Maximum epochs:

```text
15
```

---

## Phase 2: Fine-Tuning

During fine-tuning:

- Approximately the last 30% of the EfficientNetV2S layers are unfrozen
- Batch Normalization layers remain frozen
- A smaller learning rate is used

```python
learning_rate = 1e-5
```

Maximum epochs:

```text
20
```

Fine-tuning allows the model to adapt the pre-trained features specifically for driver drowsiness detection.

---

# 🛑 Callbacks Used

## Early Stopping

Training automatically stops when validation performance stops improving.

The monitored metric is:

```text
Validation AUC
```

This helps reduce unnecessary training and overfitting.

---

## Reduce Learning Rate on Plateau

If validation loss does not improve, the learning rate is automatically reduced.

This helps the model make smaller and more precise updates.

---

## Model Checkpoint

The best-performing fine-tuned model is saved automatically as:

```text
best_drowsiness_model.keras
```

The best model is selected based on:

```text
Validation AUC
```

---

# 📊 Evaluation Metrics

The following metrics are used to evaluate the model:

## Accuracy

Measures the percentage of correct predictions.

```text
Accuracy = Correct Predictions / Total Predictions
```

---

## AUC

**Area Under the ROC Curve** measures how well the model distinguishes between alert and drowsy classes.

Higher AUC values indicate better classification performance.

---

## Precision

Measures how many predicted drowsy cases were actually drowsy.

```text
Precision = TP / (TP + FP)
```

---

## Recall

Measures how many actual drowsy cases were correctly detected.

```text
Recall = TP / (TP + FN)
```

For a driver safety application, recall is especially important because failing to detect a truly drowsy driver can be dangerous.

---

# 📈 Training Visualizations

After training, the following graphs are generated:

```text
accuracy_curve.png
auc_curve.png
loss_curve.png
```

These graphs show the model's performance during both:

- Warm-up training
- Fine-tuning

The graphs include a marker indicating where fine-tuning begins.

---

# 🧪 Test Set Evaluation

After training is complete, the final model is evaluated on the unseen test dataset.

Example:

```python
results = model.evaluate(test_ds)
```

The following values are displayed:

```text
Loss
Accuracy
AUC
Precision
Recall
```

The test dataset is not used during training or validation, providing a more realistic estimate of model performance on unseen data.

---

# 💾 Output Files

After successful execution, the following files may be generated:

```text
drowsiness_model.keras
best_drowsiness_model.keras
accuracy_curve.png
auc_curve.png
loss_curve.png
```

### Main Model

```text
drowsiness_model.keras
```

This is the final trained model.

### Best Checkpoint

```text
best_drowsiness_model.keras
```

This contains the best model saved during fine-tuning based on validation AUC.

---

# 🌐 TensorFlow.js Conversion

The final model can be converted for use in a web application.

Example workflow:

```text
Train Model
     │
     ▼
drowsiness_model.keras
     │
     ▼
convert_to_tfjs.py
     │
     ▼
TensorFlow.js Model
     │
     ▼
Web Application
     │
     ▼
Camera-Based Drowsiness Detection
```

The converted model can be integrated with:

- HTML
- CSS
- JavaScript
- TensorFlow.js
- Webcam API

This allows the system to perform predictions directly from a user's webcam.

---

# 🔮 Future Improvements

The project can be further improved by adding:

- Real-time webcam detection
- Eye aspect ratio analysis
- Yawning detection
- Facial landmark detection
- Head pose estimation
- Sound alerts for drowsy drivers
- Mobile application integration
- Cloud-based monitoring
- Multi-class fatigue detection
- Temporal models using video sequences
- CNN + LSTM architecture for continuous frame analysis

---

# ⚠️ Important Notes

### Image Normalization

The images are **not manually divided by 255** because:

```python
include_preprocessing=True
```

is enabled in EfficientNetV2S.

EfficientNetV2S automatically performs the required preprocessing.

---

### Batch Normalization

During fine-tuning, Batch Normalization layers remain frozen to maintain stable learned statistics.

---

### Safety Disclaimer

This project is intended for **educational and research purposes**.

A production-grade driver safety system should be tested extensively under different real-world conditions, including:

- Low lighting
- Night driving
- Sunglasses
- Face masks
- Different camera angles
- Different facial appearances
- Motion blur
- Multiple environmental conditions

This model should not be considered a replacement for professional automotive safety systems without extensive validation and testing.

---

# 👨‍💻 How to Run

1. Open a GPU-enabled Kaggle Notebook or Google Colab notebook.
2. Attach or upload the dataset.
3. Update the dataset paths if necessary.
4. Install the required dependencies.
5. Run the training script.

```bash
python train_drowsiness_model.py
```

6. Wait for both training phases to complete.
7. Review the generated performance graphs.
8. Check the test-set metrics.
9. Use:

```text
drowsiness_model.keras
```

for further deployment or TensorFlow.js conversion.

---

# 📌 Project Workflow

```text
Dataset Collection
        │
        ▼
Dataset Analysis
        │
        ▼
Image Loading
        │
        ▼
Train / Validation / Test Split
        │
        ▼
Data Augmentation
        │
        ▼
EfficientNetV2S Transfer Learning
        │
        ▼
Warm-Up Training
        │
        ▼
Partial Fine-Tuning
        │
        ▼
Model Evaluation
        │
        ▼
Save Trained Model
        │
        ▼
TensorFlow.js Conversion
        │
        ▼
Real-Time Deployment
```

---

# 👨‍💻 Author

**THELU SAI SAKETH**

B.Tech — Computer Science and Engineering (Data Science)  
SR University, Warangal

---

## ⭐ If you found this project useful

Consider giving the repository a **star ⭐** and using it as a foundation for further research and development in intelligent driver safety systems.
