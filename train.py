# =========================================
# 🚗 DRIVER DROWSINESS DETECTION — TRAINING
# =========================================
# Run this in a GPU-enabled environment (Kaggle Notebook / Colab).
# Produces: drowsiness_model.keras  (feed this into convert_to_tfjs.py)

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# =========================================
# 📁 DATASET PATHS  (edit these to match your environment)
# =========================================
DATASET_ROOT = "/kaggle/input/datasets/rakibuleceruet/drowsiness-prediction-dataset"
ACTIVE_PATH  = os.path.join(DATASET_ROOT, "0 FaceImages/Active Subjects")
FATIGUE_PATH = os.path.join(DATASET_ROOT, "0 FaceImages/Fatigue Subjects")

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# =========================================
# 📊 DATASET ANALYSIS (sanity check before training)
# =========================================
def analyze_dataset(root):
    print("📂 Dataset Analysis:\n")
    total = 0
    for class_name in os.listdir(root):
        class_path = os.path.join(root, class_name)
        if os.path.isdir(class_path):
            n = len([f for f in os.listdir(class_path)
                     if f.lower().endswith(IMAGE_EXTENSIONS)])
            total += n
            print(f"Class: {class_name}\nNumber of images: {n}\n")
    print("=================================")
    print(f"Total number of images: {total}")
    return total


# =========================================
# 📊 LOAD IMAGES  (NO /255 — EfficientNet handles normalisation internally)
# =========================================
def load_images(folder, label):
    images, labels = [], []
    for file in os.listdir(folder):
        if not file.lower().endswith(IMAGE_EXTENSIONS):
            continue
        img_path = os.path.join(folder, file)
        try:
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
            img = tf.keras.preprocessing.image.img_to_array(img)  # keep [0, 255]
            images.append(img)
            labels.append(label)
        except Exception:
            continue
    return images, labels


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    return image, label


def plot_history(h1, h2, metric="accuracy", title=""):
    v1 = h1.history[metric] + h2.history[metric]
    v2 = h1.history[f"val_{metric}"] + h2.history[f"val_{metric}"]
    ep = range(1, len(v1) + 1)
    boundary = len(h1.history[metric])

    plt.figure(figsize=(10, 4))
    plt.plot(ep, v1, label=f"Train {metric}")
    plt.plot(ep, v2, label=f"Val {metric}")
    plt.axvline(boundary, color="gray", linestyle="--", label="Fine-tune start")
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(f"{metric}_curve.png", dpi=150)
    plt.show()


def build_model():
    base_model = tf.keras.applications.EfficientNetV2S(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
        include_preprocessing=True,  # handles rescaling internally; DO NOT /255 yourself
    )
    base_model.trainable = False  # frozen for warm-up phase

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs), base_model


def main():
    analyze_dataset(DATASET_ROOT)

    active_images, active_labels = load_images(ACTIVE_PATH, 0)    # Alert
    fatigue_images, fatigue_labels = load_images(FATIGUE_PATH, 1)  # Drowsy

    X = np.array(active_images + fatigue_images, dtype=np.float32)
    y = np.array(active_labels + fatigue_labels, dtype=np.float32)

    print(f"Total images : {len(X)}")
    print(f"Alert        : {len(active_images)}")
    print(f"Drowsy       : {len(fatigue_images)}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )
    print(f"Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train), seed=SEED)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model, base_model = build_model()
    model.summary()

    metrics = [
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]

    # ---------- Phase 1: warm-up (frozen backbone) ----------
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy", metrics=metrics)

    callbacks_warmup = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=5,
                                          restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=3, min_lr=1e-6, verbose=1),
    ]

    print("\n===== Phase 1: Warm-up (frozen backbone) =====")
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=15,
                          callbacks=callbacks_warmup)

    # ---------- Phase 2: fine-tune (partial unfreeze) ----------
    total_layers = len(base_model.layers)
    unfreeze_from = int(total_layers * 0.70)
    for layer in base_model.layers[unfreeze_from:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss="binary_crossentropy", metrics=metrics)

    callbacks_finetune = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=6,
                                          restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                              patience=3, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint("best_drowsiness_model.keras",
                                            monitor="val_auc", save_best_only=True,
                                            mode="max", verbose=1),
    ]

    print("\n===== Phase 2: Fine-tuning (partial unfreeze) =====")
    history2 = model.fit(train_ds, validation_data=val_ds, epochs=20,
                          callbacks=callbacks_finetune)

    plot_history(history1, history2, "accuracy", "Accuracy over training")
    plot_history(history1, history2, "auc", "AUC over training")
    plot_history(history1, history2, "loss", "Loss over training")

    print("\n===== Test-set evaluation =====")
    results = model.evaluate(test_ds, verbose=1)
    for name, val in zip(model.metrics_names, results):
        print(f"  {name}: {val:.4f}")

    model.save("drowsiness_model.keras")
    print("Model saved to drowsiness_model.keras")


if __name__ == "__main__":
    main()
