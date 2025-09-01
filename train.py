import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# ================== DIRECTORIES ==================
train_dir = "dataset/train"
val_dir = "dataset/validation"
test_dir = "dataset/test"

img_size = (150, 150)
batch_size = 32
epochs = 20

# ================== DATA LOADERS ==================
datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary'
)
val_gen = datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary'
)
test_gen = datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False
)

# ================== MODEL ==================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(150,150,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# ================== TRAINING ==================
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen
)

# Save model
model.save("breast_cancer_model.h5")
print("✅ Model saved as breast_cancer_model.h5")

# Save training history
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("✅ Training history saved as training_history.pkl")

# ================== EVALUATION ==================
train_loss, train_acc = model.evaluate(train_gen)
val_loss, val_acc = model.evaluate(val_gen)
test_loss, test_acc = model.evaluate(test_gen)

with open("accuracy_results.txt", "w") as f:
    f.write(f"Train Accuracy: {train_acc:.4f}\n")
    f.write(f"Validation Accuracy: {val_acc:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
print("✅ Accuracy results saved to accuracy_results.txt")

# ================== METRICS ==================
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

# Classification report
report = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"], output_dict=True)
pd.DataFrame(report).transpose().to_csv("metrics_report.csv", index=True)
print("✅ Metrics report saved to metrics_report.csv")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
pd.DataFrame(cm, index=["Benign", "Malignant"], columns=["Predicted Benign", "Predicted Malignant"]).to_csv("confusion_matrix.csv")

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.close()
print("✅ Confusion matrix saved (CSV + PNG)")

# ================== TRAINING CURVES ==================
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.close()
print("✅ Training curves saved (training_curves.png)")
