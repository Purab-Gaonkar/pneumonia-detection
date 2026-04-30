import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

print("Setting up paths...")
TEST_DIR = os.path.join("chest_xray", "test")
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, 'pneumonia_ensemble_model.h5')
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

print("Loading test dataset...")
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    shuffle=False, # Must be False to match predictions with labels
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='binary'
)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Running predictions on the test dataset...")
# Predict on the entire test dataset
predictions = model.predict(test_dataset)
predictions = predictions.flatten() # Flatten to 1D array of probabilities

# Extract true labels
y_true = np.concatenate([y for x, y in test_dataset], axis=0).flatten()

# Convert probabilities to binary predictions (threshold 0.5)
y_pred = (predictions > 0.5).astype(int)

# 1. Plot Confusion Matrix
print("Generating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal (0)', 'Pneumonia (1)'], 
            yticklabels=['Normal (0)', 'Pneumonia (1)'])
plt.title('Confusion Matrix on Test Dataset')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'))
plt.close()

# 2. Plot ROC Curve
print("Generating ROC Curve...")
fpr, tpr, thresholds = roc_curve(y_true, predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'))
plt.close()

print(f"Evaluation complete! Plots saved in the '{PLOTS_DIR}/' directory.")
