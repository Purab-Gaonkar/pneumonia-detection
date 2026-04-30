import os
import tensorflow as tf

TEST_DIR = os.path.join("chest_xray", "test")
MODEL_PATH = os.path.join("models", 'pneumonia_ensemble_model.h5')

print("Loading test dataset...")
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    shuffle=False,
    batch_size=32,
    image_size=(224, 224),
    label_mode='binary'
)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Evaluating model...")
loss, accuracy, auc = model.evaluate(test_dataset)
print(f"\\n--- Results ---")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test AUC: {auc:.4f}")
print(f"Test Loss: {loss:.4f}")
