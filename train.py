import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.ensemble import IsolationForest
import numpy as np
import joblib

# Ensure TensorFlow uses GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs available: {len(physical_devices)}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Training will run on CPU.")

DATASET_DIR = "../dataset/chest_xray"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

print("Loading datasets...")
train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='binary'
)

val_dataset = image_dataset_from_directory(
    VAL_DIR,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='binary'
)

# Apply basic data augmentation
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.1),
  tf.keras.layers.RandomZoom(0.1),
])

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

print("Building Deep Learning Ensemble Model...")
input_layer = tf.keras.Input(shape=(224, 224, 3))
# Preprocessing functions
preprocess_densenet = tf.keras.applications.densenet.preprocess_input(input_layer)
preprocess_mobilenet = tf.keras.applications.mobilenet_v2.preprocess_input(input_layer)
preprocess_resnet = tf.keras.applications.resnet_v2.preprocess_input(input_layer)

# Model 1: DenseNet121
densenet = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_tensor=preprocess_densenet)
densenet.trainable = False
x1 = tf.keras.layers.GlobalAveragePooling2D()(densenet.output)
x1 = tf.keras.layers.Dropout(0.3)(x1)

# Model 2: MobileNetV2
mobilenet = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_tensor=preprocess_mobilenet)
mobilenet.trainable = False
x2 = tf.keras.layers.GlobalAveragePooling2D()(mobilenet.output)
x2 = tf.keras.layers.Dropout(0.3)(x2)

# Model 3: ResNet50V2
resnet = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_tensor=preprocess_resnet)
resnet.trainable = False
x3 = tf.keras.layers.GlobalAveragePooling2D()(resnet.output)
x3 = tf.keras.layers.Dropout(0.3)(x3)

# Combine features
merged = tf.keras.layers.Concatenate()([x1, x2, x3])
dense = tf.keras.layers.Dense(256, activation='relu')(merged)
dense = tf.keras.layers.Dropout(0.4)(dense)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

ensemble_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

ensemble_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

print("Training Ensemble Top Layers...")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

ensemble_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stopping]
)

print("Unfreezing base models for fine-tuning...")
# Unfreeze the last few layers of each base model for fine-tuning
densenet.trainable = True
for layer in densenet.layers[:-20]: layer.trainable = False

mobilenet.trainable = True
for layer in mobilenet.layers[:-20]: layer.trainable = False

resnet.trainable = True
for layer in resnet.layers[:-20]: layer.trainable = False

ensemble_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Lower learning rate for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

ensemble_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    callbacks=[early_stopping]
)

print("Saving Ensemble Model...")
ensemble_model.save(os.path.join(MODELS_DIR, 'pneumonia_ensemble_model.h5'))

print("Training Out-of-Distribution (OOD) Detection Model (Is it an X-ray?)...")
# We use MobileNetV2 to extract features and Isolation Forest to detect anomalies
feature_extractor = tf.keras.Model(inputs=input_layer, outputs=x2)

print("Extracting features from training set for OOD detection...")
features = []
# Extract features from a subset to save time (e.g., 2000 images)
count = 0
for images, _ in train_dataset:
    batch_features = feature_extractor.predict(images, verbose=0)
    features.append(batch_features)
    count += len(images)
    if count >= 2000:
        break

features = np.vstack(features)

print("Fitting Isolation Forest...")
isolation_forest = IsolationForest(contamination=0.01, random_state=42)
isolation_forest.fit(features)

print("Saving OOD Model...")
joblib.dump(isolation_forest, os.path.join(MODELS_DIR, 'ood_isolation_forest.pkl'))

print("Training complete! Models saved in 'models/' directory.")
