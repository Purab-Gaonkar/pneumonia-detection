import tensorflow as tf
import numpy as np
from PIL import Image
import joblib

print("Loading models...")
ensemble_model = tf.keras.models.load_model('models/pneumonia_ensemble_model.h5')
ood_model = joblib.load('models/ood_isolation_forest.pkl')

input_layer = tf.keras.Input(shape=(224, 224, 3))
preprocess_mobilenet = tf.keras.applications.mobilenet_v2.preprocess_input(input_layer)

mobilenet_block = ensemble_model.get_layer('mobilenet_block')
mobilenet = tf.keras.applications.MobileNetV2(weights=None, include_top=False)
mobilenet.set_weights(mobilenet_block.get_weights())

x2 = mobilenet(preprocess_mobilenet)
x2 = tf.keras.layers.GlobalAveragePooling2D()(x2)
x2 = tf.keras.layers.Dropout(0.3)(x2)
feature_extractor = tf.keras.Model(inputs=input_layer, outputs=x2)

print("Testing all images in validation dataset...")
val_dir = 'chest_xray/val'
inliers = 0
outliers = 0

import glob
for img_path in glob.glob(val_dir + '/*/*.jpeg'):
    try:
        image = Image.open(img_path).convert('RGB')
        image = image.resize((224, 224))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        
        features = feature_extractor.predict(image_array, verbose=0)
        is_inlier = ood_model.predict(features)[0]
        
        if is_inlier == 1:
            inliers += 1
        else:
            outliers += 1
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

print(f"Total tested: {inliers + outliers}")
print(f"Inliers (X-ray): {inliers}")
print(f"Outliers (Not X-ray): {outliers}")
