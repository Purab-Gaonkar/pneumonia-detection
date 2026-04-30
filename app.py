import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import io

app = FastAPI()

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, 'pneumonia_ensemble_model.h5')
OOD_MODEL_PATH = os.path.join(MODELS_DIR, 'ood_isolation_forest.pkl')

ensemble_model = None
ood_model = None
feature_extractor = None

@app.on_event("startup")
async def load_models():
    global ensemble_model, ood_model, feature_extractor
    
    print("Loading models...")
    try:
        ensemble_model = tf.keras.models.load_model(MODEL_PATH)
        print("Ensemble model loaded.")
        
        # Recreate feature extractor from ensemble model (MobileNetV2 branch)
        input_layer = tf.keras.Input(shape=(224, 224, 3))
        preprocess_mobilenet = tf.keras.applications.mobilenet_v2.preprocess_input(input_layer)
        
        # Load the fine-tuned weights from the ensemble model
        mobilenet_block = ensemble_model.get_layer('mobilenet_block')
        mobilenet = tf.keras.applications.MobileNetV2(weights=None, include_top=False)
        mobilenet.set_weights(mobilenet_block.get_weights())
        
        x2 = mobilenet(preprocess_mobilenet)
        x2 = tf.keras.layers.GlobalAveragePooling2D()(x2)
        x2 = tf.keras.layers.Dropout(0.3)(x2)
        feature_extractor = tf.keras.Model(inputs=input_layer, outputs=x2)
        print("Feature extractor ready.")
        
        ood_model = joblib.load(OOD_MODEL_PATH)
        print("OOD model loaded.")
    except Exception as e:
        print(f"Error loading models. Please ensure train.py has been run successfully. Error: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ... (rest of the predict function)
    if ensemble_model is None or ood_model is None:
        raise HTTPException(status_code=500, detail="Models are not loaded. Run train.py first.")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read and preprocess the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize((224, 224))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        
        # 1. Out-of-Distribution Check (Is it an X-ray?)
        features = feature_extractor.predict(image_array)
        is_inlier = ood_model.predict(features)[0] # 1 for inlier (X-ray), -1 for outlier (random)
        
        if is_inlier == -1:
            return {
                "status": "success",
                "is_xray": False,
                "message": "The uploaded image does not appear to be a chest X-ray. Please upload a valid X-ray image.",
                "prediction": None,
                "confidence": None
            }
            
        # 2. Predict Pneumonia using Ensemble Model
        prediction_prob = ensemble_model.predict(image_array)[0][0]
        
        # If probability > 0.5 it's PNEUMONIA (class 1), else NORMAL (class 0)
        result_class = "Pneumonia" if prediction_prob > 0.5 else "Normal"
        confidence = float(prediction_prob) if result_class == "Pneumonia" else float(1 - prediction_prob)
        
        return {
            "status": "success",
            "is_xray": True,
            "prediction": result_class,
            "confidence": round(confidence * 100, 2),
            "message": f"Detection Complete: {result_class}."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files from the 'static' directory at the root level
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
