import os
import io
import cv2
import base64
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from contextlib import asynccontextmanager
from torchvision import transforms, models
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import load_model

# --- HELPER: Convert OpenCV Image to Base64 String ---
def encode_image(image_np):
    """Encodes a numpy image (BGR) into a base64 string for the frontend."""
    success, encoded_image = cv2.imencode('.png', image_np)
    if success:
        return base64.b64encode(encoded_image).decode('utf-8')
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a heat map for class activation.
    FIXED: Handles Keras models that return outputs as lists.
    """
    # 1. Handle Model Inputs (Fixes "Expected structure" warning)
    # If the model expects a list of inputs, wrap the image in a list.
    model_inputs = model.inputs
    if isinstance(img_array, np.ndarray):
        # If the model has multiple inputs, we might need to be careful, 
        # but for CNNs it's usually just one image.
        img_inputs = [img_array]
    else:
        img_inputs = img_array

    # 2. Handle Model Outputs (Fixes "List indices" error)
    # Ensure we are grabbing the actual tensor, not a list containing the tensor.
    model_output = model.output
    if isinstance(model_output, list):
        model_output = model_output[0]

    # 3. Create the Gradient Model
    grad_model = tf.keras.models.Model(
        inputs=model_inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model_output]
    )

    # 4. Record operations
    with tf.GradientTape() as tape:
        # Pass inputs as a list to satisfy Keras 3.x
        conv_outputs, predictions = grad_model(img_inputs)
        
        # DOUBLE CHECK: If predictions is still a list, extract the tensor
        if isinstance(predictions, list):
            predictions = predictions[0]
            
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        class_channel = predictions[:, pred_index]

    # 5. Compute Gradients
    grads = tape.gradient(class_channel, conv_outputs)

    # 6. Generate Heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Check if conv_outputs is a list (rare but possible)
    if isinstance(conv_outputs, list):
        conv_outputs = conv_outputs[0]
        
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 7. Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    """
    Overlays the heatmap on the original image.
    """
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Rescale heatmap to match original image size
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))

    # Convert to RGB
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    
    # Superimpose the heatmap on original image
    superimposed_img = jet * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8') # Ensure valid range

    return superimposed_img
# --- GLOBAL VARIABLES (To hold models) ---
models_cache = {}
preprocessors = {}

# --- 1. LIFESPAN (Load models when server starts) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Loading AI Models... This might take a minute.")
    
    # A. LOAD CHEST MODEL (PyTorch)
    try:
        chest_model = models.resnet50(weights=None)
        chest_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        chest_model.fc = nn.Linear(2048, len(chest_labels))
        
        # Load weights (Fixing keys as before)
        checkpoint = torch.load('models/chest_xray.pth', map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "").replace("base_model.", "").replace("backbone.", "").replace("fc.1.", "fc.")
            if "classifier" in name: name = name.replace("classifier", "fc")
            if "head" in name: name = name.replace("head", "fc")
            new_state_dict[name] = v
        
        chest_model.load_state_dict(new_state_dict, strict=False)
        chest_model.eval()
        models_cache['chest'] = chest_model
        models_cache['chest_labels'] = chest_labels
        print("✅ Chest X-Ray Model Loaded")
    except Exception as e:
        print(f"❌ Failed to load Chest Model: {e}")

    # B. LOAD BRAIN MODEL (Keras)
    try:
        models_cache['brain'] = load_model('models/brain_tumor_vgg16_fixed.keras')
        models_cache['brain_labels'] = {0: 'Glioma', 1: 'Meningioma', 2: 'No Tumor', 3: 'Pituitary'}
        print("✅ Brain Tumor Model Loaded")
    except Exception as e:
        print(f"❌ Failed to load Brain Model: {e}")

    # C. LOAD EYE MODELS (Ensemble)
    try:
        models_cache['eye_multi'] = load_model('models/multibranch_model_1.h5')
        models_cache['eye_cnn'] = load_model('models/cnn_model_1.h5')
        models_cache['eye_labels'] = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        print("✅ Eye Disease Models Loaded")
    except Exception as e:
        print(f"❌ Failed to load Eye Models: {e}")

    # D. LOAD SKIN MODEL + PREPROCESSORS (MobileNetV3)
    try:
        # Load Model
        models_cache['skin'] = load_model('models/best_model.keras', compile=False)
        
        # Load CSV for Encoders
        df = pd.read_csv('models/train.csv')
        df['sex'] = df['sex'].fillna('unknown')
        df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].fillna('unknown')
        df['age_approx'] = df['age_approx'].fillna(df['age_approx'].mean())
        
        # Fit Encoders
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe.fit(df[['sex', 'anatom_site_general_challenge']])
        
        scaler = StandardScaler()
        scaler.fit(df[['age_approx']])
        
        preprocessors['skin_ohe'] = ohe
        preprocessors['skin_scaler'] = scaler
        preprocessors['skin_mean_age'] = df['age_approx'].mean()
        print("✅ Skin Cancer Model & Preprocessors Loaded")
    except Exception as e:
        print(f"❌ Failed to load Skin Model: {e}")

    yield
    print("🛑 Shutting down AI Server")

# --- INITIALIZE APP ---
app = FastAPI(lifespan=lifespan)

# --- UTILITIES ---
def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image.convert('RGB')

# --- ENDPOINT 1: CHEST X-RAY ---
# --- ENDPOINT 1: CHEST X-RAY (Upgraded with Enhancement Tools) ---
@app.post("/predict/chest")
async def predict_chest(file: UploadFile = File(...)):
    model = models_cache.get('chest')
    if not model: raise HTTPException(500, "Chest model not loaded")
    
    # 1. Read Image
    image = read_imagefile(await file.read())
    
    # 2. Preprocess for Model (PyTorch Standard)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)
    
    # 3. Generate Visualizations (OpenCV)
    # Convert PIL -> OpenCV (BGR)
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    img_cv = cv2.resize(img_cv, (224, 224))

    # A. CLAHE (Contrast Enhancement - "Bone View")
    # Work on the "Lightness" channel to avoid messing up colors (even though it's grayscale)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # B. Negative Mode (Inverted - Good for Masses)
    img_neg = cv2.bitwise_not(img_cv)
    
    # 4. Predict
    with torch.no_grad():
        out = model(img_tensor)
        probs = torch.sigmoid(out)[0].tolist()
    
    # 5. Format Results
    results = []
    labels = models_cache['chest_labels']
    
    # Define thresholds for reporting (only show if confidence > 20%)
    for i, score in enumerate(probs):
        confidence = round(score * 100, 2)
        if confidence > 20.0: 
            results.append({"condition": labels[i], "confidence": confidence})
            
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    # If nothing detected, return "Normal"
    if not results:
        results.append({"condition": "No Finding", "confidence": 99.0})

    return {
        "module": "chest",
        "findings": results,
        "images": {
            "original": encode_image(img_cv),
            "clahe": encode_image(img_clahe),
            "negative": encode_image(img_neg)
        }
    }

# --- ENDPOINT 2: BRAIN TUMOR ---
# --- ENDPOINT 2: BRAIN TUMOR (Upgraded with Heatmap & Contrast) ---
@app.post("/predict/brain")
async def predict_brain(file: UploadFile = File(...)):
    model = models_cache.get('brain')
    if not model: raise HTTPException(500, "Brain model not loaded")
    
    # 1. Read Image
    image = read_imagefile(await file.read())
    
    # 2. Preprocess for VGG16 (Resize to 150x150 as per your training)
    # We keep a copy of the original size for better visualization later
    img_viz = np.array(image)
    img_viz = cv2.cvtColor(img_viz, cv2.COLOR_RGB2BGR)
    
    # Resize for Model
    img_resized = image.resize((150, 150))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # 3. Predict
    pred = model.predict(img_batch, verbose=0)
    idx = np.argmax(pred)
    label = models_cache['brain_labels'].get(idx, "Unknown")
    conf = float(np.max(pred)) * 100
    
    # 4. Generate "High Contrast" Version (Histogram Equalization)
    # MRI scans are often dark; this makes the tumor pop out.
    img_yuv = cv2.cvtColor(img_viz, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_contrast = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # 5. Generate Heatmap
    encoded_heatmap = None
    try:
        # Find the last Conv layer. For VGG16, it is usually 'block5_conv3'
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if 'Conv2D' in layer.__class__.__name__:
                last_conv_layer_name = layer.name
                break
        
        if last_conv_layer_name:
            # We resize the original image to 150x150 for the heatmap overlay to match model input
            heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer_name)
            
            # Upscale heatmap to original image size for better visuals
            viz_resized = cv2.resize(img_viz, (150, 150))
            gradcam_img = save_and_display_gradcam(viz_resized, heatmap)
            encoded_heatmap = encode_image(cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"❌ Brain Heatmap Failed: {e}")

    # 6. Return Data
    return {
        "module": "brain", 
        "diagnosis": label, 
        "confidence": round(conf, 2),
        "images": {
            "original": encode_image(img_viz),
            "contrast": encode_image(img_contrast),
            "heatmap": encoded_heatmap
        }
    }
# --- ENDPOINT 3: EYE DISEASE (UPDATED WITH CLAHE VISUALIZATION) ---
# --- ENDPOINT 3: EYE DISEASE (Fixed Heatmap Logic) ---
@app.post("/predict/eye")
async def predict_eye(file: UploadFile = File(...)):
    m_multi = models_cache.get('eye_multi')
    m_cnn = models_cache.get('eye_cnn')
    if not m_multi or not m_cnn: raise HTTPException(500, "Eye models not loaded")
    
    # 1. Read & Preprocess
    image = read_imagefile(await file.read())
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (224, 224))
    
    # 2. Generate CLAHE (Enhanced View)
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    img_clahe_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    final_rgb = cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2RGB)
    
    # 3. Predict
    norm_img = final_rgb / 255.0
    img_batch = np.expand_dims(norm_img, axis=0)
    
    p1 = m_multi.predict(img_batch, verbose=0)
    p2 = m_cnn.predict(img_batch, verbose=0)
    ensemble = (0.7 * p1) + (0.3 * p2)
    
    idx = np.argmax(ensemble)
    label = models_cache['eye_labels'][idx]
    conf = float(np.max(ensemble)) * 100

    # 4. Generate REAL Grad-CAM Heatmap (ROBUST FIX)
    encoded_heatmap = None
    try:
        last_conv_layer_name = None
        
        # Iterate backwards to find the LAST layer that is a Convolution
        for layer in reversed(m_cnn.layers):
            # We check the class name string instead of output_shape to avoid errors
            if 'Conv2D' in layer.__class__.__name__:
                last_conv_layer_name = layer.name
                break
        
        if last_conv_layer_name:
            # print(f"✅ Found Conv Layer: {last_conv_layer_name}") # Debugging
            heatmap = make_gradcam_heatmap(img_batch, m_cnn, last_conv_layer_name)
            gradcam_img = save_and_display_gradcam(img_resized, heatmap)
            encoded_heatmap = encode_image(cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))
        else:
            print("❌ No Conv2D layer found in model architecture.")

    except Exception as e:
        print(f"❌ Heatmap generation skipped: {e}")

    # 5. Return Results
    return {
        "module": "eye",
        "diagnosis": label,
        "confidence": round(conf, 2),
        "images": {
            "original": encode_image(img_resized),
            "clahe": encode_image(img_clahe_bgr),
            "heatmap": encoded_heatmap
        },
        "details": "CLAHE enhancement + Grad-CAM analysis applied."
    }

# --- ENDPOINT 4: SKIN CANCER (Multi-Input) ---
# --- ENDPOINT 4: SKIN CANCER (Upgraded with Hair Removal & Segmentation) ---
@app.post("/predict/skin")
async def predict_skin(
    file: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    site: str = Form(...)
):
    model = models_cache.get('skin')
    ohe = preprocessors.get('skin_ohe')
    if not model or not ohe: raise HTTPException(500, "Skin model/preprocessors not loaded")
    
    # 1. Read Image
    image = read_imagefile(await file.read())
    
    # Keep original size for visuals, but resize for processing
    img_viz = np.array(image)
    img_viz = cv2.cvtColor(img_viz, cv2.COLOR_RGB2BGR) # Convert to BGR
    
    # 2. FEATURE A: Digital Hair Removal (The "Cleaned View")
    # Grayscale -> BlackHat Transform -> Threshold -> Inpainting
    gray = cv2.cvtColor(img_viz, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    img_cleaned = cv2.inpaint(img_viz, thresh, 1, cv2.INPAINT_TELEA)

    # 3. FEATURE B: Lesion Segmentation (Contour Detection)
    # Convert cleaned image to grayscale -> Blur -> Otsu Threshold
    gray_clean = cv2.cvtColor(img_cleaned, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_clean, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_segmented = img_cleaned.copy()
    if contours:
        # Draw the largest contour (the lesion) in Green
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(img_segmented, [largest_contour], -1, (0, 255, 0), 2)

    # 4. Prepare for AI Model
    # Resize cleaned image to model input size (512x512)
    img_model = cv2.resize(img_cleaned, (512, 512))
    img_model = cv2.cvtColor(img_model, cv2.COLOR_BGR2RGB)
    img_arr = np.array(img_model).astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_arr, axis=0)
    
    # 5. Prepare Metadata
    mean_age = preprocessors['skin_mean_age']
    age_val = age if age > 0 else mean_age
    
    meta_df = pd.DataFrame({'sex': [sex], 'anatom_site_general_challenge': [site], 'age_approx': [age_val]})
    cat_enc = ohe.transform(meta_df[['sex', 'anatom_site_general_challenge']])
    num_scl = preprocessors['skin_scaler'].transform(meta_df[['age_approx']])
    meta_batch = np.hstack([cat_enc, num_scl])
    
    # 6. Predict
    pred = model.predict([img_batch, meta_batch], verbose=0)
    malignancy = float(pred[0][0])
    
    risk = "Malignant" if malignancy > 0.5 else "Benign"
    
    return {
        "module": "skin", 
        "diagnosis": risk, 
        "malignancy_score": round(malignancy * 100, 2),
        "details": "High Risk" if malignancy > 0.5 else "Low Risk",
        "images": {
            "original": encode_image(img_viz),
            "cleaned": encode_image(img_cleaned),   # No Hair
            "segmented": encode_image(img_segmented) # Green Outline
        }
    }

# --- ROOT ---
@app.get("/")
def home():
    return {"status": "online", "message": "Universal Disease Predictor API is running."}