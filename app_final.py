import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path
import os
from datetime import datetime
import json
import base64
import random

st.set_page_config(page_title="Skin Disease Detection", page_icon="🩺", layout="wide")

PROJECT_DIR = Path("C:/Users/ACER/OneDrive/Desktop/Skin Disease Detection")
MODEL_PATH_ADV = Path("best_model_advanced.keras")
MODEL_PATH_V2 = Path("skin_disease_model_v2.keras")
MODEL_PATH_OLD = Path("best_model.keras")
HISTORY_FILE = PROJECT_DIR / 'prediction_history.json'

# Custom objects for loading
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1.0), self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * tf.pow(1. - pt, self.gamma)
        loss = focal_weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha, 'from_logits': self.from_logits})
        return config

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

CLASS_LABELS = {
    'akiec': 'Pre-Cancer (Actinic Keratoses)',
    'bcc': 'Skin Cancer (Basal Cell)',
    'bkl': 'Non-Cancer (Benign Keratosis)',
    'df': 'Dermatofibroma',
    'nv': 'Moles (Nevus)',
    'vasc': 'Birthmarks/Vascular',
    'mel': 'Dangerous Cancer (Melanoma)'
}

DISEASE_INFO = {
    'akiec': {
        'name': 'Actinic Keratoses (Pre-Cancer)',
        'description': 'Rough, scaly patches on skin from sun damage. Can become cancer if not treated.',
        'medicines': ['Special creams', 'Freezing treatment', 'Skin surgery'],
        'prevention': 'Use sunscreen SPF 30+ daily, avoid sun',
        'severity': 'Medium - Can become cancer'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma (Skin Cancer)',
        'description': 'Most common skin cancer. Usually stays in one place but can damage skin.',
        'medicines': ['Surgery', 'Mohs surgery', 'Radiation'],
        'prevention': 'Sun protection, no tanning beds',
        'severity': 'Low-Medium - Curable if early'
    },
    'bkl': {
        'name': 'Benign Keratosis (Non-Cancer)',
        'description': 'Common non-cancerous raised spots that appear with age.',
        'medicines': ['No treatment needed', 'Freezing for looks'],
        'prevention': 'None needed',
        'severity': 'Very Low - Harmless'
    },
    'df': {
        'name': 'Dermatofibroma',
        'description': 'Small hard bumps on skin, usually on legs. May itch but not dangerous.',
        'medicines': ['No treatment', 'Surgery if bothersome'],
        'prevention': 'None needed',
        'severity': 'Very Low - Harmless'
    },
    'nv': {
        'name': 'Moles (Nevus)',
        'description': 'Common brown/black spots. Most are safe but watch for changes.',
        'medicines': ['No treatment', 'Remove if concerned'],
        'prevention': 'Watch ABCDE changes',
        'severity': 'Very Low - Usually safe'
    },
    'vasc': {
        'name': 'Vascular Lesions (Birthmarks)',
        'description': 'Red/purple marks from blood vessels.',
        'medicines': ['Laser treatment', 'Steroids'],
        'prevention': 'None needed',
        'severity': 'Very Low - Cosmetic only'
    },
    'mel': {
        'name': 'Melanoma (Dangerous Cancer)',
        'description': 'Most dangerous skin cancer. Can spread to other organs!',
        'medicines': ['Surgery', 'Immunotherapy', 'Chemo'],
        'prevention': 'Use sunscreen, avoid sun, check skin monthly',
        'severity': 'HIGH - Very dangerous!'
    }
}

def load_history():
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_history(prediction_data):
    history = load_history()
    history.append(prediction_data)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

@st.cache_resource
def load_model():
    custom_objs = {'FocalLoss': FocalLoss}
    for path in [MODEL_PATH_ADV, MODEL_PATH_V2, MODEL_PATH_OLD]:
        try:
            model = tf.keras.models.load_model(path, custom_objects=custom_objs, compile=False)
            st.info(f"Loaded model: {path.name}")
            return model
        except:
            continue
    st.error("No model found! Please train first.")
    return None

def get_img_size():
    return 224

def preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize((get_img_size(), get_img_size()), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 127.5 - 1.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def tta_predict(model, img_array, n_aug=10):
    preds = [model.predict(img_array, verbose=0)]
    for _ in range(n_aug):
        aug = img_array.copy()
        if random.random() > 0.5:
            aug = np.fliplr(aug)
        if random.random() > 0.5:
            aug = np.flipud(aug)
        if random.random() > 0.3:
            aug = np.rot90(aug, k=random.choice([1, 2, 3]), axes=(1, 2))
        preds.append(model.predict(aug, verbose=0))
    return np.mean(preds, axis=0)[0]

def predict(model, img_array):
    predictions = tta_predict(model, img_array)
    results = []
    for idx in np.argsort(predictions)[::-1][:3]:
        results.append({
            'class': CLASS_NAMES[idx],
            'confidence': float(predictions[idx]) * 100,
            'label': CLASS_LABELS[CLASS_NAMES[idx]],
            'info': DISEASE_INFO[CLASS_NAMES[idx]]
        })
    return results

st.title("Skin Disease Detection System")

model = load_model()

if model is None:
    st.error("Model not loaded! Please train the model first.")
    st.stop()

st.success("AI Model Loaded - Ready to Analyze!")

menu = st.sidebar.selectbox("Choose Option:", [
    "Disease Detection",
    "Skin Analysis & Routine",
    "Diet & Health Tips",
    "Prediction History"
])

if menu == "Disease Detection":
    st.header("Disease Detection from Image")
    
    source = st.radio("Choose method:", ["Upload Image", "Take Photo"])
    
    if source == "Upload Image":
        uploaded_file = st.file_uploader("Upload skin image...", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Take a photo")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Image", width='stretch')

        if st.button("Analyze Now"):
            processed_img = preprocess_image(image)
            predictions = predict(model, processed_img)
            
            top_pred = predictions[0]
            
            st.markdown("### Analysis Results:")
            
            if top_pred['class'] == 'mel':
                st.error(f" HIGH PRIORITY: {top_pred['info']['name']}")
            elif top_pred['class'] in ['akiec', 'bcc']:
                st.warning(f" Attention Required: {top_pred['info']['name']}")
            else:
                st.success(f" Detected: {top_pred['info']['name']}")
            
            st.markdown(f"**Confidence:** {top_pred['confidence']:.1f}%")
            
            st.markdown("### Details:")
            st.write(f"**Description:** {top_pred['info']['description']}")
            st.write(f"**Severity:** {top_pred['info']['severity']}")
            
            st.markdown("**Medicines/Products:**")
            for med in top_pred['info']['medicines']:
                st.write(f"- {med}")
            
            st.markdown(f"**Prevention:** {top_pred['info']['prevention']}")
            
            st.markdown("### Top 3 Possibilities:")
            for i, pred in enumerate(predictions):
                st.write(f"{i+1}. {pred['label']} - {pred['confidence']:.1f}%")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save to History"):
                    save_history({
                        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'disease': top_pred['info']['name'],
                        'confidence': top_pred['confidence'],
                        'severity': top_pred['info']['severity']
                    })
                    st.success("Saved!")
            
            report = f"Skin Report - {top_pred['info']['name']}\n\nDetails: {top_pred['info']['description']}\n\nMedicines: {', '.join(top_pred['info']['medicines'])}"
            st.download_button("Download Report", report, file_name="skin_report.txt")

elif menu == "Skin Analysis & Routine":
    st.header("Personalized Skin Care Routine")
    
    st.info("Answer questions to get your routine!")
    
    with st.form("quiz"):
        q1 = st.selectbox("How does skin feel after washing?", 
                         ["Tight and dry", "Shiny/Oily", "Oily in T-zone, dry elsewhere", "Comfortable/Normal", "Red/Irritated"])
        
        q2 = st.selectbox("Main concerns?", 
                         ["Acne", "Wrinkles", "Dark spots", "Dryness", "Redness", "Large pores"])
        
        q3 = st.selectbox("How does skin react to products?", 
                         ["Easily irritated", "Usually fine", "Sometimes breaks out", "Very sensitive"])
        
        submitted = st.form_submit_button("Get My Routine")
        
        if submitted:
            skin_type = "normal"
            if "dry" in q1.lower():
                skin_type = "dry"
            elif "shiny" in q1.lower() or "oily" in q1.lower():
                skin_type = "oily"
            elif "else" in q1.lower():
                skin_type = "combination"
            elif "red" in q1.lower() or "irritat" in q1.lower():
                skin_type = "sensitive"
            
            routines = {
                "dry": {
                    "morning": "Gentle cleanser → Hydrating toner → Vitamin C → Rich moisturizer → SPF 30",
                    "evening": "Oil cleanser → Gentle cleanser → Hydrating serum → Rich night cream",
                    "products": ["Heavy cream", "Hyaluronic acid", "Gentle cleanser"]
                },
                "oily": {
                    "morning": "Foaming cleanser → BHA toner → Niacinamide → Gel moisturizer → SPF 30",
                    "evening": "Oil cleanser → Foaming cleanser → Salicylic acid → Light gel",
                    "products": ["Salicylic acid", "Gel moisturizer", "Niacinamide"]
                },
                "combination": {
                    "morning": "Gentle cleanser → Hydrating toner → Vitamin C → Light moisturizer → SPF 30",
                    "evening": "Balanced cleanser → Treatment for T-zone → Night cream",
                    "products": ["Balanced cleanser", "Light moisturizer", "Treatment serums"]
                },
                "sensitive": {
                    "morning": "Water rinse → Centella toner → Soothing serum → Minimal moisturizer → SPF 30",
                    "evening": "Gentle micellar → Ultra gentle cleanser → Centella serum → Barrier cream",
                    "products": ["Fragrance-free", "Centella", "Aloe vera"]
                },
                "normal": {
                    "morning": "Gentle cleanser → Vitamin C → Light moisturizer → SPF 30",
                    "evening": "Cleanser → Retinol → Night cream",
                    "products": ["Basic cleanser", "Sunscreen", "Retinol"]
                }
            }
            
            routine = routines.get(skin_type, routines["normal"])
            
            st.success(f"Your Skin Type: {skin_type.title()}")
            
            st.markdown("### Morning Routine")
            st.write(routine["morning"])
            
            st.markdown("### Evening Routine")
            st.write(routine["evening"])
            
            st.markdown("### Recommended Products")
            for p in routine["products"]:
                st.write(f"- {p}")

elif menu == "Diet & Health Tips":
    st.header("Diet & Health for Healthy Skin")
    
    st.markdown("### Foods for Healthy Skin")
    foods = ["Oranges, strawberries (Vitamin C)", "Almonds, spinach (Vitamin E)", "Fish, walnuts (Omega-3)", "Beans, seafood (Zinc)", "Green tea", "Water (8 glasses)"]
    for f in foods:
        st.write(f"- {f}")
    
    st.markdown("### Foods to Avoid")
    bad = ["Too much sugar", "Fried foods", "Excessive dairy", "Processed foods", "Alcohol"]
    for b in bad:
        st.write(f"- {b}")
    
    st.markdown("### Daily Skin Tips")
    tips = ["Always use sunscreen", "Drink water", "Sleep 7-8 hours", "Exercise regularly", "Don't touch face with dirty hands", "Change pillowcase weekly"]
    for t in tips:
        st.write(f"- {t}")

elif menu == "Prediction History":
    st.header("Your Prediction History")
    history = load_history()
    
    if history:
        for i, item in enumerate(reversed(history)):
            with st.expander(f"{item['datetime']} - {item['disease']}"):
                st.write(f"**Disease:** {item['disease']}")
                st.write(f"**Confidence:** {item['confidence']:.1f}%")
                st.write(f"**Severity:** {item['severity']}")
    else:
        st.info("No predictions yet!")