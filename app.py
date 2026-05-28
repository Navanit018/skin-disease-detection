import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

st.set_page_config(page_title="Skin Disease Detection", layout="wide")

CLASS_NAMES = [
    'akiec',    # Actinic Keratoses
    'bcc',      # Basal Cell Carcinoma
    'bkl',      # Benign Keratosis
    'df',       # Dermatofibroma
    'nv',       # Melanocytic Nevi
    'vasc',     # Vascular Lesions
    'mel'       # Melanoma
]

DISEASE_INFO = {
    'akiec': {
        'name': 'Actinic Keratoses',
        'description': 'Precancerous skin growths that appear as rough, scaly patches on sun-exposed areas.',
        'medicines': ['5-Fluorouracil cream', 'Imiquimod cream', 'Diclofenac gel', 'Cryotherapy (freezing)', 'Photodynamic therapy'],
        'prevention': 'Use sunscreen daily (SPF 30+), avoid excessive sun exposure, wear protective clothing',
        'severity': 'Medium - can progress to skin cancer if untreated'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'description': 'Most common type of skin cancer. Rarely spreads but can damage surrounding tissue.',
        'medicines': ['Surgery (most common)', 'Mohs micrographic surgery', 'Radiation therapy', 'Topical medications (for superficial)'],
        'prevention': 'Sun protection, avoid tanning beds, regular skin self-exams',
        'severity': 'Low-Medium - usually curable when caught early'
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'description': 'Non-cancerous raised spots on the skin, also known as seborrheic keratosis.',
        'medicines': ['No treatment needed', 'Cryotherapy if cosmetic concern', 'Curettage', 'Laser therapy'],
        'prevention': 'None specific needed',
        'severity': 'Very Low - harmless'
    },
    'df': {
        'name': 'Dermatofibroma',
        'description': 'Benign skin nodules, usually found on legs. Often itchy and firm.',
        'medicines': ['Usually no treatment needed', 'Surgical excision if symptomatic', 'Corticosteroid injection'],
        'prevention': 'None specific',
        'severity': 'Very Low - harmless'
    },
    'nv': {
        'name': 'Melanocytic Nevi (Moles)',
        'description': 'Common moles, usually brown or black spots on the skin.',
        'medicines': ['No treatment needed', 'Surgical removal if concerned about changes', 'Laser removal for cosmetic'],
        'prevention': 'Monitor for ABCDE changes (Asymmetry, Border, Color, Diameter, Evolving)',
        'severity': 'Very Low - usually harmless but monitor for changes'
    },
    'vasc': {
        'name': 'Vascular Lesions',
        'description': 'Birthmarks or hemangiomas from abnormal blood vessels.',
        'medicines': ['Laser therapy (pulsed dye laser)', 'Corticosteroids', 'Beta-blockers (for large hemangiomas)'],
        'prevention': 'None specific',
        'severity': 'Very Low - usually cosmetic concern only'
    },
    'mel': {
        'name': 'Melanoma',
        'description': 'Most dangerous form of skin cancer. Can spread to other organs if not caught early.',
        'medicines': ['Surgery (wide excision)', 'Immunotherapy (Pembrolizumab, Nivolumab)', 'Targeted therapy (BRAF/MEK inhibitors)', 'Chemotherapy', 'Radiation therapy'],
        'prevention': 'Sun protection (SPF 30+), avoid tanning beds, regular skin checks, monitor moles',
        'severity': 'HIGH - deadliest form of skin cancer if untreated'
    }
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model():
    try:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 7)
        )
        ckpt = torch.load('skin_disease_model_pt.pt', map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model not found. Please train the model first. Error: {e}")
        return None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_disease(model, img):
    img_tensor = transform(img.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
    top_3_idx = np.argsort(probabilities)[::-1][:3]
    results = []
    for idx in top_3_idx:
        results.append({
            'class': CLASS_NAMES[idx],
            'confidence': float(probabilities[idx]) * 100,
            'info': DISEASE_INFO[CLASS_NAMES[idx]]
        })
    return results

st.title("Skin Disease Detection System")

st.markdown("Upload a skin image to detect potential diseases and get recommendations.")

st.info("Upload a clear photo of the affected skin area. For best results, use good lighting.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    model = load_model()

    if model is not None:
        predictions = predict_disease(model, image)

        st.subheader("Prediction Results")

        top_prediction = predictions[0]

        if top_prediction['confidence'] > 60:
            if top_prediction['class'] == 'mel':
                st.error(f"HIGH PRIORITY: {top_prediction['info']['name']}")
            elif top_prediction['class'] in ['akiec', 'bcc']:
                st.warning(f"Attention: {top_prediction['info']['name']}")
            else:
                st.info(f"Detected: {top_prediction['info']['name']}")
        else:
            st.warning("Low confidence prediction. Please consult a dermatologist.")

        for i, pred in enumerate(predictions):
            with st.expander(f"#{i+1}: {pred['info']['name']} ({pred['confidence']:.1f}%)"):
                st.write(f"**Description:** {pred['info']['description']}")
                st.write(f"**Severity:** {pred['info']['severity']}")
                st.write(f"**Recommended Medicines:** {', '.join(pred['info']['medicines'])}")
                st.write(f"**Prevention:** {pred['info']['prevention']}")

        st.subheader("Top 3 Predictions")
        for i, pred in enumerate(predictions):
            st.write(f"{i+1}. **{pred['info']['name']}** - {pred['confidence']:.1f}%")

st.markdown("---")

st.subheader("Common Skin Conditions Guide")

cols = st.columns(3)
for i, (key, info) in enumerate(DISEASE_INFO.items()):
    with cols[i % 3]:
        st.markdown(f"**{info['name']}**")
        st.caption(f"Severity: {info['severity']}")

st.markdown("---")
st.warning("Disclaimer: This app is for educational purposes only. Please consult a dermatologist for proper diagnosis and treatment.")
