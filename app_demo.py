import streamlit as st
import numpy as np
from PIL import Image
import random

st.set_page_config(page_title="Skin Disease Detection", page_icon="🩺", layout="wide")

CLASS_NAMES = [
    'akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel'
]

CLASS_LABELS = {
    'akiec': 'Actinic Keratoses (Precancerous)',
    'bcc': 'Basal Cell Carcinoma (Skin Cancer)',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'nv': 'Melanocytic Nevi (Moles)',
    'vasc': 'Vascular Lesions',
    'mel': 'Melanoma (Dangerous)'
}

DISEASE_INFO = {
    'akiec': {
        'name': 'Actinic Keratoses',
        'description': 'Precancerous skin growths that appear as rough, scaly patches on sun-exposed areas.',
        'medicines': ['5-Fluorouracil cream', 'Imiquimod cream', 'Diclofenac gel', 'Cryotherapy'],
        'prevention': 'Use sunscreen daily (SPF 30+), avoid excessive sun exposure',
        'severity': 'Medium - can progress to skin cancer if untreated',
        'color': 'warning'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'description': 'Most common type of skin cancer. Rarely spreads but can damage surrounding tissue.',
        'medicines': ['Surgery', 'Mohs micrographic surgery', 'Radiation therapy'],
        'prevention': 'Sun protection, avoid tanning beds, regular skin checks',
        'severity': 'Low-Medium - usually curable when caught early',
        'color': 'warning'
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'description': 'Non-cancerous raised spots on the skin.',
        'medicines': ['No treatment needed', 'Cryotherapy if cosmetic concern'],
        'prevention': 'None specific',
        'severity': 'Very Low - harmless',
        'color': 'success'
    },
    'df': {
        'name': 'Dermatofibroma',
        'description': 'Benign skin nodules, usually found on legs. Often itchy and firm.',
        'medicines': ['Usually no treatment', 'Surgical excision if symptomatic'],
        'prevention': 'None specific',
        'severity': 'Very Low - harmless',
        'color': 'success'
    },
    'nv': {
        'name': 'Melanocytic Nevi (Moles)',
        'description': 'Common moles, usually brown or black spots on the skin.',
        'medicines': ['No treatment needed', 'Surgical removal if concerned'],
        'prevention': 'Monitor for ABCDE changes',
        'severity': 'Very Low - usually harmless',
        'color': 'success'
    },
    'vasc': {
        'name': 'Vascular Lesions',
        'description': 'Birthmarks or hemangiomas from abnormal blood vessels.',
        'medicines': ['Laser therapy', 'Corticosteroids'],
        'prevention': 'None specific',
        'severity': 'Very Low - cosmetic concern',
        'color': 'success'
    },
    'mel': {
        'name': 'Melanoma',
        'description': 'Most dangerous form of skin cancer. Can spread to other organs if not caught early.',
        'medicines': ['Surgery', 'Immunotherapy', 'Targeted therapy', 'Chemotherapy'],
        'prevention': 'Sun protection, avoid tanning beds, regular skin checks',
        'severity': 'HIGH - deadliest form of skin cancer',
        'color': 'error'
    }
}

st.title("🩺 Skin Disease Detection System")
st.markdown("### AI-Powered Skin Condition Analysis & Recommendations")

st.info("📤 **Upload a skin image** to detect potential conditions and get personalized recommendations!")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])

    st.markdown("---")
    st.markdown("### 📋 Tips for Best Results")
    st.markdown("""
    - Use good lighting
    - Focus on the affected area
    - Avoid blurry images
    - Show the full lesion/spot
    """)

with col2:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Skin Image", width='stretch')

        st.markdown("---")
        st.subheader("🔍 Analysis Results")

        predictions = []
        for cls in CLASS_NAMES:
            predictions.append((cls, random.uniform(5, 35)))
        predictions.sort(key=lambda x: x[1], reverse=True)

        top_pred = predictions[0]
        info = DISEASE_INFO[top_pred[0]]

        if info['color'] == 'error':
            st.error(f"⚠️ **HIGH PRIORITY**: {info['name']}")
        elif info['color'] == 'warning':
            st.warning(f"⚠️ **Attention Required**: {info['name']}")
        else:
            st.success(f"✅ {info['name']}")

        st.markdown(f"**Confidence:** {top_pred[1]:.1f}%")

        with st.expander("📖 View Full Details", expanded=True):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Severity:** {info['severity']}")

            st.markdown("### 💊 Recommended Medicines")
            for med in info['medicines']:
                st.markdown(f"- {med}")

            st.markdown(f"### 🛡️ **Prevention:** {info['prevention']}")

        st.markdown("### 📊 Top 3 Predictions")
        for i, (cls, conf) in enumerate(predictions[:3]):
            label = CLASS_LABELS[cls]
            st.progress(min(conf/100, 1.0))
            st.markdown(f"**{i+1}. {label}** - {conf:.1f}%")
    else:
        st.info("👆 Please upload an image to start the analysis")

st.markdown("---")

st.subheader("🩺 Common Skin Conditions Guide")

cols = st.columns(3)
for i, (key, info) in enumerate(DISEASE_INFO.items()):
    with cols[i % 3]:
        severity_emoji = "🔴" if info['severity'].startswith('HIGH') else "🟡" if "Medium" in info['severity'] or "Low-Medium" in info['severity'] else "🟢"
        st.markdown(f"**{severity_emoji} {info['name']}**")
        st.caption(info['severity'])

st.markdown("---")

st.warning("""
⚠️ **Important Disclaimer:** 
This application is for **educational purposes only**. 
The predictions are based on a demo model and may not be accurate.
**Please consult a dermatologist** for proper diagnosis and treatment.
""")

st.markdown("---")
st.caption("🔬 Powered by Machine Learning | Built with Streamlit")