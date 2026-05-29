import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import sys
import os
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from modules.skin_type_analyzer import detect_skin_type, detect_skin_type_from_questionnaire
from modules.allergy_detector import detect_allergy, check_ingredient_allergy, ALLERGY_CATEGORIES
from modules.ingredient_intelligence import analyze_ingredient_list, get_ingredient_recommendations, get_harmful_warnings
from modules.product_recommender import recommend_products, PRODUCT_CATEGORIES, get_product_by_concern
from modules.treatment_planner import get_personalized_plan, SUN_PROTECTION_ADVICE, get_hydration_advice
from modules.image_analyzer import perform_full_analysis
from modules.report_generator import generate_report, format_report_text

st.set_page_config(page_title="AI Skin Disease Detection", page_icon="", layout="wide")

PROJECT_DIR = Path(__file__).parent
HISTORY_FILE = PROJECT_DIR / 'prediction_history.json'

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
CLASS_LABELS = {
    'akiec': 'Actinic Keratoses (Precancerous)',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'nv': 'Melanocytic Nevi (Moles)',
    'vasc': 'Vascular Lesions',
    'mel': 'Melanoma'
}
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

EXTENDED_CONDITIONS = {
    'acne': {
        'name': 'Acne Vulgaris',
        'description': 'Common skin condition causing pimples, blackheads, and whiteheads due to clogged pores.',
        'medicines': ['Benzoyl peroxide', 'Salicylic acid', 'Retinoids', 'Antibiotics', 'Isotretinoin (severe)'],
        'prevention': 'Clean face twice daily, avoid touching face, use non-comedogenic products',
        'severity': 'Varies - Mild to Severe'
    },
    'eczema': {
        'name': 'Eczema (Atopic Dermatitis)',
        'description': 'Chronic condition causing itchy, inflamed, and cracked skin.',
        'medicines': ['Moisturizers', 'Topical corticosteroids', 'Calcineurin inhibitors', 'Antihistamines', 'Phototherapy'],
        'prevention': 'Moisturize regularly, avoid triggers, use gentle skincare',
        'severity': 'Mild to Severe - chronic condition'
    },
    'psoriasis': {
        'name': 'Psoriasis',
        'description': 'Autoimmune condition causing red, scaly patches on skin.',
        'medicines': ['Topical corticosteroids', 'Vitamin D analogs', 'Phototherapy', 'Biologics', 'Systemic medications'],
        'prevention': 'Moisturize, manage stress, avoid triggers (alcohol, smoking)',
        'severity': 'Mild to Severe - chronic autoimmune'
    },
    'rosacea': {
        'name': 'Rosacea',
        'description': 'Chronic condition causing facial redness, visible blood vessels, and bumps.',
        'medicines': ['Metronidazole cream', 'Azelaic acid', 'Doxycycline', 'Ivermectin', 'Laser therapy'],
        'prevention': 'Avoid triggers (spicy food, alcohol, sun), gentle skincare',
        'severity': 'Mild to Moderate - chronic'
    },
    'vitiligo': {
        'name': 'Vitiligo',
        'description': 'Condition causing loss of skin pigment in patches.',
        'medicines': ['Topical corticosteroids', 'Calcineurin inhibitors', 'Phototherapy (UVB)', 'Excimer laser', 'Skin grafting'],
        'prevention': 'Sun protection on depigmented areas, manage stress',
        'severity': 'Progressive condition'
    },
    'melasma': {
        'name': 'Melasma',
        'description': 'Brown or gray-brown patches on face, often triggered by hormones and sun.',
        'medicines': ['Hydroquinone', 'Tretinoin', 'Azelaic acid', 'Chemical peels', 'Laser therapy'],
        'prevention': 'Strict sun protection, avoid hormonal triggers',
        'severity': 'Cosmetic - can be persistent'
    },
    'hyperpigmentation': {
        'name': 'Hyperpigmentation',
        'description': 'Dark patches or spots on skin from excess melanin production.',
        'medicines': ['Vitamin C serum', 'Kojic acid', 'Alpha arbutin', 'Niacinamide', 'Chemical peels'],
        'prevention': 'Daily sunscreen, avoid picking at skin',
        'severity': 'Mild to Moderate'
    },
    'dark_circles': {
        'name': 'Dark Circles',
        'description': 'Dark discoloration under the eyes from various causes.',
        'medicines': ['Vitamin C eye cream', 'Retinol eye cream', 'Caffeine eye cream', 'Hyaluronic acid', 'Laser therapy'],
        'prevention': 'Sleep 7-8 hours, manage allergies, sun protection',
        'severity': 'Mild - cosmetic concern'
    },
    'hair_loss': {
        'name': 'Hair Loss (Alopecia)',
        'description': 'Partial or complete loss of hair from scalp or body.',
        'medicines': ['Minoxidil (Rogaine)', 'Finasteride', 'Low-level laser therapy', 'Platelet-rich plasma (PRP)', 'Hair transplant'],
        'prevention': 'Balanced diet, reduce stress, gentle hair care',
        'severity': 'Varies - progressive'
    },
    'dandruff': {
        'name': 'Dandruff',
        'description': 'Flaky scalp condition often caused by yeast overgrowth or dry skin.',
        'medicines': ['Ketoconazole shampoo', 'Zinc pyrithione shampoo', 'Salicylic acid shampoo', 'Selenium sulfide shampoo', 'Tea tree oil'],
        'prevention': 'Regular washing, manage stress, reduce sugar intake',
        'severity': 'Mild - manageable'
    },
    'fungal_infection': {
        'name': 'Fungal Skin Infection',
        'description': 'Infection caused by fungi like ringworm, athlete\'s foot, or yeast.',
        'medicines': ['Clotrimazole cream', 'Terbinafine', 'Fluconazole', 'Ketoconazole', 'Miconazole'],
        'prevention': 'Keep skin clean and dry, avoid sharing towels, wear breathable fabrics',
        'severity': 'Mild to Moderate - contagious'
    },
    'sun_damage': {
        'name': 'Sun Damage (Photoaging)',
        'description': 'Skin damage from UV exposure causing premature aging, spots, and texture changes.',
        'medicines': ['Vitamin C serum', 'Retinoids', 'Sunscreen SPF 50', 'Chemical peels', 'Laser resurfacing'],
        'prevention': 'Daily sunscreen, protective clothing, avoid tanning beds',
        'severity': 'Cumulative - preventable'
    }
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model():
    model_paths = [
        PROJECT_DIR / 'skin_disease_model_pt.pt',
        PROJECT_DIR / 'best_model_pt.pt',
        PROJECT_DIR / 'best_model.keras',
        PROJECT_DIR / 'best_model_advanced.keras',
        PROJECT_DIR / 'skin_disease_model.keras'
    ]
    for path in model_paths:
        if path.exists():
            if path.suffix == '.pt':
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
                    ckpt = torch.load(str(path), map_location=device, weights_only=True)
                    model.load_state_dict(ckpt)
                    model.to(device)
                    model.eval()
                    return model
                except Exception as e:
                    st.warning(f"Could not load PyTorch model {path.name}: {e}")
            else:
                try:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(str(path), compile=False)
                    return model
                except:
                    continue
    st.warning("No pre-trained model found. Using demo mode with simulated predictions.")
    return 'demo'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_disease(model, img):
    if isinstance(model, str) and model == 'demo':
        import random
        predictions = []
        for cls in CLASS_NAMES:
            predictions.append({'class': cls, 'confidence': random.uniform(1, 50)})
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        for p in predictions:
            p['info'] = DISEASE_INFO[p['class']]
            p['label'] = CLASS_LABELS[p['class']]
        return predictions[:3]

    if isinstance(model, torch.nn.Module):
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
                'info': DISEASE_INFO[CLASS_NAMES[idx]],
                'label': CLASS_LABELS[CLASS_NAMES[idx]]
            })
        return results

    try:
        import tensorflow as tf
        img_array = np.array(img.convert('RGB').resize((224, 224)), dtype=np.float32) / 127.5 - 1.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array, verbose=0)[0]
        results = []
        for idx in np.argsort(predictions)[::-1][:3]:
            results.append({
                'class': CLASS_NAMES[idx],
                'confidence': float(predictions[idx]) * 100,
                'info': DISEASE_INFO[CLASS_NAMES[idx]],
                'label': CLASS_LABELS[CLASS_NAMES[idx]]
            })
        return results
    except:
        import random
        predictions = []
        for cls in CLASS_NAMES:
            predictions.append({'class': cls, 'confidence': random.uniform(1, 50)})
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        for p in predictions:
            p['info'] = DISEASE_INFO[p['class']]
            p['label'] = CLASS_LABELS[p['class']]
        return predictions[:3]

def load_history():
    from modules.database import load_history as db_load
    return db_load()

def save_history(data):
    from modules.database import save_prediction
    save_prediction(
        data.get('datetime', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        data.get('type', 'disease'),
        data.get('disease', ''),
        data.get('confidence', 0.0),
        data.get('severity', ''),
        details_dict={k: v for k, v in data.items() if k not in ['datetime', 'type', 'disease', 'confidence', 'severity']}
    )

model = load_model()

st.sidebar.title("AI Skin Disease Detection")
st.sidebar.markdown("Complete dermatology analysis system")

menu = st.sidebar.radio("Main Menu", [
    "Disease Detection",
    "Skin Type Analysis",
    "Extended Conditions",
    "Allergy Detection",
    "Ingredient Intelligence",
    "Product Recommendations",
    "Treatment Planner",
    "Dermatology Report",
    "Prediction History"
])

st.sidebar.markdown("---")

if isinstance(model, torch.nn.Module):
    st.sidebar.success("PyTorch Model Loaded")
elif isinstance(model, str):
    st.sidebar.info("Running in Demo Mode")
else:
    st.sidebar.success("TensorFlow Model Loaded")

hide_st_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

/* Global Font and Theme */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    font-family: 'Outfit', sans-serif !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Custom Premium Buttons */
.stButton>button, [data-testid="baseButton-secondary"] {
    background: linear-gradient(135deg, #38bdf8 0%, #3b82f6 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px 0 rgba(59, 130, 246, 0.3) !important;
}

.stButton>button:hover, [data-testid="baseButton-secondary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px 0 rgba(59, 130, 246, 0.4) !important;
}

/* Glassmorphism Expander styling */
.stExpander, [data-testid="stExpander"] {
    border-radius: 12px !important;
    border: 1px solid rgba(240, 246, 252, 0.1) !important;
    background-color: rgba(22, 27, 34, 0.4) !important;
    backdrop-filter: blur(10px) !important;
    margin-bottom: 12px !important;
}

/* Header Gradient */
h1, h2, h3, [data-testid="stHeader"] {
    font-weight: 700 !important;
}

div[data-testid="stMarkdownContainer"] > h1 {
    background: linear-gradient(135deg, #38bdf8 0%, #3b82f6 50%, #4f46e5 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-size: 2.8rem !important;
    padding-bottom: 10px !important;
}

/* Clean sidebar styling */
[data-testid="stSidebar"] {
    background-color: rgba(13, 17, 23, 0.95) !important;
    border-right: 1px solid rgba(240, 246, 252, 0.1) !important;
}

/* Visual highlights */
[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-weight: 700 !important;
}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

if menu == "Disease Detection":
    st.title("Disease Detection")
    st.markdown("Upload a skin image to detect potential diseases using AI")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        source = st.radio("Image Source", ["Upload Image", "Take Photo"])
        if source == "Upload Image":
            uploaded_file = st.file_uploader("Choose skin image...", type=["jpg", "jpeg", "png"])
        else:
            uploaded_file = st.camera_input("Take a photo")
            
        if uploaded_file:
            file_name = uploaded_file.name if hasattr(uploaded_file, 'name') else 'camera'
            if st.session_state.get('last_uploaded_file') != file_name:
                st.session_state['last_uploaded_file'] = file_name
                if 'analysis_results' in st.session_state:
                    del st.session_state['analysis_results']
                    
    with col2:
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=400)
        else:
            st.info("Upload or capture a skin image to begin analysis")

    if uploaded_file and st.button("Analyze Image"):
        image = Image.open(uploaded_file)
        with st.spinner("Running AI analysis..."):
            predictions = predict_disease(model, image)
            
            # Generate Grad-CAM if PyTorch
            heatmap_img = None
            if isinstance(model, torch.nn.Module):
                from modules.grad_cam import generate_gradcam_heatmap
                img_tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
                heatmap_img, _ = generate_gradcam_heatmap(model, image, img_tensor)
                
            st.session_state['analysis_results'] = {
                'predictions': predictions,
                'heatmap': heatmap_img,
                'image': image
            }

    if 'analysis_results' in st.session_state:
        results = st.session_state['analysis_results']
        predictions = results['predictions']
        heatmap_img = results['heatmap']
        image = results['image']
        top = predictions[0]
        risk_color = "error" if top['class'] == 'mel' else ("warning" if top['class'] in ['akiec', 'bcc'] else "success")

        st.markdown("### Results")
        if risk_color == "error":
            st.error(f"HIGH PRIORITY: {top['info']['name']}")
        elif risk_color == "warning":
            st.warning(f"Attention Required: {top['info']['name']}")
        else:
            st.success(f"Detected: {top['info']['name']}")

        # Render original vs heatmap side by side
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col_img2:
            if heatmap_img is not None:
                st.image(heatmap_img, caption="AI Interpretability (Grad-CAM Heatmap)", use_container_width=True)
            else:
                st.info("AI Grad-CAM interpretation requires an active deep learning model.")

        cols = st.columns(3)
        with cols[0]:
            st.metric("Confidence", f"{top['confidence']:.1f}%")
        with cols[1]:
            st.metric("Severity", top['info']['severity'].split('-')[0].strip())
        with cols[2]:
            st.metric("Category", "Skin Disease")

        with st.expander("Full Details", expanded=True):
            st.write(f"**Description:** {top['info']['description']}")
            st.write(f"**Severity:** {top['info']['severity']}")
            st.markdown("**Medicines:**")
            for med in top['info']['medicines']:
                st.markdown(f"- {med}")
            st.write(f"**Prevention:** {top['info']['prevention']}")

        st.markdown("### Top 3 Possibilities")
        for i, pred in enumerate(predictions):
            st.progress(pred['confidence'] / 100, text=f"{i+1}. {pred['label']} - {pred['confidence']:.1f}%")

        if st.button("Save to History"):
            save_history({
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'disease',
                'disease': top['info']['name'],
                'confidence': top['confidence'],
                'severity': top['info']['severity']
            })
            st.success("Saved to history!")

elif menu == "Skin Type Analysis":
    st.title("Skin Type Analysis")
    st.markdown("Detect your skin type using image analysis or questionnaire")

    tab1, tab2 = st.tabs(["Image Analysis", "Questionnaire"])

    with tab1:
        st.markdown("Upload a photo of your face for AI skin type analysis")
        img_source = st.radio("Source", ["Upload Image", "Take Photo"], key="st_source")
        if img_source == "Upload Image":
            skin_img = st.file_uploader("Choose face image...", type=["jpg", "jpeg", "png"], key="st_upload")
        else:
            skin_img = st.camera_input("Take a photo", key="st_cam")

        if skin_img:
            image = Image.open(skin_img)
            st.image(image, caption="Image for Analysis", width=300)

            if st.button("Analyze Skin Type"):
                with st.spinner("Analyzing skin characteristics..."):
                    result = detect_skin_type(image)

                st.markdown(f"### Your Skin Type: **{result['skin_type']}**")
                st.progress(result['confidence'] / 100, text=f"Confidence: {result['confidence']:.1f}%")

                scores = result['analysis']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Oil Level", scores['oil_level'].title())
                    st.metric("Shine", scores['shine'].title())
                    st.metric("Pores", scores['pore_visibility'].title())
                with col2:
                    st.metric("Dryness", scores['dryness'].title())
                    st.metric("Flakiness", scores['flakiness'].title())
                    st.metric("Texture", scores['texture'].title())
                with col3:
                    st.metric("Redness", scores['redness'].title())

                st.markdown("**Composition Scores:**")
                st.markdown(f"- Oily Score: {result['oily_score']}/6")
                st.markdown(f"- Dry Score: {result['dry_score']}/6")
                st.markdown(f"- Sensitive Score: {result['sensitive_score']}/6")

    with tab2:
        st.markdown("Answer a few questions to determine your skin type")

        with st.form("skin_type_quiz"):
            q_oil = st.selectbox("How does your skin feel after washing?", ["Normal", "Oily", "Very Oily", "Dry", "Very Dry"])
            q_sensitivity = st.selectbox("How does your skin react to products?", ["Normal", "Sometimes irritated", "Sensitive", "Very Sensitive"])
            q_dryness = st.selectbox("Do you experience dry or flaky patches?", ["No", "Sometimes", "Yes, Very Dry"])
            q_pores = st.selectbox("How visible are your pores?", ["Small/Minimal", "Somewhat Visible", "Large/Very Visible"])
            q_shine = st.selectbox("Does your skin get shiny during the day?", ["No", "A Little", "Yes, Very Shiny"])

            if st.form_submit_button("Determine Skin Type"):
                stype = detect_skin_type_from_questionnaire(q_oil, q_sensitivity, q_dryness, q_pores, q_shine)
                st.success(f"Your Skin Type: **{stype}**")
                st.info(get_hydration_advice(stype))

elif menu == "Extended Conditions":
    st.title("Extended Skin Conditions")
    st.markdown("Information on common skin conditions beyond HAM10000")

    condition = st.selectbox("Select a condition", list(EXTENDED_CONDITIONS.keys()),
        format_func=lambda x: EXTENDED_CONDITIONS[x]['name'])

    if condition:
        info = EXTENDED_CONDITIONS[condition]
        st.subheader(info['name'])
        st.write(f"**Description:** {info['description']}")
        st.write(f"**Severity:** {info['severity']}")
        st.markdown("**Medicines/Treatments:**")
        for med in info['medicines']:
            st.markdown(f"- {med}")
        st.write(f"**Prevention:** {info['prevention']}")

        recs = get_product_by_concern(condition.replace('_', ' '))
        if recs:
            st.markdown("**Recommended Products:**")
            for r in recs:
                st.markdown(f"- {r}")

elif menu == "Allergy Detection":
    st.title("Allergy Detection")
    st.markdown("Identify potential allergies from symptoms, triggers, and ingredients")

    tab1, tab2 = st.tabs(["Symptom Checker", "Allergy Categories"])

    with tab1:
        st.markdown("Select your symptoms to identify potential allergies")

        all_symptoms = list(set(
            sym for cat in ALLERGY_CATEGORIES.values()
            for sym in cat['symptoms']
        ))
        selected_symptoms = st.multiselect("Your Symptoms", sorted(all_symptoms))

        st.markdown("Any known triggers?")
        trigger_input = st.text_input("Enter triggers (comma separated, e.g., soap, pollen, dairy)", placeholder="e.g., new face cream, dust, strawberries")

        if st.button("Check Allergies"):
            if selected_symptoms or trigger_input:
                triggers = [t.strip() for t in trigger_input.split(',') if t.strip()] if trigger_input else None
                result = detect_allergy(selected_symptoms, triggers)

                if result:
                    for r in result:
                        sev_color = "error" if r['severity'] == 'Severe' else ("warning" if r['severity'] == 'Moderate' else "info")
                        with st.expander(f"{r['name']} - {r['severity']}", expanded=True):
                            st.markdown(f"Severity: :{sev_color}[**{r['severity']}**] (Score: {r['severity_score']})")
                            st.markdown(f"**Matched Symptoms:** {', '.join(r['matched_symptoms'])}")
                            st.markdown("**Common Triggers:**")
                            for t in r['common_triggers'][:5]:
                                st.markdown(f"- {t}")
                            st.markdown("**Precautions:**")
                            for p in r['precautions'][:4]:
                                st.markdown(f"- {p}")
                            st.markdown("**Ingredients to Avoid:**")
                            for ing in r['ingredients_to_avoid'][:4]:
                                st.markdown(f"- {ing}")
                else:
                    st.info("No specific allergies matched your symptoms. Consult a dermatologist for personalized assessment.")

    with tab2:
        st.markdown("Browse allergy categories and their details")
        category = st.selectbox("Select Category", list(ALLERGY_CATEGORIES.keys()),
            format_func=lambda x: ALLERGY_CATEGORIES[x]['name'])
        if category:
            data = ALLERGY_CATEGORIES[category]
            st.subheader(data['name'])
            st.markdown("**Symptoms:**")
            for s in data['symptoms']:
                st.markdown(f"- {s}")
            st.markdown("**Common Triggers:**")
            for t in data['common_triggers']:
                st.markdown(f"- {t}")
            st.markdown("**Precautions:**")
            for p in data['precautions']:
                st.markdown(f"- {p}")
            st.markdown("**Ingredients to Avoid:**")
            for ing in data['ingredients_to_avoid']:
                st.markdown(f"- {ing}")

elif menu == "Ingredient Intelligence":
    st.title("Ingredient Intelligence")
    st.markdown("Analyze skincare ingredients and get recommendations")

    tab1, tab2, tab3 = st.tabs(["Analyze Product", "Useful Ingredients", "Harmful Ingredients"])

    with tab1:
        st.markdown("Paste a product's ingredient list to analyze it")
        ingredient_text = st.text_area("Ingredient List (paste comma or newline separated)", height=200,
            placeholder="e.g., Water, Niacinamide, Hyaluronic Acid, Alcohol Denat, Fragrance...")

        if st.button("Analyze Ingredients"):
            if ingredient_text:
                result = analyze_ingredient_list(ingredient_text)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Ingredients", result['total_analyzed'])
                with col2:
                    st.metric("Useful Found", result['useful_count'])
                with col3:
                    st.metric("Harmful Found", result['harmful_count'])

                if result['has_harmful']:
                    st.error("This product contains potentially harmful ingredients!")
                    st.markdown("### Harmful Ingredients")
                    for h in result['harmful_ingredients']:
                        st.warning(f"**{h['matched_as']}** - {h['concern']} (Severity: {h['severity']})")

                if result['useful_ingredients']:
                    st.markdown("### Useful Ingredients")
                    for u in result['useful_ingredients']:
                        with st.expander(f"{u['matched_as']}"):
                            st.markdown(f"**Benefits:** {', '.join(u['benefits'])}")
                            st.markdown(f"**Good For:** {', '.join(u['good_for'])}")
                            st.markdown(f"**Usage:** {u['usage']}")

    with tab2:
        st.markdown("Browse beneficial skincare ingredients")
        ingredient = st.selectbox("Select Ingredient",
            ['Niacinamide', 'Hyaluronic Acid', 'Salicylic Acid', 'Vitamin C', 'Ceramides',
             'Retinol', 'Benzoyl Peroxide', 'Alpha Arbutin', 'Peptides', 'Azelaic Acid',
             'Centella Asiatica', 'Squalane'])
        if ingredient:
            info = get_ingredient_recommendations('Normal')
            for rec in info:
                if rec['ingredient'] == ingredient:
                    st.subheader(ingredient)
                    st.markdown(f"**Benefits:** {', '.join(rec['benefits'])}")
                    st.markdown(f"**Usage:** {rec['usage']}")
                    st.markdown(f"**Concentration:** {rec['concentration']}")

    with tab3:
        warnings_list = get_harmful_warnings('Sensitive')
        for warning in warnings_list:
            with st.expander(f"{warning['ingredient']} - {warning['severity']}"):
                st.write(f"**Concern:** {warning['concern']}")
                st.write(f"**Severity:** {warning['severity']}")

elif menu == "Product Recommendations":
    st.title("Product Recommendations")
    st.markdown("Get personalized product recommendations based on your skin profile")

    with st.form("product_form"):
        col1, col2 = st.columns(2)
        with col1:
            p_skin_type = st.selectbox("Skin Type", ["Normal", "Oily", "Dry", "Combination", "Sensitive"])
            p_climate = st.selectbox("Climate/Environment", ["Normal", "Humid", "Dry/Arid", "Cold", "Hot", "Polluted"])
            p_age = st.number_input("Age", min_value=10, max_value=100, value=25)
        with col2:
            p_allergies = st.multiselect("Allergies", ["Fragrance", "Alcohol", "Parabens", "Essential Oils", "Sulfates"])
            p_concerns = st.multiselect("Concerns", ["Acne", "Wrinkles", "Dark spots", "Dryness", "Redness", "Large pores", "Dullness", "Blackheads", "Dark circles", "Hair loss", "Dandruff"])
            p_acne = st.selectbox("Acne Severity", ["None", "Mild", "Moderate", "Severe"])
            p_pigment = st.selectbox("Pigmentation Level", ["None", "Mild", "Moderate", "Severe"])

        submitted = st.form_submit_button("Get Recommendations")

    if submitted:
        with st.spinner("Generating personalized recommendations..."):
            recs = recommend_products(
                p_skin_type, p_climate, p_age,
                allergies=p_allergies if p_allergies else None,
                concerns=p_concerns if p_concerns else None,
                acne_severity=p_acne if p_acne != "None" else None,
                pigmentation_level=p_pigment if p_pigment != "None" else None
            )

        st.success(f"Recommendations for {p_skin_type} skin in {p_climate} climate")

        for cat_key, cat_data in recs.items():
            if cat_key == 'climate_advice':
                st.info(f"Climate Advice: {cat_data}")
            else:
                with st.expander(f"{cat_data['category']}", expanded=True):
                    for prod in cat_data['recommended_products']:
                        st.markdown(f"- {prod}")

elif menu == "Treatment Planner":
    st.title("Personalized Treatment Planner")
    st.markdown("Get a complete skincare routine and treatment plan")

    with st.form("treatment_form"):
        col1, col2 = st.columns(2)
        with col1:
            t_skin_type = st.selectbox("Your Skin Type", ["Normal", "Oily", "Dry", "Combination", "Sensitive"])
            t_age = st.number_input("Age", min_value=10, max_value=100, value=25, key="t_age")
        with col2:
            t_concerns = st.multiselect("Primary Concerns", ["Acne", "Aging/Wrinkles", "Dark spots/Pigmentation", "Dryness", "Redness", "Dullness", "Large pores", "Blackheads"])

        t_include_diet = st.checkbox("Include diet suggestions", True)
        t_include_lifestyle = st.checkbox("Include lifestyle tips", True)

        submitted = st.form_submit_button("Generate Treatment Plan")

    if submitted:
        with st.spinner("Creating your personalized treatment plan..."):
            plan = get_personalized_plan(t_skin_type, t_age, t_concerns)

        st.success(f"Treatment Plan for {t_skin_type} Skin")

        st.markdown("### Morning Routine")
        for step in plan['morning_routine']:
            st.info(f"**{step[0]}: {step[1]}** - {step[2]}")

        st.markdown("### Night Routine")
        for step in plan['night_routine']:
            st.info(f"**{step[0]}: {step[1]}** - {step[2]}")

        if t_include_diet and plan['diet_suggestions']:
            st.markdown("### Diet Suggestions")
            for d in plan['diet_suggestions']:
                st.markdown(f"- {d}")

        if t_include_lifestyle and plan['lifestyle_tips']:
            st.markdown("### Lifestyle Tips")
            for tip in plan['lifestyle_tips'][:8]:
                st.markdown(f"- {tip}")

        st.markdown("### Sun Protection Advice")
        spf_advice = SUN_PROTECTION_ADVICE.get(t_skin_type, SUN_PROTECTION_ADVICE['general'])
        for a in spf_advice:
            st.markdown(f"- {a}")

        st.markdown("### Hydration Advice")
        st.info(get_hydration_advice(t_skin_type))

elif menu == "Dermatology Report":
    st.title("Comprehensive Dermatology Report")
    st.markdown("Generate a complete dermatology analysis report")

    col1, col2 = st.columns(2)
    with col1:
        report_img = st.file_uploader("Upload skin image for analysis", type=["jpg", "jpeg", "png"])
    with col2:
        r_skin_type = st.selectbox("Skin Type (optional)", ["Auto-Detect", "Normal", "Oily", "Dry", "Combination", "Sensitive"])
        r_allergy_symptoms = st.multiselect("Any allergy symptoms?", ["Redness", "Itching", "Swelling", "Rash", "Burning", "Dryness", "Hives"])

    if report_img:
        image = Image.open(report_img)
        st.image(image, caption="Image for Report", width=300)

        if st.button("Generate Full Report"):
            with st.spinner("Running comprehensive analysis..."):
                disease_results = predict_disease(model, image)

                if r_skin_type == "Auto-Detect":
                    skin_type_result = detect_skin_type(image)
                elif r_skin_type:
                    skin_type_result = {
                        'skin_type': r_skin_type,
                        'confidence': 100,
                        'analysis': {},
                        'oily_score': 0,
                        'dry_score': 0,
                        'sensitive_score': 0
                    }
                else:
                    skin_type_result = None

                skin_analysis = perform_full_analysis(image)

                allergy_risks = detect_allergy(r_allergy_symptoms) if r_allergy_symptoms else []

                stype = skin_type_result['skin_type'] if skin_type_result else 'Normal'
                product_recs = recommend_products(stype)
                treatment_plan = get_personalized_plan(stype)

                report = generate_report(disease_results, skin_analysis, skin_type_result, allergy_risks, product_recs, treatment_plan)

                report_text = format_report_text(report)

            st.success("Report Generated Successfully!")

            st.download_button(
                "Download Report (TXT)",
                report_text,
                file_name=f"dermatology_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

            for section in report['sections']:
                if section['type'] == 'disease':
                    c = section['content']
                    risk = "error" if 'HIGH' in c.get('severity', '') else ("warning" if 'Medium' in c.get('severity', '') else "success")
                    if risk == "error":
                        st.error(f"Primary Diagnosis: {c['primary_diagnosis']} ({c['confidence']})")
                    elif risk == "warning":
                        st.warning(f"Primary Diagnosis: {c['primary_diagnosis']} ({c['confidence']})")
                    else:
                        st.success(f"Primary Diagnosis: {c['primary_diagnosis']} ({c['confidence']})")

                elif section['type'] == 'analysis':
                    with st.expander("Skin Image Analysis", expanded=True):
                        c = section['content']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Texture", c.get('skin_texture', {}).get('value', ''))
                            st.metric("Wrinkles", f"{c.get('wrinkles', {}).get('severity', '')} ({c.get('wrinkles', {}).get('count', 0)})")
                            st.metric("Acne", f"{c.get('acne', {}).get('severity', '')} ({c.get('acne', {}).get('count', 0)} spots)")
                        with col2:
                            st.metric("Pigmentation", f"{c.get('pigmentation', {}).get('severity', '')}")
                            st.metric("Pores", c.get('pores', {}).get('severity', ''))
                            st.metric("Redness", c.get('redness', {}).get('severity', ''))
                        with col3:
                            st.metric("Scars", c.get('scars', {}).get('severity', ''))
                            st.metric("Lesions", c.get('lesions', {}).get('count', 0))
                            st.metric("Fungal Pattern", c.get('fungal_patterns', {}).get('assessment', '').split('-')[0].strip())

                elif section['type'] == 'skin_type':
                    with st.expander("Skin Type Assessment", expanded=True):
                        c = section['content']
                        st.info(f"Detected: **{c['skin_type']}** (Confidence: {c['confidence']})")

                elif section['type'] == 'allergy':
                    with st.expander("Allergy Risk Assessment", expanded=True):
                        for item in section['content']:
                            st.warning(f"{item['category']} - Severity: {item['severity']}")

                elif section['type'] == 'treatment':
                    with st.expander("Treatment Plan", expanded=True):
                        st.markdown("**Morning:**")
                        for s in section['content']['morning_routine'][:3]:
                            st.markdown(f"- {s['action']}: {s['detail']}")
                        st.markdown("**Night:**")
                        for s in section['content']['night_routine'][:3]:
                            st.markdown(f"- {s['action']}: {s['detail']}")

elif menu == "Prediction History":
    st.title("Prediction History")
    history = load_history()
    if history:
        for i, item in enumerate(reversed(history)):
            with st.expander(f"{item.get('datetime', 'Unknown')} - {item.get('disease', 'Unknown')}"):
                st.write(f"**Disease/Condition:** {item.get('disease', 'N/A')}")
                st.write(f"**Confidence:** {item.get('confidence', 0):.1f}%")
                st.write(f"**Severity:** {item.get('severity', 'N/A')}")
    else:
        st.info("No prediction history yet. Run an analysis to build your history.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Info
Reports and predictions are for educational purposes only.
Always consult a dermatologist for medical advice.
""")
