import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

from modules.grad_cam import (
    generate_gradcam_heatmap, find_target_layer, find_multi_target_layers,
    build_gradcam_views, colormap_legend_html, enhanced_overlay,
    upscale_heatmap_highres
)
from preprocessing import remove_hair, color_normalization

st.set_page_config(page_title="DermaVision — AI Skin Lesion Analysis", layout="wide", page_icon="🔬")

NUM_CLASSES = 7


def build_plain_model():
    model = models.efficientnet_b0(weights=None)
    inf = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(inf, 512), nn.ReLU(inplace=True),
        nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU(inplace=True),
        nn.Dropout(0.2), nn.Linear(256, NUM_CLASSES)
    )
    return model

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', system-ui, sans-serif; }
    .stApp { background: #0a0a0f; }
    .main > div { padding: 0 2rem; }
    .block-container { max-width: 1400px; padding-top: 1.5rem !important; }
    
    .header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border-bottom: 1px solid #30363d;
        padding: 1.2rem 2rem; margin: -1.5rem -2rem 1.5rem -2rem;
        display: flex; align-items: center; gap: 1rem;
    }
    .header h1 { font-size: 1.6rem; font-weight: 800; color: #f0f6fc; margin: 0;
        letter-spacing: -0.5px; background: linear-gradient(90deg, #58a6ff, #3fb950);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .header .tagline { color: #8b949e; font-size: 0.85rem; margin-left: auto; }
    
    .card {
        background: linear-gradient(145deg, #161b22, #0d1117);
        border: 1px solid #30363d; border-radius: 16px; padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3); transition: border-color .2s;
    }
    .card:hover { border-color: #58a6ff40; }
    
    .severity-critical {
        background: linear-gradient(135deg, #3b0a0a, #1a0404);
        border: 1px solid #ef4444; border-radius: 16px; padding: 1.5rem;
        box-shadow: 0 0 30px #ef444420;
    }
    .severity-warning {
        background: linear-gradient(135deg, #3b2d0a, #1a1504);
        border: 1px solid #f59e0b; border-radius: 16px; padding: 1.5rem;
    }
    .severity-low {
        background: linear-gradient(135deg, #0a3b1a, #041a0e);
        border: 1px solid #10b981; border-radius: 16px; padding: 1.5rem;
    }
    
    .badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        padding: 0.25rem 1rem; border-radius: 999px;
        font-size: 0.85rem; font-weight: 600;
    }
    .badge-confidence { background: rgba(88,166,255,0.15); color: #58a6ff; }
    .badge-severity-high { background: rgba(239,68,68,0.15); color: #ef4444; }
    .badge-severity-med { background: rgba(245,158,11,0.15); color: #f59e0b; }
    .badge-severity-low { background: rgba(16,185,129,0.15); color: #10b981; }
    
    .metric-tile {
        text-align: center; padding: 0.6rem; border-radius: 10px; margin: 2px;
        transition: transform .15s, box-shadow .15s;
    }
    .metric-tile:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
    .metric-tile .label { font-size: 0.65rem; color: #8b949e; line-height: 1.2; }
    .metric-tile .value { font-size: 1.15rem; font-weight: 700; line-height: 1.4; }
    
    .rank-card {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border-radius: 12px; padding: 0.9rem 1.2rem; margin-bottom: 0.5rem;
        border-left: 4px solid; transition: transform .15s;
    }
    .rank-card:hover { transform: translateX(4px); }
    .rank-card .rank-num { font-size: 0.75rem; color: #8b949e; margin-bottom: 0.1rem; }
    .rank-card .rank-name { font-size: 1rem; font-weight: 700; }
    .rank-card .rank-meta { display: flex; gap: 1rem; margin-top: 0.2rem; font-size: 0.8rem; color: #8b949e; }
    .rank-card .rank-metrics { display: flex; gap: 0.8rem; margin-top: 0.2rem; font-size: 0.7rem; color: #484f58; }
    
    .conf-gauge {
        height: 4px; border-radius: 2px; margin: 0.5rem 0; background: #21262d;
        overflow: hidden;
    }
    .conf-gauge-fill { height: 100%; border-radius: 2px; transition: width .5s ease; }
    
    .footer { text-align: center; color: #484f58; font-size: 0.75rem; padding: 2rem 0; }
    
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 800 !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #8b949e !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: #0d1117; padding: 0.3rem; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 0.4rem 1rem; font-size: 0.8rem; }
    .stTabs [aria-selected="true"] { background: #21262d !important; }
    .stFileUploader > div { background: #0d1117 !important; border: 1px dashed #30363d !important;
        border-radius: 12px !important; padding: 2rem !important; }
    .stFileUploader:hover > div { border-color: #58a6ff !important; }
    .stCheckbox label { font-size: 0.85rem !important; }
    .stSlider label { font-size: 0.85rem !important; }
    
    .section-title { font-size: 1.1rem; font-weight: 700; color: #f0f6fc; margin: 1.5rem 0 1rem 0;
        display: flex; align-items: center; gap: 0.5rem; }
    .section-title::before { content: ''; display: inline-block; width: 3px; height: 1.1rem;
        background: #58a6ff; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = [
    'akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel'
]
CLASS_LABELS = {
    'akiec': 'Actinic Keratoses', 'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis', 'df': 'Dermatofibroma',
    'nv': 'Melanocytic Nevi', 'vasc': 'Vascular Lesions', 'mel': 'Melanoma'
}

try:
    _EVAL_PATH = (__import__('pathlib').Path(__file__).resolve().parent / 'f2_scores.pkl')
    with open(_EVAL_PATH, 'rb') as f:
        EVAL_METRICS = __import__('pickle').load(f)
except:
    EVAL_METRICS = {'f2_macro': 0.0, 'f2_per_class': {}, 'iou_macro': 0.0, 'iou_per_class': {}, 'dice_macro': 0.0, 'dice_per_class': {}}

DISEASE_INFO = {
    'akiec': {
        'name': 'Actinic Keratoses',
        'description': 'Precancerous skin growths that appear as rough, scaly patches on sun-exposed areas.',
        'medicines': ['5-Fluorouracil cream', 'Imiquimod cream', 'Diclofenac gel', 'Cryotherapy', 'Photodynamic therapy'],
        'prevention': 'Use sunscreen daily (SPF 30+), avoid excessive sun exposure, wear protective clothing',
        'severity': 'Medium - can progress to skin cancer if untreated'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'description': 'Most common type of skin cancer. Rarely spreads but can damage surrounding tissue.',
        'medicines': ['Surgery', 'Mohs micrographic surgery', 'Radiation therapy', 'Topical medications'],
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
        'medicines': ['No treatment needed', 'Surgical removal if concerned', 'Laser removal for cosmetic'],
        'prevention': 'Monitor for ABCDE changes (Asymmetry, Border, Color, Diameter, Evolving)',
        'severity': 'Very Low - usually harmless but monitor for changes'
    },
    'vasc': {
        'name': 'Vascular Lesions',
        'description': 'Birthmarks or hemangiomas from abnormal blood vessels.',
        'medicines': ['Laser therapy', 'Corticosteroids', 'Beta-blockers for large hemangiomas'],
        'prevention': 'None specific',
        'severity': 'Very Low - usually cosmetic concern only'
    },
    'mel': {
        'name': 'Melanoma',
        'description': 'Most dangerous form of skin cancer. Can spread to other organs if not caught early.',
        'medicines': ['Surgery (wide excision)', 'Immunotherapy', 'Targeted therapy', 'Chemotherapy', 'Radiation therapy'],
        'prevention': 'Sun protection (SPF 30+), avoid tanning beds, regular skin checks, monitor moles',
        'severity': 'HIGH - deadliest form of skin cancer if untreated'
    }
}

EVAL_DIR = str(__import__('pathlib').Path(__file__).resolve().parent / 'evaluation')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

FRAMEWORK_IMG_SIZE = 224


def make_input_tensor(pil_img, size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(pil_img)  # noqa: B950


@st.cache_resource
def load_model(choice='best_model_pt_f2.pt'):
    candidates = {
        'Plain F2 (best_model_pt_f2.pt)': 'best_model_pt_f2.pt',
        'Plain EMA (best_model_pt.pt)': 'best_model_pt.pt',
    }
    requested = candidates.get(choice, choice)
    order = [requested] if __import__('pathlib').Path(requested).exists() else []
    order += [c for c in ['best_model_pt_f2.pt', 'best_model_pt.pt'] if c not in order]

    for ckpt_path in order:
        if __import__('pathlib').Path(ckpt_path).exists():
            try:
                model = build_plain_model()
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                model.load_state_dict(ckpt)
                model.to(device)
                model.eval()
                return model, find_target_layer(model)
            except Exception:
                continue

    st.error("No trained model found. Run train_f2_plain_fast.py first.")
    return None, None


def enhance_heatmap(cam, img_w, img_h, blur_ksize=7):
    heatmap = cv2.resize(cam, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.GaussianBlur(heatmap, (blur_ksize, blur_ksize), 0)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return heatmap_color, heatmap


def predict_disease(model, img, target_layer, cam_method='gradcam_pp', smooth_iters=12):
    # Model is trained on CLEAN images (hair removal + color normalization) @224,
    # so the cleaning pipeline is always applied before inference.

    # 'Cleaned' version is always computed for the display tab too.
    cleaned = img
    try:
        cleaned = remove_hair(img)
        cleaned = color_normalization(cleaned)
    except Exception:
        pass

    model_input = cleaned
    input_size = FRAMEWORK_IMG_SIZE  # 224

    img_tensor = make_input_tensor(model_input, input_size).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()

    views = build_gradcam_views(
        model, cleaned, img_tensor, target_layer,
        alpha=0.5, method=cam_method, cmap='severity',
        roi_threshold=0.5,
        target_layers=find_multi_target_layers(model),
        multi_layer=True,
        smooth_iters=smooth_iters, noise_sigma=0.10
    )

    top_3_idx = np.argsort(probabilities)[::-1][:3]
    results = []
    for idx in top_3_idx:
        results.append({
            'class': CLASS_NAMES[idx],
            'confidence': float(probabilities[idx]) * 100,
            'info': DISEASE_INFO[CLASS_NAMES[idx]]
        })

    return results, views, cleaned, probabilities


st.markdown("""
<div class="header">
    <div style="font-size:1.8rem;font-weight:800;background:linear-gradient(90deg,#58a6ff,#3fb950);-webkit-background-clip:text;-webkit-text-fill-color:transparent">DermaVision</div>
    <span style="color:#8b949e;font-size:0.85rem;font-weight:500">AI-Powered Skin Lesion Analysis</span>
    <span class="tagline">🔬 Grad-CAM · F2-Score · Dropout</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://placehold.co/60x60/0d1117/58a6ff?text=DV&font=raleway", width=60)
    st.markdown("<div style='font-size:0.7rem;color:#484f58;margin-top:-0.5rem'>v2.0 · EfficientNet-B0</div>", unsafe_allow_html=True)
    st.divider()
    uploaded_file = st.file_uploader("Upload skin image", type=["jpg", "jpeg", "png"],
                                     help="Dermoscopic or clinical photo of the lesion")
    st.divider()
    model_choice = st.selectbox(
        "Model",
        ["Plain F2 (best_model_pt_f2.pt)",
         "Plain EMA (best_model_pt.pt)"],
        index=0, help="Trained @224 on cleaned images (hair removal + color normalization)."
    )
    st.divider()
    cam_method = st.selectbox(
        "Grad-CAM method",
        ["SmoothGrad-CAM++ (default)", "Grad-CAM++", "Eigen-CAM", "Score-CAM", "Grad-CAM"],
        index=0,
        help="SmoothGrad-CAM++ is the most stable. Eigen-CAM is gradient-free and very stable. Score-CAM is slow."
    )
    st.divider()
    conf_threshold = st.slider("Confidence threshold", 0, 100, 30, 5,
                               help="Predictions below this are flagged as low confidence")
    st.divider()
    st.markdown("<div style='font-size:0.75rem;color:#484f58'><b>Key Metrics</b></div>", unsafe_allow_html=True)
    if EVAL_METRICS['f2_macro'] > 0:
        st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:0.8rem;padding:0.2rem 0'><span style='color:#8b949e'>F2</span><span style='color:#58a6ff;font-weight:700'>{EVAL_METRICS['f2_macro']:.4f}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:0.8rem;padding:0.2rem 0'><span style='color:#8b949e'>IoU</span><span style='color:#3fb950;font-weight:700'>{EVAL_METRICS.get('iou_macro',0):.4f}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:0.8rem;padding:0.2rem 0'><span style='color:#8b949e'>Dice</span><span style='color:#d2a8ff;font-weight:700'>{EVAL_METRICS.get('dice_macro',0):.4f}</span></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='font-size:0.7rem;color:#484f58'>⚠ Educational purposes only.<br>Consult a dermatologist.</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    model, target_layer = load_model(model_choice)

    if model is not None:
        _CAM_MAP = {
            "SmoothGrad-CAM++ (default)": "gradcam_pp",
            "Grad-CAM++": "gradcam_pp",
            "Eigen-CAM": "eigencam",
            "Score-CAM": "scorecam",
            "Grad-CAM": "gradcam",
        }
        _smooth = cam_method == "SmoothGrad-CAM++ (default)"
        with st.spinner("Analyzing with Grad-CAM ..."):
            predictions, grad_cam_views, processed_image, probs = predict_disease(
                model, image, target_layer,
                cam_method=_CAM_MAP[cam_method],
                smooth_iters=12 if _smooth else 0
            )

        top_prediction = predictions[0]
        is_low_conf = top_prediction['confidence'] < conf_threshold

        info = top_prediction['info']
        sev_class = "severity-critical" if top_prediction['class'] == 'mel' else ("severity-warning" if top_prediction['class'] in ['akiec','bcc'] else "severity-low")
        sev_color = "#ef4444" if top_prediction['class'] == 'mel' else ("#f59e0b" if top_prediction['class'] in ['akiec','bcc'] else "#10b981")
        sev_tag = "HIGH" if top_prediction['class'] == 'mel' else ("MEDIUM" if top_prediction['class'] in ['akiec','bcc'] else "LOW")
        sev_badge_class = "badge-severity-high" if top_prediction['class'] == 'mel' else ("badge-severity-med" if top_prediction['class'] in ['akiec','bcc'] else "badge-severity-low")

        st.markdown(f"<div class='{sev_class}' style='text-align:center;margin-bottom:1.5rem'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:2rem;font-weight:800;color:{sev_color}'>{info['name']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='display:flex;justify-content:center;gap:0.8rem;margin-top:0.5rem;flex-wrap:wrap'>"
                    f"<span class='badge badge-confidence'>{top_prediction['confidence']:.1f}% confidence</span>"
                    f"<span class='badge {sev_badge_class}'>{sev_tag}</span>"
                    f"</div>", unsafe_allow_html=True)
        conf_pct = top_prediction['confidence'] / 100
        st.markdown(f"<div style='max-width:400px;margin:0.5rem auto'><div class='conf-gauge'><div class='conf-gauge-fill' style='width:{conf_pct*100}%;background:linear-gradient(90deg,{sev_color}80,{sev_color})'></div></div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        subtitle = "Top-3 Predictions · Grad-CAM"
        st.markdown("<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.8rem'>"
                    "<span style='font-size:1.1rem;font-weight:700;color:#f0f6fc'>Visual Analysis</span>"
                    f"<span style='font-size:0.7rem;color:#484f58'>{subtitle}</span>"
                    "</div>", unsafe_allow_html=True)

        all_tabs = st.tabs(["Original", "Cleaned", "Grad-CAM"])

        with all_tabs[0]:
            st.image(image, width=280)

        with all_tabs[1]:
            st.image(processed_image, width=280)

        with all_tabs[2]:
            if grad_cam_views is not None:
                v = grad_cam_views
                stats = v['stats']
                bbox = v['roi_bbox']

                alpha_slider = st.slider(
                    "Heatmap opacity", 0.05, 0.95, 0.40, 0.05,
                    help="Blend strength of the Grad-CAM heatmap over the cleaned image (recommended α = 0.40)"
                )
                orig_np = np.array(processed_image.convert('RGB'))
                hm_interactive = v['heatmap_highres']
                enhanced_view = enhanced_overlay(
                    orig_np, hm_interactive, alpha=alpha_slider,
                    cmap=v['cmap'], contour=True, contour_thresh=0.45,
                    lesion_mask=v['lesion_mask']
                )

                cam_cols = st.columns(3)
                with cam_cols[0]:
                    st.image(image, width=260, caption="Original")
                with cam_cols[1]:
                    st.image(v['heatmap_only'], width=260, caption="Heatmap Only")
                with cam_cols[2]:
                    st.image(enhanced_view, width=260,
                             caption=f"Enhanced Overlay (α={alpha_slider:.2f})")

                st.markdown(colormap_legend_html(
                    v['cmap'], labels=('Less affected', 'Moderate', 'Most affected')),
                    unsafe_allow_html=True)

                st.markdown("<div class='section-title'>Actual Class Probabilities</div>", unsafe_allow_html=True)
                prob_html = "".join(
                    f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem'>"
                    f"<span style='width:60px;font-size:0.72rem;color:#8b949e'>{CLASS_LABELS[c]}</span>"
                    f"<div style='flex:1;background:#21262d;border-radius:4px;height:12px;overflow:hidden'>"
                    f"<div style='width:{p * 100:.1f}%;height:100%;background:linear-gradient(90deg,#58a6ff,#3fb950)'></div></div>"
                    f"<span style='width:50px;text-align:right;font-size:0.72rem;color:#f0f6fc;font-weight:600'>{p * 100:.1f}%</span>"
                    f"</div>"
                    for c, p in zip(CLASS_NAMES, probs)
                )
                st.markdown(f"<div class='card' style='padding:0.8rem'>{prob_html}</div>", unsafe_allow_html=True)

                st.markdown("<div class='section-title'>Grad-CAM Activation Region</div>", unsafe_allow_html=True)
                loc_cols = st.columns(2)
                with loc_cols[0]:
                    st.image(v['roi_image'], width=360, caption="Dominant Activation Region (Grad-CAM hotspot, not a lesion mask)")
                with loc_cols[1]:
                    if bbox is not None:
                        st.markdown(f"""
                        <div class='card' style='padding:1rem'>
                            <div style='font-size:0.9rem;font-weight:700;color:#f0f6fc;margin-bottom:0.6rem'>Activation Region Details</div>
                            <table style='font-size:0.78rem;color:#8b949e;width:100%'>
                                <tr><td>Bounding Box</td><td style='color:#58a6ff;font-weight:600'>({bbox['x']}, {bbox['y']}) → ({bbox['x']+bbox['w']}, {bbox['y']+bbox['h']})</td></tr>
                                <tr><td>Size (w × h)</td><td style='color:#f0f6fc'>{bbox['w']} × {bbox['h']} px</td></tr>
                                <tr><td>Activation Area</td><td style='color:#f0f6fc'>{bbox['area_pct']:.1f}% of image</td></tr>
                                <tr><td>Mean Activation</td><td style='color:#f0f6fc'>{bbox['intensity_mean']:.3f}</td></tr>
                                <tr><td>Center</td><td style='color:#f0f6fc'>({bbox['center'][0]}, {bbox['center'][1]})</td></tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.caption("No dominant activation region above threshold.")
                    st.markdown(f"""
                    <div class='card' style='padding:1rem;margin-top:0.5rem'>
                        <div style='font-size:0.9rem;font-weight:700;color:#f0f6fc;margin-bottom:0.6rem'>XAI Metrics</div>
                        <table style='font-size:0.78rem;color:#8b949e;width:100%'>
                            <tr><td>Confidence (top pred)</td><td style='color:#58a6ff;font-weight:600'>{top_prediction['confidence']:.1f}% — {CLASS_LABELS[top_prediction['class']]}</td></tr>
                            <tr><td>Peak Focus</td><td style='color:#f0f6fc'>{stats['peak']:.3f} (max activation)</td></tr>
                            <tr><td>Average Focus</td><td style='color:#f0f6fc'>{stats['mean']:.3f} (mean activation)</td></tr>
                            <tr><td>Activation Area (≥{stats['threshold']:.2f})</td><td style='color:#f0f6fc'>{stats['coverage_pct']:.1f}% of image</td></tr>
                            <tr><td>Activation IoU</td><td style='color:#8b949e'>N/A — no independent lesion mask</td></tr>
                            <tr><td>In-lesion Activation</td><td style='color:#8b949e'>N/A — no independent lesion mask</td></tr>
                            <tr><td>Background Activation</td><td style='color:#8b949e'>N/A — no independent lesion mask</td></tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(
                        "<div class='card' style='padding:0.8rem;margin-top:0.5rem;border:1px solid #f59e0b40'>"
                        "<span style='color:#f59e0b;font-weight:700'>⚠ Localization not independently verified:</span> "
                        "<span style='color:#c9d1d9'>HAM10000 has no pixel-level ground-truth lesion masks. The "
                        "Grad-CAM heatmap is shown qualitatively; IoU / in-lesion / background metrics are N/A and "
                        "no localization accuracy claim is made.</span></div>",
                        unsafe_allow_html=True)
            else:
                st.caption("Grad-CAM unavailable")

        st.markdown("<div style='margin-top:1rem;border-top:1px solid #21262d;padding-top:1rem'>", unsafe_allow_html=True)
        pred_cols = st.columns(3)
        for i, pred in enumerate(predictions):
            p_sev = pred['info']['severity']
            p_color = "#ef4444" if 'HIGH' in p_sev else ("#f59e0b" if 'Medium' in p_sev else "#10b981")
            cls = pred['class']
            clr_iou = EVAL_METRICS.get('iou_per_class', {}).get(cls, 0)
            clr_dice = EVAL_METRICS.get('dice_per_class', {}).get(cls, 0)
            with pred_cols[i]:
                st.markdown(f"""
                <div class="rank-card" style="border-left-color:{p_color}">
                    <div class="rank-num"># {i+1}</div>
                    <div class="rank-name" style="color:{p_color}">{pred['info']['name']}</div>
                    <div class="rank-meta">
                        <span>{pred['confidence']:.1f}%</span>
                        <span>{pred['info']['severity'].split(' - ')[0] if ' - ' in pred['info']['severity'] else pred['info']['severity']}</span>
                    </div>
                    <div class="rank-metrics">
                        <span>IoU: <b style="color:#58a6ff">{clr_iou:.4f}</b></span>
                        <span>Dice: <b style="color:#d2a8ff">{clr_dice:.4f}</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Clinical Details</div>", unsafe_allow_html=True)
        for i, pred in enumerate(predictions):
            p_color = "#ef4444" if 'HIGH' in pred['info']['severity'] else ("#f59e0b" if 'Medium' in pred['info']['severity'] else "#10b981")
            expander_label = f"{'🔴' if pred['class']=='mel' else ('🟡' if pred['class'] in ['akiec','bcc'] else '🟢')} {pred['info']['name']} ({pred['confidence']:.1f}%)"
            with st.expander(expander_label, expanded=(i == 0)):
                st.markdown(f"<div style='font-size:0.85rem;color:#c9d1d9'>{pred['info']['description']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top:0.5rem;font-size:0.8rem'><b style='color:#f0f6fc'>Severity:</b> <span style='color:{p_color}'>{pred['info']['severity']}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top:0.3rem;font-size:0.8rem'><b style='color:#f0f6fc'>Therapeutics:</b> <span style='color:#8b949e'>{', '.join(pred['info']['medicines'])}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top:0.3rem;font-size:0.8rem'><b style='color:#f0f6fc'>Prevention:</b> <span style='color:#8b949e'>{pred['info']['prevention']}</span></div>", unsafe_allow_html=True)

st.divider()
st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)

if EVAL_METRICS['f2_macro'] > 0:
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Macro F2", f"{EVAL_METRICS['f2_macro']:.4f}")
    with mcol2:
        st.metric("Macro IoU", f"{EVAL_METRICS.get('iou_macro', 0):.4f}")
    with mcol3:
        st.metric("Macro Dice", f"{EVAL_METRICS.get('dice_macro', 0):.4f}")
    with mcol4:
        n_classes = len(CLASS_NAMES)
        avg_conf = np.mean([EVAL_METRICS.get('f2_per_class', {}).get(c, 0) for c in CLASS_NAMES])
        st.metric("Avg Confidence", f"{avg_conf:.4f}")

    tabs = st.tabs(["F2-Score", "IoU", "Dice"])
    for ti, (key, label) in enumerate([('f2_per_class', 'F2'), ('iou_per_class', 'IoU'), ('dice_per_class', 'Dice')]):
        with tabs[ti]:
            cols = st.columns(len(CLASS_NAMES))
            for i, cls in enumerate(CLASS_NAMES):
                val = EVAL_METRICS.get(key, {}).get(cls, 0)
                bg = "#3b0a0a" if cls == "mel" else ("#3b2d0a" if cls in ["akiec","bcc"] else "#0a3b1a")
                border = "#ef4444" if cls == "mel" else ("#f59e0b" if cls in ["akiec","bcc"] else "#10b981")
                cols[i].markdown(
                    f"<div class='metric-tile' style='background:{bg};border:1px solid {border}40'>"
                    f"<div class='label'>{CLASS_LABELS[cls]}</div>"
                    f"<div class='value' style='color:{border}'>{val:.4f}</div></div>",
                    unsafe_allow_html=True
                )

    with st.expander("Diagnostic Plots", expanded=False):
        col_cm, col_curves = st.columns([1, 1])
        with col_cm:
            cm_path = f'{EVAL_DIR}/confusion_matrix.png'
            if __import__('pathlib').Path(cm_path).exists():
                st.image(cm_path, width=420)
        with col_curves:
            pr_path = f'{EVAL_DIR}/pr_curves.png'
            if __import__('pathlib').Path(pr_path).exists():
                st.image(pr_path, width=420)
else:
    st.caption("No evaluation data available")

st.divider()
col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    st.markdown("<div class='card' style='padding:1rem;text-align:center'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.2rem;font-weight:700'>EfficientNet-B0</div><div style='font-size:0.75rem;color:#8b949e'>Backbone with custom classifier head (Dropout + FC 1280→512→256→7)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with col_b:
    st.markdown("<div class='card' style='padding:1rem;text-align:center'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.2rem;font-weight:700'>Grad-CAM + Dropout</div><div style='font-size:0.75rem;color:#8b949e'>Explainable heatmaps with dropout active during inference</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with col_c:
    st.markdown("<div class='card' style='padding:1rem;text-align:center'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.2rem;font-weight:700'>F2-Score Optimized</div><div style='font-size:0.75rem;color:#8b949e'>Primary metric weights recall 2× precision for maximum melanoma sensitivity</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>DermaVision v2.0 · EfficientNet-B0 · HAM10000 · 🔬 For educational use only · Always consult a qualified dermatologist.</div>", unsafe_allow_html=True)
