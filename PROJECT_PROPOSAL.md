# AI-Powered Skin Disease Detection & Skincare Assistant
## Project Approval Proposal

---

## 1. Project Title
**AI-Powered Skin Disease Detection & Personalized Skincare Assistant**

---

## 2. Project Objective
To develop an intelligent system that uses deep learning and computer vision to:
- Detect skin diseases from images
- Analyze skin type
- Identify potential allergies
- Provide personalized skincare recommendations
- Generate comprehensive dermatology reports

---

## 3. Problem Statement
- Skin diseases are among the most common health issues worldwide, yet access to dermatologists is limited
- Many people self-diagnose using unreliable sources
- There is no single platform that combines disease detection, skin analysis, allergy detection, and product recommendations
- Existing solutions focus on only one aspect (usually just disease classification)

---

## 4. Scope of Work

### 4.1 Modules Implemented (Completed)

| # | Module | Description | Technology |
|---|--------|-------------|------------|
| 1 | **Disease Detection** | Classifies 7 skin lesions using deep learning | EfficientNet-B0, PyTorch/TensorFlow, HAM10000 dataset |
| 2 | **Extended Disease Conditions** | Information and guidance for 12 additional skin conditions | Knowledge base |
| 3 | **Skin Type Analysis** | Detects Oily/Dry/Combination/Sensitive/Normal via image + questionnaire | OpenCV computer vision |
| 4 | **Allergy Detection** | Identifies allergies from 8 categories with severity scoring | Rule-based expert system |
| 5 | **Ingredient Intelligence** | Analyzes product ingredient lists for useful/harmful components | Knowledge base |
| 6 | **Product Recommendation** | Recommends 8 product categories personalized by skin type, climate, age, allergies | Rule-based system |
| 7 | **Treatment Planner** | Generates morning/night routines, diet, lifestyle, sun protection advice | Knowledge base |
| 8 | **Image Analysis** | Analyzes texture, wrinkles, acne, pigmentation, pores, redness, scars, lesions, fungal patterns | OpenCV |
| 9 | **Report Generator** | Produces downloadable comprehensive dermatology report | Report engine |
| 10 | **Prediction History** | Saves and displays all analysis results | JSON storage |

### 4.2 Modules Proposed (To Be Implemented)

| # | Module | Description | Priority |
|---|--------|-------------|----------|
| 1 | **User Authentication** | Email/password login system for multi-user support | High |
| 2 | **Skin Progress Tracker** | Track skin changes over time with photo comparison and metrics | High |
| 3 | **Explainable AI (Grad-CAM)** | Visual heatmaps showing model decision regions | High |
| 4 | **Database Migration** | Replace JSON file with SQLite database | Medium |
| 5 | **REST API Backend** | FastAPI backend for mobile/web app integration | Medium |
| 6 | **Multilingual Support** | Hindi + English language toggle | Medium |
| 7 | **Real-Time Webcam Detection** | Live video analysis | Medium |
| 8 | **Barcode Product Scanner** | Scan product barcode for ingredient analysis | Low |
| 9 | **Doctor Consultation Integration** | Connect users with dermatologists | Low |

---

## 5. Technology Stack

| Component | Technology Used | Purpose |
|-----------|----------------|---------|
| Frontend | Streamlit (Python) | Web-based user interface |
| Deep Learning | PyTorch, TensorFlow | Disease classification model |
| Computer Vision | OpenCV | Skin texture/acne/pigmentation analysis |
| Image Processing | Pillow (PIL), NumPy | Image preprocessing |
| Data Storage | JSON (current) → SQLite (proposed) | User history |
| Pre-trained Model | EfficientNet-B0 | Transfer learning for skin lesion classification |
| Dataset | HAM10000 (10,015 images, 7 classes) | Training disease detection model |

---

## 6. System Architecture

```
User Uploads Image
        │
        ▼
┌─────────────────────────────────────┐
│         Image Preprocessing         │
│   (Resize, Normalize, Augment)      │
└────────┬────────────────────────────┘
         │
    ┌────┴────────────────────────────┐
    │         Feature Extraction      │
    │   (CNN / OpenCV Features)       │
    └────┬────────────────────────────┘
         │
    ┌────┴────────────────────────────┐
    │    Parallel Analysis Pipeline   │
    │                                 │
    │  ┌─────────────────────────┐    │
    │  │ Disease Classification  │    │
    │  │ (EfficientNet-B0)       │    │
    │  └─────────────────────────┘    │
    │  ┌─────────────────────────┐    │
    │  │ Skin Type Analysis      │    │
    │  │ (OpenCV + Rules)        │    │
    │  └─────────────────────────┘    │
    │  ┌─────────────────────────┐    │
    │  │ Allergy Risk Detection  │    │
    │  │ (Symptom Matching)      │    │
    │  └─────────────────────────┘    │
    │  ┌─────────────────────────┐    │
    │  │ Product Recommendation  │    │
    │  │ (Rule-based Engine)     │    │
    │  └─────────────────────────┘    │
    │  ┌─────────────────────────┐    │
    │  │ Treatment Planning      │    │
    │  │ (Expert Knowledge Base) │    │
    │  └─────────────────────────┘    │
    └────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      Final Dermatology Report       │
│   (All results combined + export)   │
└─────────────────────────────────────┘
```

---

## 7. Dataset Details

| Dataset | Source | Images | Classes | Used For |
|---------|--------|--------|---------|----------|
| HAM10000 | ISIC Archive | 10,015 | 7 skin lesion types | Disease classification model training |
| Custom Image Input | User uploads | N/A | N/A | Skin type analysis, image analysis |

**Disease Classes (HAM10000):**
1. Actinic Keratoses (akiec)
2. Basal Cell Carcinoma (bcc)
3. Benign Keratosis (bkl)
4. Dermatofibroma (df)
5. Melanocytic Nevi (nv)
6. Vascular Lesions (vasc)
7. Melanoma (mel)

---

## 8. Model Performance

| Metric | Value |
|--------|-------|
| Architecture | EfficientNet-B0 |
| Validation Accuracy | ~67% |
| Framework | PyTorch |
| Training Hardware | NVIDIA RTX 4050 GPU |
| Input Size | 224x224 pixels |
| Output | Top-3 predictions with confidence scores |

---

## 9. Deliverables

### 9.1 Completed Deliverables
- `app_comprehensive.py` — Main application with all features
- `modules/skin_type_analyzer.py` — Skin type detection module
- `modules/allergy_detector.py` — Allergy detection module
- `modules/ingredient_intelligence.py` — Ingredient analysis module
- `modules/product_recommender.py` — Product recommendation engine
- `modules/treatment_planner.py` — Treatment plan generator
- `modules/image_analyzer.py` — Computer vision image analysis
- `modules/report_generator.py` — Report generation engine
- Pre-trained model files (`.pt`, `.keras`)

### 9.2 Proposed Deliverables
- User authentication system
- Skin progress tracker with timeline charts
- Grad-CAM explainable AI visualization
- SQLite database backend
- FastAPI REST API
- Multilingual interface (Hindi + English)

---

## 10. Project Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| **Phase 1** (Complete) | Week 1 | Core features implementation, model training, basic UI |
| **Phase 2** (Proposed) | Week 2-3 | User authentication, progress tracker, Grad-CAM, database |
| **Phase 3** (Proposed) | Week 4 | REST API, multilingual support, webcam detection |
| **Phase 4** (Proposed) | Week 5 | Testing, deployment, documentation, final submission |

---

## 11. Testing & Validation

| Test Type | Description |
|-----------|-------------|
| Model Accuracy | Validate disease classification accuracy on test split |
| Module Unit Tests | Each module tested independently for correct output |
| Integration Testing | Full pipeline from image upload to report generation |
| User Acceptance Testing | Test with sample users for feedback |
| Performance Testing | Response time analysis for real-time prediction |

---

## 12. Conclusion
This project delivers a comprehensive AI-powered skincare platform combining deep learning for disease detection with computer vision and rule-based expert systems for skin analysis, allergy detection, product recommendations, and personalized treatment planning. The system is fully functional with 10 core modules and has a clear roadmap for future enhancements.

---

**Submitted by:** [Your Name]
**Date:** May 2026
**Project Type:** Academic Research / Development Project

---

### Approval Signatures

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Project Guide | | | |
| Head of Department | | | |
| Principal/Dean | | | |
