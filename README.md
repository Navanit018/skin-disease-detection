# Skin Disease Detection System

AI-powered skin disease detection from skin lesion images (HAM10000 dataset).

## Features
- 7-class skin disease classification (EfficientNet-B0)
- Top-3 predictions with confidence scores
- Disease info, severity, medicines, and prevention tips
- GPU-accelerated training (PyTorch)

## How to Run Locally
1. Install: `pip install -r requirements.txt`
2. Run: `streamlit run app.py`

### Colab quick-smoke test
Use `train_colab.ipynb` for a fast PyTorch smoke test on a sampled subset of HAM10000.
Open in Colab, set runtime to GPU, and run cells in order; see cell 6 for smoke test and cell 7 for full training instructions.

## Deploy to Render
1. Push this repo to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Connect your GitHub repo
4. Set **Runtime** to `Python 3`
5. **Build Command**: `pip install -r requirements.txt`
6. **Start Command**: `streamlit run app_comprehensive.py --server.port $PORT --server.address 0.0.0.0`
7. Deploy!

## Model
- **Dataset**: HAM10000 (10,015 skin lesion images, 7 classes)
- **Architecture**: EfficientNet-B0 with transfer learning
- **Framework**: PyTorch (trained on NVIDIA RTX 4050 GPU)
- **Validation Accuracy**: ~90%

## Classes
- akiec - Actinic Keratoses
- bcc - Basal Cell Carcinoma
- bkl - Benign Keratosis
- df - Dermatofibroma
- nv - Melanocytic Nevi (Moles)
- vasc - Vascular Lesions
- mel - Melanoma

## Disclaimer
This app is for educational purposes only. Please consult a dermatologist for proper diagnosis.
