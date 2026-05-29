import numpy as np
from PIL import Image
import cv2

SKIN_TYPES = ['Oily', 'Dry', 'Combination', 'Sensitive', 'Normal']

def analyze_skin_oil_level(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var > 100:
        return 'high'
    elif laplacian_var > 50:
        return 'medium'
    return 'low'

def analyze_skin_texture(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    texture_score = np.std(gray - blur)
    if texture_score > 30:
        return 'rough'
    elif texture_score > 15:
        return 'moderate'
    return 'smooth'

def analyze_pore_visibility(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pore_density = np.sum(thresh > 0) / thresh.size
    if pore_density > 0.4:
        return 'large'
    elif pore_density > 0.2:
        return 'visible'
    return 'small'

def analyze_dryness(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    flakiness = np.sum(gray > 200) / gray.size
    saturation = np.mean(hsv[:, :, 1])
    if flakiness > 0.15 or saturation < 30:
        return 'high'
    elif flakiness > 0.05:
        return 'moderate'
    return 'low'

def analyze_redness(img_array):
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    red_mask = cv2.inRange(hsv, (0, 30, 30), (10, 255, 255))
    red_ratio = np.sum(red_mask > 0) / red_mask.size
    if red_ratio > 0.15:
        return 'high'
    elif red_ratio > 0.05:
        return 'moderate'
    return 'low'

def analyze_shine(img_array):
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    value = hsv[:, :, 2]
    shine_pixels = np.sum(value > 200) / value.size
    if shine_pixels > 0.2:
        return 'high'
    elif shine_pixels > 0.1:
        return 'moderate'
    return 'low'

def analyze_flakiness(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    white_pixels = np.sum(gray > 220) / gray.size
    flake_score = edge_density * 0.4 + white_pixels * 0.6
    if flake_score > 0.2:
        return 'high'
    elif flake_score > 0.1:
        return 'moderate'
    return 'low'

def detect_skin_type(image):
    img_array = np.array(image.convert('RGB'))
    img_array = cv2.resize(img_array, (224, 224))

    oil = analyze_skin_oil_level(img_array)
    texture = analyze_skin_texture(img_array)
    pores = analyze_pore_visibility(img_array)
    dryness = analyze_dryness(img_array)
    redness = analyze_redness(img_array)
    shine = analyze_shine(img_array)
    flakiness = analyze_flakiness(img_array)

    scores = {
        'oil_level': oil,
        'texture': texture,
        'pore_visibility': pores,
        'dryness': dryness,
        'redness': redness,
        'shine': shine,
        'flakiness': flakiness
    }

    oily_score = 0
    dry_score = 0
    sensitive_score = 0

    if oil in ['high', 'medium']:
        oily_score += 2 if oil == 'high' else 1
    if pores == 'large':
        oily_score += 2
    elif pores == 'visible':
        oily_score += 1
    if shine == 'high':
        oily_score += 2
    elif shine == 'moderate':
        oily_score += 1

    if dryness == 'high':
        dry_score += 2
    elif dryness == 'moderate':
        dry_score += 1
    if flakiness == 'high':
        dry_score += 2
    elif flakiness == 'moderate':
        dry_score += 1
    if texture == 'rough':
        dry_score += 1

    if redness == 'high':
        sensitive_score += 2
    elif redness == 'moderate':
        sensitive_score += 1

    max_score = max(oily_score, dry_score, sensitive_score)

    if max_score == 0:
        skin_type = 'Normal'
    else:
        if oily_score >= dry_score and oily_score >= sensitive_score:
            if dry_score > 1:
                skin_type = 'Combination'
            else:
                skin_type = 'Oily'
        elif dry_score >= sensitive_score:
            if oily_score > 1:
                skin_type = 'Combination'
            else:
                skin_type = 'Dry'
        else:
            skin_type = 'Sensitive'

    confidence = (max_score / 6) * 100 if max_score > 0 else 60

    return {
        'skin_type': skin_type,
        'confidence': min(confidence, 95),
        'analysis': scores,
        'oily_score': oily_score,
        'dry_score': dry_score,
        'sensitive_score': sensitive_score
    }

def detect_skin_type_from_questionnaire(oil, sensitivity, dryness, pores, shine):
    oily_score = 0
    dry_score = 0
    sensitive_score = 0

    oil_map = {'Very Oily': 3, 'Oily': 2, 'Normal': 0, 'Dry': -1, 'Very Dry': -2}
    oily_score += max(0, oil_map.get(oil, 0))
    dry_score += max(0, -oil_map.get(oil, 0))

    if sensitivity in ['Very Sensitive', 'Sensitive']:
        sensitive_score += 2

    dryness_map = {'Yes, Very Dry': 3, 'Sometimes': 1, 'No': 0}
    dry_score += dryness_map.get(dryness, 0)

    if pores in ['Large/Very Visible']:
        oily_score += 2
    elif pores in ['Somewhat Visible']:
        oily_score += 1

    if shine == 'Yes, Very Shiny':
        oily_score += 2
    elif shine == 'A Little':
        oily_score += 1

    if sensitive_score >= 2 and oily_score == 0 and dry_score == 0:
        return 'Sensitive'

    if oily_score > 1 and dry_score > 1:
        return 'Combination'
    if oily_score > dry_score and oily_score >= sensitive_score:
        return 'Oily'
    if dry_score > oily_score and dry_score >= sensitive_score:
        return 'Dry'
    if sensitive_score > oily_score and sensitive_score > dry_score:
        return 'Sensitive'

    return 'Normal'
