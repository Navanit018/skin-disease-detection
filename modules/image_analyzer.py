import numpy as np
import cv2
from PIL import Image

def analyze_skin_texture_detail(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_score = np.std(gray - blur)
    edge_intensity = np.mean(np.abs(laplacian))

    if texture_score > 35:
        return 'Rough/Uneven', round(texture_score, 1)
    elif texture_score > 20:
        return 'Moderately Smooth', round(texture_score, 1)
    return 'Smooth', round(texture_score, 1)

def analyze_wrinkles(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=10)

    wrinkle_count = len(lines) if lines is not None else 0

    if wrinkle_count > 30:
        severity = 'High'
    elif wrinkle_count > 15:
        severity = 'Moderate'
    elif wrinkle_count > 5:
        severity = 'Mild'
    else:
        severity = 'Minimal'

    return severity, wrinkle_count

def analyze_acne_count(img_array):
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, dark_spots = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    acne_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if 10 < area < 500:
            acne_count += 1

    if acne_count > 15:
        severity = 'Severe'
    elif acne_count > 8:
        severity = 'Moderate'
    elif acne_count > 3:
        severity = 'Mild'
    else:
        severity = 'Minimal'

    return severity, acne_count

def analyze_pigmentation(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    dark_mask = cv2.inRange(gray, 0, 80)
    dark_pixels = np.sum(dark_mask > 0) / dark_mask.size

    if dark_pixels > 0.15:
        severity = 'Severe'
    elif dark_pixels > 0.08:
        severity = 'Moderate'
    elif dark_pixels > 0.03:
        severity = 'Mild'
    else:
        severity = 'Minimal'

    return severity, round(dark_pixels * 100, 1)

def analyze_pores_detail(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray, 5)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pore_density = np.sum(thresh > 0) / thresh.size

    if pore_density > 0.35:
        severity = 'Large/Very Visible'
    elif pore_density > 0.2:
        severity = 'Visible'
    elif pore_density > 0.1:
        severity = 'Somewhat Visible'
    else:
        severity = 'Minimal'

    return severity, round(pore_density * 100, 1)

def analyze_redness_detail(img_array):
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, (0, 20, 20), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 20, 20), (180, 255, 255))
    red_mask = mask1 + mask2

    redness_pixels = np.sum(red_mask > 0) / red_mask.size
    mean_red = np.mean(img_array[:, :, 0])

    if redness_pixels > 0.12 or mean_red > 150:
        severity = 'High'
    elif redness_pixels > 0.05 or mean_red > 120:
        severity = 'Moderate'
    elif redness_pixels > 0.02:
        severity = 'Mild'
    else:
        severity = 'Minimal'

    return severity, round(redness_pixels * 100, 1)

def analyze_scars(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(gray, blur)
    _, scar_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    scar_pixels = np.sum(scar_mask > 0) / scar_mask.size

    if scar_pixels > 0.1:
        severity = 'Significant'
    elif scar_pixels > 0.05:
        severity = 'Moderate'
    elif scar_pixels > 0.02:
        severity = 'Mild'
    else:
        severity = 'Minimal'

    return severity, round(scar_pixels * 100, 1)

def analyze_lesions(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    _, binary = cv2.threshold(blur, 0, 255, thresh_flag)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lesion_count = 0
    total_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            lesion_count += 1
            total_area += area

    return lesion_count, round(total_area / (img_array.shape[0] * img_array.shape[1]) * 100, 1)

def analyze_fungal_patterns(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 150)

    circular_patterns = 0
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour) >= 5:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.7 and area > 50:
                    circular_patterns += 1

    if circular_patterns > 5:
        suspicion = 'High - Possible fungal involvement'
    elif circular_patterns > 2:
        suspicion = 'Moderate - Monitor for circular patterns'
    else:
        suspicion = 'Low - No significant fungal patterns detected'

    return suspicion, circular_patterns

def analyze_skin_borders(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    irregular_borders = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.5:
                    irregular_borders += 1

    if irregular_borders > 3:
        assessment = 'Irregular borders detected - monitor closely'
    elif irregular_borders > 1:
        assessment = 'Some irregular borders present'
    else:
        assessment = 'Borders appear regular'

    return assessment, irregular_borders

def perform_full_analysis(image):
    img_array = np.array(image.convert('RGB'))
    img_array = cv2.resize(img_array, (512, 512))

    texture, texture_score = analyze_skin_texture_detail(img_array)
    wrinkles, wrinkle_count = analyze_wrinkles(img_array)
    acne_severity, acne_count = analyze_acne_count(img_array)
    pigmentation, pigmentation_pct = analyze_pigmentation(img_array)
    pores, pore_score = analyze_pores_detail(img_array)
    redness, redness_pct = analyze_redness_detail(img_array)
    scars, scar_pct = analyze_scars(img_array)
    lesion_count, lesion_area = analyze_lesions(img_array)
    fungal_suspicion, fungal_count = analyze_fungal_patterns(img_array)
    border_assessment, irregular_borders = analyze_skin_borders(img_array)

    return {
        'skin_texture': {'value': texture, 'score': texture_score},
        'wrinkles': {'severity': wrinkles, 'count': wrinkle_count},
        'acne': {'severity': acne_severity, 'count': acne_count},
        'pigmentation': {'severity': pigmentation, 'percentage': pigmentation_pct},
        'pores': {'severity': pores, 'score': pore_score},
        'redness': {'severity': redness, 'percentage': redness_pct},
        'scars': {'severity': scars, 'percentage': scar_pct},
        'lesions': {'count': lesion_count, 'area_percentage': lesion_area},
        'fungal_patterns': {'assessment': fungal_suspicion, 'circular_count': fungal_count},
        'border_assessment': {'description': border_assessment, 'irregular_count': irregular_borders}
    }
