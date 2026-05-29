ALLERGY_CATEGORIES = {
    'cosmetics': {
        'name': 'Cosmetics Allergy',
        'symptoms': ['Redness', 'Itching', 'Swelling', 'Burning', 'Rash on face'],
        'common_triggers': ['Foundation', 'Mascara', 'Lipstick', 'Eye shadow', 'Blush', 'Concealer'],
        'precautions': [
            'Use hypoallergenic products',
            'Patch test before using new products',
            'Check expiry dates',
            'Avoid sharing makeup',
            'Clean brushes regularly'
        ],
        'ingredients_to_avoid': ['Parabens', 'Fragrance', 'Formaldehyde', 'Toluene', 'Phthalates']
    },
    'soaps': {
        'name': 'Soap & Detergent Allergy',
        'symptoms': ['Dryness', 'Redness', 'Itching', 'Flaking', 'Burning sensation'],
        'common_triggers': ['Bar soaps', 'Liquid hand washes', 'Laundry detergents', 'Body washes', 'Shampoos'],
        'precautions': [
            'Use fragrance-free soaps',
            'Wear gloves when cleaning',
            'Use gentle, pH-balanced cleansers',
            'Moisturize after washing',
            'Rinse thoroughly'
        ],
        'ingredients_to_avoid': ['Sodium Lauryl Sulfate', 'Sodium Laureth Sulfate', 'Fragrance', 'Dyes', 'Formaldehyde']
    },
    'food': {
        'name': 'Food Allergy',
        'symptoms': ['Hives', 'Swelling', 'Itching', 'Redness', 'Rash', 'Digestive issues'],
        'common_triggers': ['Dairy', 'Eggs', 'Nuts', 'Shellfish', 'Soy', 'Gluten', 'Spicy foods'],
        'precautions': [
            'Identify trigger foods through elimination diet',
            'Read food labels carefully',
            'Keep a food diary',
            'Consult an allergist',
            'Avoid known allergens'
        ],
        'ingredients_to_avoid': ['Milk protein', 'Egg albumin', 'Gluten', 'Soy lecithin', 'Nut oils']
    },
    'medicines': {
        'name': 'Medication Allergy',
        'symptoms': ['Rash', 'Hives', 'Itching', 'Swelling', 'Redness', 'Difficulty breathing'],
        'common_triggers': ['Antibiotics', 'NSAIDs', 'Aspirin', 'Anticonvulsants', 'Chemotherapy drugs'],
        'precautions': [
            'Always inform doctors about allergies',
            'Wear medical alert bracelet',
            'Carry antihistamines',
            'Read medication labels',
            'Report adverse reactions immediately'
        ],
        'ingredients_to_avoid': ['Penicillin', 'Sulfa', 'Codeine', 'Ibuprofen', 'Aspirin']
    },
    'pollen': {
        'name': 'Pollen Allergy',
        'symptoms': ['Itchy eyes', 'Runny nose', 'Sneezing', 'Watery eyes', 'Skin rash'],
        'common_triggers': ['Tree pollen', 'Grass pollen', 'Weed pollen', 'Flower pollen', 'Seasonal changes'],
        'precautions': [
            'Stay indoors during high pollen counts',
            'Close windows',
            'Shower after outdoor activities',
            'Use air purifiers',
            'Wear sunglasses outdoors'
        ],
        'ingredients_to_avoid': ['Pollen-based skincare', 'Bee pollen', 'Propolis', 'Royal jelly']
    },
    'dust': {
        'name': 'Dust Allergy',
        'symptoms': ['Sneezing', 'Coughing', 'Itchy skin', 'Red eyes', 'Congestion'],
        'common_triggers': ['Dust mites', 'Cockroach droppings', 'Mold spores', 'Pet dander', 'Carpet fibers'],
        'precautions': [
            'Use allergen-proof covers on bedding',
            'Vacuum regularly with HEPA filter',
            'Wash bedding in hot water weekly',
            'Reduce humidity below 50%',
            'Remove carpets if possible'
        ],
        'ingredients_to_avoid': ['Lanolin', 'Wool', 'Certain botanical extracts']
    },
    'sunlight': {
        'name': 'Sun Allergy (Photosensitivity)',
        'symptoms': ['Redness', 'Blisters', 'Hives', 'Itching', 'Burning after sun exposure'],
        'common_triggers': ['Direct sunlight', 'Tanning beds', 'Certain medications + sun', 'Chemical sunscreens'],
        'precautions': [
            'Use physical/mineral sunscreens (zinc oxide, titanium dioxide)',
            'Wear UPF clothing',
            'Avoid sun during peak hours (10am-4pm)',
            'Wear wide-brimmed hats',
            'Gradually increase sun exposure'
        ],
        'ingredients_to_avoid': ['Oxybenzone', 'Octinoxate', 'Retinol', 'AHAs', 'Benzoyl peroxide']
    },
    'metals': {
        'name': 'Metal Allergy',
        'symptoms': ['Contact dermatitis', 'Redness', 'Itching', 'Swelling', 'Blisters at contact site'],
        'common_triggers': ['Nickel', 'Cobalt', 'Chromium', 'Copper', 'Gold'],
        'precautions': [
            'Wear hypoallergenic jewelry (surgical steel, titanium)',
            'Use barrier coatings on metal objects',
            'Avoid cheap metal accessories',
            'Check clothing for metal snaps/buttons',
            'Use plastic or silicone alternatives'
        ],
        'ingredients_to_avoid': ['Nickel sulfate', 'Cobalt chloride', 'Potassium dichromate', 'Copper sulfate']
    },
    'skincare_ingredients': {
        'name': 'Skincare Ingredient Allergy',
        'symptoms': ['Burning', 'Stinging', 'Redness', 'Dryness', 'Breakouts', 'Peeling'],
        'common_triggers': ['Fragrance', 'Essential oils', 'Alcohol', 'Retinol', 'AHAs/BHAs'],
        'precautions': [
            'Patch test all new products',
            'Introduce one product at a time',
            'Use fragrance-free products',
            'Avoid alcohol-based products',
            'Start with low concentrations of active ingredients'
        ],
        'ingredients_to_avoid': ['Fragrance/Parfum', 'Denatured Alcohol', 'Essential oils', 'Menthol', 'Camphor']
    }
}

ALLERGY_SEVERITY_LEVELS = ['Mild', 'Moderate', 'Severe']

def assess_allergy_severity(symptoms_count, duration_days, affected_area, spread):
    score = 0
    if symptoms_count >= 4:
        score += 3
    elif symptoms_count >= 2:
        score += 2
    else:
        score += 1

    if duration_days > 7:
        score += 3
    elif duration_days > 3:
        score += 2
    else:
        score += 1

    area_map = {'Localized': 1, 'Multiple areas': 2, 'Widespread': 3}
    score += area_map.get(affected_area, 1)

    if spread == 'Spreading':
        score += 2

    if score >= 8:
        return 'Severe', score
    elif score >= 5:
        return 'Moderate', score
    return 'Mild', score

def detect_allergy(symptoms, trigger_candidates=None):
    matched_allergies = []

    for key, allergy in ALLERGY_CATEGORIES.items():
        symptom_matches = [s for s in symptoms if s.lower() in [sym.lower() for sym in allergy['symptoms']]]
        if trigger_candidates and any(t.lower() in [tg.lower() for tg in allergy['common_triggers']] for t in trigger_candidates):
            symptom_matches = symptom_matches or ['matched_trigger']
        if symptom_matches:
            severity, sev_score = assess_allergy_severity(
                len(symptom_matches), 3, 'Localized', 'No'
            )
            matched_allergies.append({
                'category': key,
                'name': allergy['name'],
                'matched_symptoms': symptom_matches if symptom_matches != ['matched_trigger'] else ['Known trigger match'],
                'severity': severity,
                'severity_score': sev_score,
                'common_triggers': allergy['common_triggers'],
                'precautions': allergy['precautions'],
                'ingredients_to_avoid': allergy['ingredients_to_avoid']
            })

    matched_allergies.sort(key=lambda x: x['severity_score'], reverse=True)

    return matched_allergies

def check_ingredient_allergy(ingredient_list, user_allergies):
    warnings = []
    for ingredient in ingredient_list:
        ingredient_lower = ingredient.lower()
        for allergy_key, allergy_data in ALLERGY_CATEGORIES.items():
            for bad_ing in allergy_data['ingredients_to_avoid']:
                if bad_ing.lower() in ingredient_lower:
                    warnings.append({
                        'ingredient': ingredient,
                        'allergy_category': allergy_data['name'],
                        'reason': f'{bad_ing} found in ingredient list'
                    })
    return warnings
