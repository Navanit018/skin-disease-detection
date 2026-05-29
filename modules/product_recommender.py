PRODUCT_CATEGORIES = {
    'cleanser': {
        'name': 'Cleanser',
        'concerns_map': {
            'Oily': ['Salicylic acid cleanser', 'Foaming gel cleanser', 'Clay-based cleanser'],
            'Dry': ['Cream cleanser', 'Milk cleanser', 'Oil-based cleanser'],
            'Combination': ['Gentle gel cleanser', 'Balancing cleanser', 'Low-pH cleanser'],
            'Sensitive': ['Micellar water', 'Non-foaming cleanser', 'Soothing cleanser'],
            'Normal': ['Gentle foaming cleanser', 'Water-based cleanser']
        }
    },
    'moisturizer': {
        'name': 'Moisturizer',
        'concerns_map': {
            'Oily': ['Oil-free gel moisturizer', 'Water-based moisturizer', 'Mattifying moisturizer'],
            'Dry': ['Rich cream moisturizer', 'Barrier repair cream', 'Nourishing balm'],
            'Combination': ['Lightweight lotion', 'Gel-cream hybrid', 'Balancing moisturizer'],
            'Sensitive': ['Fragrance-free moisturizer', 'Ceramide cream', 'Soothing balm'],
            'Normal': ['Daily moisturizer', 'Multi-vitamin cream']
        }
    },
    'sunscreen': {
        'name': 'Sunscreen',
        'concerns_map': {
            'Oily': ['Matte finish sunscreen', 'Oil-free SPF', 'Gel-based sunscreen'],
            'Dry': ['Moisturizing SPF', 'Cream sunscreen', 'Hydrating sunscreen'],
            'Combination': ['Lightweight SPF', 'Daily SPF moisturizer'],
            'Sensitive': ['Mineral sunscreen (Zinc oxide)', 'Titanium dioxide SPF', 'Fragrance-free SPF'],
            'Normal': ['Daily SPF 30+', 'Broad spectrum sunscreen']
        }
    },
    'serum': {
        'name': 'Serum',
        'concerns_map': {
            'Oily': ['Niacinamide serum', 'Salicylic acid serum', 'Retinol serum (night)'],
            'Dry': ['Hyaluronic acid serum', 'Squalane serum', 'Ceramide serum'],
            'Combination': ['Vitamin C serum', 'Niacinamide serum', 'Hydrating serum'],
            'Sensitive': ['Centella asiatica serum', 'Azelaic acid serum', 'Soothing serum'],
            'Normal': ['Vitamin C serum', 'Peptide serum', 'Antioxidant serum']
        }
    },
    'toner': {
        'name': 'Toner',
        'concerns_map': {
            'Oily': ['BHA toner', 'Astringent toner', 'Exfoliating toner'],
            'Dry': ['Hydrating toner', 'Milky toner', 'Alcohol-free toner'],
            'Combination': ['Balancing toner', 'pH-balancing toner', 'Hydrating toner'],
            'Sensitive': ['Soothing toner', 'Alcohol-free toner', 'Calming mist'],
            'Normal': ['Hydrating toner', 'Essence toner']
        }
    },
    'night_cream': {
        'name': 'Night Cream',
        'concerns_map': {
            'Oily': ['Retinol night cream', 'Light gel night cream', 'Oil-control night cream'],
            'Dry': ['Rich night cream', 'Overnight mask', 'Nourishing night balm'],
            'Combination': ['Balancing night cream', 'Light night cream', 'Rejuvenating cream'],
            'Sensitive': ['Barrier repair night cream', 'Soothing night mask', 'Calming night cream'],
            'Normal': ['Anti-aging night cream', 'Vitamin night cream']
        }
    },
    'shampoo': {
        'name': 'Shampoo',
        'concerns_map': {
            'Oily': ['Clarifying shampoo', 'Volumizing shampoo', 'Tea tree shampoo'],
            'Dry': ['Moisturizing shampoo', 'Sulfate-free shampoo', 'Nourishing shampoo'],
            'Combination': ['Balancing shampoo', 'Gentle daily shampoo'],
            'Sensitive': ['Hypoallergenic shampoo', 'Fragrance-free shampoo', 'Soothing shampoo'],
            'Normal': ['Gentle shampoo', 'Herbal shampoo']
        }
    },
    'treatment_cream': {
        'name': 'Treatment Cream',
        'concerns_map': {
            'Oily': ['Acne spot treatment', 'Oil control cream', 'Retinol treatment'],
            'Dry': ['Hydrating treatment', 'Barrier repair cream', 'Eczema cream'],
            'Combination': ['Spot treatment', 'Multi-action cream'],
            'Sensitive': ['Calming treatment', 'Anti-redness cream', 'Azelaic acid cream'],
            'Normal': ['Brightening cream', 'Anti-aging treatment']
        }
    }
}

CLIMATE_ADJUSTMENTS = {
    'Humid': {
        'Oily': 'Focus on lightweight, gel-based, non-comedogenic products',
        'Dry': 'Light hydration with humectants still needed',
        'Combination': 'Use gel textures, avoid heavy creams on T-zone',
        'Sensitive': 'Stick to minimal, soothing routine',
        'Normal': 'Lightweight hydration works well'
    },
    'Dry/Arid': {
        'Oily': 'May need more hydration than usual - use hydrating serums',
        'Dry': 'Rich creams and occlusives essential, layer hydration',
        'Combination': 'Focus on hydration, use richer products on dry areas',
        'Sensitive': 'Barrier repair crucial, use gentle occlusives',
        'Normal': 'Increase hydration with richer moisturizer'
    },
    'Cold': {
        'Oily': 'Add extra moisturizer, skin may become combination',
        'Dry': 'Use heavy creams and face oils for protection',
        'Combination': 'Richer products needed especially on cheeks',
        'Sensitive': 'Barrier protection essential, avoid extreme temps',
        'Normal': 'Upgrade to richer moisturizer for cold months'
    },
    'Hot': {
        'Oily': 'Light gel moisturizer, oil-control products essential',
        'Dry': 'Light hydration, focus on water-based products',
        'Combination': 'Lightweight throughout, mattify T-zone',
        'Sensitive': 'Cooling soothing products, avoid heat irritation',
        'Normal': 'Switch to lighter textures for summer'
    },
    'Polluted': {
        'Oily': 'Antioxidants essential, double cleanse at night',
        'Dry': 'Barrier protection + antioxidants crucial',
        'Combination': 'Antioxidant serum, thorough cleansing',
        'Sensitive': 'Extra protection needed, barrier support',
        'Normal': 'Antioxidant protection, double cleanse'
    }
}

def recommend_products(skin_type, climate='Normal', age=25, allergies=None, concerns=None, acne_severity=None, pigmentation_level=None, hydration_level=None):
    if skin_type == 'Sensitive':
        skin_type = 'Sensitive'
    elif skin_type == 'Combination':
        skin_type = 'Combination'
    elif skin_type not in PRODUCT_CATEGORIES['cleanser']['concerns_map']:
        skin_type = 'Normal'

    recommendations = {}

    for category_key, category_data in PRODUCT_CATEGORIES.items():
        products = category_data['concerns_map'].get(skin_type, category_data['concerns_map']['Normal'])
        recommendations[category_key] = {
            'category': category_data['name'],
            'recommended_products': products
        }

    if age > 40:
        recommendations['serum']['recommended_products'].append('Retinol serum for anti-aging')
        recommendations['night_cream']['recommended_products'].append('Advanced anti-aging night cream')
    elif age > 30:
        recommendations['serum']['recommended_products'].append('Peptide serum for collagen support')

    if allergies:
        for cat_key, cat_rec in recommendations.items():
            filtered = [p for p in cat_rec['recommended_products'] if not any(a.lower() in p.lower() for a in allergies)]
            if filtered:
                cat_rec['recommended_products'] = filtered
            cat_rec['recommended_products'].append('Fragrance-free option recommended')

    if acne_severity and acne_severity == 'Severe':
        recommendations['cleanser']['recommended_products'].extend(['Medicated acne cleanser', 'Dermatologist-recommended cleanser'])
        recommendations['treatment_cream']['recommended_products'].extend(['Prescription-strength treatment', 'Benzoyl peroxide cream'])
    elif acne_severity == 'Moderate':
        recommendations['cleanser']['recommended_products'].append('Salicylic acid cleanser')
        recommendations['treatment_cream']['recommended_products'].append('Acne spot treatment')

    if pigmentation_level and pigmentation_level in ['Moderate', 'Severe']:
        recommendations['serum']['recommended_products'].extend(['Vitamin C serum for brightening', 'Alpha arbutin serum', 'Kojic acid treatment'])
        recommendations['treatment_cream']['recommended_products'].append('Brightening treatment cream')

    if climate in CLIMATE_ADJUSTMENTS:
        adjustment = CLIMATE_ADJUSTMENTS[climate].get(skin_type, CLIMATE_ADJUSTMENTS[climate].get('Normal', ''))
        recommendations['climate_advice'] = adjustment

    return recommendations

def get_product_by_concern(concern):
    concern_product_map = {
        'acne': ['Salicylic acid cleanser', 'Benzoyl peroxide spot treatment', 'Niacinamide serum'],
        'wrinkles': ['Retinol serum', 'Peptide cream', 'Hyaluronic acid serum'],
        'dark spots': ['Vitamin C serum', 'Alpha arbutin serum', 'Kojic acid cream'],
        'dryness': ['Hyaluronic acid serum', 'Rich moisturizer', 'Squalane oil'],
        'redness': ['Centella asiatica serum', 'Azelaic acid cream', 'Soothing moisturizer'],
        'large pores': ['Niacinamide serum', 'BHA toner', 'Clay mask'],
        'dullness': ['Vitamin C serum', 'AHA toner', 'Brightening moisturizer'],
        'blackheads': ['BHA toner', 'Salicylic acid cleanser', 'Niacinamide serum'],
        'dark circles': ['Caffeine eye cream', 'Vitamin C eye cream', 'Retinol eye cream'],
        'hair loss': ['Minoxidil treatment', 'Biotin shampoo', 'Scalp treatment serum'],
        'dandruff': ['Ketoconazole shampoo', 'Salicylic acid shampoo', 'Zinc pyrithione shampoo']
    }
    return concern_product_map.get(concern.lower(), ['Consult a dermatologist for specific concerns'])
