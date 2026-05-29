USEFUL_INGREDIENTS = {
    'Niacinamide': {
        'category': 'Multi-benefit',
        'benefits': ['Reduces pores', 'Controls oil', 'Brightens skin', 'Strengthens barrier', 'Reduces redness'],
        'good_for': ['Oily', 'Combination', 'Acne-prone', 'Large pores'],
        'avoid_with': ['Pure Vitamin C (can be used at different times)'],
        'usage': 'Use morning and evening',
        'concentration': '2-10%'
    },
    'Hyaluronic Acid': {
        'category': 'Hydration',
        'benefits': ['Deep hydration', 'Plumps skin', 'Reduces fine lines', 'Soothes dryness'],
        'good_for': ['Dry', 'Dehydrated', 'Mature', 'Sensitive'],
        'avoid_with': [],
        'usage': 'Apply on damp skin, morning and evening',
        'concentration': '1-2%'
    },
    'Salicylic Acid': {
        'category': 'Exfoliant',
        'benefits': ['Unclogs pores', 'Reduces acne', 'Exfoliates', 'Controls oil', 'Blackhead removal'],
        'good_for': ['Oily', 'Acne-prone', 'Blackheads', 'Combination'],
        'avoid_with': ['Other acids', 'Retinol (same routine)', 'Benzoyl peroxide (same time)'],
        'usage': 'Use 2-3 times a week, evening only',
        'concentration': '0.5-2%'
    },
    'Vitamin C': {
        'category': 'Antioxidant',
        'benefits': ['Brightens skin', 'Reduces dark spots', 'Boosts collagen', 'Protects from pollution', 'Fades hyperpigmentation'],
        'good_for': ['Dull skin', 'Hyperpigmentation', 'Aging', 'Sun damage'],
        'avoid_with': ['Niacinamide (same time - use separately)', 'AHAs/BHAs'],
        'usage': 'Use in morning before sunscreen',
        'concentration': '10-20%'
    },
    'Ceramides': {
        'category': 'Barrier Repair',
        'benefits': ['Repairs skin barrier', 'Locks in moisture', 'Reduces sensitivity', 'Protects skin'],
        'good_for': ['Dry', 'Sensitive', 'Damaged barrier', 'Eczema-prone'],
        'avoid_with': [],
        'usage': 'Use morning and evening',
        'concentration': '0.1-1%'
    },
    'Retinol': {
        'category': 'Anti-aging',
        'benefits': ['Reduces wrinkles', 'Speeds cell turnover', 'Fades dark spots', 'Improves texture', 'Boosts collagen'],
        'good_for': ['Aging', 'Sun damage', 'Fine lines', 'Uneven texture'],
        'avoid_with': ['AHAs/BHAs (same routine)', 'Benzoyl peroxide', 'Vitamin C (same routine)'],
        'usage': 'Use at night only, start 2-3 times/week',
        'concentration': '0.01-1%'
    },
    'Benzoyl Peroxide': {
        'category': 'Acne Treatment',
        'benefits': ['Kills acne bacteria', 'Reduces inflammation', 'Dries out pimples', 'Prevents breakouts'],
        'good_for': ['Inflammatory acne', 'Cystic acne', 'Pimples'],
        'avoid_with': ['Retinol', 'AHAs/BHAs (same routine)', 'Vitamin C'],
        'usage': 'Use on affected areas only, start with low concentration',
        'concentration': '2.5-10%'
    },
    'Alpha Arbutin': {
        'category': 'Brightening',
        'benefits': ['Fades dark spots', 'Reduces hyperpigmentation', 'Even skin tone', 'Gentle brightening'],
        'good_for': ['Hyperpigmentation', 'Dark spots', 'Melasma', 'Sun damage'],
        'avoid_with': [],
        'usage': 'Use morning and evening',
        'concentration': '1-2%'
    },
    'Peptides': {
        'category': 'Anti-aging',
        'benefits': ['Boosts collagen', 'Reduces wrinkles', 'Firms skin', 'Improves elasticity'],
        'good_for': ['Aging', 'Fine lines', 'Loss of firmness', 'Mature skin'],
        'avoid_with': [],
        'usage': 'Use morning and evening',
        'concentration': '0.1-5%'
    },
    'Azelaic Acid': {
        'category': 'Multi-benefit',
        'benefits': ['Reduces redness', 'Treats rosacea', 'Fades dark spots', 'Anti-acne'],
        'good_for': ['Rosacea', 'Redness', 'Acne', 'Hyperpigmentation', 'Sensitive'],
        'avoid_with': [],
        'usage': 'Use morning and evening',
        'concentration': '10-20%'
    },
    'Centella Asiatica': {
        'category': 'Soothing',
        'benefits': ['Calms irritation', 'Heals wounds', 'Reduces redness', 'Anti-inflammatory'],
        'good_for': ['Sensitive', 'Irritated', 'Inflamed', 'Post-procedure'],
        'avoid_with': [],
        'usage': 'Use morning and evening',
        'concentration': '0.1-1%'
    },
    'Squalane': {
        'category': 'Moisturizer',
        'benefits': ['Lightweight hydration', 'Non-comedogenic', 'Softens skin', 'Suitable for all types'],
        'good_for': ['All skin types', 'Dehydrated', 'Oily (lightweight)'],
        'avoid_with': [],
        'usage': 'Use morning and evening',
        'concentration': '0.5-5%'
    }
}

HARMFUL_INGREDIENTS = {
    'Alcohol Denat': {
        'concern': 'Strips skin barrier',
        'severity': 'High',
        'avoid_for': ['Dry', 'Sensitive', 'Dehydrated']
    },
    'Fragrance/Parfum': {
        'concern': 'Common allergen, irritant',
        'severity': 'High',
        'avoid_for': ['Sensitive', 'Allergic', 'Eczema-prone']
    },
    'Parabens': {
        'concern': 'Potential endocrine disruptor',
        'severity': 'Medium',
        'avoid_for': ['Sensitive', 'Pregnant']
    },
    'Sulfates (SLS/SLES)': {
        'concern': 'Strips natural oils, irritates',
        'severity': 'Medium',
        'avoid_for': ['Dry', 'Sensitive', 'Eczema-prone']
    },
    'Menthol': {
        'concern': 'Can cause irritation, sensitizing',
        'severity': 'Low-Medium',
        'avoid_for': ['Sensitive', 'Rosacea']
    },
    'Essential Oils': {
        'concern': 'Common sensitizers, can cause reactions',
        'severity': 'Medium',
        'avoid_for': ['Sensitive', 'Allergic']
    },
    'Oxybenzone': {
        'concern': 'Potential hormone disruptor, skin irritant',
        'severity': 'Medium',
        'avoid_for': ['Sensitive', 'Pregnant']
    },
    'Octinoxate': {
        'concern': 'Potential hormone disruptor',
        'severity': 'Low-Medium',
        'avoid_for': ['Sensitive', 'Pregnant']
    },
    'Urea (high concentration)': {
        'concern': 'Can irritate broken skin',
        'severity': 'Low',
        'avoid_for': ['Broken skin', 'Eczema flare-up']
    },
    'Lactic Acid (high concentration)': {
        'concern': 'Can cause stinging and irritation',
        'severity': 'Low-Medium',
        'avoid_for': ['Sensitive', 'Damaged barrier']
    },
    'Propylene Glycol': {
        'concern': 'Common irritant for sensitive skin',
        'severity': 'Low-Medium',
        'avoid_for': ['Sensitive', 'Eczema-prone']
    },
    'Denatured Alcohol': {
        'concern': 'Very drying and irritating',
        'severity': 'High',
        'avoid_for': ['Dry', 'Sensitive', 'Dehydrated', 'Acne-prone']
    }
}

USEFUL_ALIASES = {
    'Niacinamide': ['niacinamide', 'niacin', 'vitamin b3', 'nicotinamide'],
    'Hyaluronic Acid': ['hyaluronic acid', 'sodium hyaluronate', 'hyaluronate'],
    'Salicylic Acid': ['salicylic acid', 'salicylate', 'bha'],
    'Vitamin C': ['vitamin c', 'ascorbic acid', 'l-ascorbic acid', 'ascorbyl palmitate', 'magnesium ascorbyl phosphate', 'tetrahexyldecyl ascorbate'],
    'Ceramides': ['ceramide', 'ceramides', 'phytosphingosine'],
    'Retinol': ['retinol', 'retinyl palmitate', 'retinoid', 'tretinoin', 'adapalene'],
    'Benzoyl Peroxide': ['benzoyl peroxide'],
    'Alpha Arbutin': ['alpha arbutin', 'arbutin'],
    'Peptides': ['peptide', 'peptides', 'copper tripeptide', 'palmitoyl oligopeptide'],
    'Azelaic Acid': ['azelaic acid'],
    'Centella Asiatica': ['centella asiatica', 'cica', 'madecassoside', 'asiaticoside', 'asiatic acid', 'madecassic acid'],
    'Squalane': ['squalane', 'squalene']
}

HARMFUL_ALIASES = {
    'Alcohol Denat': ['alcohol denat', 'denatured alcohol', 'isopropyl alcohol', 'sd alcohol'],
    'Fragrance/Parfum': ['fragrance', 'parfum', 'perfume', 'essential oil', 'linalool', 'limonene', 'geraniol'],
    'Parabens': ['paraben', 'methylparaben', 'propylparaben', 'butylparaben', 'ethylparaben', 'isobutylparaben'],
    'Sulfates (SLS/SLES)': ['sulfate', 'sulfates', 'sls', 'sles', 'sodium lauryl sulfate', 'sodium laureth sulfate', 'ammonium lauryl sulfate'],
    'Menthol': ['menthol', 'peppermint oil'],
    'Essential Oils': ['essential oil', 'lavender oil', 'tea tree oil', 'eucalyptus oil', 'rosemary oil', 'lemon oil'],
    'Oxybenzone': ['oxybenzone', 'benzophenone-3'],
    'Octinoxate': ['octinoxate', 'ethylhexyl methoxycinnamate'],
    'Urea (high concentration)': ['urea'],
    'Lactic Acid (high concentration)': ['lactic acid'],
    'Propylene Glycol': ['propylene glycol'],
    'Denatured Alcohol': ['denatured alcohol', 'alcohol denat', 'sd alcohol']
}

def analyze_ingredient_list(ingredient_text):
    lines = [l.strip() for l in ingredient_text.replace(',', '\n').split('\n') if l.strip()]
    
    found_useful = []
    found_harmful = []

    for ingredient in lines:
        ingredient_lower = ingredient.lower()
        
        # Match useful ingredients
        for useful_key, useful_data in USEFUL_INGREDIENTS.items():
            aliases = USEFUL_ALIASES.get(useful_key, [useful_key.lower()])
            if any(alias in ingredient_lower for alias in aliases):
                found_useful.append({
                    'ingredient': useful_key,
                    'matched_as': ingredient,
                    'benefits': useful_data['benefits'],
                    'good_for': useful_data['good_for'],
                    'usage': useful_data['usage'],
                    'concentration': useful_data['concentration']
                })
                break

        # Match harmful ingredients
        for harmful_key, harmful_data in HARMFUL_INGREDIENTS.items():
            aliases = HARMFUL_ALIASES.get(harmful_key, [harmful_key.lower()])
            if any(alias in ingredient_lower for alias in aliases):
                found_harmful.append({
                    'ingredient': harmful_key,
                    'matched_as': ingredient,
                    'concern': harmful_data['concern'],
                    'severity': harmful_data['severity']
                })
                break

    skin_type_recommendation = {}
    for useful in found_useful:
        for st in useful['good_for']:
            skin_type_recommendation[st] = skin_type_recommendation.get(st, 0) + 1

    return {
        'useful_ingredients': found_useful,
        'harmful_ingredients': found_harmful,
        'has_harmful': len(found_harmful) > 0,
        'total_analyzed': len(lines),
        'useful_count': len(found_useful),
        'harmful_count': len(found_harmful)
    }

def get_ingredient_recommendations(skin_type, concerns=None):
    recommended = []
    for key, data in USEFUL_INGREDIENTS.items():
        if skin_type in data['good_for']:
            recommended.append({
                'ingredient': key,
                'benefits': data['benefits'],
                'usage': data['usage'],
                'concentration': data['concentration']
            })
        elif concerns:
            match = any(c.lower() in ' '.join(data['good_for']).lower() for c in concerns)
            if match:
                recommended.append({
                    'ingredient': key,
                    'benefits': data['benefits'],
                    'usage': data['usage'],
                    'concentration': data['concentration']
                })

    return recommended

def get_harmful_warnings(skin_type):
    warnings = []
    for key, data in HARMFUL_INGREDIENTS.items():
        if skin_type in data['avoid_for']:
            warnings.append({
                'ingredient': key,
                'concern': data['concern'],
                'severity': data['severity']
            })
    return warnings
