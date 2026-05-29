from datetime import datetime

def generate_report(disease_results, skin_analysis, skin_type_result, allergy_risks, product_recs, treatment_plan, ingredients_analysis=None):
    report = {
        'report_id': datetime.now().strftime('RPT-%Y%m%d-%H%M%S'),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sections': []
    }

    report['sections'].append({
        'title': '1. Patient Information',
        'type': 'info',
        'content': {
            'report_date': report['generated_at'],
            'report_id': report['report_id']
        }
    })

    if disease_results and len(disease_results) > 0:
        top = disease_results[0]
        disease_section = {
            'title': '2. Disease Detection Results',
            'type': 'disease',
            'content': {
                'primary_diagnosis': top.get('info', {}).get('name', top.get('label', 'Unknown')),
                'confidence': f"{top.get('confidence', 0):.1f}%",
                'description': top.get('info', {}).get('description', ''),
                'severity': top.get('info', {}).get('severity', ''),
                'medicines': top.get('info', {}).get('medicines', []),
                'prevention': top.get('info', {}).get('prevention', ''),
                'top_predictions': [
                    {
                        'name': p.get('info', {}).get('name', p.get('label', 'Unknown')),
                        'confidence': f"{p.get('confidence', 0):.1f}%"
                    }
                    for p in disease_results[:3]
                ]
            }
        }
        report['sections'].append(disease_section)

    if skin_analysis:
        analysis_section = {
            'title': '3. Skin Image Analysis',
            'type': 'analysis',
            'content': skin_analysis
        }
        report['sections'].append(analysis_section)

    if skin_type_result:
        st_section = {
            'title': '4. Skin Type Assessment',
            'type': 'skin_type',
            'content': {
                'skin_type': skin_type_result.get('skin_type', 'Unknown'),
                'confidence': f"{skin_type_result.get('confidence', 0):.1f}%",
                'analysis': skin_type_result.get('analysis', {}),
                'oily_score': skin_type_result.get('oily_score', 0),
                'dry_score': skin_type_result.get('dry_score', 0),
                'sensitive_score': skin_type_result.get('sensitive_score', 0)
            }
        }
        report['sections'].append(st_section)

    if allergy_risks and len(allergy_risks) > 0:
        allergy_section = {
            'title': '5. Allergy Risk Assessment',
            'type': 'allergy',
            'content': [
                {
                    'category': a.get('name', ''),
                    'severity': a.get('severity', ''),
                    'severity_score': a.get('severity_score', 0),
                    'symptoms': a.get('matched_symptoms', []),
                    'common_triggers': a.get('common_triggers', []),
                    'precautions': a.get('precautions', []),
                    'ingredients_to_avoid': a.get('ingredients_to_avoid', [])
                }
                for a in allergy_risks
            ]
        }
        report['sections'].append(allergy_section)

    if product_recs:
        product_section = {
            'title': '6. Product Recommendations',
            'type': 'products',
            'content': []
        }
        for cat_key, rec in product_recs.items():
            if cat_key != 'climate_advice':
                product_section['content'].append({
                    'category': rec.get('category', cat_key),
                    'recommended_products': rec.get('recommended_products', [])
                })
        if 'climate_advice' in product_recs:
            product_section['climate_advice'] = product_recs['climate_advice']
        report['sections'].append(product_section)

    if treatment_plan:
        treatment_section = {
            'title': '7. Personalized Treatment Plan',
            'type': 'treatment',
            'content': {
                'morning_routine': [
                    {'step': s[0], 'action': s[1], 'detail': s[2]}
                    for s in treatment_plan.get('morning_routine', [])
                ],
                'night_routine': [
                    {'step': s[0], 'action': s[1], 'detail': s[2]}
                    for s in treatment_plan.get('night_routine', [])
                ],
                'diet_suggestions': treatment_plan.get('diet_suggestions', []),
                'lifestyle_tips': treatment_plan.get('lifestyle_tips', [])
            }
        }
        report['sections'].append(treatment_section)

    if ingredients_analysis:
        ingredient_section = {
            'title': '8. Ingredient Intelligence',
            'type': 'ingredients',
            'content': {
                'useful_ingredients': ingredients_analysis.get('useful_ingredients', []),
                'harmful_ingredients': ingredients_analysis.get('harmful_ingredients', []),
                'total_analyzed': ingredients_analysis.get('total_analyzed', 0),
                'has_harmful': ingredients_analysis.get('has_harmful', False)
            }
        }
        report['sections'].append(ingredient_section)

    report['sections'].append({
        'title': '9. Disclaimer',
        'type': 'disclaimer',
        'content': 'This report is generated by an AI system for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Please consult a qualified dermatologist for proper medical evaluation.'
    })

    return report


def format_report_text(report):
    lines = []
    lines.append('=' * 60)
    lines.append('DERMATOLOGY ANALYSIS REPORT')
    lines.append('=' * 60)
    lines.append(f'Report ID: {report.get("report_id", "")}')
    lines.append(f'Generated: {report.get("generated_at", "")}')
    lines.append('-' * 60)
    lines.append('')

    for section in report.get('sections', []):
        lines.append(section['title'])
        lines.append('-' * 40)

        if section['type'] == 'disease':
            c = section['content']
            lines.append(f'Primary Diagnosis: {c.get("primary_diagnosis", "N/A")}')
            lines.append(f'Confidence: {c.get("confidence", "N/A")}')
            lines.append(f'Severity: {c.get("severity", "N/A")}')
            lines.append(f'Description: {c.get("description", "N/A")}')
            lines.append(f'Prevention: {c.get("prevention", "N/A")}')
            lines.append('Medicines:')
            for m in c.get('medicines', []):
                lines.append(f'  - {m}')
            lines.append('Top Predictions:')
            for p in c.get('top_predictions', []):
                lines.append(f'  - {p["name"]} ({p["confidence"]})')

        elif section['type'] == 'skin_type':
            c = section['content']
            lines.append(f'Detected Skin Type: {c.get("skin_type", "N/A")}')
            lines.append(f'Confidence: {c.get("confidence", "N/A")}')

        elif section['type'] == 'allergy':
            for item in section['content']:
                lines.append(f'Category: {item.get("category", "")} (Severity: {item.get("severity", "")})')
                lines.append(f'  Triggers: {", ".join(item.get("common_triggers", [])[:3])}')

        elif section['type'] == 'treatment':
            c = section['content']
            lines.append('Morning Routine:')
            for s in c.get('morning_routine', []):
                lines.append(f'  {s["step"]}: {s["action"]} - {s["detail"]}')
            lines.append('Night Routine:')
            for s in c.get('night_routine', []):
                lines.append(f'  {s["step"]}: {s["action"]} - {s["detail"]}')

        elif section['type'] == 'disclaimer':
            lines.append(section['content'])

        lines.append('')

    lines.append('=' * 60)
    lines.append('END OF REPORT')
    lines.append('=' * 60)

    return '\n'.join(lines)
