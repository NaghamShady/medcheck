"""
Common brand / layman names → interaction-dataset generic names.

Keys and values must be normalize_drug_name()-style (lowercase, spaced).
Keep this list growing — fuzzy matching alone is unreliable for brands.
"""

from __future__ import annotations

# brand / synonym → preferred dataset generic
BRAND_ALIASES = {
    # Pain / fever / NSAIDs
    "aspirin": "acetylsalicylic acid",
    "acetylsalicylic acid": "acetylsalicylic acid",
    "paracetamol": "acetaminophen",
    "acetaminophen": "acetaminophen",
    "tylenol": "acetaminophen",
    "panadol": "acetaminophen",
    "calpol": "acetaminophen",
    "fevadol": "acetaminophen",
    "adol": "acetaminophen",
    "crocin": "acetaminophen",
    "doliprane": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "brufen": "ibuprofen",
    "nurofen": "ibuprofen",
    "profinal": "ibuprofen",
    "nsaid": "ibuprofen",
    "voltarin": "diclofenac",
    "voltaren": "diclofenac",
    "cataflam": "diclofenac",
    "aleve": "naproxen",
    "naprosyn": "naproxen",

    # Antiplatelet / anticoagulant
    "plavix": "clopidogrel",
    "coumadin": "warfarin",
    "marevan": "warfarin",
    "heparin": "heparin",
    "clexane": "enoxaparin",
    "lovenox": "enoxaparin",
    "eliquis": "apixaban",
    "xarelto": "rivaroxaban",
    "pradaxa": "dabigatran",

    # Statins / lipids
    "lipitor": "atorvastatin",
    "crestor": "rosuvastatin",
    "zocor": "simvastatin",
    "pravachol": "pravastatin",
    "lescol": "fluvastatin",

    # Blood pressure / heart
    "norvasc": "amlodipine",
    "istin": "amlodipine",
    "amlor": "amlodipine",
    "amlovas": "amlodipine",
    "amlong": "amlodipine",
    "stamlo": "amlodipine",
    "atenolol": "atenolol",
    "tenormin": "atenolol",
    "concor": "bisoprolol",
    "zabesta": "bisoprolol",
    "lopressor": "metoprolol",
    "betaloc": "metoprolol",
    "cozaar": "losartan",
    "hyzaar": "losartan",
    "diovan": "valsartan",
    "coversyl": "perindopril",
    "tritace": "ramipril",
    "altace": "ramipril",
    "capoten": "captopril",
    "zestril": "lisinopril",
    "prinivil": "lisinopril",
    "vasotec": "enalapril",
    "cardura": "doxazosin",
    "inderal": "propranolol",

    # Diuretics
    "lasix": "furosemide",
    "frusemide": "furosemide",
    "frusamide": "furosemide",
    "furosemide": "furosemide",
    "aldactone": "spironolactone",
    "hydrochlorothiazide": "hydrochlorothiazide",
    "hctz": "hydrochlorothiazide",
    "microzide": "hydrochlorothiazide",

    # Diabetes
    "glucophage": "metformin",
    "glucophage xr": "metformin",
    "diaformin": "metformin",
    "amaryl": "glimepiride",
    "daonil": "glibenclamide",
    "glyburide": "glibenclamide",
    "januvia": "sitagliptin",
    "galvus": "vildagliptin",
    "jardiance": "empagliflozin",
    "forxiga": "dapagliflozin",
    "victoza": "liraglutide",
    "ozempic": "semaglutide",
    "lantus": "insulin glargine",
    "humalog": "insulin lispro",
    "novorapid": "insulin aspart",

    # GI
    "prilosec": "omeprazole",
    "losec": "omeprazole",
    "nexium": "esomeprazole",
    "pantoloc": "pantoprazole",
    "controloc": "pantoprazole",
    "protonix": "pantoprazole",
    "zantac": "ranitidine",
    "pepcid": "famotidine",
    "motilium": "domperidone",
    "zofran": "ondansetron",

    # Antibiotics / infection
    "augmentin": "amoxicillin",
    "amoxil": "amoxicillin",
    "zithromax": "azithromycin",
    "zithro": "azithromycin",
    "cipro": "ciprofloxacin",
    "flagyl": "metronidazole",
    "bactrim": "sulfamethoxazole",
    "septra": "sulfamethoxazole",

    # Respiratory
    "ventolin": "salbutamol",
    "proair": "salbutamol",
    "seretide": "fluticasone",
    "symbicort": "budesonide",
    "singulair": "montelukast",
    "claritin": "loratadine",
    "zyrtec": "cetirizine",
    "telfast": "fexofenadine",
    "allegra": "fexofenadine",

    # Thyroid / hormones
    "synthroid": "levothyroxine",
    "eltroxin": "levothyroxine",
    "euthyrox": "levothyroxine",

    # Psych / neuro
    "prozac": "fluoxetine",
    "zoloft": "sertraline",
    "cipralex": "escitalopram",
    "lexapro": "escitalopram",
    "xanax": "alprazolam",
    "valium": "diazepam",
    "ativan": "lorazepam",
    "ambien": "zolpidem",
    "stilnox": "zolpidem",

    # Vitamins / supplements
    "vitamin c": "ascorbic acid",
    "vit c": "ascorbic acid",
    "vitaminc": "ascorbic acid",
    "ascorbic acid": "ascorbic acid",
    "ascorbate": "ascorbic acid",
    # Interaction CSV uses chemical names, not "Vitamin D"
    "vitamin d": "cholecalciferol",
    "vit d": "cholecalciferol",
    "vitd": "cholecalciferol",
    "vitamin d3": "cholecalciferol",
    "vit d3": "cholecalciferol",
    "vitamin d2": "ergocalciferol",
    "vit d2": "ergocalciferol",
    "cholecalciferol": "cholecalciferol",
    "ergocalciferol": "ergocalciferol",
}
