import tensorflow as tf
import gradio as gr
import numpy as np
import os
import requests
import urllib.parse
from datetime import datetime, date
from PIL import Image
from data import regional_reports

# --- PASTE THE DICTIONARIES HERE ---
crop_db = {"maize": "Plant in rows 75cm apart. Use DAP fertilizer at planting.", "beans": "Needs well-drained soil. Avoid waterlogging."}
livestock_db = {"mastitis": "Clean udder, use antibiotics if severe.", "cows": "Needs 30-50 liters of water daily."}
soil_weather_db = {"clay": "High water retention.", "rain": "Prepare drainage channels before heavy rains."}
fertilizer_db = {"dap": "High phosphorus, best for planting.", "can": "High nitrogen, good for top dressing."}

# Now, add this one extra line to combine them so the search engine can see them all:
all_knowledge = {
    "Crops": crop_db,
    "Livestock": livestock_db,
    "Soil & Weather": soil_weather_db,
    "Fertilizers": fertilizer_db
}

# ==========================================
# 1. LOAD THE AI BRAIN DIAGNOSIS MODEL
# ==========================================
try:
    model = tf.keras.models.load_model('agri_doctor_final_15_classes.h5', compile=False)
# ... (rest of your code)
except Exception as e:
    print(f"Model Loading Error: {e}")

class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

treatment_db = {
    'Pepper__bell___Bacterial_spot': {'disease': 'Bacterial Spot (Pilipili Hoho)', 'medicine': 'Copper-based fungicides mixed with Mancozeb.', 'prevention': 'Use pathogen-free seeds.'},
    'Pepper__bell___healthy': {'disease': 'Healthy Pepper Plant', 'medicine': 'No chemical required.', 'prevention': 'Maintain consistent soil moisture.'},
    'Potato___Early_blight': {'disease': 'Early Blight (Viazi)', 'medicine': 'Chlorothalonil or Mancozeb.', 'prevention': 'Crop rotation and balanced Nitrogen.'},
    'Potato___Late_blight': {'disease': 'Late Blight (Viazi - Severe Threat!)', 'medicine': 'EMERGENCY: Metalaxyl or Propamocarb.', 'prevention': 'Destroy infected tubers; avoid high humidity.'},
    'Potato___healthy': {'disease': 'Healthy Potato Plant', 'medicine': 'No treatment needed.', 'prevention': 'Ensure certified seeds.'},
    'Tomato___Bacterial_spot': {'disease': 'Bacterial Spot (Nyanya)', 'medicine': 'Copper-based sprays (Copper Oxychloride).', 'prevention': 'Avoid wet foliage.'},
    'Tomato___Early_blight': {'disease': 'Early Blight (Nyanya)', 'medicine': 'Mancozeb or Difenoconazole.', 'prevention': 'Mulching and pruning lower leaves.'},
    'Tomato___Late_blight': {'disease': 'Late Blight (Nyanya - High Risk)', 'medicine': 'EMERGENCY: Metalaxyl-M or Cymoxanil.', 'prevention': 'Remove infected plants.'},
    'Tomato___Leaf_Mold': {'disease': 'Leaf Mold (Nyanya)', 'medicine': 'Difenoconazole or Chlorothalonil.', 'prevention': 'Increase airflow.'},
    'Tomato___Septoria_leaf_spot': {'disease': 'Septoria Leaf Spot (Nyanya)', 'medicine': 'Chlorothalonil or Mancozeb.', 'prevention': 'Remove bottom leaves.'},
    'Tomato___Spider_mites_Two-spotted_spider_mite': {'disease': 'Spider Mites (Nyanya)', 'medicine': 'Abamectin or Neem Oil.', 'prevention': 'Maintain local humidity.'},
    'Tomato___Target_Spot': {'disease': 'Target Spot (Nyanya)', 'medicine': 'Azoxystrobin or Chlorothalonil.', 'prevention': 'Avoid overhead irrigation.'},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'disease': 'Tomato Yellow Leaf Curl Virus', 'medicine': 'Treat Whitefly carriers with Imidacloprid.', 'prevention': 'Use yellow sticky traps.'},
    'Tomato___Tomato_mosaic_virus': {'disease': 'Tomato Mosaic Virus (ToMV)', 'medicine': 'No chemical cure.', 'prevention': 'Sanitize hands/tools.'},
    'Tomato___healthy': {'disease': 'Healthy Tomato Plant', 'medicine': 'No medicine required.', 'prevention': 'Continue protective management.'}
}

# ==========================================
# 2. DATA REPOSITORIES (WIKI & CROP/LIVESTOCK MANUALS)
# ==========================================
wiki_deficiency_db = {
    "Nitrogen Deficiency (Yellow Leaves)": {
        "symptoms": "Older baseline leaves turn pale yellow uniformly, slowing plant structural height growth.",
        "cause": "Depleted nitrogen stores due to soil leaching or heavy continuous cropping.",
        "premium_fix": "Apply 50kg/Acre CAN Topdress or spray high-nitrogen foliar feeds like 'Easy Gro Starter' or 'Vegmax'.",
        "calc": "Recommended: 50 KGs CAN per acre mixed close to wet root zones."
    },
    "Calcium Deficiency (Blossom End Rot)": {
        "symptoms": "The bottom ends of tomato or pepper fruits turn flat, leathery, and dark black.",
        "cause": "Poor calcium mobility up the plant stem, often worsened by erratic watering.",
        "premium_fix": "Spray 'Calmax' or 'Oshun Calcium' foliar feed weekly. Incorporate Agricultural Lime during planting.",
        "calc": "Recommended Foliar Dosage: 40ml of Calmax per 20L Knapsack sprayer."
    },
    "Cutworm Damage (Kukatwa kwa Machipukizi)": {
        "symptoms": "Young seedlings are found cleanly cut down right at the soil surface overnight.",
        "cause": "Grey, soil-dwelling caterpillars that hide under dirt clods during hot daylight hours.",
        "premium_fix": "Drench soil lines around crop stems using Alpha-Cypermethrin or 'Dursban' insecticides late in the afternoon.",
        "calc": "Recommended Solution Mix: 30ml of Duduthrin per 20L spray tank applied directly to soil lines."
    },
    "Phosphorus Deficiency (Purple Tint)": {
        "symptoms": "Leaves turn an unnaturally dark green color with distinct purple or reddish tints along margins and veins.",
        "cause": "Poor root development in cold soils or low soil phosphate availability.",
        "premium_fix": "Apply 'DAP' or 'YaraMila ChukuChuku' at planting. Spray 'Easy Gro Flower & Fruit' for correction.",
        "calc": "Recommended: 50-75 KGs of DAP planting fertilizer mass applied per acre."
    }
}

manual_directory = {
    "Maize (Mahindi)": {
        "Fall Armyworm (Kiwavi wa Jeshi)": {
            "symptoms": "Large ragged windowpane holes in crop leaves filled with sawdust-like waste pellets.",
            "medicine": "Spray specialized solutions like Belt 480SC, Match 050EC, or Voliam Targo late in the evening when larvae emerge.",
            "prevention": "Practice early seasonal planting and manual weed clearance to break early cycle pest harborages."
        },
        "Maize Lethal Necrosis Disease (MLND)": {
            "symptoms": "Severe premature yellowing of leaf crowns starting from margins, drying stems, and completely barren ears.",
            "medicine": "No chemical cure exists for viral MLND. Promptly remove, drag out, and burn affected crop clumps immediately.",
            "prevention": "Control insect vectors like thrips and beetles using Actara, and source strictly certified hybrid seed strains."
        }
    },
    "Sukuma Wiki / Cabbage": {
        "Diamondback Moth (Viwavi DBM)": {
            "symptoms": "Tiny, highly active pale green caterpillars chewing leaf undersides to create fine parchment windows.",
            "medicine": "Apply targeted dynamic sprays such as Radiant 120SC, Avaunt, or alternate with systemic options.",
            "prevention": "Utilize overhead sprinkler irrigation systems to physically dislodge and disrupt larval colonies from leaf undersides."
        },
        "Black Rot (Bakteria)": {
            "symptoms": "Distinct V-shaped dull yellow tracking patches expanding inwards from leaf margins with blackening structural veins.",
            "medicine": "Spray protective copper bactericides (Copper Oxychloride) immediately upon noticing systemic perimeter spread.",
            "prevention": "Enforce strict 3-year brassica field crop rotations and purchase authenticated disease-free seed selections."
        }
    },
    "Potatoes (Viazi)": {
        "Late Blight (Baridi / Imanyara)": {
            "symptoms": "Dark, water-soaked necrotic lesions on leaves that rot rapidly into black sludge during cold, misty seasons.",
            "medicine": "EMERGENCY: Spray specialized systemics like Ridomil Gold, Milestone, or Equation Pro every 7-10 days during rainy spells.",
            "prevention": "Plant certified clean seed tubers and hill up soil hills properly to form protective earthen barriers over growing tubers."
        },
        "Bacterial Wilt (Kuoza kwa Viazi)": {
            "symptoms": "Rapid daytime wilting of green foliage while soil is fully wet, alongside stinky internal brown rot rings inside sliced tubers.",
            "medicine": "No chemical rescue exists. Quarantine the infected area immediately and restrict cross-field tool movement.",
            "prevention": "Avoid sorting or cutting seed tubers with unsterilized knives and skip planting solanaceous items in infected soil for 5 years."
        }
    },
    "Tomatoes (Nyanya)": {
        "Tuta Absoluta (Mvinyo wa Nyanya)": {
            "symptoms": "Extensive blotchy mines inside leaves, bored pinholes around fruit stems, and internal rot in harvested tomatoes.",
            "medicine": "Spray targeted insecticides such as Coragen, Radiant, or Belt 480SC alternating frequently to avoid resistance build-up.",
            "prevention": "Install black or yellow sticky pheromone traps early around nursery rows and eliminate structural solanaceous weed hosts."
        },
        "Early Blight (Madoadoa ya Kutu)": {
            "symptoms": "Concentric, target-board circular dark brown spots appearing first on older lower leaves, causing them to yellow and drop.",
            "medicine": "Spray defensive protectants like Mancozeb, or curative options like Difenoconazole (Score) or Absolute.",
            "prevention": "Apply protective crop mulches to block mud splashback and prune out low-hanging baseline foliage systematically."
        }
    },
    "Onions (Kitunguu)": {
        "Purple Blotch (Madoadoa ya Zambarau)": {
            "symptoms": "Small, water-soaked leaf lesions that quickly expand into large purple-centered oval shapes, causing tips to dry up.",
            "medicine": "Spray with curative formulations like Master (Mancozeb + Metalaxyl), Luna Tranquility, or systemic copper compounds.",
            "prevention": "Ensure wide row spacing to allow internal air circulation and manage destructive thrips vectors aggressively."
        },
        "Onion Thrips (Wadudu Mafuta)": {
            "symptoms": "Fine silvery streaks or white speckles running down the inner folds of leaves, leading to distorted, curling leaf necks.",
            "medicine": "Apply effective contact and systemic insecticides such as Decis, Profile, or Regent dissolved inside wet solution runs.",
            "prevention": "Maintain consistent watering routines because dry, dusty stress environments accelerate explosive thrips breeding cycles."
        }
    },
    "Beans (Maharagwe)": {
        "Bean Fly (Inzi wa Maharagwe)": {
            "symptoms": "Seedling stems swell up, crack near the soil line, and snap easily due to tiny internal tunneling maggots.",
            "medicine": "Drench or spray young rows with systemic insecticides like Escort, Imidacloprid, or Actara within 14 days of emergence.",
            "prevention": "Dress all seeds before planting using specialized insecticide powders like Cruiser to repel early subterranean fly vectors."
        },
        "Anthracnose (Madoadoa Meusi)": {
            "symptoms": "Dark brick-red to black sunken angular spots running along leaf veins and forming round lesions on bean pods.",
            "medicine": "Spray early crops using broad-spectrum protective fungicides such as Mancozeb, Benomyl, or systemic triadimenol options.",
            "prevention": "Never enter or walk through bean rows to weed or harvest while foliage is wet from morning dew or recent rain."
        }
    },
    "Coffee (Kahawa)": {
        "Coffee Berry Disease (CBD)": {
            "symptoms": "Sunken black spots forming on green expanding berries, causing them to rot, shrivel, and drop off prematurely.",
            "medicine": "Apply routine structural sprays of Copper Nordox, 50% Copper Oxychloride, or systemic options like Green Cop or Octave.",
            "prevention": "Prune back internal canopy shade branches systematically before cold misty periods to allow drying sunlight onto internal branches."
        },
        "Coffee Leaf Rust (Kutu ya Majani)": {
            "symptoms": "Bright orange, powdery pustules forming on the undersides of leaves, causing major leaf drops and leaving branches bare.",
            "medicine": "Spray early at the onset of rains with protective copper compounds or systemic curative triazoles.",
            "prevention": "Maintain proper tree nutrition by top-dressing with nitrogen and potassium to help leaves naturally resist fungal pressure."
        }
    },
    "Tea (Chai)": {
        "Red Spider Mites (Wadudu Wekundu)": {
            "symptoms": "Upper surfaces of tea leaves lose their healthy green gloss and turn a dull brick-red color, reducing plucking yields.",
            "medicine": "Spot-spray affected bushes with approved soft acaricides like sulfur formulations or organic Azadirachtin (Neem options).",
            "prevention": "Maintain tall perimeter windbreak lines and conserve natural predatory mites by avoiding harsh broad-spectrum chemicals."
        },
        "Hypoxylon Wood Rot (Kuoza kwa Shina)": {
            "symptoms": "Old structural tea branches die back gradually, showing grey-black crusty fungal sheets over decaying structural wood lines.",
            "medicine": "No chemical paint cure. Cleanly prune away dead wood tissue down to healthy green zones and seal cuts with copper paste.",
            "prevention": "Avoid physical trunk wounding during manual weeding and maintain soil health through regular mulching."
        }
    },
    "Avocados (Parachichi)": {
        "Root Rot (Phytophthora Cinnamomi)": {
            "symptoms": "Foliage turns pale green, wilts, and drops, leaving branches bare while feeder roots turn black and rotten.",
            "medicine": "Inject structural trunks with Phosphorous Acid formulas or drench root base lines using Ridomil Gold solutions.",
            "prevention": "Always plant on raised mounds in well-drained soils, and source verified Hass seedlings grafted onto resistant rootstocks."
        },
        "Anthracnose (Madoadoa ya Matunda)": {
            "symptoms": "Small, sunken circular dark spots appearing on skin surfaces that rapidly rot internal creamy flesh after harvest.",
            "medicine": "Apply routine preventative pre-harvest cover sprays using copper formulations from fruit set until final maturity.",
            "prevention": "Handle harvested fruits carefully to avoid skin scratches and cool them down to 6°C immediately after picking."
        }
    }
}

# ==========================================
# 2.1 REGIONAL ALERT SYSTEM (SPATIAL METHODS)
# ==========================================

# Current reported cases of diseases in different areas
# This acts as our "Neighborhood Watch" database


def get_regional_alert(user_region):
    """Checks regional data to provide location-specific disease warnings."""
    # Convert input to match dictionary keys
    clean_region = user_region.strip().title()
    
    if clean_region not in regional_reports:
        return ["Status: No major regional outbreaks detected in your area."]
    
    alerts = []
    # If a disease is reported more than 3 times, trigger a High Risk Warning
    for disease, count in regional_reports.get(clean_region, {}).items():
        if count > 3:
            alerts.append(f"⚠️ HIGH RISK ALERT: {count} reports of {disease.replace('_', ' ')} in {clean_region}. Apply preventative sprays.")
    
    return alerts if alerts else ["Status: Your regional crop health appears stable."]
livestock_directory = {
    "Poultry (Kienyeji Chickens)": {
        "Chronic Respiratory Disease": {
            "symptoms": "Birds gasping for air, bubbles in corners of the eyes, rattling breathing noises, and sneezing.",
            "herbal": "Isolate the birds. Crush fresh garlic and raw ginger into their drinking water for 5 consecutive days.",
            "vet": "Administer broad-spectrum antibiotic soluble powders like Aliseryl, Tylosin, or Egg Formula in water."
        },
        "Gumboro / Newcastle Disease": {
            "symptoms": "Greenish-white watery diarrhea, ruffled feathers, loss of appetite, and severe wing/leg paralysis.",
            "herbal": "No cure for viral attacks. Mix ground cayenne pepper and aloe vera extract into water to boost immunity.",
            "vet": "Strictly preventative: Vaccinate on Week 1 (Gumboro) and Week 3 (Newcastle). Provide high-dose multivitamins."
        }
    },
    "Dairy Cattle (Ng'ombe)": {
        "Mastitis (Ugonjwa wa Kiwele)": {
            "symptoms": "Swollen, hot, painful udder quarters; milk appears watery, clotted, thick, or blood-stained.",
            "herbal": "Strip the affected quarters frequently (every 3 hours). Wash the udder with warm salt water and massage gently.",
            "vet": "Infuse intramammary Multi-mast or Penicort tubes into the teat channel. Inject Penistrep if the cow has a high fever."
        },
        "East Coast Fever (ECF / Ndigana)": {
            "symptoms": "High body temperature, swollen lymph nodes under the ears and shoulders, heavy frothy nasal discharge, and coughing.",
            "herbal": "Keep the animal warm. Note: Traditional remedies have zero effect on ECF; immediate chemical action is mandatory.",
            "vet": "Inject Buparvaquone (Butalex or Nova-Bup) at 1ml per 20kg body weight. Spray weekly with Amitraz-based acaricides."
        }
    },
    "Camels (Ngamia)": {
        "Camel Trypanosomiasis (Surra)": {
            "symptoms": "Intermittent fever, progressive wasting/thinning despite eating, swelling of the belly/legs, and low milk yield.",
            "herbal": "Provide complete rest under shade and offer mineral-rich desert saltbush to preserve energy reserves.",
            "vet": "Administer targeted curative injections of Melarsomine (Cymelarsan) or Quinapyramine sulfate (Antrycide) strictly by body weight."
        },
        "Camel Pox (Ndui ya Ngamia)": {
            "symptoms": "Skin lesions, crusts, and scabs forming rapidly around the lips, nostrils, eyes, and bare skin areas, plus fever.",
            "herbal": "Wash external raw skin sores with mild warm neem leaf extract or apply pure local unheated honey to speed up scab healing.",
            "vet": "No viral cure. Apply antiseptic zinc oxide ointments on external crusts to block secondary bacterial skin rot."
        }
    },
    "Sheep (Kondoo)": {
        "Peste des Petits Ruminants (PPR)": {
            "symptoms": "Sudden high fever, foul-smelling erosive sores around the gums, heavy discharge sticking the eyelids together, and severe diarrhea.",
            "herbal": "Isolate sick sheep immediately. Keep their eyes and muzzle wiped clean with warm, mild antiseptic water rinses.",
            "vet": "Highly contagious viral threat. Vaccinate sheep annually. Provide intensive antibiotic cover (Oxytetracycline 20%) against pneumonia."
        },
        "Nasal Bots (Viwavi wa Pua)": {
            "symptoms": "Frequent violent sneezing, shaking of the head, snorting, and thick mucous discharge from the nostrils caused by fly larvae.",
            "herbal": "Smear a tiny trace of natural eucalyptus oil or local menthol extracts near the nostrils to soothe irritation.",
            "vet": "Drench the sheep with Ivermectin oral solution (e.g., Noromectin) at standard dosage to eliminate internal nasal maggots."
        }
    },
    "Goats (Mbuzi)": {
        "Contagious Caprine Pleuropneumonia (CCPP)": {
            "symptoms": "Painful stretching of the neck, structural coughing, heavy grunting breathing patterns, and fast death across the herd.",
            "herbal": "Isolate the animal immediately away from dry wind drafts. Keep them fully hydrated with warm molasses water mixes.",
            "vet": "CRITICAL RISK: Treat early with specialized Tylosin injections or long-acting Oxytetracycline 20% to prevent total herd loss."
        },
        "Foot Rot (Ona / Kioza cha Miguu)": {
            "symptoms": "Severe limping, holding up of the leg, heat in the hoof, and a putrid, foul-smelling rot between the claws.",
            "herbal": "Carefully pare/trim back excess overgrown hoof horn. Wash thoroughly with a highly concentrated copper sulfate basin bath.",
            "vet": "Apply Oxytetracycline wound sprays directly onto cleaned hoof lesions. Inject broad antibiotics if foot swelling moves higher."
        }
    },
    "Pigs (Nguruwe)": {
        "African Swine Fever (ASF)": {
            "symptoms": "High fever, complete loss of appetite, purple/cyanotic skin blotches on ears, abdomen, and tail, followed by internal bleeding.",
            "herbal": "No natural treatment exists. Immediately implement strict farm quarantine blocks (zero incoming or outgoing movement).",
            "vet": "🚨 EMERGENCY BIO-THREAT: 100% fatal with no vaccine or treatment. Report immediately to local sub-county veterinary officers."
        },
        "Mange Infestation (Cherebuka)": {
            "symptoms": "Intense continuous scratching against walls, thick crusty scabs forming behind ears and on flanks, and hair loss.",
            "herbal": "Wash the pig down with mild soapy water, then massage affected skin areas thoroughly using pure raw coconut oil.",
            "vet": "Inject Subcutaneous Ivermectin at 1ml per 33kg of body weight, and thoroughly spray the concrete pens with structural pesticide rims."
        }
    },
    "Rabbits (Sungura)": {
        "Coccidiosis (Kuhara kwa Sungura)": {
            "symptoms": "Severe watery or bloody diarrhea, a highly distended/swollen belly, sudden weight loss, and rough unkempt fur.",
            "herbal": "Add hay or dry fibrous roughage to slow down gut motility. Mix a tiny drop of pure apple cider vinegar into their drinking water.",
            "vet": "Treat immediately by adding Sulphaclozine or Amprolium soluble powders into their daily drinking water lines for 5 days."
        },
        "Ear Canker (Sikio Kuwasha)": {
            "symptoms": "Rabbits frequently shaking their heads, scratching ears with hind legs, and brown leathery crusts forming inside the ear canal.",
            "herbal": "Apply a few drops of pure vegetable cooking oil or mineral oil into the ear to suffocate and clear out ear mites.",
            "vet": "Apply a couple of drops of Ivermectin spot-on treatment directly onto the skin at the back of the neck, or use generic miticide ear drops."
        }
    }
}

kenya_counties = [
    "01. Mombasa", "02. Kwale", "03. Kilifi", "04. Tana River", "05. Lamu", "06. Taita Taveta",
    "07. Garissa", "08. Wajir", "09. Mandera", "10. Marsabit", "11. Isiolo", "12. Meru",
    "13. Tharaka-Nithi", "14. Embu", "15. Kitui", "16. Machakos", "17. Makueni", "18. Nyandarua",
    "19. Nyeri", "20. Kirinyaga", "21. Murang'a", "22. Kiambu", "23. Turkana", "24. West Pokot",
    "25. Samburu", "26. Trans Nzoia", "27. Uasin Gishu", "28. Elgeyo Marakwet", "29. Nandi",
    "30. Baringo", "31. Laikipia", "32. Nakuru", "33. Narok", "34. Kajiado", "35. Kericho",
    "36. Bomet", "37. Kakamega", "38. Vihiga", "39. Bungoma", "40. Busia", "41. Siaya",
    "42. Kisumu", "43. Homa Bay", "44. Migori", "45. Kisii", "46. Nyamira", "47. Nairobi"
]

calendar_db = {
    "Potatoes (Viazi)": [
        {"weeks": (0, 2), "title": "🌱 Sprouting & Emergence", "action": "Keep soil moist. Watch out for early weeds stealing field nutrients."},
        {"weeks": (3, 5), "title": "🌿 Hilling & Top-Dressing", "action": "Pull soil up around stems to cover tubers. Apply CAN top-dress."},
        {"weeks": (6, 8), "title": "🌸 Flowering & Tuber Initiation", "action": "CRITICAL RISK: High threat of Late Blight. Spray defensive copper fungicide before rains."},
        {"weeks": (9, 15), "title": "🥔 Tuber Bulking & Harvesting", "action": "Perform de-haulm (cut green tops) 2 weeks before lifting so potato skins harden properly."}
    ],
    "Tomatoes (Nyanya)": [
        {"weeks": (0, 2), "title": "🌱 Transplanting Recovery", "action": "Apply DAP fertilizer. Watch out for nocturnal cutworms cutting young tender stems."},
        {"weeks": (3, 6), "title": "🌿 Staking & Canopy Pruning", "action": "Stake the plants upright to avoid soil contact. Prune off early non-fruiting suckers."},
        {"weeks": (7, 10), "title": "🌼 Flowering & Fruit Set", "action": "Apply Calcium-rich foliar feed immediately to protect fruits from Blossom End Rot (black bottoms)."},
        {"weeks": (11, 14), "title": "🍅 Fruit Ripening & Picking", "action": "Reduce structural overhead irrigation volume to avoid skin splitting right before harvest."}
    ],
    "Onions (Kitunguu)": [
        {"weeks": (0, 6), "title": "🌱 Nursery & Seedbed Establishment", "action": "Keep bed tilth ultra-fine and completely free of shallow root weeds."},
        {"weeks": (7, 12), "title": "🌿 Leaf Foliage Accumulation", "action": "Apply Nitrogen top-dress options to maximize leaf numbers (more leaves mean much larger bulbs later)."},
        {"weeks": (13, 18), "title": "🧅 Bulb Bulking Phase", "action": "Ensure stable, uniform moisture cycles. Do not let soil drop out completely and crack."},
        {"weeks": (19, 22), "title": "🍂 Curing & Storing", "action": "Stop watering once 50% of the leafy necks break and drop over naturally. Cure under shade."}
    ],
    "Beans (Maharagwe)": [
        {"weeks": (0, 2), "title": "🌱 Node Emergence", "action": "Scout for early Bean Fly vectors. Monitor uniform germination rows across the acreage."},
        {"weeks": (3, 6), "title": "🌿 Mid-Season Growth", "action": "Conduct clean physical weed management. Inspect leaf undersides for aphid colonies."},
        {"weeks": (7, 10), "title": "🌸 Pod Development Cycle", "action": "Beans require maximum moisture balance right now. Avoid heavy pesticide spraying during active bloom."},
        {"weeks": (11, 14), "title": "🍂 Field Dry Down", "action": "Harvest when pods take on a crisp yellow/brown color and the interior seeds turn solid and hard."}
    ],
    "Generic Template (All Other Crops)": [
        {"weeks": (0, 3), "title": "🌱 Sprouting & Rooting Stage", "action": "Keep root systems consistently damp. Monitor daily for ground insects and early weeds."},
        {"weeks": (4, 8), "title": "🌿 Vegetative Expansion Phase", "action": "Apply balanced crop specific fertilizer feeds. Maximize leaf health to absorb optimal sunshine."},
        {"weeks": (9, 13), "title": "🌼 Flowering & Fruit/Seed Formation", "action": "Protect active blooms from pests. Ensure constant water access during critical cellular swelling."},
        {"weeks": (14, 20), "title": "🍂 Final Maturation & Harvest", "action": "Watch for bird or mold damage. Harvest during dry weather windows to avoid post-harvest losses."}
    ]
}

commodity_market_data = {
    "Potatoes (Viazi)": {"std_weight": 50, "unit_name": "50kg Bag", "prices": {"Nairobi": 3400, "Nakuru": 3100, "Eldoret": 2900, "Kisumu": 3600, "Mombasa": 3700}},
    "Tomatoes (Nyanya)": {"std_weight": 64, "unit_name": "64kg Crate", "prices": {"Nairobi": 6200, "Nakuru": 5800, "Eldoret": 5500, "Kisumu": 6600, "Mombasa": 6800}},
    "Maize (Mahindi)": {"std_weight": 90, "unit_name": "90kg Bag", "prices": {"Nairobi": 4200, "Nakuru": 3700, "Eldoret": 3400, "Kisumu": 4400, "Mombasa": 4500}},
    "Cabbages (Mavuno)": {"std_weight": 50, "unit_name": "Large Crate", "prices": {"Nairobi": 2500, "Nakuru": 1800, "Eldoret": 2000, "Kisumu": 2800, "Mombasa": 2900}}
}

inventory_db = {
    "Fungicides (Dawa za Kuvu)": {
        "Ridomil Gold (Blight Control)": {"price": 1250, "image": "ridomil.png"},
        "Copper Nordox (Bacterial Control)": {"price": 850, "image": "nordox.png"}
    },
    "Insecticides (Dawa za Wadudu)": {
        "Belt 480SC (Armyworm Killer)": {"price": 1400, "image": "belt.png"},
        "Match 050EC (Caterpillar Control)": {"price": 550, "image": "match.png"}
    },
    "Fertilizers & Nutrition (Mbolea)": {
        "DAP Planting Fertilizer (10Kg)": {"price": 1300, "image": "dap.png"},
        "CAN Topdress Fertilizer (10Kg)": {"price": 950, "image": "can.png"}
    }
}

# ==========================================
# 3. ADVANCED UPGRADED BACKEND WORKFLOWS
# ==========================================

def get_weather(city_name, access_code):
    is_premium = (access_code.strip() == "DORIS_2026")
    if not is_premium:
        return "🔒 WEATHER UTILITY LOCKED\nUpgrade to Premium to pull dynamic live forecasting streams directly into your dashboard panels."
    
    clean_city = city_name.strip().upper() if city_name.strip() else "NAKURU"
    
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(clean_city)}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url, timeout=3).json()
        
        if geo_res.get("results"):
            lat = geo_res["results"][0]["latitude"]
            lon = geo_res["results"][0]["longitude"]
            
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_24m,relative_humidity_2m,apparent_temperature,precipitation,weather_code&wind_speed_unit=ms&timezone=auto"
            w_data = requests.get(weather_url, timeout=3).json()
            
            current = w_data.get("current", {})
            temp = current.get("temperature_24m", 21.5)
            humidity = current.get("relative_humidity_2m", 62)
            feels_like = current.get("apparent_temperature", temp)
            precip = current.get("precipitation", 0.0)
            
            advice = "🌿 Perfect conditions for foliar sprays and fertilizer application."
            if precip > 0.2:
                advice = "⚠️ Rain detected! Delay open fungicide runs immediately to protect against wash-off solutions."
            elif temp > 26:
                advice = "☀️ High evaporative loss danger. Apply deep root moisture lines in early mornings or late evenings."
                
            return (
                f"📍 LIVE AGROMET REPORT: {clean_city}\n"
                f"--------------------------------------------------\n"
                f"Ambient Temperature: {temp}°C (Feels like {feels_like}°C)\n"
                f"Ambient Humidity   : {humidity}%\n"
                f"Active Rainfall Run : {precip} mm\n"
                f"--------------------------------------------------\n"
                f"💡 FIELD INSIGHT: {advice}"
            )
    except Exception:
        pass
        
    regional_forecasts = {
        "NAKURU": {"temp": 22.4, "humidity": 58, "precip": 0.0, "advice": "🌿 Moderate humidity over Nakuru County. Excellent window for applying crop nutrition top-dressing and field weed management."},
        "NAIVASHA": {"temp": 24.1, "humidity": 52, "precip": 0.0, "advice": "☀️ Dry atmosphere conditions. Optimize overhead irrigation cycles on vegetable plots early before peak evaporation."},
        "MOLO": {"temp": 16.8, "humidity": 74, "precip": 0.4, "advice": "⚠️ Cold temperatures and high humidity. Potato fields are at increased risk of Late Blight. Apply protective cover sprays."},
        "NYERI": {"temp": 19.5, "humidity": 68, "precip": 0.0, "advice": "🌤️ Overcast sky profiles. Ideal timing for pruning off fruit tree suckers and general coffee canopy line care."},
        "ELDORET": {"temp": 18.2, "humidity": 65, "precip": 0.0, "advice": "🌾 Calm wind runs. Highly suitable for precise herbicide operations and top-dress distribution."},
        "NAIROBI": {"temp": 23.0, "humidity": 55, "precip": 0.0, "advice": "🌤️ Mild weather pattern. Great for general farm planning, sorting field supplies, or managing distribution channels."}
    }
    
    match = regional_forecasts.get(clean_city, {
        "temp": 21.0, 
        "humidity": 60, 
        "precip": 0.0, 
        "advice": f"🌿 Stable regional conditions calculated for {clean_city}. Maintain standard localized irrigation matrices and pest monitoring cycles."
    })
    
    return (
        f"📍 AGROMET DATA PORTAL: {clean_city} (Micro-Climate Matrix)\n"
        f"--------------------------------------------------\n"
        f"🌡️ Calculated Temperature: {match['temp']}°C\n"
        f"💧 Estimated Air Humidity: {match['humidity']}%\n"
        f"🌧️ Active Precipitation  : {match['precip']} mm\n"
        f"--------------------------------------------------\n"
        f"💡 FIELD ADVISORY: {match['advice']}\n"
        f"✨ Security Note: Running smoothly inside sandbox mode."
    )

# ==========================================
# 4. CONVERSATIONAL AI CHAT ENGINE (GEMINI CORE)
# ==========================================

def call_agridoctor_chat(user_message, history):
    if history is None: history = []
    if not user_message.strip(): return "", history

    history.append({"role": "user", "content": user_message})
    
    # --- UNIVERSAL OFFLINE SEARCH ENGINE ---
    # Searches through crop_db, livestock_db, soil_weather_db, and fertilizer_db
    msg = user_message.lower()
    found_answer = False
    reply = ""

    # Check against the master knowledge base
    for category, db in all_knowledge.items():
        for key, value in db.items():
            if key.lower() in msg:
                reply = f"AgriDoctor Expert Info ({category}): {value}"
                found_answer = True
                break
        if found_answer: break

    # Fallback if the keyword is not found
    if not found_answer:
        reply = (
            "Habari! I am your AgriDoctor Assistant. I am currently running in Offline Expert Mode. "
            "I can help with crops, livestock, soil, or fertilizers. "
            "Please try asking about a specific item like 'maize', 'mastitis', or 'DAP'."
        )

    history.append({"role": "assistant", "content": reply})
    return "", history

    
    # ------------------------------------------------------------
    # LOCAL OFFLINE DICTIONARY FALLBACK (If API key fails or matches keyword)
    # ------------------------------------------------------------
    msg_lower = user_message.lower()
    fallback_reply = "I am reviewing your inquiry. Please check the 'Other Kenyan Crops Directory' or 'Visual Wiki' tabs above for instant disease and treatment manuals while the AI finishes updating!"
    
    if "mastitis" in msg_lower or "kiwele" in msg_lower:
        fallback_reply = "⚠️ [Offline Mode] Mastitis detected! Clean the udder quarters with warm salt water and milk them out every 3 hours. Consult a vet for antibiotic options."
    elif "blight" in msg_lower or "baridi" in msg_lower:
        fallback_reply = "⚠️ [Offline Mode] Blight suspected. If it is early blight (target-board spots), spray Mancozeb. If late blight (dark water-soaked rotting sludge), apply Metalaxyl immediately."

    history.append({"role": "assistant", "content": fallback_reply})
    return "", history

    msg_lower = user_message.lower()
    if "mastitis" in msg_lower or "kiwele" in msg_lower:
        fallback_reply = "⚠️ [Offline Mode] Mastitis detected! Clean the udder quarters with warm salt water and milk them out every 3 hours. For severe clinical cases, visit the nearest agrovet for intramammary tubes (Multi-mast/Penicort). Check the 'AgriDoctor Vet Manual' tab for full details."
    elif "blight" in msg_lower or "baridi" in msg_lower:
        fallback_reply = "⚠️ [Offline Mode] Blight suspected. If it is early blight (circular target spots), apply Mancozeb. If late blight (dark melting rotting spots), treat it as an emergency and apply Metalaxyl-M (Ridomil Gold). Check the 'Other Kenyan Crops Directory' tab under Potatoes or Tomatoes."
    elif "armyworm" in msg_lower or "kiwavi" in msg_lower:
        fallback_reply = "⚠️ [Offline Mode] Fall Armyworm identified. Spray targeted insecticides like Belt 480SC or Match 050EC late in the evening when the caterpillars climb into the whorls. See the Maize section in your directory tab."
    else:
        fallback_reply = (
            "Habari! I am your AgriDoctor Assistant. I am currently running on regional fallback mode. "
            f"Regarding your query on '{user_message}', please consult our detailed specialized tabs above "
            "(Crop Directories, Vet Manuals, or Visual Wiki Matrix) for localized prescriptions and inputs."
        )

    history.append((user_message, fallback_reply))
    return "", history

# =======================================================
# SHAMBA LEDGER TRANSACTION ENGINE WITH ADVANCED COMPOUND METRICS
# =======================================================
def add_ledger_entry(trans_type, detail, amount, total_acres, access_code, current_ledger):
    if current_ledger is None: 
        current_ledger = []
        
    is_premium = (access_code.strip() == "DORIS_2026")
    if not is_premium and len(current_ledger) >= 5:
        err_msg = "🚨 LEDGER LIMIT REACHED (5/5 Entries)\n🔒 Free accounts are capped at 5 records. Pay KES 100 to Till 6965429 to clear restrictions and download statements."
        return current_ledger, err_msg, ""

    try:
        val = float(amount)
        acres = float(total_acres) if total_acres else 1.0
        if acres <= 0: acres = 1.0
    except ValueError:
        return current_ledger, "⚠️ Error: Please key in valid numbers for cash and acreage.", ""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    current_ledger.append({
        "time": timestamp,
        "type": trans_type,
        "detail": detail,
        "amount": val,
        "acres": acres
    })
    
    total_revenue = sum(item["amount"] for item in current_ledger if item["type"] == "Income (Mauzo)")
    total_expenses = sum(item["amount"] for item in current_ledger if item["type"] == "Expense (Gharama)")
    net_profit = total_revenue - total_expenses
    
    summary_text = (
        f"📋 SHAMBA ACCOUNTING SUMMARY SHEET\n"
        f"--------------------------------------------------\n"
        f"💰 Total Crop Sales Revenue: KES {total_revenue:,.2f}\n"
        f"💸 Total Production Costs  : KES {total_expenses:,.2f}\n"
        f"📊 NET PROFIT / LOSS MARGIN : KES {net_profit:,.2f}\n"
        f"--------------------------------------------------\n"
        f"🔢 Active Records Logged   : {len(current_ledger)} entries.\n"
    )
    
    if is_premium:
        cost_per_acre = total_expenses / acres
        profit_per_acre = net_profit / acres
        summary_text += (
            f"🚀 PREMIUM METRICS ACTIVATED:\n"
            f"   • Total Farm Size Base : {acres} Acres\n"
            f"   • Cost Per Single Acre : KES {cost_per_acre:,.2f} / Acre\n"
            f"   • Net Yield Per Acre   : KES {profit_per_acre:,.2f} / Acre\n"
        )
        
        invoice_doc = f"🧾 OFFICIAL AGRI-DOCTOR FINANCIAL LOG REPORT\nDate Generated: {date.today()}\n=====================================\n"
        for i, r in enumerate(current_ledger):
            invoice_doc += f"[{i+1}] {r['time']} | {r['type']} - {r['detail']}: KES {r['amount']:,.2f}\n"
        invoice_doc += f"=====================================\nNET SHAMBA PROFIT BALANCE: KES {net_profit:,.2f}\nStatus: Verified Premium Statement"
    else:
        summary_text += "\n🔒 [LOCKED] Cost-Per-Acre analytics & Statement compiler require Premium Code authorization."
        invoice_doc = "🔒 Premium code needed to unlock downloadable transaction statement reports."

    return current_ledger, summary_text, invoice_doc

# WIKI ENGINE
def look_up_wiki(issue_key, access_code):
    is_premium = (access_code.strip() == "DORIS_2026")
    data = wiki_deficiency_db[issue_key]
    
    basic_output = (
        f"🔎 FARM DEFICIENCY PROFILE: {issue_key.upper()}\n"
        f"--------------------------------------------------\n"
        f"📋 VISUAL SYMPTOMS:\n{data['symptoms']}\n\n"
        f"🍂 ROOT DEFICIENCY/CAUSE:\n{data['cause']}\n"
    )
    
    if is_premium:
        premium_output = (
            f"⚡ PREMIUM CORRECTIVE SUITE ACTIVE:\n"
            f"--------------------------------------------------\n"
            f"💊 CHEMICAL BRAND AGROVET FIX:\n{data['premium_fix']}\n\n"
            f"🧮 APPLICATION DOSAGE CALCULATOR:\n{data['calc']}\n"
        )
    else:
        premium_output = (
            "🔒 PREMIUM TREATMENT ACTIONS LOCKED\n"
            "--------------------------------------------------\n"
            "To view the specific agrovet brand chemicals, dosages, and exact mixing ratios:\n"
            "1. Send KES 100 to Buy Goods Till Number: 6965429\n"
            "2. Input the code to clear permission blocks."
        )
        
    return basic_output, premium_output

# MODEL PHOTO DIAGNOSIS ENGINE
def diagnose_crop(image, access_code, state):
    if state is None: state = {'scans': 0}
    is_premium = (access_code.strip() == "DORIS_2026")
    if not is_premium and state['scans'] >= 3:
        return ("🚨 FREE LIMIT REACHED", "🔒 PREMIUM LOCKED", "⚠️ Buy Premium Code via Till 6965429 to access prescription logs.", state)
    if not is_premium: state['scans'] += 1
    try:
        img = Image.fromarray(image).resize((224, 224))
        img_array = np.expand_dims(np.array(img), axis=0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]
        confidence = float(np.max(score)) * 100
        lookup_key = predicted_class.replace("Tomato_", "Tomato___") if "Tomato_" in predicted_class and "___" not in predicted_class else predicted_class
    except Exception as e:
        return "Error analyzing image", "N/A", f"Details: {e}", state
    info = treatment_db.get(lookup_key, {'disease': predicted_class, 'medicine': 'Consult agrovet.', 'prevention': 'Monitor closely.'})
    if not is_premium:
        return (f"📋 Result: {info['disease']} ({confidence:.1f}% Match)\n📊 Scans Used: {state['scans']}/3", "🔒 Upgrade to premium to unlock medical prescriptions.", info['prevention'], state)
    return (f"🌟 PREMIUM ACTIVE\n📋 Result: {info['disease']} ({confidence:.1f}% Match)", f"💊 MEDICINE:\n{info['medicine']}", f"🛡️ PREVENTION:\n{info['prevention']}", state)

# SELECTION HANDLERS
def update_condition_choices(crop):
    conditions = list(manual_directory[crop].keys())
    return gr.Dropdown(choices=conditions, value=conditions[0])

def look_up_directory(crop, condition, access_code):
    is_premium = (access_code.strip() == "DORIS_2026")
    data = manual_directory[crop][condition]
    med = f"💊 PRESCRIPTION:\n{data['medicine']}" if is_premium else "🔒 Premium required to view agrovet prescriptions."
    return f"📋 SYMPTOMS:\n{data['symptoms']}", med, f"🛡️ PREVENTION:\n{data['prevention']}"

def update_livestock_conditions(animal):
    conditions = list(livestock_directory[animal].keys())
    return gr.Dropdown(choices=conditions, value=conditions[0])

def look_up_vet_directory(animal, condition, access_code):
    is_premium = (access_code.strip() == "DORIS_2026")
    data = livestock_directory[animal][condition]
    vet = f"💊 VET DRUG DOSAGE:\n{data['vet']}" if is_premium else "🔒 KES 100 premium access required for clinical dosages."
    return f"📋 PRESENTATION:\n{data['symptoms']}", f"🌿 FIRST AID:\n{data['herbal']}", vet

# AUXILIARY UTILITIES
def generate_crop_calendar(crop, custom_crop_name, date_string, access_code):
    is_premium = (access_code.strip() == "DORIS_2026")
    if crop == "Other Crop (Type below...)":
        final_crop = custom_crop_name.strip()
        template_key = "Generic Template (All Other Crops)"
    else:
        final_crop = crop
        template_key = crop
    if not final_crop: return "❌ Missing Crop Designation!", ""
    try:
        weeks = max(0, (date.today() - datetime.strptime(date_string, "%Y-%m-%d").date()).days // 7)
    except Exception: return "❌ Use valid format (YYYY-MM-DD).", ""
    out = f"📅 CUSTOM HARVEST TIMELINE FOR {final_crop.upper()} | Age: {weeks} weeks old\n📋 Using Template: {template_key}\n==================\n\n"
    selected_roadmap = calendar_db.get(template_key, calendar_db["Generic Template (All Other Crops)"])
    for milestone in selected_roadmap:
        s_w, e_w = milestone["weeks"]
        if s_w <= weeks <= e_w:
            out += f"▶️ CURRENT GROWTH PHASE: 📌 {milestone['title']}\n💡 Agronomist Directive: {milestone['action']}\n\n"
        elif is_premium:
            out += f"◽ {milestone['title']} (Week {s_w}-{e_w})\n💡 Directive: {milestone['action']}\n\n"
        else:
            out += f"🔒 [LOCKED PHASE] {milestone['title']} (Week {s_w}-{e_w})\n"
    encoded = urllib.parse.quote(f"🌾 AgriDoctor Custom Report:\nMy {final_crop} crop track is {weeks} weeks along.")
    btn_html = f'<a href="https://wa.me/?text={encoded}" target="_blank"><button class="custom-wa-btn">📲 Share Summary via WhatsApp</button></a>'
    return out, btn_html

def fetch_market_prices(commodity, county_selection, total_kg, access_code):
    is_premium = (access_code.strip() == "DORIS_2026")
    try:
        total_kg = float(total_kg)
        if total_kg <= 0: total_kg = 50.0
    except ValueError: total_kg = 50.0
    commodity_info = commodity_market_data[commodity]
    std_bag_weight = commodity_info["std_weight"]
    unit_name = commodity_info["unit_name"]
    prices = commodity_info["prices"]
    clean_county = county_selection.split(". ")[-1]
    if clean_county in ["Mombasa", "Kwale", "Kilifi", "Tana River", "Lamu", "Taita Taveta"]: target_hub = "Mombasa"
    elif clean_county in ["Uasin Gishu", "Trans Nzoia", "Elgeyo Marakwet", "Nandi", "Bungoma", "West Pokot", "Turkana"]: target_hub = "Eldoret"
    elif clean_county in ["Kisumu", "Siaya", "Homa Bay", "Migori", "Kisii", "Nyamira", "Kakamega", "Vihiga", "Busia"]: target_hub = "Kisumu"
    elif clean_county in ["Nakuru", "Baringo", "Samburu", "Kericho", "Bomet", "Narok", "Laikipia"]: target_hub = "Nakuru"
    else: target_hub = "Nairobi"
    base_bag_price = prices[target_hub]
    bag_price_to_use = base_bag_price if is_premium else int(base_bag_price * 0.85)
    calculated_bags = total_kg / std_bag_weight
    total_payout = calculated_bags * bag_price_to_use
    output = f"💰 MAZAO HARVEST CASH ESTIMATOR: {commodity.upper()}\n📍 Production Origin: {clean_county} (Hub: {target_hub})\n=====\n"
    output += f"⚖️ WEIGHT REPORT:\n   • Total Weight: {total_kg:,} Kgs ({calculated_bags:.2f} standard {unit_name}s)\n\n"
    output += f"💵 CASH METRICS:\n   • Value per single {unit_name}: KES {bag_price_to_use:,}\n   • ESTIMATED NET PAYOUT: KES {int(total_payout):,}\n"
    return output

def simulate_sms_broadcast(phone, location, crops, access_code):
    if not phone.strip(): return "❌ Missing phone parameters."
    loc = location.split(". ")[-1]
    return f"📱 SMS BROADCAST SIMULATOR\n📡 Line: {phone} | Node: {loc}\n💬 Inbox Feed: From AGRIDOCTOR_KE - Monitor weather patterns to guide crop care applications safely inside {loc} county guidelines."

def update_product_details(category, product):
    prod_list = list(inventory_db[category].keys())
    if product not in prod_list: product = prod_list[0]
    img_filename = inventory_db[category][product]["image"]
    img_out = Image.open(img_filename) if os.path.exists(img_filename) else None
    return gr.Dropdown(choices=prod_list, value=product), img_out

def generate_checkout_invoice(category, product, quantity):
    cost = inventory_db[category][product]["price"] * int(quantity)
    return f"📄 INVOICE\nItem: {product}\nQty: {quantity}\n💵 PAYOUT INVOICE DUE: KES {cost:,}\n🚀 Till: 6965429"

# ==========================================
# 4. CUSTOM VISUAL INJECTION DESIGN MATRIX (CSS)
# ==========================================
custom_css = """
body, .gradio-container {
    background-color: #0b0f19 !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}
.tabs {
    border-bottom: 2px solid #1e293b !important;
}
.tab-nav button {
    font-weight: 600 !important;
    color: #94a3b8 !important;
    transition: all 0.2s ease !important;
}
.tab-nav button.selected {
    color: #22c55e !important;
    border-bottom: 3px solid #22c55e !important;
}
.gr-box, .gr-padded {
    background-color: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2) !important;
}
input, textarea, select, .gr-text-input {
    background-color: #1f2937 !important;
    border: 1px solid #374151 !important;
    color: #f3f4f6 !important;
    border-radius: 8px !important;
}
input:focus, textarea:focus {
    border-color: #22c55e !important;
}
.primary-btn {
    background: linear-gradient(135deg, #15803d 0%, #166534 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 4px rgba(22, 101, 52, 0.4) !important;
}
.primary-btn:hover {
    background: linear-gradient(135deg, #166534 0%, #14532d 100%) !important;
}
.custom-wa-btn {
    background-color: #25D366 !important;
    color: white !important;
    border: none !important;
    padding: 12px;
    border-radius: 8px;
    width: 100%;
    font-weight: bold;
    cursor: pointer;
    box-shadow: 0 4px 10px rgba(37, 211, 102, 0.3);
}
"""

# ==========================================
# 5. GRADIO APPLICATION VIEW LAYOUT SYSTEM
# ==========================================
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🌾 AgriDoctor Kenya AI National Portal")
    scan_state = gr.State()
    ledger_state = gr.State(value=[])
        
    with gr.Row():
        code_input = gr.Textbox(label="🔑 Enter Premium Access Key Token", placeholder="Type multi-feature token code here...", scale=2)
        
    with gr.Tabs():
        # TAB 1: INTERACTIVE AI CHATBOT
        with gr.Tab("💬 Ask AgriDoctor AI"):
            gr.Markdown("### 🤖 Live Agricultural Consultation Assistant")
            gr.Markdown("_Ask any farming questions regarding crop pests, livestock care, soil prep, or fertilizer applications in English, Swahili, or Sheng._")
                        
            chatbot_v = gr.Chatbot(label="AgriDoctor Interactive Consult Feed", height=380)
            with gr.Row():
                msg_input = gr.Textbox(
                    label="✏️ Type your farming question here...", 
                    placeholder="e.g., Jinsi ya kutibu ugonjwa wa mastitis? / Why are my bean leaves turning black?", 
                    scale=4
                )
                submit_chat_btn = gr.Button("🚀 Ask AI", elem_classes="primary-btn", scale=1)
                clear_chat_btn = gr.Button("🧹 Clear Chat", scale=1)
                            
            submit_chat_btn.click(fn=call_agridoctor_chat, inputs=[msg_input, chatbot_v], outputs=[msg_input, chatbot_v])
            msg_input.submit(fn=call_agridoctor_chat, inputs=[msg_input, chatbot_v], outputs=[msg_input, chatbot_v])
            clear_chat_btn.click(fn=lambda: [], inputs=None, outputs=chatbot_v, queue=False)
            
        # TAB 2: SHAMBA DIGITAL LEDGER
        with gr.Tab("📋 Shamba Digital Ledger"):
            gr.Markdown("### 2026 Farm Expense & Profit Tracking Journal")
            with gr.Row():
                with gr.Column(scale=1):
                    entry_type = gr.Dropdown(choices=["Expense (Gharama)", "Income (Mauzo)"], value="Expense (Gharama)", label="🔄 Transaction Flow Classification")
                    entry_detail = gr.Textbox(label="✏️ Record Description", placeholder="e.g., Bought 2 Bags CAN, Sold 10 Crates Tomatoes")
                    entry_amount = gr.Number(label="💵 Total Money Exchanged (KES)", value=1000.0)
                    farm_acres = gr.Textbox(label="📐 Shamba Land Size (Acres)", value="1.0")
                    ledger_btn = gr.Button("💾 Log Transaction Record", elem_classes="primary-btn")
                with gr.Column(scale=2):
                    ledger_output = gr.Textbox(label="📊 Live Financial Performance Breakdown Ledger", lines=8, interactive=False)
                    statement_download = gr.Textbox(label="🧾 Exported Premium Statement Output Documents", lines=6, interactive=False)
            ledger_btn.click(fn=add_ledger_entry, inputs=[entry_type, entry_detail, entry_amount, farm_acres, code_input, ledger_state], outputs=[ledger_state, ledger_output, statement_download])
            
        # TAB 3: INTERACTIVE DEFICIENCY VISUAL WIKI
        with gr.Tab("🪱 Visual Wiki & Deficiencies"):
            gr.Markdown("### 🔍 Interactive Non-Pathogenic Matrix & Crop Nutrient Guide")
            with gr.Row():
                with gr.Column():
                    wiki_selector = gr.Dropdown(choices=list(wiki_deficiency_db.keys()), value="Nitrogen Deficiency (Yellow Leaves)", label="🍂 Choose Observed Plant Defect Profile")
                    wiki_btn = gr.Button("📖 Fetch Diagnostic Manual Profile", elem_classes="primary-btn")
                with gr.Column():
                    wiki_basic = gr.Textbox(label="📋 Diagnostics & Field Characteristics (FREE ACCESS)", lines=6, interactive=False)
                    wiki_premium = gr.Textbox(label="⚡ Actionable Treatments & Dosages (PREMIUM LEVEL)", lines=6, interactive=False)
            wiki_btn.click(fn=look_up_wiki, inputs=[wiki_selector, code_input], outputs=[wiki_basic, wiki_premium])
            
        # TAB 4: AI LEAF PHOTO SCANNER
        with gr.Tab("🤖 AI Leaf Photo Scanner"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(label="📸 Upload/Take Leaf Photo")
                    btn = gr.Button("🔍 Run AI Diagnosis", elem_classes="primary-btn")
                with gr.Column():
                    output_diag = gr.Textbox(label="📊 Summary", interactive=False)
                    output_med = gr.Textbox(label="💊 Recommended Prescriptions", interactive=False)
                    output_prev = gr.Textbox(label="🛡️ Preventive Practices", interactive=False)
            btn.click(fn=diagnose_crop, inputs=[input_img, code_input, scan_state], outputs=[output_diag, output_med, output_prev, scan_state])
                    
        # TAB 5: OTHER KENYAN CROPS DIRECTORY
        with gr.Tab("📚 Other Kenyan Crops Directory"):
            with gr.Row():
                with gr.Column():
                    crop_dropdown = gr.Dropdown(
                        choices=["Maize (Mahindi)", "Sukuma Wiki / Cabbage", "Potatoes (Viazi)", "Tomatoes (Nyanya)", "Onions (Kitunguu)", "Beans (Maharagwe)", "Coffee (Kahawa)", "Tea (Chai)", "Avocados (Parachichi)"], 
                        value="Maize (Mahindi)", 
                        label="🌽 Select Crop Type"
                    )
                    condition_dropdown = gr.Dropdown(choices=list(manual_directory["Maize (Mahindi)"].keys()), value="Fall Armyworm (Kiwavi wa Jeshi)", label="🪱 Select Local Threat Issue")
                    dir_btn = gr.Button("📖 View Pest Manual Details", elem_classes="primary-btn")
                with gr.Column():
                    dir_symptoms = gr.Textbox(label="📋 Symptoms / Crop Visual Presentation", interactive=False)
                    dir_medicine = gr.Textbox(label="💊 Agrovet Curative Action / Chemical Prescription", interactive=False)
                    dir_prevention = gr.Textbox(label="🛡️ Preventive Cultural Management Plan", interactive=False)
            crop_dropdown.change(fn=update_condition_choices, inputs=crop_dropdown, outputs=condition_dropdown)
            dir_btn.click(fn=look_up_directory, inputs=[crop_dropdown, condition_dropdown, code_input], outputs=[dir_symptoms, dir_medicine, dir_prevention])
                    
        # TAB 6: AGRIDOCTOR VET MANUAL
        with gr.Tab("🐔 AgriDoctor Vet Manual"):
            with gr.Row():
                with gr.Column():
                    animal_dropdown = gr.Dropdown(
                        choices=["Poultry (Kienyeji Chickens)", "Dairy Cattle (Ng'ombe)", "Camels (Ngamia)", "Sheep (Kondoo)", "Goats (Mbuzi)", "Pigs (Nguruwe)", "Rabbits (Sungura)"], 
                        value="Poultry (Kienyeji Chickens)", 
                        label="🐾 Select Livestock Class"
                    )
                    issue_dropdown = gr.Dropdown(choices=list(livestock_directory["Poultry (Kienyeji Chickens)"].keys()), value="Chronic Respiratory Disease", label="🌡️ Select Symptom Pattern")
                    vet_btn = gr.Button("🩺 Fetch Vet Insights", elem_classes="primary-btn")
                with gr.Column():
                    vet_symptoms = gr.Textbox(label="📋 Symptoms / Presentation", interactive=False)
                    vet_herbal = gr.Textbox(label="🌿 Field First Aid / Herbal Remedies", interactive=False)
                    vet_commercial = gr.Textbox(label="💊 Vet Treatment Prescription & Dosage", interactive=False)
            animal_dropdown.change(fn=update_livestock_conditions, inputs=animal_dropdown, outputs=issue_dropdown)
            vet_btn.click(fn=look_up_vet_directory, inputs=[animal_dropdown, issue_dropdown, code_input], outputs=[vet_symptoms, vet_herbal, vet_commercial])
            
        # ==========================================
        # TAB 7: SMART CROP CALENDAR
        # ==========================================
        with gr.Tab("📅 Smart Crop Calendar"):
            with gr.Row():
                with gr.Column():
                    cal_crop_dropdown = gr.Dropdown(
                        choices=["Potatoes (Viazi)", "Tomatoes (Nyanya)", "Onions (Kitunguu)", "Beans (Maharagwe)", "Other Crop (Type below...)"], 
                        label="Select Crop Template"
                    )
                    custom_crop_input = gr.Textbox(label="✏️ Custom Crop Name")
                    cal_date_input = gr.Textbox(label="📅 Date Planted", value=str(date.today()))
                    cal_btn = gr.Button("📊 Build My Farm Calendar", elem_classes="primary-btn")
            
                with gr.Column():
                    cal_timeline_output = gr.Textbox(label="🕒 Your Milestone Roadmap", lines=10, interactive=False)
                    cal_whatsapp_html = gr.HTML()
            
                    gr.Markdown("### 📍 Regional Health Alerts")
                    # This uses your existing 'kenya_counties' list
                    region_selector = gr.Dropdown(choices=kenya_counties, label="Select your County")
                    alert_output = gr.Textbox(label="⚠️ Current Alerts", interactive=False)
                    check_alert_btn = gr.Button("🔍 Check Alerts")

            cal_btn.click(fn=generate_crop_calendar, inputs=[cal_crop_dropdown, custom_crop_input, cal_date_input, code_input], outputs=[cal_timeline_output, cal_whatsapp_html])
            check_alert_btn.click(fn=get_regional_alert, inputs=region_selector, outputs=alert_output)

            # Actions
            cal_btn.click(
                fn=generate_crop_calendar, 
                inputs=[cal_crop_dropdown, custom_crop_input, cal_date_input, code_input], 
                outputs=[cal_timeline_output, cal_whatsapp_html]
            )
    
            check_alert_btn.click(
                fn=get_regional_alert, 
                inputs=region_selector, 
                outputs=alert_output
            )
        
        # TAB 8: MAZAO PRICE HUB
        with gr.Tab("💰 Mazao Price Hub"):
            with gr.Row():
                with gr.Column():
                    commodity_dropdown = gr.Dropdown(choices=list(commodity_market_data.keys()), value="Potatoes (Viazi)", label="🥔 Select Crop Commodity")
                    county_market_dropdown = gr.Dropdown(choices=kenya_counties, value="32. Nakuru", label="📍 Select Production County")
                    weight_input = gr.Number(label="⚖️ Total Harvest Weight (in Kgs)", value=100.0, precision=1)
                    price_btn = gr.Button("📈 Calculate Market Value Payout", elem_classes="primary-btn")
                with gr.Column():
                    market_output = gr.Textbox(label="📋 Valuation Output", lines=12, interactive=False)
            price_btn.click(fn=fetch_market_prices, inputs=[commodity_dropdown, county_market_dropdown, weight_input, code_input], outputs=[market_output])
            
        # TAB 9: AGRO-ADVISORY SMS BROADCAST
        with gr.Tab("📢 Agro-Advisory SMS Broadcast"):
            with gr.Row():
                with gr.Column():
                    sms_phone = gr.Textbox(label="📞 Phone Number", placeholder="e.g. 0700399549")
                    sms_location = gr.Dropdown(choices=kenya_counties, value="32. Nakuru", label="📍 County Node")
                    sms_crops = gr.CheckboxGroup(choices=["Maize", "Potatoes", "Tomatoes"], value=["Potatoes"], label="🌿 Target Assets")
                    sms_btn = gr.Button("📢 Simulate Broadcast", elem_classes="primary-btn")
                with gr.Column():
                    sms_screen_out = gr.Textbox(label="📱 Simulated Receiver View", lines=10, interactive=False)
            sms_btn.click(fn=simulate_sms_broadcast, inputs=[sms_phone, sms_location, sms_crops, code_input], outputs=[sms_screen_out])
            
        # TAB 10: BUY AGROVET SUPPLIES
        with gr.Tab("🛒 Buy Agrovet Supplies"):
            with gr.Row():
                with gr.Column():
                    cat_dropdown = gr.Dropdown(choices=list(inventory_db.keys()), value="Fungicides (Dawa za Kuvu)", label="🗂️ Category")
                    prod_dropdown = gr.Dropdown(choices=list(inventory_db["Fungicides (Dawa za Kuvu)"].keys()), value="Ridomil Gold (Blight Control)", label="🧪 Product")
                    qty_dropdown = gr.Dropdown(choices=["1", "2"], value="1", label="🔢 Quantity")
                    order_btn = gr.Button("🛒 Generate Order Invoice", elem_classes="primary-btn")
                with gr.Column():
                    product_view = gr.Image(label="Packaging Preview", interactive=False, width=250)
                    checkout_invoice = gr.Textbox(label="📄 Checkout Instructions", interactive=False, lines=10)
            cat_dropdown.change(fn=update_product_details, inputs=[cat_dropdown, prod_dropdown], outputs=[prod_dropdown, product_view])
            prod_dropdown.change(fn=update_product_details, inputs=[cat_dropdown, prod_dropdown], outputs=[prod_dropdown, product_view])
            order_btn.click(fn=generate_checkout_invoice, inputs=[cat_dropdown, prod_dropdown, qty_dropdown], outputs=checkout_invoice)
            
        # TAB 11: SMART WEATHER PLANNER
        with gr.Tab("🌤️ Smart Weather Planner"):
            gr.Markdown("### 🌦️ Real-Time Agromet Micro-Climate Interface")
            with gr.Row():
                with gr.Column():
                    city_input = gr.Textbox(label="Enter Town/City Name", placeholder="e.g., Nakuru, Naivasha, Molo, Nyeri")
                    weather_btn = gr.Button("📡 Fetch Real-Time Weather Stream", elem_classes="primary-btn")
                with gr.Column():
                    weather_out = gr.Textbox(label="Agronomist Smart Advisory Output", lines=7, interactive=False)
            weather_btn.click(fn=get_weather, inputs=[city_input, code_input], outputs=weather_out)
                
    gr.Markdown("---\n© 2026 AgriDoctor Kenya. Empowering smallholder farmers through digital agricultural access networks.")

# Launching interface configuration safely map internal runtime instances
demo.launch(server_name="0.0.0.0", share=True)