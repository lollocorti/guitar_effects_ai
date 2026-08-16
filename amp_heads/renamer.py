import os
import re

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

# 1. Mappatura Marche
BRAND_MAP = {
    "FENDER": "FENDER",
    "FNDR": "FENDER",
    "MARSHALL": "MARSHALL",
    "MRSH": "MARSHALL",
    "TONE TRUCK": "TONE TRUCK",
    "TRCK": "TONE TRUCK",
    "BENSON": "BENSON",
    "BEN": "BENSON",
    "MESA BOOGIE": "MESA BOOGIE",
    "MESA": "MESA BOOGIE",
    "VOX": "VOX",
    "LANEY": "LANEY",
    "ORANGE": "ORANGE",
    "CARR": "CARR",
    "TONE KING": "TONE KING",
    "SILVERTONE": "SILVERTONE"
}

# 2. Mappatura Modelli
MODEL_MAP = {
    # Fender
    "TWEED DELUXE": "TWEED DELUXE",
    "TWDLX": "TWEED DELUXE",
    "DELUXE REVERB": "DELUXE REVERB",
    "BLUES JUNIOR": "BLUES JUNIOR",
    "BLUES JR": "BLUES JUNIOR",
    
    # Mesa Boogie
    "DUAL RECTIFIER MULTIWATT": "DUAL RECTIFIER MULTIWATT",
    "DUAL REC MULTIWATT": "DUAL RECTIFIER MULTIWATT",
    "DUAL RECTIFIER": "DUAL RECTIFIER",
    "DUAL REC": "DUAL RECTIFIER",
    "RECTIFIER": "RECTIFIER",
    "LONESTAR": "LONESTAR",
    
    # Vox & Marshall
    "AC30": "AC30",
    "AC50": "AC50",
    "JTM100": "JTM100",
    "JT100": "JTM100",
    "JTM45": "JTM45",
    "J45": "JTM45",
    "1959": "1959 SUPER LEAD",
    
    # Altri
    "LA30BL": "LA30BL",
    "OR30": "OR30",
    "MERCURY V": "MERCURY V",
    "IMPERIAL MKII": "IMPERIAL MKII",
    "1484 TWIN TWELVE": "1484 TWIN TWELVE",
    "1484": "1484 TWIN TWELVE",
    
    # Tone Truck & Benson
    "STEEL STRING SINGER": "STEEL STRING SINGER",
    "SSS": "STEEL STRING SINGER",
    "TRIPLE CROWN": "TRIPLE CROWN",
    "TCL": "TRIPLE CROWN",
    "OVERDRIVE SPECIAL 50": "OVERDRIVE SPECIAL 50",
    "OD50": "OVERDRIVE SPECIAL 50",
    "CHIMERA": "CHIMERA",
    "CHI": "CHIMERA"
}

# 3. Conversione Impostazioni / Canali / Gain
SETTING_CONVERSIONS = {
    "HG": "HIGH GAIN",
    "LG": "LOW GAIN",
    "MG": "MID GAIN",
    "CL": "CLEAN",
    "OD": "OVERDRIVE",
    "CR": "CRUNCH",
    "LD": "LEAD",
    "DRV": "DRIVE",
    "IN": "INPUT",
    "BR": "BRIGHT",
    "BRI": "BRIGHT",
    "MOD": "MODERN",
    "VNTG": "VINTAGE",
    "VTG": "VINTAGE"
}

# Parole inutili o residui da eliminare dai canali/settings
NOISE_TOKENS = {
    "HEAD", "RH", "BAL", "BALANCED", "DI", "CAB", "A2", "TC4", "B2", "T7", "V7", "V8",
    "SLAMMIN", "D.I", "S", "VOL", "MASTER", "AMP", "UNKNOWN", "MERA"
}

def parse_and_standardize(filename):
    name, ext = os.path.splitext(filename)
    if ext.lower() != '.nam':
        return None

    name_upper = name.upper()

    # Determina tipo di cattura
    capture_type = "DI"
    if "[CAB]" in name_upper or " CAB" in name_upper:
        capture_type = "CAB"

    # Pulizia iniziale da parentesi e separatori
    cleaned = re.sub(r'\[.*?\]', '', name_upper)
    cleaned = re.sub(r'[\-_]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # A. ESTRAZIONE MARCA
    brand_found = None
    for key in sorted(BRAND_MAP.keys(), key=len, reverse=True):
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, cleaned):
            brand_found = BRAND_MAP[key]
            cleaned = re.sub(pattern, '', cleaned)
            break

    # B. ESTRAZIONE MODELLO
    model_found = None
    for key in sorted(MODEL_MAP.keys(), key=len, reverse=True):
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, cleaned):
            model_found = MODEL_MAP[key]
            cleaned = re.sub(pattern, '', cleaned)
            break

    # Fallback se la marca è rimasta ignota ma il modello fa capo a un marchio noto
    if not brand_found and model_found:
        if model_found in ["JTM45", "JTM100", "1959 SUPER LEAD"]:
            brand_found = "MARSHALL"

    brand_str = brand_found if brand_found else "UNKNOWN"
    model_str = model_found if model_found else "AMP"

    # C. ESTRAZIONE CANALE E IMPOSTAZIONI
    tokens = [t.strip() for t in cleaned.split() if t.strip()]
    final_settings = []

    for token in tokens:
        if token in NOISE_TOKENS or re.match(r'^[A-Z]\d+$', token):
            continue

        if token in SETTING_CONVERSIONS:
            final_settings.append(SETTING_CONVERSIONS[token])
        elif token.startswith("CH") and len(token) > 2 and token[2:].isdigit():
            final_settings.append(f"CH {token[2:]}")
        else:
            final_settings.append(token)

    # Rimuove duplicati consecutivi o ridondanti mantenendo l'ordine
    unique_settings = []
    for item in final_settings:
        if item not in unique_settings:
            unique_settings.append(item)

    setting_str = " ".join(unique_settings).strip()
    if not setting_str:
        setting_str = "GENERAL"

    # Risultato rigido: MARCA - MODELLO - IMPOSTAZIONE [TIPO].NAM
    return f"{brand_str} - {model_str} - {setting_str} [{capture_type}]{ext.upper()}"

def run_rename(dry_run=False):
    for filename in os.listdir(FOLDER_PATH):
        old_path = os.path.join(FOLDER_PATH, filename)
        if os.path.isfile(old_path):
            new_name = parse_and_standardize(filename)
            if new_name and new_name != filename:
                new_path = os.path.join(FOLDER_PATH, new_name)
                print(f"RINOMINATO: {filename}")
                print(f"       -->  {new_name}\n" + "-"*60)
                if not dry_run:
                    os.rename(old_path, new_path)

if __name__ == "__main__":
    # Esegue subito la rinomina effettiva su disco
    run_rename(dry_run=False)