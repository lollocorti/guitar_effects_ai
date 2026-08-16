import os
import re

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

BRAND_MAP = {
    "AMPEG": "AMPEG",
    "ASHDOWN": "ASHDOWN",
    "FENDER": "FENDER",
    "FRAMUS": "FRAMUS",
    "MARSHALL": "MARSHALL",
    "MESA BOOGIE": "MESA BOOGIE",
    "MESA": "MESA BOOGIE",
    "ORANGE": "ORANGE",
    "PEAVEY": "PEAVEY"
}

SPEAKER_MAP = {
    "V30": "V30",
    "T75": "G12T75",
    "G12T75": "G12T75",
    "G12M GREENBACK": "G12M GREENBACK",
    "G12M": "G12M GREENBACK",
    "JENSEN P10R": "JENSEN P10R",
    "JENP10R": "JENSEN P10R",
    "BLACK WIDOW": "BLACK WIDOW",
    "BW": "BLACK WIDOW"
}

MIC_MAP = {
    "SE4400A": "SE4400A",
    "BD300": "BD300",
    "SM57": "SM57",
    "57": "SM57",
    "R121": "R121",
    "121": "R121",
    "MD421": "MD421",
    "421": "MD421",
    "D11L": "D11L"
}

# Parole/Codici spazzatura da RIMUOVERE
NOISE_TOKENS = {"DB", "PLUS", "AND", "+", "STD", "AMP", "UNKNOWN"}

def parse_and_standardize_ir(filename):
    name, ext = os.path.splitext(filename)
    if ext.lower() not in ['.wav', '.flac']:
        return None

    name_upper = name.upper()
    cleaned = re.sub(r'[\-_]+', ' ', name_upper)

    # 1. RILEVAMENTO MARCHE (SINGOLA O MIX)
    detected_brands = []
    for key in sorted(BRAND_MAP.keys(), key=len, reverse=True):
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, cleaned):
            brand_val = BRAND_MAP[key]
            if brand_val not in detected_brands:
                detected_brands.append(brand_val)

    is_mix = len(detected_brands) > 1

    if is_mix:
        brand_str = "MIX"
    else:
        brand_str = detected_brands[0] if detected_brands else "GENERIC"
        # Rimuove il marchio dal corpo se non è un mix
        for key in BRAND_MAP.keys():
            pattern = r'\b' + re.escape(key) + r'\b'
            cleaned = re.sub(pattern, '', cleaned)

    # 2. MICROFONI
    mics_found = []
    for key, val in MIC_MAP.items():
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, cleaned):
            if val not in mics_found:
                mics_found.append(val)
            cleaned = re.sub(pattern, '', cleaned)

    # 3. SPEAKER / CONI
    speakers_found = []
    for key, val in SPEAKER_MAP.items():
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, cleaned):
            if val not in speakers_found:
                speakers_found.append(val)
            cleaned = re.sub(pattern, '', cleaned)

    # 4. PULIZIA TOKEN E CLASSIFICAZIONE NOTE VS CAB
    cleaned = re.sub(r'[()\[\]]', ' ', cleaned)
    tokens = [t.strip() for t in cleaned.split() if t.strip()]

    cab_tokens = []
    notes_tokens = []

    for t in tokens:
        if t in ["CLEAN", "EQ'D"]:
            if t not in notes_tokens:
                notes_tokens.append(t)
        elif t == "OS":
            # OS = Oversized, va con le caratteristiche della cassa
            if "OS" not in cab_tokens:
                cab_tokens.append("OS")
        elif t not in NOISE_TOKENS:
            if t not in cab_tokens:
                cab_tokens.append(t)

    # Ricostruzione Cabinet
    cab_str = " ".join(cab_tokens).strip()

    # Aggiunta speaker se assenti nella stringa cabinet
    for spk in speakers_found:
        if spk not in cab_str:
            cab_str += f" {spk}"

    cab_str = re.sub(r'\s+', ' ', cab_str).strip()

    # Gestione Tag Mix
    if is_mix and "BLEND" not in notes_tokens:
        notes_tokens.insert(0, "BLEND")

    # Ricostruzione Blocco Parentesi [MIC / NOTE]
    mic_str = " + ".join(mics_found)
    if notes_tokens:
        mic_str = f"{mic_str} {' '.join(notes_tokens)}".strip()

    # Assegna [STD] SOLO se non ci sono microfoni né note particolari
    if not mic_str:
        mic_str = "[STD]"
    else:
        mic_str = f"[{mic_str}]"

    cab_str = cab_str if cab_str else "CAB"

    return f"{brand_str} - {cab_str} {mic_str}{ext.upper()}"

def run_rename(dry_run=False):
    for filename in os.listdir(FOLDER_PATH):
        old_path = os.path.join(FOLDER_PATH, filename)
        if os.path.isfile(old_path):
            new_name = parse_and_standardize_ir(filename)
            if new_name and new_name != filename:
                new_path = os.path.join(FOLDER_PATH, new_name)
                print(f"RINOMINATO: {filename}")
                print(f"       -->  {new_name}\n" + "-"*60)
                if not dry_run:
                    os.rename(old_path, new_path)

if __name__ == "__main__":
    run_rename(dry_run=False)