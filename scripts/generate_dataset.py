import glob
import json
import os
import re
import sys
import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve

from nam.models import init_from_nam
from pedalboard import (
    Chorus,
    Delay,
    Distortion,
    HighpassFilter,
    Limiter,
    Pedalboard,
    Reverb,
)

# --- Percorsi dei File ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

IDMT_DATASET_DIR = os.path.join(PROJECT_ROOT, "IDMT-SMT-GUITAR_V2")
AMP_HEADS_DIR = os.path.join(PROJECT_ROOT, "amp_heads")
IR_DIR = os.path.join(PROJECT_ROOT, "ir")
OUTPUT_DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

SAMPLE_RATE = 48000
CHUNK_DURATION = 4.0
NUM_SAMPLES_TO_GENERATE = 5000


def parse_amp_metadata(nam_path: str) -> dict:
    """Estrae Marca, Modello, Impostazione e Tipo di cattura dal nome del file .nam standardizzato."""
    if not nam_path:
        return None

    filename = os.path.splitext(os.path.basename(nam_path))[0]
    
    # Formato standard: MARCA - MODELLO - IMPOSTAZIONE [TIPO]
    capture_type = "DI"
    match_type = re.search(r'\[(.*?)\]', filename)
    if match_type:
        capture_type = match_type.group(1)
        filename = re.sub(r'\[.*?\]', '', filename).strip()

    parts = [p.strip() for p in filename.split(" - ") if p.strip()]

    brand = parts[0] if len(parts) > 0 else "UNKNOWN"
    model = parts[1] if len(parts) > 1 else "AMP"
    setting = parts[2] if len(parts) > 2 else "GENERAL"

    return {
        "raw_file": os.path.relpath(nam_path, PROJECT_ROOT),
        "brand": brand,
        "model": model,
        "setting": setting,
        "capture_type": capture_type
    }


def parse_ir_metadata(ir_path: str) -> dict:
    """Estrae Marca, Cabinet/Cono e Microfono/Note dal nome dell'IR standardizzato."""
    if not ir_path:
        return None

    filename = os.path.splitext(os.path.basename(ir_path))[0]

    # Formato standard: MARCA - CABINET E CONI [MICROFONI O NOTE]
    mic_or_notes = "STD"
    match_mic = re.search(r'\[(.*?)\]', filename)
    if match_mic:
        mic_or_notes = match_mic.group(1)
        filename = re.sub(r'\[.*?\]', '', filename).strip()

    parts = [p.strip() for p in filename.split(" - ") if p.strip()]

    brand = parts[0] if len(parts) > 0 else "GENERIC"
    cabinet = parts[1] if len(parts) > 1 else "CAB"

    return {
        "raw_file": os.path.relpath(ir_path, PROJECT_ROOT),
        "brand": brand,
        "cabinet": cabinet,
        "microphone_notes": mic_or_notes
    }


def apply_ir_convolution(audio: np.ndarray, ir_path: str) -> np.ndarray:
    """Applica l'IR della cassa/microfono tramite convoluzione FFT."""
    try:
        ir_audio, ir_sr = sf.read(ir_path)
        if ir_audio.ndim > 1:
            ir_audio = np.mean(ir_audio, axis=1)

        convolved = fftconvolve(audio, ir_audio, mode="full")[: len(audio)]
        return convolved
    except Exception as e:
        print(f"\n[WARNING] Impossibile leggere il file IR '{os.path.basename(ir_path)}': {e}")
        return audio


def patch_nam_config_for_legacy(config: dict) -> dict:
    """Normalizza la struttura del file .nam per garantire la compatibilità."""
    arch = config.get("architecture")

    if arch == "SlimmableContainer":
        submodels = config.get("config", {}).get("submodels", [])
        if not submodels:
            raise ValueError("Modello SlimmableContainer privo di 'submodels'.")
        
        submodel_entry = submodels[0]
        inner_model = submodel_entry.get("model", {})
        
        weights = (
            inner_model.get("weights") 
            or submodel_entry.get("weights") 
            or config.get("weights", [])
        )

        if not weights:
            raise ValueError("Array dei pesi vuoto o non trovato nel modello SlimmableContainer.")

        config = {
            "version": config.get("version", inner_model.get("version", "0.7.0")),
            "architecture": inner_model.get("architecture", "WaveNet"),
            "config": inner_model.get("config", {}),
            "weights": weights
        }
        arch = config.get("architecture")

    if arch == "WaveNet":
        net_config = config.get("config", {})
        layers = net_config.get("layers", [])
        
        for layer in layers:
            if "head" not in layer or layer["head"] is None:
                layer["head"] = {
                    "out_channels": layer.get("head_size", 1),
                    "kernel_size": 1,
                    "bias": layer.get("head_bias", True)
                }

    return config


def process_nam_native(audio_data: np.ndarray, nam_path: str) -> np.ndarray:
    """Carica ed esegue l'inferenza di un modello .nam."""
    try:
        with open(nam_path, "r", encoding="utf-8") as fp:
            config = json.load(fp)

        config = patch_nam_config_for_legacy(config)

        model = init_from_nam(config)
        model.eval()

        input_tensor = torch.from_numpy(audio_data).float()

        with torch.no_grad():
            output_tensor = model(input_tensor)

        if isinstance(output_tensor, torch.Tensor):
            output_data = output_tensor.cpu().numpy().squeeze()
        else:
            output_data = np.squeeze(output_tensor)

        if output_data.shape != audio_data.shape:
            output_data = np.pad(output_data, (0, max(0, len(audio_data) - len(output_data))))[:len(audio_data)]

        return output_data

    except Exception as e:
        print(f"\n[WARNING] Errore durante l'inferenza NAM con '{os.path.basename(nam_path)}': {e}")
        return audio_data


def process_and_generate():
    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

    clean_files = glob.glob(os.path.join(IDMT_DATASET_DIR, "**", "*.wav"), recursive=True)
    if not clean_files:
        raise FileNotFoundError(f"Nessun file .wav trovato in {IDMT_DATASET_DIR}")

    nam_files = glob.glob(os.path.join(AMP_HEADS_DIR, "**", "*.nam"), recursive=True)
    ir_files = glob.glob(os.path.join(IR_DIR, "**", "*.wav"), recursive=True)

    print(f"Sorgenti audio 'clean' trovate: {len(clean_files)}")
    print(f"File IR trovati: {len(ir_files)}")
    print(f"Modelli NAM totali trovati su disco: {len(nam_files)}")

    metadata = []
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)

    for i in range(NUM_SAMPLES_TO_GENERATE):
        audio_clean = None
        random_clean_file = None
        while audio_clean is None:
            random_clean_file = np.random.choice(clean_files)
            try:
                audio_clean, sr = sf.read(random_clean_file)
            except Exception as e:
                print(f"\n[WARNING] Impossibile leggere '{os.path.basename(random_clean_file)}': {e}")

        if audio_clean.ndim > 1:
            audio_clean = np.mean(audio_clean, axis=1)

        if len(audio_clean) < chunk_samples:
            repeats = int(np.ceil(chunk_samples / len(audio_clean)))
            audio_clean = np.tile(audio_clean, repeats)[:chunk_samples]
        else:
            max_start = len(audio_clean) - chunk_samples
            start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            audio_clean = audio_clean[start_idx : start_idx + chunk_samples]

        # Parametri casuali degli effetti
        hp_active = bool(np.random.rand() < 0.50)
        dist_active = bool(np.random.rand() < 0.50)
        chorus_active = bool(np.random.rand() < 0.50)
        delay_active = bool(np.random.rand() < 0.50)
        reverb_active = bool(np.random.rand() < 0.50)

        cutoff_freq = float(np.random.uniform(80.0, 350.0)) if hp_active else 0.0
        drive_db = float(np.random.uniform(6.0, 30.0)) if dist_active else 0.0

        chorus_rate = float(np.random.uniform(0.5, 2.2)) if chorus_active else 0.0
        chorus_depth = float(np.random.uniform(0.15, 0.50)) if chorus_active else 0.0
        chorus_delay = float(np.random.uniform(7.0, 20.0)) if chorus_active else 0.0
        chorus_feedback = float(np.random.uniform(0.10, 0.40)) if chorus_active else 0.0
        chorus_mix = float(np.random.uniform(0.20, 0.50)) if chorus_active else 0.0

        delay_time = float(np.random.uniform(0.18, 0.55)) if delay_active else 0.0
        delay_feedback = float(np.random.uniform(0.15, 0.45)) if delay_active else 0.0
        delay_mix = float(np.random.uniform(0.15, 0.45)) if delay_active else 0.0

        reverb_room = float(np.random.uniform(0.2, 0.8)) if reverb_active else 0.0
        reverb_damping = float(np.random.uniform(0.2, 0.7)) if reverb_active else 0.0
        reverb_wet = float(np.random.uniform(0.15, 0.50)) if reverb_active else 0.0
        reverb_dry = float(np.random.uniform(0.8, 1.0)) if reverb_active else 0.0
        reverb_width = float(np.random.uniform(0.5, 1.0)) if reverb_active else 0.0

        audio_processing = audio_clean.copy()

        # 1. PRE-EFFETTI
        pre_effects = []
        if hp_active:
            pre_effects.append(HighpassFilter(cutoff_frequency_hz=cutoff_freq))
        if dist_active:
            pre_effects.append(Distortion(drive_db=drive_db))

        if pre_effects:
            board_pre = Pedalboard(pre_effects)
            audio_processing = board_pre(
                np.expand_dims(audio_processing, axis=0), SAMPLE_RATE
            ).squeeze()

        # 2. AMPLIFICATORE NAM
        selected_nam_file = None
        if nam_files:
            selected_nam_file = np.random.choice(nam_files)
            audio_processing = process_nam_native(audio_processing, selected_nam_file)

        # 3. IR CAB / MIC CONVOLUTION
        selected_ir_file = None
        if ir_files:
            selected_ir_file = np.random.choice(ir_files)
            audio_processing = apply_ir_convolution(audio_processing, selected_ir_file)

        # 4. POST-EFFETTI
        post_effects = []
        if chorus_active:
            post_effects.append(
                Chorus(
                    rate_hz=chorus_rate,
                    depth=chorus_depth,
                    centre_delay_ms=chorus_delay,
                    feedback=chorus_feedback,
                    mix=chorus_mix,
                )
            )
        if delay_active:
            post_effects.append(
                Delay(
                    delay_seconds=delay_time,
                    feedback=delay_feedback,
                    mix=delay_mix,
                )
            )
        if reverb_active:
            post_effects.append(
                Reverb(
                    room_size=reverb_room,
                    damping=reverb_damping,
                    wet_level=reverb_wet,
                    dry_level=reverb_dry,
                    width=reverb_width,
                )
            )
        post_effects.append(Limiter(threshold_db=-1.0))

        board_post = Pedalboard(post_effects)
        audio_final = board_post(
            np.expand_dims(audio_processing, axis=0), SAMPLE_RATE
        ).squeeze()

        max_val = np.max(np.abs(audio_final))
        if max_val > 0:
            audio_final = audio_final / max_val * 0.90

        out_audio_name = f"sample_{i:05d}.wav"
        sf.write(
            os.path.join(OUTPUT_DATASET_DIR, out_audio_name),
            audio_final,
            SAMPLE_RATE,
        )

        # 5. ESTRAZIONE E SALVATAGGIO LABELS STRUTTURATE
        amp_info = parse_amp_metadata(selected_nam_file)
        ir_info = parse_ir_metadata(selected_ir_file)

        metadata.append(
            {
                "audio_file": out_audio_name,
                "source_file": os.path.relpath(random_clean_file, PROJECT_ROOT),
                "amplifier": amp_info,
                "cabinet_ir": ir_info,
                "effects": {
                    "highpass": {
                        "enabled": hp_active,
                        "cutoff_frequency_hz": cutoff_freq,
                    },
                    "distortion": {
                        "enabled": dist_active,
                        "drive_db": drive_db,
                    },
                    "chorus": {
                        "enabled": chorus_active,
                        "rate_hz": chorus_rate,
                        "depth": chorus_depth,
                        "centre_delay_ms": chorus_delay,
                        "feedback": chorus_feedback,
                        "mix": chorus_mix,
                    },
                    "delay": {
                        "enabled": delay_active,
                        "delay_seconds": delay_time,
                        "feedback": delay_feedback,
                        "mix": delay_mix,
                    },
                    "reverb": {
                        "enabled": reverb_active,
                        "room_size": reverb_room,
                        "damping": reverb_damping,
                        "wet_level": reverb_wet,
                        "dry_level": reverb_dry,
                        "width": reverb_width,
                    },
                },
            }
        )

        if (i + 1) % 100 == 0 or (i + 1) == NUM_SAMPLES_TO_GENERATE:
            print(f"Avanzamento: {i + 1}/{NUM_SAMPLES_TO_GENERATE} campioni generati.")

    with open(
        os.path.join(OUTPUT_DATASET_DIR, "dataset_labels.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=4)

    print("\nGenerazione del dataset completata con successo!")


if __name__ == "__main__":
    process_and_generate()