import os
import json
import glob
import numpy as np
import librosa
import soundfile as sf
from pedalboard import Pedalboard, Distortion, Chorus, Delay, Reverb, HighpassFilter, Limiter

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IDMT_DATASET_DIR = os.path.join(BASE_DIR, "IDMT-SMT-GUITAR_V2")
OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "dataset_pedalboard")

SAMPLE_RATE = 44100
CHUNK_DURATION = 4.0
NUM_SAMPLES_TO_GENERATE = 2000

HOP_LENGTH = 512  # Migliore risoluzione temporale per Chorus/Delay
N_FFT = 2048
N_MELS = 128


def process_and_generate():
    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

    clean_files = glob.glob(os.path.join(IDMT_DATASET_DIR, "**", "*.wav"), recursive=True)
    if not clean_files:
        raise FileNotFoundError(f"No .wav files found in {IDMT_DATASET_DIR}")

    print(f"Found {len(clean_files)} clean audio sources[cite: 1]. Generating {NUM_SAMPLES_TO_GENERATE} samples...")

    metadata = []
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)

    for i in range(NUM_SAMPLES_TO_GENERATE):
        random_clean_file = np.random.choice(clean_files)
        audio_clean, _ = librosa.load(random_clean_file, sr=SAMPLE_RATE, mono=True)

        if len(audio_clean) < chunk_samples:
            repeats = int(np.ceil(chunk_samples / len(audio_clean)))
            audio_clean = np.tile(audio_clean, repeats)[:chunk_samples]
        else:
            max_start = len(audio_clean) - chunk_samples
            start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            audio_clean = audio_clean[start_idx : start_idx + chunk_samples]

        audio_input = np.expand_dims(audio_clean, axis=0)

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

        active_effects = []
        if hp_active:
            active_effects.append(HighpassFilter(cutoff_frequency_hz=cutoff_freq))
        if dist_active:
            active_effects.append(Distortion(drive_db=drive_db))
        if chorus_active:
            active_effects.append(Chorus(
                rate_hz=chorus_rate, depth=chorus_depth, centre_delay_ms=chorus_delay,
                feedback=chorus_feedback, mix=chorus_mix
            ))
        if delay_active:
            active_effects.append(Delay(
                delay_seconds=delay_time, feedback=delay_feedback, mix=delay_mix
            ))
        if reverb_active:
            active_effects.append(Reverb(
                room_size=reverb_room, damping=reverb_damping, wet_level=reverb_wet,
                dry_level=reverb_dry, width=reverb_width
            ))
        active_effects.append(Limiter(threshold_db=-1.0))

        board = Pedalboard(active_effects)
        audio_effected = board(audio_input, SAMPLE_RATE).squeeze()

        max_val = np.max(np.abs(audio_effected))
        if max_val > 0:
            audio_effected = audio_effected / max_val * 0.90

        out_audio_name = f"sample_{i:05d}.wav"
        out_mel_name = f"sample_{i:05d}_mel.npy"

        sf.write(os.path.join(OUTPUT_DATASET_DIR, out_audio_name), audio_effected, SAMPLE_RATE)

        mel_spec = librosa.feature.melspectrogram(
            y=audio_effected, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_norm = np.clip((mel_spec_db + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)

        np.save(os.path.join(OUTPUT_DATASET_DIR, out_mel_name), mel_norm)

        metadata.append({
            "audio_file": out_audio_name,
            "mel_file": out_mel_name,
            "source_file": os.path.relpath(random_clean_file, BASE_DIR),
            "effects": {
                "highpass": {"enabled": hp_active, "cutoff_frequency_hz": cutoff_freq},
                "distortion": {"enabled": dist_active, "drive_db": drive_db},
                "chorus": {
                    "enabled": chorus_active, "rate_hz": chorus_rate, "depth": chorus_depth,
                    "centre_delay_ms": chorus_delay, "feedback": chorus_feedback, "mix": chorus_mix
                },
                "delay": {
                    "enabled": delay_active, "delay_seconds": delay_time,
                    "feedback": delay_feedback, "mix": delay_mix
                },
                "reverb": {
                    "enabled": reverb_active, "room_size": reverb_room, "damping": reverb_damping,
                    "wet_level": reverb_wet, "dry_level": reverb_dry, "width": reverb_width
                }
            }
        })

        if (i + 1) % 100 == 0 or (i + 1) == NUM_SAMPLES_TO_GENERATE:
            print(f"Progress: {i + 1}/{NUM_SAMPLES_TO_GENERATE} samples generated.")

    with open(os.path.join(OUTPUT_DATASET_DIR, "dataset_labels.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("Dataset generation complete.")


if __name__ == "__main__":
    process_and_generate()