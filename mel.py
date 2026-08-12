import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

PARAM_RANGES = {
    "highpass_cutoff": (80.0, 350.0),
    "distortion_drive": (6.0, 30.0),
    "chorus_rate": (0.5, 2.2),
    "chorus_depth": (0.15, 0.50),
    "chorus_delay": (7.0, 20.0),
    "chorus_feedback": (0.10, 0.40),
    "chorus_mix": (0.20, 0.50),
    "delay_time": (0.18, 0.55),
    "delay_feedback": (0.15, 0.45),
    "delay_mix": (0.15, 0.45),
    "reverb_room": (0.2, 0.8),
    "reverb_damping": (0.2, 0.7),
    "reverb_wet": (0.15, 0.50),
    "reverb_dry": (0.8, 1.0),
    "reverb_width": (0.5, 1.0)
}


def normalize_val(val, min_v, max_v):
    if max_v == min_v:
        return 0.0
    return float(np.clip((val - min_v) / (max_v - min_v), 0.0, 1.0))


class GuitarDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        json_path = os.path.join(dataset_dir, "dataset_labels.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Metadata file not found: {json_path}")

        with open(json_path, "r") as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        spec_path = os.path.join(self.dataset_dir, item["mel_file"])
        mel_norm = np.load(spec_path)
        x_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)

        fx = item["effects"]
        onoff_targets = [
            float(fx["highpass"]["enabled"]),
            float(fx["distortion"]["enabled"]),
            float(fx["chorus"]["enabled"]),
            float(fx["delay"]["enabled"]),
            float(fx["reverb"]["enabled"])
        ]

        hp = fx["highpass"]
        dist = fx["distortion"]
        ch = fx["chorus"]
        dl = fx["delay"]
        rv = fx["reverb"]

        params_normalized = [
            normalize_val(hp.get("cutoff_frequency_hz", 0.0), *PARAM_RANGES["highpass_cutoff"]),
            normalize_val(dist.get("drive_db", 0.0), *PARAM_RANGES["distortion_drive"]),
            normalize_val(ch.get("rate_hz", 0.0), *PARAM_RANGES["chorus_rate"]),
            normalize_val(ch.get("depth", 0.0), *PARAM_RANGES["chorus_depth"]),
            normalize_val(ch.get("centre_delay_ms", 0.0), *PARAM_RANGES["chorus_delay"]),
            normalize_val(ch.get("feedback", 0.0), *PARAM_RANGES["chorus_feedback"]),
            normalize_val(ch.get("mix", 0.0), *PARAM_RANGES["chorus_mix"]),
            normalize_val(dl.get("delay_seconds", 0.0), *PARAM_RANGES["delay_time"]),
            normalize_val(dl.get("feedback", 0.0), *PARAM_RANGES["delay_feedback"]),
            normalize_val(dl.get("mix", 0.0), *PARAM_RANGES["delay_mix"]),
            normalize_val(rv.get("room_size", 0.0), *PARAM_RANGES["reverb_room"]),
            normalize_val(rv.get("damping", 0.0), *PARAM_RANGES["reverb_damping"]),
            normalize_val(rv.get("wet_level", 0.0), *PARAM_RANGES["reverb_wet"]),
            normalize_val(rv.get("dry_level", 0.0), *PARAM_RANGES["reverb_dry"]),
            normalize_val(rv.get("width", 0.0), *PARAM_RANGES["reverb_width"])
        ]

        return x_tensor, torch.tensor(onoff_targets, dtype=torch.float32), torch.tensor(params_normalized, dtype=torch.float32)


# --- TEST RAPIDO DEL LOADER ---
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset_pedalboard")
    
    if not os.path.exists(DATASET_DIR):
        print(f" Cartella '{DATASET_DIR}' non trovata. Genera prima il dataset con lo script 2!")
        exit()

    dataset = GuitarDataset(dataset_dir=DATASET_DIR)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Preleva il primo batch
    x, y_onoff, y_params = next(iter(dataloader))

    # Verifica le dimensioni    
    print(" DataLoader Funzionante con Successo!")
    print(f"• Shape Spettrogramma (Input): {x.shape}        -> [Batch, Channel, Mel_Bins, Time]")
    print(f"• Shape ON/OFF (Target 1):     {y_onoff.shape}   -> [Batch, 5 Pedali]")
    print(f"• Shape Manopole (Target 2):   {y_params.shape}  -> [Batch, 15 Manopole]")

    # Preleva il primo spettrogramma del batch (rimuovendo la dimensione del canale)
    # Shape iniziale: [1, 128, 173] -> Shape processata: [128, 173]
    mel_image = x[0, 0].numpy() 

    plt.figure(figsize=(10, 4))
    plt.imshow(mel_image, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(format='%+2.0f dB', label='Intensità Normalizzata')
    plt.title('Spettrogramma Mel di Esempio')
    plt.xlabel('Frame Temporali')
    plt.ylabel('Bin di Frequenza Mel')
    plt.tight_layout()
    plt.show()
