import os
import torch
import numpy as np
import librosa
import torchaudio
import torchaudio.transforms as T
import gradio as gr
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import io
from PIL import Image

from model import GuitarEffectsNet
from mel import PARAM_RANGES

# 1. CARICAMENTO MODELLO
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "pod_go_model.pth"

model = GuitarEffectsNet().to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Modello caricato con successo")

def denormalize(val, min_v, max_v):
    return val * (max_v - min_v) + min_v

def compute_mel_spectrogram(waveform, sr=44100):
    """ Calcola lo spettrogramma Mel normalizzato concordamente al dataset """
    mel_transform = T.MelSpectrogram(
        sample_rate=sr, n_fft=2048, hop_length=512, n_mels=128
    )
    mel_spec = mel_transform(waveform)
    mel_db = T.AmplitudeToDB()(mel_spec)
    # Normalizzazione [0, 1] coerente con il dataset di training
    mel_norm = torch.clamp((mel_db + 80.0) / 80.0, 0.0, 1.0)
    return mel_norm.squeeze(0).cpu().numpy()

def process_and_predict(audio_path):
    if audio_path is None or not os.path.exists(MODEL_PATH):
        return None, "0%", "Carica un audio o addestra prima il modello (.pth)!"

    # A. Caricamento Audio
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample se necessario
    if sr != 44100:
        resampler = T.Resample(sr, 44100)
        waveform = resampler(waveform)
        sr = 44100

    # B. Estrazione Feature
    mel_img = compute_mel_spectrogram(waveform, sr)
    x_input = torch.tensor(mel_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # C. Inference IA
    with torch.no_grad():
        logits_onoff, raw_params = model(x_input)
        probs_onoff = torch.sigmoid(logits_onoff).squeeze(0).cpu().numpy()
        params = raw_params.squeeze(0).cpu().numpy()

    # D. Soglia di Attivazione (Thresholding a 0.5)
    states = (probs_onoff > 0.5)

    # E. Denormalizzazione dei Parametri Predetti
    p_hp = denormalize(params[0], *PARAM_RANGES["highpass_cutoff"])
    p_dist = denormalize(params[1], *PARAM_RANGES["distortion_drive"])
    p_ch_rate = denormalize(params[2], *PARAM_RANGES["chorus_rate"])
    p_ch_depth = denormalize(params[3], *PARAM_RANGES["chorus_depth"])
    p_dl_time = denormalize(params[7], *PARAM_RANGES["delay_time"])
    p_dl_mix = denormalize(params[9], *PARAM_RANGES["delay_mix"])
    p_rv_room = denormalize(params[10], *PARAM_RANGES["reverb_room"])

    # F. Generazione Report Markdown
    report_text = f"""
    ### Effetti e Parametri Predetti dal Modello
    
    | Effetto | Stato Predetto | Probabilità | Parametro Relevante |
    | :--- | :--- | :--- | :--- |
    | **Highpass Filter** | {"🟢 ON" if states[0] else "🔴 BYPASS"} | {probs_onoff[0]*100:.1f}% | Cutoff: {p_hp:.1f} Hz |
    | **Distortion** | {"🟢 ON" if states[1] else "🔴 BYPASS"} | {probs_onoff[1]*100:.1f}% | Drive: {p_dist:.1f} dB |
    | **Chorus** | {"🟢 ON" if states[2] else "🔴 BYPASS"} | {probs_onoff[2]*100:.1f}% | Rate: {p_ch_rate:.2f} Hz / Depth: {p_ch_depth:.2f} |
    | **Delay** | {"🟢 ON" if states[3] else "🔴 BYPASS"} | {probs_onoff[3]*100:.1f}% | Time: {p_dl_time:.2f}s / Mix: {p_dl_mix*100:.0f}% |
    | **Reverb** | {"🟢 ON" if states[4] else "🔴 BYPASS"} | {probs_onoff[4]*100:.1f}% | Room Size: {p_rv_room:.2f} |
    """

    # Grafico Spettrogramma
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(mel_img, x_axis='time', y_axis='mel', sr=sr, ax=ax, cmap='viridis')
    ax.set_title("Spettrogramma Mel dell'Audio Inserito")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return Image.open(buf), "100%", report_text