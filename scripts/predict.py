import os
import io
import torch
import numpy as np
import librosa
import soundfile as sf
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image

# Libreria per il DSP in Python
from pedalboard import Pedalboard, HighpassFilter, Distortion, Chorus, Delay, Reverb
from model import GuitarEffectsNet
from mel import PARAM_RANGES, compute_mel_feature

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODELS_DIR = r"C:\Users\lollo\OneDrive\Documenti\guitar_effects_ai\models"

def get_available_models(models_dir=DEFAULT_MODELS_DIR):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    return models if models else ["Nessun modello trovato (.pth)"]

def load_selected_model(model_name, models_dir=DEFAULT_MODELS_DIR):
    if not model_name or "Nessun modello" in model_name:
        return None, {}, "Nessun modello selezionato."

    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path):
        return None, {}, f"File non trovato: {model_path}"

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        id_to_amp = {}
        num_amp_classes = 10

        if isinstance(checkpoint, dict) and 'amp_to_id' in checkpoint:
            id_to_amp = checkpoint['id_to_amp']
            num_amp_classes = len(checkpoint['amp_to_id'])

        net = GuitarEffectsNet(num_amp_classes=num_amp_classes).to(DEVICE)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
        else:
            net.load_state_dict(checkpoint)

        net.eval()
        return net, id_to_amp, "OK"
    except Exception as e:
        return None, {}, str(e)

def denormalize(val, min_v, max_v):
    return float(val * (max_v - min_v) + min_v)

def plot_spectrogram(mel_img, title, sr=48000):
    fig, ax = plt.subplots(figsize=(7, 2.5))
    img = librosa.display.specshow(mel_img, x_axis='time', y_axis='mel', sr=sr, ax=ax, cmap='viridis')
    ax.set_title(title, fontsize=10)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

def find_dry_audio_path(wet_audio_path, dataset_dir=r"C:\Users\lollo\OneDrive\Documenti\guitar_effects_ai\dataset"):
    """
    Cerca il file pulito (dry) corrispondente al file effettato.
    Se non lo trova, fa il fallback sul file caricato dall'utente.
    """
    filename = os.path.basename(wet_audio_path)
    
    # Esempio per IDMT: se il file si chiama 'G53-52100-1111-12345.wav',
    # la versione dry corrisponde alla catena con tutti gli effetti disattivati (es. 'G53-52100-0000-0000.wav' o simile)
    # Oppure se hai una cartella /dry/ con lo stesso nome base:
    dry_candidate = os.path.join(dataset_dir, "raw_audio", "dry", filename)
    if os.path.exists(dry_candidate):
        return dry_candidate

    # Cerca per pattern nella cartella del dataset se presente la traccia senza effetti
    # Fallback: restituisce il file stesso se non trova la corrispondenza dry
    return wet_audio_path

def process_and_recreate(audio_path, selected_model_name):
    if audio_path is None:
        return None, None, None, "⚠️ Carica un file audio di test (.wav)!"

    # 1. Caricamento Modello
    net, id_to_amp, status_msg = load_selected_model(selected_model_name)
    if net is None:
        return None, None, None, f"⚠️ Errore modello: {status_msg}"

    # 2. Caricamento e Analisi Spettrogramma dell'Audio Effettato Inserito (WET)
    audio_wet, sr = sf.read(audio_path)
    mel_input = compute_mel_feature(audio_wet, sr=sr)
    img_input = plot_spectrogram(mel_input, title=f"Spettrogramma Audio Inserito (Wet)")

    # 3. Inferenza IA sull'audio effettato
    x_input = torch.tensor(mel_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits_amp, logits_onoff, raw_params = net(x_input)
        
        pred_amp_id = torch.argmax(logits_amp, dim=1).item()
        amp_name = id_to_amp.get(pred_amp_id, f"Classe #{pred_amp_id}")
        
        probs_onoff = torch.sigmoid(logits_onoff).squeeze(0).cpu().numpy()
        params = raw_params.squeeze(0).cpu().numpy()

    states = (probs_onoff > 0.5)

    # 4. Estrazione Parametri e Costruzione Pedalboard
    p_hp_cutoff = denormalize(params[0], *PARAM_RANGES["highpass_cutoff"])
    p_dist_drive = denormalize(params[1], *PARAM_RANGES["distortion_drive"])
    p_ch_rate = denormalize(params[2], *PARAM_RANGES["chorus_rate"])
    p_ch_depth = denormalize(params[3], *PARAM_RANGES["chorus_depth"])
    p_ch_delay = denormalize(params[4], *PARAM_RANGES["chorus_delay"])
    p_ch_fb = denormalize(params[5], *PARAM_RANGES["chorus_feedback"])
    p_ch_mix = denormalize(params[6], *PARAM_RANGES["chorus_mix"])
    p_dl_time = denormalize(params[7], *PARAM_RANGES["delay_time"])
    p_dl_fb = denormalize(params[8], *PARAM_RANGES["delay_feedback"])
    p_dl_mix = denormalize(params[9], *PARAM_RANGES["delay_mix"])
    p_rv_room = denormalize(params[10], *PARAM_RANGES["reverb_room"])
    p_rv_damp = denormalize(params[11], *PARAM_RANGES["reverb_damping"])
    p_rv_wet = denormalize(params[12], *PARAM_RANGES["reverb_wet"])

    board_plugins = []
    rows = []

    if states[0]:
        board_plugins.append(HighpassFilter(cutoff_frequency_hz=p_hp_cutoff))
        rows.append(f"| **Highpass Filter** | {probs_onoff[0]*100:.1f}% | Cutoff: **{p_hp_cutoff:.1f} Hz** |")

    if states[1]:
        board_plugins.append(Distortion(drive_db=p_dist_drive))
        rows.append(f"| **Distortion** | {probs_onoff[1]*100:.1f}% | Drive: **{p_dist_drive:.1f} dB** |")

    if states[2]:
        board_plugins.append(Chorus(rate_hz=p_ch_rate, depth=p_ch_depth, centre_delay_ms=p_ch_delay, feedback=p_ch_fb, mix=p_ch_mix))
        rows.append(f"| **Chorus** | {probs_onoff[2]*100:.1f}% | Rate: **{p_ch_rate:.2f} Hz** \| Depth: **{p_ch_depth:.2f}** |")

    if states[3]:
        board_plugins.append(Delay(delay_seconds=p_dl_time, feedback=p_dl_fb, mix=p_dl_mix))
        rows.append(f"| **Delay** | {probs_onoff[3]*100:.1f}% | Time: **{p_dl_time:.2f}s** \| Mix: **{p_dl_mix*100:.0f}%** |")

    if states[4]:
        board_plugins.append(Reverb(room_size=p_rv_room, damping=p_rv_damp, wet_level=p_rv_wet))
        rows.append(f"| **Reverb** | {probs_onoff[4]*100:.1f}% | Room Size: **{p_rv_room:.2f}** |")

    if rows:
        table_content = "\n".join(rows)
        report_text = f"### 🎸 Amplificatore: **{amp_name}**\n\n| Effetto Attivo | Confidenza | Parametri Predetti |\n| :--- | :--- | :--- |\n{table_content}"
    else:
        report_text = f"### 🎸 Amplificatore: **{amp_name}**\n\n#### 🎛️ Tutti gli effetti sono BYPASSATI (OFF)"

    # 5. CARICAMENTO DEL SUONO PULITO (DRY) PER APPLICARE GLI EFFETTI
    dry_path = find_dry_audio_path(audio_path)
    audio_dry, sr_dry = sf.read(dry_path)

    # 6. Applicazione DSP al Suono PULITO
    board = Pedalboard(board_plugins)
    audio_float = audio_dry.astype(np.float32)
    if audio_float.ndim == 1:
        audio_float = np.expand_dims(audio_float, axis=0)
    else:
        audio_float = audio_float.T

    recreated_audio = board(audio_float, sr_dry)
    
    output_wav_path = "recreated_output.wav"
    sf.write(output_wav_path, recreated_audio.T, sr_dry)

    # Spettrogramma del nuovo suono effettato partendo dal dry
    mel_recreated = compute_mel_feature(recreated_audio.T, sr=sr_dry)
    img_recreated = plot_spectrogram(mel_recreated, title="Spettrogramma Suono Ricreato dal Dry (DSP)")

    return img_input, output_wav_path, img_recreated, report_text

# Costruzione Interfaccia Gradio
available_models = get_available_models()

with gr.Blocks(title="Guitar AI - Predict & Recreate") as demo:
    gr.Markdown("# 🎸 Guitar Effects AI - Predict & Recreate DSP")
    
    # SEZIONE 1: CONFIGURAZIONE & INPUT AUDIO
    gr.Markdown("---")
    gr.Markdown("### 1️⃣ Selezione Modello e Input Audio")
    
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=available_models,
                value=available_models[0] if available_models else None,
                label="Seleziona Modello AI (.pth)",
                interactive=True
            )
            refresh_btn = gr.Button("🔄 Aggiorna Modelli")
            audio_input = gr.Audio(type="filepath", label="Carica Audio da Analizzare (.wav)")
            predict_btn = gr.Button("🚀 Analizza e Ricrea Suono", variant="primary")

        with gr.Column(scale=1):
            image_input_spec = gr.Image(type="pil", label="Spettrogramma Audio Inserito")

    # SEZIONE 2: SUONO RICREATO & PARAMETRI
    gr.Markdown("---")
    gr.Markdown("### 2️⃣ Suono Ricreato e Parametri Applicati")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_output = gr.Audio(label="🎧 Suono Ricreato (DSP Pedalboard)")
            image_recreated_spec = gr.Image(type="pil", label="Spettrogramma Suono Ricreato")
            
        with gr.Column(scale=1):
            markdown_output = gr.Markdown(label="Report Predizione")

    # Eventi GUI
    def refresh_models_list():
        models = get_available_models()
        return gr.Dropdown(choices=models, value=models[0] if models else None)

    refresh_btn.click(fn=refresh_models_list, outputs=model_dropdown)
    predict_btn.click(
        fn=process_and_recreate,
        inputs=[audio_input, model_dropdown],
        outputs=[image_input_spec, audio_output, image_recreated_spec, markdown_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)