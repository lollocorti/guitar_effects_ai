import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from mel import GuitarDataset
from model import GuitarEffectsNet
import matplotlib.pyplot as plt

def masked_mse_loss(pred_params, target_params, target_onoff):
    mask_list = [
        target_onoff[:, 0:1],                  # Highpass
        target_onoff[:, 1:2],                  # Distortion
        target_onoff[:, 2:3].repeat(1, 5),     # Chorus
        target_onoff[:, 3:4].repeat(1, 3),     # Delay
        target_onoff[:, 4:5].repeat(1, 5)      # Reverb
    ]
    mask = torch.cat(mask_list, dim=1)
    squared_diff = (pred_params - target_params) ** 2
    masked_diff = squared_diff * mask

    total_active_params = mask.sum()
    if total_active_params > 0:
        return masked_diff.sum() / total_active_params
    return torch.tensor(0.0, device=pred_params.device)

def get_next_model_path(models_dir):
    """
    Scansiona la cartella models e restituisce i percorsi dinamici per il modello
    e il grafico della loss (es. model1.pth, loss_curve_model1.png).
    """
    os.makedirs(models_dir, exist_ok=True)
    existing_indices = []
    
    for filename in os.listdir(models_dir):
        if filename.startswith("model") and filename.endswith(".pth"):
            num_part = filename[5:-4]
            if num_part.isdigit():
                existing_indices.append(int(num_part))
    
    next_idx = max(existing_indices) + 1 if existing_indices else 1
    model_name = f"model{next_idx}.pth"
    plot_name = f"loss_curve_model{next_idx}.png"
    
    return os.path.join(models_dir, model_name), os.path.join(models_dir, plot_name)


def train_model(dataset_dir=None, models_dir=None):
    # Rilevamento automatico ambiente Google Colab
    is_colab = os.path.exists("/content")

    if dataset_dir is None:
        dataset_dir = "/content/dataset" if is_colab else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))

    if models_dir is None:
        models_dir = "/content/drive/MyDrive/guitar_effects_ai/models" if is_colab else os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))

    # Generazione dei percorsi incrementali
    save_path, plot_path = get_next_model_path(models_dir)

    print(f"I file verranno salvati in:\n - Modello: {save_path}\n - Grafico: {plot_path}")

    batch_size = 32
    epochs = 60
    learning_rate = 5e-4
    weight_decay = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Device: {device}")
    print(f"Dataset Path: {dataset_dir}")

    # Istanziazione dataset (SpecAugment attivo solo per il train set)
    train_dataset_full = GuitarDataset(dataset_dir=dataset_dir, is_train=True)
    val_dataset_full = GuitarDataset(dataset_dir=dataset_dir, is_train=False)

    total_samples = len(train_dataset_full)
    train_size = int(0.85 * total_samples)

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_samples, generator=generator).tolist()

    train_subset = Subset(train_dataset_full, indices[:train_size])
    val_subset = Subset(val_dataset_full, indices[train_size:])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    num_amp_classes = len(train_dataset_full.amp_to_id)
    model = GuitarEffectsNet(num_amp_classes=num_amp_classes).to(device)

    criterion_amp = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_onoff = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float("inf")
    patience = 12
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x, y_amp, y_onoff, y_params in train_loader:
            x = x.to(device)
            y_amp = y_amp.to(device)
            y_onoff = y_onoff.to(device)
            y_params = y_params.to(device)

            optimizer.zero_grad()

            logits_amp, logits_onoff, pred_params = model(x)

            loss_amp = criterion_amp(logits_amp, y_amp)
            loss_onoff = criterion_onoff(logits_onoff, y_onoff)
            loss_params = masked_mse_loss(pred_params, y_params, y_onoff)

            # Pesi bilanciati delle tre loss
            total_loss = 2.0 * loss_amp + 1.0 * loss_onoff + 0.5 * loss_params

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * x.size(0)

        epoch_train_loss = running_loss / len(train_subset)

        # Fase di Validazione
        model.eval()
        val_loss = 0.0
        amp_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for x, y_amp, y_onoff, y_params in val_loader:
                x = x.to(device)
                y_amp = y_amp.to(device)
                y_onoff = y_onoff.to(device)
                y_params = y_params.to(device)

                logits_amp, logits_onoff, pred_params = model(x)

                loss_amp = criterion_amp(logits_amp, y_amp)
                loss_onoff = criterion_onoff(logits_onoff, y_onoff)
                loss_params = masked_mse_loss(pred_params, y_params, y_onoff)

                total_loss = 2.0 * loss_amp + 1.0 * loss_onoff + 0.5 * loss_params
                val_loss += total_loss.item() * x.size(0)

                preds_amp = torch.argmax(logits_amp, dim=1)
                amp_correct += (preds_amp == y_amp).sum().item()
                total_val_samples += x.size(0)

        epoch_val_loss = val_loss / len(val_subset)
        amp_acc = (amp_correct / total_val_samples) * 100.0

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)

        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Amp Acc: {amp_acc:.2f}%")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'amp_to_id': train_dataset_full.amp_to_id,
                'id_to_amp': train_dataset_full.id_to_amp
            }, save_path)
            print(f"Model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Nessun miglioramento per {patience} epoche. Training interrotto.")
                break

    # Generazione del grafico di loss con nome dinamico abbinato
    plt.figure(figsize=(9, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training vs Validation Loss ({os.path.basename(save_path)})')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Grafico delle loss salvato in: {plot_path}")
    print("Training completato!")


if __name__ == "__main__":
    train_model()