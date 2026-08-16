import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from mel import GuitarDataset
from model import GuitarEffectsNet

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


def train_model(dataset_dir=None, save_path=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if dataset_dir is None:
        project_root = os.path.abspath(os.path.join(base_dir, ".."))
        dataset_dir = os.path.join(project_root, "datasets")

    if save_path is None:
        save_path = os.path.join(base_dir, "pod_go_model.pth")

    batch_size = 32
    epochs = 60
    learning_rate = 5e-4
    weight_decay = 1e-2  # Aumentato da 1e-4 a 1e-2 per forte regolarizzazione L2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Device: {device}")
    print(f"Dataset Path: {dataset_dir}")

    # Inizializzazione separata per disattivare la data augmentation sul validation set
    full_dataset = GuitarDataset(dataset_dir=dataset_dir, is_train=True)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Disattiva SpecAugment per le istanze di validazione
    val_dataset.dataset.is_train = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_amp_classes = len(full_dataset.amp_to_id)
    model = GuitarEffectsNet(num_amp_classes=num_amp_classes).to(device)

    criterion_amp = nn.CrossEntropyLoss()
    criterion_onoff = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    best_val_loss = float("inf")

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

            # Bilanciamento pesi rimodulato per favorire la generalizzazione
            total_loss = loss_amp + 1.0 * loss_onoff + 0.8 * loss_params

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * x.size(0)

        epoch_train_loss = running_loss / train_size

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

                total_loss = loss_amp + 1.0 * loss_onoff + 0.8 * loss_params
                val_loss += total_loss.item() * x.size(0)

                preds_amp = torch.argmax(logits_amp, dim=1)
                amp_correct += (preds_amp == y_amp).sum().item()
                total_val_samples += x.size(0)

        epoch_val_loss = val_loss / val_size
        amp_acc = (amp_correct / total_val_samples) * 100.0
        scheduler.step(epoch_val_loss)

        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Amp Acc: {amp_acc:.2f}%")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'amp_to_id': full_dataset.amp_to_id,
                'id_to_amp': full_dataset.id_to_amp
            }, save_path)
            print(f"Model saved to {save_path}")
    print("Training complete!")