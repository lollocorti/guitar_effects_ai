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


def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset_pedalboard")
    model_save_path = os.path.join(base_dir, "pod_go_model.pth")

    batch_size = 64
    epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    dataset = GuitarDataset(dataset_dir=dataset_dir)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GuitarEffectsNet().to(device)
    
    # BCEWithLogitsLoss include la sigmoide internamente per maggiore stabilità numerica
    criterion_onoff = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Scheduler per abbassare il learning rate quando la val loss va in plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x, y_onoff, y_params in train_loader:
            x, y_onoff, y_params = x.to(device), y_onoff.to(device), y_params.to(device)

            optimizer.zero_grad()
            
            logits_onoff, pred_params = model(x)
            
            loss_onoff = criterion_onoff(logits_onoff, y_onoff)
            loss_params = masked_mse_loss(pred_params, y_params, y_onoff)
            
            # Diamo pari peso o leggermente prioritario alla classificazione ON/OFF
            total_loss = loss_onoff + 1.0 * loss_params

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * x.size(0)

        epoch_train_loss = running_loss / train_size

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y_onoff, y_params in val_loader:
                x, y_onoff, y_params = x.to(device), y_onoff.to(device), y_params.to(device)

                logits_onoff, pred_params = model(x)
                loss_onoff = criterion_onoff(logits_onoff, y_onoff)
                loss_params = masked_mse_loss(pred_params, y_params, y_onoff)
                
                total_loss = loss_onoff + 1.0 * loss_params
                val_loss += total_loss.item() * x.size(0)

        epoch_val_loss = val_loss / val_size
        scheduler.step(epoch_val_loss)

        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    train_model()