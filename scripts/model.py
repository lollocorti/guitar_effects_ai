import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class GuitarEffectsNet(nn.Module):
    def __init__(self, num_amp_classes=10):
        super().__init__()

        # Extractor Feature Convolutional (ResNet Backbone)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ResBlock(32, 64, stride=2)
        self.layer2 = ResBlock(64, 128, stride=2)
        self.layer3 = ResBlock(128, 256, stride=2)
        self.layer4 = ResBlock(256, 512, stride=2)

        # Pooling solo sulla dimensione delle frequenze (F -> 1), preservando il tempo (T)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # Modulo Ricorrente (BiGRU) per analizzare l'evoluzione temporale dell'audio
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Attention Mechanism per pesare i frame temporali rilevanti (es. attacco della nota)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # FC Condivisa con Dropout incrementato per evitare overfitting
        self.shared_fc = nn.Sequential(
            nn.Linear(512, 256),  # 256 * 2 (dovuto a bidirectional GRU)
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4)
        )

        # Head 1: Classificazione Amplificatore
        self.head_amp = nn.Linear(256, num_amp_classes)

        # Head 2: Classificazione Pedali ON/OFF (Logits)
        self.head_onoff = nn.Linear(256, 5)

        # Head 3: Regressione Parametri Manopole [0, 1]
        self.head_params = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 15),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. Extractor Convoluzionale
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # Shape: [B, 512, F, T]

        # 2. Pooling Frequenziale
        x = self.freq_pool(x).squeeze(2)  # Shape: [B, 512, T]
        x = x.permute(0, 2, 1)            # Shape: [B, T, 512] (compatibile con GRU)

        # 3. Processamento Ricorrente Temporal-Aware
        gru_out, _ = self.gru(x)          # Shape: [B, T, 512]

        # 4. Temporal Attention Pooling (Sostituisce torch.mean)
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)  # Shape: [B, T, 1]
        feat_temporal = torch.sum(gru_out * attn_weights, dim=1)      # Shape: [B, 512]

        # 5. Dense Layer condiviso
        feat = self.shared_fc(feat_temporal)

        # 6. Output Heads
        logits_amp = self.head_amp(feat)
        logits_onoff = self.head_onoff(feat)
        pred_params = self.head_params(feat)

        return logits_amp, logits_onoff, pred_params