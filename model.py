import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class GuitarEffectsNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Backbone Extractor (4 blocchi conv)
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )

        # Invece di ridurre a 1x1, riduciamo solo la dimensione frequenziale (Mel Bins) a 1
        # mantenendo la dimensione temporale per percepire ritmi ed echi!
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None)) 

        # Linear Adapter con flattening
        self.dense_shared = nn.Sequential(
            nn.LazyLinear(256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

        # Head 1: ON/OFF Classification (Restituisce LOGITS, no Sigmoid qui!)
        self.head_onoff = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 5)
        )

        # Head 2: Knobs Regression (I parametri continuano a usare Sigmoid per [0, 1])
        self.head_params = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 15),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.freq_pool(x)  # Riduce frequenza ma preserva il tempo
        x = torch.flatten(x, 1)

        shared_feat = self.dense_shared(x)

        logits_onoff = self.head_onoff(shared_feat)
        pred_params = self.head_params(shared_feat)

        return logits_onoff, pred_params