import math
import torch
import torch.nn as nn


def build_model(input_dim, window_size, num_classes=3):
    return _Transformer(input_dim, window_size, num_classes)


class _Transformer(nn.Module):
    def __init__(self, input_dim, window_size, num_classes, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, 128, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, num_classes))

    def forward(self, x):
        x = self.proj(x) + self.pe[:, :x.size(1)]
        return self.head(self.enc(x).mean(dim=1))
