import torch
import torch.nn as nn


def build_model(input_dim, window_size, num_classes=3):
    return _LSTM(input_dim, num_classes)


class _LSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, num_classes))

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])
