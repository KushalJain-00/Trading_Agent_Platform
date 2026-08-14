import torch.nn as nn


def build_model(input_dim, window_size, num_classes=3):
    return nn.Sequential(
        _ConvBackbone(input_dim),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes),
    )


class _ConvBackbone(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2))
