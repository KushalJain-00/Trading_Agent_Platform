import torch.nn as nn


def build_model(input_dim, window_size, num_classes=3):
    return _CNNLSTM(input_dim, num_classes)


class _CNNLSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, num_classes))

    def forward(self, x):
        c = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        _, (h_n, _) = self.lstm(c)
        return self.head(h_n[-1])
