import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_feat):
        super(Model, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(num_feat, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.BatchNorm1d(2),
            nn.Linear(2, 1),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1)
        return x
