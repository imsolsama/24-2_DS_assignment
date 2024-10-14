import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_features=28*28):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # 과적합 방지를 위해 Dropout 추가
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # 학습 안정화를 위해 Batch Normalization 추가
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # 추가 Dropout
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # 추가 레이어와 Batch Normalization
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 이진 분류를 위해 Sigmoid
        )

    def forward(self, z):
        return self.disc(z)
