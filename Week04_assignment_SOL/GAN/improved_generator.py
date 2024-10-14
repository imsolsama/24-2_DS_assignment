import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=64, img_dim=28*28):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),  # 학습 안정화를 위해 Batch Normalization 추가
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),  # Batch Normalization 추가
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),  # Batch Normalization 추가
            nn.ReLU(True),
            nn.Linear(1024, img_dim),
            nn.Tanh()  # 출력 이미지를 [-1, 1] 범위로 정규화
        )

    def forward(self, z):
        return self.gen(z)
