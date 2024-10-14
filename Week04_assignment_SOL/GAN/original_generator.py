import torch
import torch.nn as nn

class Original_Generator(nn.Module):
    def __init__(self, z_dim=64, img_dim=28*28):  # z_dim은 latent vector 크기
        super(Original_Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, img_dim),
            nn.Tanh()  # 생성된 이미지 값을 -1 ~ 1로 정규화하기 위해 Tanh 사용
        )

    def forward(self, x):
        return self.gen(x)
