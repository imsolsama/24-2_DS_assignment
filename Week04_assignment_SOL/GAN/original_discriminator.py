import torch
import torch.nn as nn

class Original_Discriminator(nn.Module):
    def __init__(self, in_features=28*28):  # Fashion MNIST의 이미지 크기는 28x28
        super(Original_Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)
