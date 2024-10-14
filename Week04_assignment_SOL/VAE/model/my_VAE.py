import torch.nn as nn
import torch
import torch.nn.functional as F

# VAE 정의
class my_VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(my_VAE, self).__init__()
        
        # 인코더 정의
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # 잠재 공간의 평균
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # 잠재 공간의 분산
        
        # 디코더 정의
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # N(0, I)에서 샘플링
        return mu + eps * std
    
    def decode(self, z):
        h2 = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h2))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 손실 함수 정의
def my_loss_function(recon_x, x, mu, logvar):
    # 재구성 손실 (BCE 로스)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # Kullback-Leibler divergence (KL 로스)
    # KL divergence는 잠재 공간의 분포를 N(0, I)에 맞추기 위한 손실
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD
