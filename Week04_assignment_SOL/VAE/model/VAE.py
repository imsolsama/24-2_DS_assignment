import torch.nn as nn
import torch
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=64, num_layers=2):
        super(VAE, self).__init__()
        
        # 인코더: 더 깊은 신경망을 구성
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # 배치 정규화 추가
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3))  # 드롭아웃 추가
            
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 디코더: 대칭적으로 더 깊은 구조 사용
        dec_layers = []
        for i in range(num_layers):
            if i == 0:
                dec_layers.append(nn.Linear(latent_dim, hidden_dim))
            else:
                dec_layers.append(nn.Linear(hidden_dim, hidden_dim))
            dec_layers.append(nn.BatchNorm1d(hidden_dim))
            dec_layers.append(nn.ReLU())
            dec_layers.append(nn.Dropout(p=0.3))
            
        self.decoder = nn.Sequential(*dec_layers)
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder(z)
        return torch.sigmoid(self.fc_out(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 손실 함수 정의
def loss_function(recon_x, x, mu, logvar, input_dim=784):
    # BCE 재구성 손실
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD
