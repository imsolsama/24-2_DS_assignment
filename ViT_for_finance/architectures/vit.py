import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

class EmbeddingLayer(nn.Module):
    def __init__(self, img_size:int, patch_size:int, in_channels:int, embed_dim:int):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b e h w -> b (h w) e')
        b = x.shape[0]
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_token, x), dim=1)
        return x + self.pos_emb   

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).transpose(2, 1)
        q, k, v = qkv.unbind(2)
        attn_weights = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = attn_weights @ v
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.fc_out(out)

class MLP(nn.Module):
    def __init__(self, embed_dim:int, hidden_dim:int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, hidden_dim:int):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = MLP(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.mlp(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size:int, patch_size:int, in_channels:int, embed_dim:int, num_heads:int, num_layers:int, hidden_dim:int, num_classes:int):
        super().__init__()
        self.embedding = EmbeddingLayer(img_size, patch_size, in_channels, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        return self.fc_out(x)
