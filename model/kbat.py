import torch
import torch.nn as nn

class KBAT(nn.Module):
    def __init__(self, embed_dim=512, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4), nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim)
        )

    def forward(self, x, knowledge):
        attn_out, _ = self.attn(x, knowledge, knowledge)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x