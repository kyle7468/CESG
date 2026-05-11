import torch
import torch.nn as nn

class CommonsenseGraphEnhance(nn.Module):
    def __init__(self, embed_dim=512, num_entity=1000):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entity, embed_dim)
        self.graph_attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, visual_feat, entity_ids):
        ent_feat = self.entity_emb(entity_ids)
        g_feat, _ = self.graph_attn(visual_feat.unsqueeze(1), ent_feat, ent_feat)
        g_feat = g_feat.squeeze(1)
        return self.fusion(torch.cat([visual_feat, g_feat], dim=-1))