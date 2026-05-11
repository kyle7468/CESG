from .clip_encoder import CLIPEncoder
from .kbat import KBAT
from .cesg import CommonsenseGraphEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F

class CESG_KBAT_CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.clip = CLIPEncoder(config.device)
        self.kbat = KBAT(config.embed_dim)
        self.cesg = CommonsenseGraphEnhance(config.embed_dim, config.num_entities)
        self.temp = nn.Parameter(torch.ones([]) * config.temp)

    def forward(self, video_frames, text_tokens, entity_ids):
        vid = self.clip.encode_video(video_frames)
        txt = self.clip.encode_text(text_tokens)
        vid_kbat = self.kbat(vid, vid)
        vid_enhanced = self.cesg(vid_kbat, entity_ids)

        vid_enhanced = F.normalize(vid_enhanced, dim=-1)
        txt = F.normalize(txt, dim=-1)
        sim = vid_enhanced @ txt.T * self.temp.exp()
        return sim, vid_enhanced, txt