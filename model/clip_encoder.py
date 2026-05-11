import clip
import torch
import torch.nn as nn

class CLIPEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)
        for param in self.model.parameters():
            param.requires_grad = False

    def encode_video(self, video_frames):
        B, T, C, H, W = video_frames.shape
        frames = video_frames.reshape(B * T, C, H, W)
        feat = self.model.encode_image(frames)
        feat = feat.reshape(B, T, -1).mean(dim=1)
        return feat

    def encode_text(self, text_tokens):
        return self.model.encode_text(text_tokens)