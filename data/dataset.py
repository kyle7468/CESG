import torch
from torch.utils.data import Dataset

class VideoTextDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        video = torch.randn(8, 3, 224, 224)  # T=8帧
        text = "a video about action"
        entities = torch.randint(0, 1000, (10,))
        return video, text, entities