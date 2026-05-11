from .dataset import VideoTextDataset
import clip

def collate_fn(batch):
    videos, texts, entities = zip(*batch)
    videos = torch.stack(videos)
    tokens = clip.tokenize(texts, truncate=True)
    entities = torch.stack(entities)
    return videos, tokens, entities