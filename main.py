import torch
from config import Config
from model import CESG_KBAT_CLIP
from data import VideoTextDataset, collate_fn
from trainer import train_one_epoch, evaluate
from utils import print_metrics
from torch.utils.data import DataLoader

def main():
    cfg = Config()
    model = CESG_KBAT_CLIP(cfg).to(cfg.device)
    dataset = VideoTextDataset()
    loader = DataLoader(dataset, cfg.batch_size, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs+1):
        loss = train_one_epoch(model, loader, optimizer, cfg.device)
        metrics = evaluate(model(next(iter(loader))[0].to(cfg.device),
                                 next(iter(loader))[1].to(cfg.device),
                                 next(iter(loader))[2].to(cfg.device))[0])
        print_metrics(epoch, loss, metrics)

if __name__ == "__main__":
    main()