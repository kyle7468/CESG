import torch
import torch.nn.functional as F

def info_nce(sim):
    labels = torch.arange(sim.size(0)).to(sim.device)
    loss_v = F.cross_entropy(sim, labels)
    loss_t = F.cross_entropy(sim.T, labels)
    return (loss_v + loss_t) / 2

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for video, text, ent in loader:
        video, text, ent = video.to(device), text.to(device), ent.to(device)
        sim, _, _ = model(video, text, ent)
        loss = info_nce(sim)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)