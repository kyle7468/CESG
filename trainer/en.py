import torch

def evaluate(sim_matrix):
    B = sim_matrix.size(0)
    labels = torch.arange(B).to(sim_matrix.device)
    ranks = torch.argsort(sim_matrix, dim=-1, descending=True)
    correct = (ranks == labels.unsqueeze(1))

    r1 = correct[:,0].float().mean().item()
    r5 = correct[:,:5].any(dim=1).float().mean().item()
    r10 = correct[:,:10].any(dim=1).float().mean().item()
    return {"R@1": r1, "R@5": r5, "R@10": r10}