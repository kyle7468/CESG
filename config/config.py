import torch
class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_dim = 512
    batch_size = 16
    epochs = 20
    lr = 1e-4
    num_entities = 1000  # 知识图谱实体数
    temp = 0.07          # 对比学习温度