import torch



def get_num_params(m):
    return sum(p.numel() for p in m.parameters())