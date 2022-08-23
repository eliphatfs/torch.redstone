import torch


if not hasattr(torch, 'square'):
    torch.square = lambda x: x * x


if not hasattr(torch, 'broadcast_to'):
    torch.broadcast_to = lambda tensor, shape: tensor.expand(shape)
