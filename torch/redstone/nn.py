import torch.nn


class Lambda(torch.nn.Module):
    def __init__(self, lam) -> None:
        super().__init__()
        self.lam = lam

    def forward(self, inputs):
        return self.lam(inputs)
