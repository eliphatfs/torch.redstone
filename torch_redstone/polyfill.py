import torch


class Polyfill:
    @staticmethod
    def cdist2(src: torch.Tensor, dst: torch.Tensor):
        """
        Computes batched the squared 2-norm distance between each pair of the two collections of row vectors.
        src (Tensor): input tensor of shape [B, N, C].
        dst (Tensor): input tensor of shape [B, M, C].
        Output: per-point square distance of shape [B, N, M].
        """
        B, M, _ = dst.shape
        dist = torch.baddbmm(torch.sum(src ** 2, -1, keepdim=True), src, dst.permute(0, 2, 1), alpha=-2)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    @staticmethod
    def cdist(src: torch.Tensor, dst: torch.Tensor):
        """
        Computes batched the 2-norm distance between each pair of the two collections of row vectors.
        src (Tensor): input tensor of shape [B, N, C].
        dst (Tensor): input tensor of shape [B, M, C].
        Output: per-point distance of shape [B, N, M].
        """
        return Polyfill.cdist2(src, dst).sqrt_()

    @staticmethod
    def square(x: torch.Tensor):
        return x * x

    @staticmethod
    def broadcast_to(tensor: torch.Tensor, shape):
        return tensor.expand(shape)

    @staticmethod
    def autocast_cuda_only(device_type: str, enabled: bool = True):
        return torch.cuda.amp.autocast(enabled=enabled)


# Polyfill impls
if not hasattr(torch, 'square'):
    torch.square = Polyfill.square
if not hasattr(torch, 'broadcast_to'):
    torch.broadcast_to = Polyfill.broadcast_to
if not hasattr(torch, 'cdist'):
    torch.cdist = Polyfill.cdist
if not hasattr(torch, 'autocast'):
    torch.autocast = Polyfill.autocast_cuda_only

# Aliases
if not hasattr(torch, 'absolute'):
    torch.absolute = torch.abs
if not hasattr(torch, 'arccos'):
    torch.arccos = torch.acos
if not hasattr(torch, 'arcsin'):
    torch.arcsin = torch.asin
if not hasattr(torch, 'arctan'):
    torch.arctan = torch.atan
if not hasattr(torch, 'arctan2'):
    torch.arctan2 = torch.atan2
