import torch
from torch import Tensor
from lpips import LPIPS

from image_evaluators.utils import check_tensor


lpips_module = LPIPS()

def lpips_fn(images1: Tensor, images2: Tensor) -> Tensor:
    """Computes the Learned Perceptual Image Patch Similarity (LPIPS) between two batches of images.

    Args:
        images1: Tensor of shape (B, C, H, W) and range [0, 1].
        images2: Tensor of shape (B, C, H, W) and range [0, 1].

    Returns:
        lpips: Tensor of shape (B, ).

    References:
        https://arxiv.org/abs/1801.03924
        https://github.com/richzhang/PerceptualSimilarity
    """
    # check image tensors
    check_tensor(images1, ndim=4, dtype=torch.float32, min_value=0., max_value=1.)
    check_tensor(images2, ndim=4, dtype=torch.float32, min_value=0., max_value=1.)
    # compute lpips
    lpips_module.to(images1.device)
    lpips = lpips_module(images1, images2)[:, 0, 0, 0]
    return lpips
