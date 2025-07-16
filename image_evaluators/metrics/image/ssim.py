import torch
from torch import Tensor
from skimage.metrics import structural_similarity as ssim_skimage

from image_evaluators.utils import check_tensor


def ssim_fn(images1: Tensor, images2: Tensor) -> Tensor:
    """Computes the Structural Similarity (SSIM) between two batches of images.

    Args:
        images1: Tensor of shape (B, C, H, W) and range [0, 1].
        images2: Tensor of shape (B, C, H, W) and range [0, 1].

    Returns:
        ssim: Tensor of shape (B, ).
    """
    # check image tensors
    check_tensor(images1, ndim=4, dtype=torch.float32, min_value=0., max_value=1.)
    check_tensor(images2, ndim=4, dtype=torch.float32, min_value=0., max_value=1.)
    # convert to numpy arrays
    images1_np = images1.permute(0, 2, 3, 1).cpu().numpy()
    images2_np = images2.permute(0, 2, 3, 1).cpu().numpy()
    # compute ssim
    ssim = torch.tensor([
        ssim_skimage(image1_np, image2_np, data_range=1.0, channel_axis=2)
        for image1_np, image2_np in zip(images1_np, images2_np)
    ]).to(images1.device)
    return ssim
