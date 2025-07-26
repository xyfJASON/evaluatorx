import torch
from torch import Tensor
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage

from ...utils import check_tensor


def psnr_fn(images1: Tensor, images2: Tensor) -> Tensor:
    """Computes the Peak Signal-to-Noise Ratio (PSNR) between two batches of images.

    Args:
        images1: Tensor of shape (B, C, H, W) and range [0, 1].
        images2: Tensor of shape (B, C, H, W) and range [0, 1].

    Returns:
        psnr: Tensor of shape (B, ).
    """
    # check image tensors
    check_tensor(images1, ndim=4, dtype=torch.float32, min_value=0., max_value=1.)
    check_tensor(images2, ndim=4, dtype=torch.float32, min_value=0., max_value=1.)
    # convert to numpy arrays
    images1_np = images1.permute(0, 2, 3, 1).cpu().numpy()
    images2_np = images2.permute(0, 2, 3, 1).cpu().numpy()
    # compute psnr
    psnr = torch.tensor([
        psnr_skimage(image1_np, image2_np, data_range=1.0)
        for image1_np, image2_np in zip(images1_np, images2_np)
    ]).to(images1.device)
    return psnr
