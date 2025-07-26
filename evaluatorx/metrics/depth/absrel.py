import torch
from torch import Tensor

from ...utils import check_tensor


def absrel_fn(preds: Tensor, targets: Tensor, valid_masks: Tensor = None) -> Tensor:
    """Computes the Absolute Relative Difference (AbsRel) between two batches of depth images.

    Args:
        preds: Predicted images. Tensor of shape (B, H, W) and range [0, +inf].
        targets: Target images. Tensor of shape (B, H, W) and range [0, +inf].
        valid_masks: Optional masks. Tensor of shape (B, H, W) and dtype bool.

    Returns:
        absrel: Tensor of shape (B, ).
    """
    # check input tensors
    check_tensor(preds, ndim=3, dtype=torch.float32, min_value=0.)
    check_tensor(targets, ndim=3, dtype=torch.float32, min_value=0.)
    check_tensor(valid_masks, ndim=3, dtype=torch.bool)
    # compute absrel and average over valid pixels
    absrel = torch.abs(preds - targets) / targets
    if valid_masks is not None:
        absrel[~valid_masks] = 0
        n = valid_masks.sum((-1, -2))
    else:
        n = preds.shape[-1] * preds.shape[-2]
    absrel = torch.sum(absrel, (-1, -2)) / n
    return absrel
