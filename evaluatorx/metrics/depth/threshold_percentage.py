import torch
from torch import Tensor

from ...utils import check_tensor


def threshold_percentage_fn(preds: Tensor, targets: Tensor, threshold: float, valid_masks: Tensor = None) -> Tensor:
    """Computes the Threshold Percentage between two batches of depth images.

    Args:
        preds: Predicted images. Tensor of shape (B, H, W) and range [0, +inf].
        targets: Target images. Tensor of shape (B, H, W) and range [0, +inf].
        threshold: Threshold value. A float value.
        valid_masks: Optional masks. Tensor of shape (B, H, W) and dtype bool.

    Returns:
        results: Tensor of shape (B, ).
    """
    # check input tensors
    check_tensor(preds, ndim=3, dtype=torch.float32, min_value=0.)
    check_tensor(targets, ndim=3, dtype=torch.float32, min_value=0.)
    check_tensor(valid_masks, ndim=3, dtype=torch.bool)
    # compute threshold percentage and average over valid pixels
    d1 = preds / targets
    d2 = targets / preds
    max_d1_d2 = torch.max(d1, d2)
    bit_mat = max_d1_d2 < threshold
    if valid_masks is not None:
        bit_mat[~valid_masks] = 0
        n = valid_masks.sum((-1, -2))
    else:
        n = preds.shape[-1] * preds.shape[-2]
    results = torch.sum(bit_mat, (-1, -2)) / n
    return results
