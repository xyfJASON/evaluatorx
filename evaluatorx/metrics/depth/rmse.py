import torch
from torch import Tensor

from ...utils import check_tensor


def rmse_fn(preds: Tensor, targets: Tensor, valid_masks: Tensor = None) -> Tensor:
    """Computes the Root Mean Square Error (RMSE) between two batches of depth images.

    Args:
        preds: Predicted images. Tensor of shape (B, H, W) and range [0, +inf].
        targets: Target images. Tensor of shape (B, H, W) and range [0, +inf].
        valid_masks: Optional masks. Tensor of shape (B, H, W) and dtype bool.

    Returns:
        rmse: Tensor of shape (B, ).
    """
    # check input tensors
    check_tensor(preds, ndim=3, dtype=torch.float32, min_value=0.)
    check_tensor(targets, ndim=3, dtype=torch.float32, min_value=0.)
    check_tensor(valid_masks, ndim=3, dtype=torch.bool)
    # compute rmse and average over valid pixels
    diff = preds - targets
    if valid_masks is not None:
        diff[~valid_masks] = 0
        n = valid_masks.sum((-1, -2))
    else:
        n = preds.shape[-1] * preds.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse
