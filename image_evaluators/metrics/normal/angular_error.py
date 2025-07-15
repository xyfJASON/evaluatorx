import torch
from torch import Tensor

from ...utils import check_tensor


def angular_error_fn(preds: Tensor, targets: Tensor) -> Tensor:
    """Computes the Angular Error between two batches of normal maps.

    Args:
        preds: Predicted normal maps. Tensor of shape (B, 3, H, W) and range [-1, 1].
        targets: Target normal maps. Tensor of shape (B, 3, H, W) and range [-1, 1].

    Returns:
        angular_error: Tensor of shape (B, 1, H, W).
    """
    # check input tensors
    check_tensor(preds, ndim=4, dtype=torch.float32, min_value=-1., max_value=1.)
    check_tensor(targets, ndim=4, dtype=torch.float32, min_value=-1., max_value=1.)
    # compute angular error
    angular_error = torch.cosine_similarity(preds, targets, dim=1)
    angular_error = torch.clamp(angular_error, min=-1.0, max=1.0)
    angular_error = torch.acos(angular_error) * 180.0 / torch.pi
    angular_error = angular_error.unsqueeze(1)
    return angular_error
