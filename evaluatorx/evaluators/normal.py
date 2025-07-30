import torch
from torch import Tensor

from ..metrics.normal import angular_error_fn
from ..utils import check_tensor


class NormalEvaluator:
    """Evaluator for surface normal estimation.

    References:
      - https://github.com/baegwangbin/DSINE/blob/main/utils/utils.py
      - https://github.com/EnVision-Research/Lotus/blob/main/evaluation/evaluation.py
      - https://github.com/EnVision-Research/Lotus/blob/main/evaluation/util/normal_utils.py
    """

    METRICS = ["mean", "median", "rmse", "a1", "a2", "a3", "a4", "a5"]

    def __init__(self):
        self.total_normal_errors = []

    def update(self, preds: Tensor, targets: Tensor, valid_masks: Tensor):
        """
        Args:
            preds: Tensor of shape (B, 3, H, W) with predicted normals.
            targets: Tensor of shape (B, 3, H, W) with ground-truth normals.
            valid_masks: Tensor of shape (B, 1, H, W) with boolean values indicating valid pixels.
        """
        # check input tensors
        check_tensor(preds, ndim=4, dtype=torch.float32, min_value=-1., max_value=1.)
        check_tensor(targets, ndim=4, dtype=torch.float32, min_value=-1., max_value=1.)
        check_tensor(valid_masks, ndim=4, dtype=torch.bool)
        # compute angular error
        angular_error = angular_error_fn(preds, targets)  # (B, 1, H, W)
        angular_error = angular_error[valid_masks]  # (B * H * W, )
        self.total_normal_errors.append(angular_error)

    def compute(self):
        total_normal_errors = torch.cat(self.total_normal_errors, dim=0)
        return {
            "mean": torch.mean(total_normal_errors),
            "median": torch.median(total_normal_errors),
            "rmse": torch.sqrt(torch.mean(total_normal_errors * total_normal_errors)),
            "a1": 100.0 * torch.mean((total_normal_errors < 5).float()),
            "a2": 100.0 * torch.mean((total_normal_errors < 7.5).float()),
            "a3": 100.0 * torch.mean((total_normal_errors < 11.25).float()),
            "a4": 100.0 * torch.mean((total_normal_errors < 22.5).float()),
            "a5": 100.0 * torch.mean((total_normal_errors < 30).float()),
        }
