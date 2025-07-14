import numpy as np

import torch
from torch import Tensor

from .meters import AverageMeter
from .metrics.depth import absrel_fn, rmse_fn, threshold_percentage_fn
from .utils import check_tensor


class DepthEvaluator:
    """Evaluator for affine-invariant depth estimation.

    The estimated depth predictions are first aligned to the ground truths using least-squares fitting in either
    depth or disparity space. Then metrics are computed on the valid pixels of the aligned predictions.

    References:
      - https://github.com/prs-eth/Marigold/blob/main/script/depth/eval.py
      - https://github.com/prs-eth/Marigold/blob/main/src/util/metric.py
      - https://github.com/EnVision-Research/Lotus/blob/main/evaluation/evaluation.py
      - https://github.com/EnVision-Research/Lotus/blob/main/evaluation/util/metric.py
    """

    METRICS = ["absrel", "rmse", "delta1", "delta2", "delta3"]

    def __init__(
            self,
            dataset_min_depth: float,
            dataset_max_depth: float,
            alignment_max_res: int = None,
            disparity: bool = False,
    ):
        """
        Args:
            dataset_min_depth: Minimum depth value in the dataset.
            dataset_max_depth: Maximum depth value in the dataset.
            alignment_max_res: Maximum resolution for least square alignment.
            disparity: If True, the input is treated as disparity instead of depth.
        """
        self.dataset_min_depth = dataset_min_depth
        self.dataset_max_depth = dataset_max_depth
        self.alignment_max_res = alignment_max_res
        self.disparity = disparity

        self.average_meters = {metric: AverageMeter() for metric in self.METRICS}

    def update(self, preds: Tensor, targets: Tensor, valid_masks: Tensor):
        """
        Args:
            preds: Tensor of shape (B, H, W) with predicted depth/disparity values.
            targets: Tensor of shape (B, H, W) with ground-truth depth values.
            valid_masks: Tensor of shape (B, H, W) with boolean values indicating valid pixels.
        """
        # check input tensors
        check_tensor(preds, ndim=3, dtype=torch.float32, min_value=0.)
        check_tensor(targets, ndim=3, dtype=torch.float32, min_value=0.)
        check_tensor(valid_masks, ndim=3, dtype=torch.bool)
        for pred_ts, gt_ts, valid_mask_ts in zip(preds, targets, valid_masks):
            # tensor to numpy array
            pred_arr = pred_ts.detach().cpu().numpy()
            gt_arr = gt_ts.detach().cpu().numpy()
            valid_mask_arr = valid_mask_ts.detach().cpu().numpy()
            # least square alignment
            if not self.disparity:
                # align in depth space
                depth_pred, scale, shift = self.align_least_square(
                    gt_arr=gt_arr,
                    pred_arr=pred_arr,
                    valid_mask_arr=valid_mask_arr,
                    return_scale_shift=True,
                    max_resolution=self.alignment_max_res,
                )
            else:
                # align in disparity space
                gt_disparity, gt_non_neg_mask = self.depth2disparity(depth=gt_arr, return_mask=True)
                pred_non_neg_mask = pred_arr > 0
                valid_nonnegative_mask = valid_mask_arr & gt_non_neg_mask & pred_non_neg_mask
                disparity_pred, scale, shift = self.align_least_square(
                    gt_arr=gt_disparity,
                    pred_arr=pred_arr,
                    valid_mask_arr=valid_nonnegative_mask,
                    return_scale_shift=True,
                    max_resolution=self.alignment_max_res,
                )
                disparity_pred = np.clip(disparity_pred, a_min=1e-3, a_max=None)
                depth_pred = self.disparity2depth(disparity_pred)
            # clip to dataset range
            depth_pred = np.clip(depth_pred, a_min=self.dataset_min_depth, a_max=self.dataset_max_depth)
            depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)
            # numpy array to tensor
            depth_pred_ts = torch.from_numpy(depth_pred).unsqueeze(0).to(preds.device)  # [1, H, W]
            gt_ts = gt_ts.unsqueeze(0)  # [1, H, W]
            valid_mask_ts = valid_mask_ts.unsqueeze(0)  # [1, H, W]
            # compute metrics
            for metric in self.METRICS:
                if metric == "absrel":
                    result = absrel_fn(depth_pred_ts, gt_ts, valid_mask_ts)
                elif metric == "rmse":
                    result = rmse_fn(depth_pred_ts, gt_ts, valid_mask_ts)
                elif metric == "delta1":
                    result = threshold_percentage_fn(depth_pred_ts, gt_ts, 1.25, valid_mask_ts)
                elif metric == "delta2":
                    result = threshold_percentage_fn(depth_pred_ts, gt_ts, 1.25**2, valid_mask_ts)
                elif metric == "delta3":
                    result = threshold_percentage_fn(depth_pred_ts, gt_ts, 1.25**3, valid_mask_ts)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                self.average_meters[metric].update(result)

    def compute(self):
        results = {metric: self.average_meters[metric].average for metric in self.METRICS}
        return results

    @staticmethod
    def align_least_square(
        gt_arr: np.ndarray,
        pred_arr: np.ndarray,
        valid_mask_arr: np.ndarray,
        return_scale_shift=True,
        max_resolution=None,
    ):
        ori_shape = pred_arr.shape  # input shape

        gt = gt_arr.squeeze()  # [H, W]
        pred = pred_arr.squeeze()
        valid_mask = valid_mask_arr.squeeze()

        # Downsample
        if max_resolution is not None:
            scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
            if scale_factor < 1:
                downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
                gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
                pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
                valid_mask = (
                    downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                    .bool()
                    .numpy()
                )

        assert (
            gt.shape == pred.shape == valid_mask.shape
        ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

        gt_masked = gt[valid_mask].reshape((-1, 1))
        pred_masked = pred[valid_mask].reshape((-1, 1))

        # numpy solver
        _ones = np.ones_like(pred_masked)
        A = np.concatenate([pred_masked, _ones], axis=-1)
        X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
        scale, shift = X

        aligned_pred = pred_arr * scale + shift

        # restore dimensions
        aligned_pred = aligned_pred.reshape(ori_shape)

        if return_scale_shift:
            return aligned_pred, scale, shift
        else:
            return aligned_pred

    @staticmethod
    def depth2disparity(depth, return_mask=False):
        if isinstance(depth, torch.Tensor):
            disparity = torch.zeros_like(depth)
        elif isinstance(depth, np.ndarray):
            disparity = np.zeros_like(depth)
        else:
            raise TypeError(f"Unsupported type: {type(depth)}")
        non_negtive_mask = depth > 0
        disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
        if return_mask:
            return disparity, non_negtive_mask
        else:
            return disparity

    def disparity2depth(self, disparity, **kwargs):
        return self.depth2disparity(disparity, **kwargs)
