import torch
from torch import Tensor

from .utils import check_tensor


class SegmentationEvaluator:
    """Evaluator for semantic segmentation.

    References:
      - https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/sem_seg_evaluation.py
    """

    METRICS = ["miou", "fwiou", "class-iou", "macc", "pacc", "class-acc"]

    def __init__(self, num_classes: int, ignore_label: int = -1):
        """
        Args:
            num_classes: Number of classes in the dataset.
            ignore_label: Label to ignore in the ground truth, e.g., background or unlabeled pixels.
        """
        self.num_classes = num_classes
        self.ignore_label = ignore_label

        self.confusion_matrix = torch.zeros((self.num_classes + 1, self.num_classes + 1), dtype=torch.long)

    def update(self, preds: Tensor, targets: Tensor):
        """
        Args:
            preds: LongTensor of shape (B, H, W) with predicted class indices.
            targets: LongTensor of shape (B, H, W) with ground truth class indices.
        """
        # check input tensors
        check_tensor(preds, ndim=3, dtype=torch.long)
        check_tensor(targets, ndim=3, dtype=torch.long)
        # set ignore label to num_classes
        targets[targets == self.ignore_label] = self.num_classes
        check_tensor(preds, min_value=0, max_value=self.num_classes - 1)
        check_tensor(targets, min_value=0, max_value=self.num_classes)
        self.confusion_matrix = self.confusion_matrix.to(preds.device)
        # compute confusion matrix
        preds = preds.view(-1)
        targets = targets.view(-1)
        self.confusion_matrix += torch.bincount(
            (self.num_classes + 1) * preds + targets,
            minlength=(self.num_classes + 1) ** 2,
        ).view(self.num_classes + 1, self.num_classes + 1)

    def compute(self):
        device = self.confusion_matrix.device
        acc = torch.full((self.num_classes, ), torch.nan, dtype=torch.float32, device=device)
        iou = torch.full((self.num_classes, ), torch.nan, dtype=torch.float32, device=device)
        tp = self.confusion_matrix.diagonal()[:-1].float()
        pos_gt = torch.sum(self.confusion_matrix[:-1, :-1], dim=0).float()
        class_weights = pos_gt / torch.sum(pos_gt)
        pos_pred = torch.sum(self.confusion_matrix[:-1, :-1], dim=1).float()
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = torch.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = torch.sum(acc[acc_valid]) / torch.sum(acc_valid)
        miou = torch.sum(iou[iou_valid]) / torch.sum(iou_valid)
        fiou = torch.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = torch.sum(tp) / torch.sum(pos_gt)
        return {
            "miou": miou * 100.,
            "fwiou": fiou * 100.,
            "class-iou": iou * 100.,
            "macc": macc * 100.,
            "pacc": pacc * 100.,
            "class-acc": acc * 100.,
        }
