import torch
from torch import Tensor

from ..meters import AverageMeter
from ..metrics.image.clip_score import CLIPIScore, CLIPTScore
from ..metrics.image.dino_score import DINOScore
from ..utils import check_tensor


class SubjectDrivenEvaluator:
    """Evaluator for subject-driven generation.

    Reference:
      - https://github.com/google/dreambooth/issues/3#issuecomment-1804546726
      - https://github.com/OSU-NLP-Group/MagicBrush/blob/main/evaluation/image_eval.py
    """

    METRICS = ["dino", "clip-i", "clip-t"]

    def __init__(self):
        self.dino_score_fn = DINOScore()
        self.clip_i_score_fn = CLIPIScore("openai/clip-vit-base-patch32")
        self.clip_t_score_fn = CLIPTScore("openai/clip-vit-base-patch32")
        self.average_meters = {metric: AverageMeter() for metric in self.METRICS}

    def update(self, gens: Tensor, refs: Tensor, prompts: list[str]):
        """
        Args:
            gens: Tensor of shape (B, C, H, W) with generated images.
            refs: Tensor of shape (B, C, H, W) with reference images.
            prompts: List of strings with prompts corresponding to each generated image.
        """
        # check input tensors
        check_tensor(gens, ndim=4, dtype=torch.float32, min_value=0., max_value=1.)
        check_tensor(refs, ndim=4, dtype=torch.float32, min_value=0., max_value=1.)
        assert len(gens) == len(refs) == len(prompts)
        # compute metrics
        for metric in self.METRICS:
            if metric == "dino":
                score = self.dino_score_fn(gens, refs)  # (B, )
            elif metric == "clip-i":
                score = self.clip_i_score_fn(gens, refs)  # (B, )
            elif metric == "clip-t":
                score = self.clip_t_score_fn(gens, prompts)  # (B, )
            else:
                raise ValueError(f"Unknown metric: {metric}")
            self.average_meters[metric].update(score)

    def compute(self):
        results = {metric: self.average_meters[metric].average for metric in self.METRICS}
        return results
