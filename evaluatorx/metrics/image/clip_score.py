import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import CLIPProcessor, CLIPModel

from ...utils import check_tensor


class CLIPIScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    def forward(self, images1: Tensor, images2: Tensor) -> Tensor:
        """Computes the CLIP score between two batches of images.

        Args:
            images1: Tensor of shape (B, C, H, W) and range [0, 1].
            images2: Tensor of shape (B, C, H, W) and range [0, 1].

        Returns:
            score: Tensor of shape (B, ).

        Reference:
          - https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/multimodal/clip_score.py
          - https://github.com/xichenpan/Kosmos-G/blob/main/eval/clip_score.py
        """
        # check image tensors
        check_tensor(images1, ndim=4, dtype=torch.float32, min_value=0., max_value=1.)
        check_tensor(images2, ndim=4, dtype=torch.float32, min_value=0., max_value=1.)
        device = images1.device
        # process images
        images1 = self.processor(images=images1, return_tensors="pt", do_rescale=False)
        images2 = self.processor(images=images2, return_tensors="pt", do_rescale=False)
        # get features
        self.to(device)
        with torch.no_grad():
            features1 = self.model.get_image_features(images1["pixel_values"].to(device))
            features2 = self.model.get_image_features(images2["pixel_values"].to(device))
        # compute cosine similarity
        score = F.cosine_similarity(features1, features2, dim=1)
        return score


class CLIPTScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    def forward(self, images: Tensor, texts: list[str]) -> Tensor:
        """Computes the CLIP score between a batch of images and a batch of texts.

        Args:
            images: Tensor of shape (B, C, H, W) and range [0, 1].
            texts: List of strings.

        Returns:
            score: Tensor of shape (B, ).

        Reference:
          - https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/multimodal/clip_score.py
          - https://github.com/xichenpan/Kosmos-G/blob/main/eval/clip_score.py
        """
        # check inputs
        check_tensor(images, ndim=4, dtype=torch.float32, min_value=0., max_value=1.)
        assert isinstance(texts, list) and all(isinstance(t, str) for t in texts)
        assert len(images) == len(texts)
        device = images.device
        # process images and texts
        processed = self.processor(text=texts, images=images, return_tensors="pt", padding=True, do_rescale=False)
        # get features
        self.to(device)
        with torch.no_grad():
            image_features = self.model.get_image_features(processed["pixel_values"].to(device))
            text_features = self.model.get_text_features(processed["input_ids"].to(device), processed["attention_mask"].to(device))
        # compute cosine similarity
        score = F.cosine_similarity(image_features, text_features, dim=1)
        return score


clip_i_score_fn = CLIPIScore()
clip_t_score_fn = CLIPTScore()
clip_score_fn = CLIPTScore()
