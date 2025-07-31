import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch import Tensor
from transformers import ViTModel

from ...utils import check_tensor


class DINOScore(nn.Module):
    """Computes the DINO score between two batches of images.

    Reference:
      - https://github.com/google/dreambooth/issues/3#issuecomment-1804546726
      - https://github.com/OSU-NLP-Group/MagicBrush/blob/main/evaluation/image_eval.py
    """
    def __init__(self):
        super().__init__()

        # load DINO ViT-S/16
        self.model = ViTModel.from_pretrained("facebook/dino-vits16", add_pooling_layer=False)
        self.model.eval()

        # DINO transforms
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def forward(self, images1: Tensor, images2: Tensor) -> Tensor:
        """
        Args:
            images1: Tensor of shape (B, C, H, W) and range [0, 1].
            images2: Tensor of shape (B, C, H, W) and range [0, 1].

        Returns:
            score: Tensor of shape (B, ).
        """
        # check image tensors
        check_tensor(images1, ndim=4, dtype=torch.float32, min_value=0., max_value=1.)
        check_tensor(images2, ndim=4, dtype=torch.float32, min_value=0., max_value=1.)
        # apply transforms
        images1 = torch.stack([self.transform(image) for image in images1], dim=0).to(images1.device)
        images2 = torch.stack([self.transform(image) for image in images2], dim=0).to(images2.device)
        # get features
        self.to(images1.device)
        with torch.no_grad():
            features1 = self.model(images1).last_hidden_state[:, 0]
            features2 = self.model(images2).last_hidden_state[:, 0]
        # compute cosine similarity
        score = F.cosine_similarity(features1, features2, dim=1)
        return score
