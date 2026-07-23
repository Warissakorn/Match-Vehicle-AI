"""Turn a vehicle crop into an L2-normalized appearance embedding.

The embedding is the heart of the Re-ID: two crops of the *same* vehicle (even
from different camera angles) should land close together, different vehicles far
apart. Closeness is measured later with cosine similarity in ``matcher.py``.

``Embedder`` is an abstract interface so the appearance model is swappable. The
default ``ResNet50Embedder`` uses ImageNet-pretrained torchvision weights: a
reliable, easy-to-download baseline. A dedicated vehicle Re-ID model (OSNet via
torchreid, CLIP, ...) can be dropped in later by implementing the same
interface, without touching the detector, matcher, GUI, or pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Maps a BGR vehicle crop to a 1-D float32 unit vector."""

    #: Dimensionality of the produced embedding.
    dim: int

    @abstractmethod
    def embed(self, crop: np.ndarray) -> np.ndarray:
        """Embed a single BGR image. Returns an L2-normalized 1-D array."""

    def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        """Embed many crops. Returns an (N, dim) L2-normalized array.

        Default implementation loops; subclasses may override for real batching.
        """
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.embed(c) for c in crops]).astype(np.float32)


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(norm, eps)


class ResNet50Embedder(Embedder):
    """torchvision ResNet50 with the classifier head removed (2048-d features)."""

    dim = 2048

    def __init__(self, device: str | None = None):
        self._device = device
        self._model = None
        self._transform = None
        self._torch = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from torchvision import transforms
        from torchvision.models import ResNet50_Weights, resnet50

        self._torch = torch
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        # Replace the 1000-class classifier with identity -> penultimate 2048-d.
        model.fc = torch.nn.Identity()
        model.eval().to(self._device)
        self._model = model

        # ImageNet preprocessing; crops arrive as BGR (OpenCV) so we convert.
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _preprocess(self, crop: np.ndarray):
        # OpenCV gives BGR; torchvision/ImageNet expects RGB.
        rgb = crop[:, :, ::-1].copy()
        return self._transform(rgb)

    def embed(self, crop: np.ndarray) -> np.ndarray:
        return self.embed_batch([crop])[0]

    def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_model()
        torch = self._torch
        batch = torch.stack([self._preprocess(c) for c in crops]).to(self._device)
        with torch.no_grad():
            feats = self._model(batch).cpu().numpy().astype(np.float32)
        return _l2_normalize(feats)


def get_default_embedder(device: str | None = None) -> Embedder:
    """Factory for the default appearance model. Swap here to change globally."""
    return ResNet50Embedder(device=device)
