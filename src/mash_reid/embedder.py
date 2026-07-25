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

import config


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

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = config.DEFAULT_EMBED_BATCH_SIZE,
        use_half_on_cuda: bool = config.USE_HALF_PRECISION_ON_CUDA,
    ):
        self._device = device
        # Bounds how many crops go through one forward pass -- a busy frame
        # can produce far more detections than fit comfortably in GPU memory
        # at once, and this used to be unbounded (one batch per frame).
        self._batch_size = max(1, batch_size)
        self._use_half_on_cuda = use_half_on_cuda
        self._use_half = False  # resolved once the device is known
        self._model = None
        self._transform = None
        self._torch = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from torchvision import transforms
        from torchvision.models import ResNet50_Weights, resnet50

        from mash_reid.device import resolve_device

        self._torch = torch
        self._device = resolve_device(self._device)
        self._use_half = self._use_half_on_cuda and self._device.startswith("cuda")

        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        # Replace the 1000-class classifier with identity -> penultimate 2048-d.
        model.fc = torch.nn.Identity()
        model.eval().to(self._device)
        if self._use_half:
            model = model.half()
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
        tensor = self._transform(rgb)
        return tensor.half() if self._use_half else tensor

    def embed(self, crop: np.ndarray) -> np.ndarray:
        return self.embed_batch([crop])[0]

    def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_model()
        torch = self._torch

        # Chunk instead of one forward pass for the whole frame's detections
        # -- see the ``batch_size`` note in __init__.
        chunks = []
        for start in range(0, len(crops), self._batch_size):
            chunk_crops = crops[start:start + self._batch_size]
            batch = torch.stack([self._preprocess(c) for c in chunk_crops]).to(self._device)
            with torch.no_grad():
                feats = self._model(batch)
            chunks.append(feats.float().cpu().numpy().astype(np.float32))
        feats = np.concatenate(chunks, axis=0)
        return _l2_normalize(feats)


def get_default_embedder(
    device: str | None = None,
    batch_size: int = config.DEFAULT_EMBED_BATCH_SIZE,
) -> Embedder:
    """Factory for the default appearance model. Swap here to change globally."""
    return ResNet50Embedder(device=device, batch_size=batch_size)
