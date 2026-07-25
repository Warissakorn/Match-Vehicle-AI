"""Tests for the embedder's batching. Needs torch (skipped where it isn't
installed, e.g. CI's lightweight test env) but never downloads real ResNet
weights or touches a real device -- the model is swapped out with a fake
that just records how big each forward-pass batch was.
"""

import numpy as np
import pytest


def test_get_default_embedder_passes_through_batch_size():
    from mash_reid.embedder import ResNet50Embedder, get_default_embedder

    emb = get_default_embedder(device="cpu", batch_size=32)
    assert isinstance(emb, ResNet50Embedder)
    assert emb._batch_size == 32


def test_batch_size_is_clamped_to_at_least_one():
    from mash_reid.embedder import ResNet50Embedder

    emb = ResNet50Embedder(batch_size=0)
    assert emb._batch_size == 1


def test_embed_batch_chunks_large_inputs():
    torch = pytest.importorskip("torch")
    from mash_reid.embedder import ResNet50Embedder

    call_sizes = []

    class _FakeModel:
        def __call__(self, batch):
            call_sizes.append(batch.shape[0])
            return torch.zeros((batch.shape[0], 8))

    emb = ResNet50Embedder(device="cpu", batch_size=64)
    emb.dim = 8
    # Bypass the real (network-downloading) model load -- embed_batch's
    # internal _ensure_model() is a no-op once _model is already set.
    emb._model = _FakeModel()
    emb._torch = torch
    emb._device = "cpu"
    emb._transform = lambda rgb: torch.zeros(3, 4, 4)

    crops = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(200)]
    feats = emb.embed_batch(crops)

    assert feats.shape == (200, 8)
    # 200 crops at batch_size=64 -> three full chunks plus one remainder.
    assert call_sizes == [64, 64, 64, 8]


def test_embed_batch_single_chunk_when_under_batch_size():
    torch = pytest.importorskip("torch")
    from mash_reid.embedder import ResNet50Embedder

    call_sizes = []

    class _FakeModel:
        def __call__(self, batch):
            call_sizes.append(batch.shape[0])
            return torch.zeros((batch.shape[0], 8))

    emb = ResNet50Embedder(device="cpu", batch_size=64)
    emb.dim = 8
    emb._model = _FakeModel()
    emb._torch = torch
    emb._device = "cpu"
    emb._transform = lambda rgb: torch.zeros(3, 4, 4)

    crops = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(5)]
    feats = emb.embed_batch(crops)

    assert feats.shape == (5, 8)
    assert call_sizes == [5]


def test_embed_batch_empty_returns_empty_without_touching_model():
    from mash_reid.embedder import ResNet50Embedder

    emb = ResNet50Embedder(device="cpu")
    feats = emb.embed_batch([])
    assert feats.shape == (0, emb.dim)
    assert emb._model is None  # never loaded -- nothing to embed
