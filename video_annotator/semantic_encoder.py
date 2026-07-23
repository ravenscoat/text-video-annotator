"""Optional frozen semantic image/text encoder interface.

Weights are loaded only when the caller explicitly constructs the encoder.
Tests can inject a small mock implementing the same methods.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class SemanticEncoder(Protocol):
    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray: ...
    def encode_text(self, texts: Sequence[str]) -> np.ndarray: ...


class FrozenClipEncoder:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        self.model_id, self.device = model_id, device
        self._model = None
        self._processor = None

    def load(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("Install Transformers with CLIP support before enabling semantic embeddings") from exc
        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    @staticmethod
    def _as_tensor(output):
        """Handle both tensor and structured outputs across Transformers versions."""
        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds
        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            return output.text_embeds
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        return output

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        self._ensure_loaded()
        import torch
        inputs = self._processor(images=list(images), return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            embeddings = self._as_tensor(self._model.get_image_features(**inputs))
            embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
        return embeddings.cpu().numpy()

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_loaded()
        import torch
        inputs = self._processor(text=list(texts), return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.inference_mode():
            embeddings = self._as_tensor(self._model.get_text_features(**inputs))
            embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
        return embeddings.cpu().numpy()
