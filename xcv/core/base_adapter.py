"""Base abstractions for plugging any CV model into xcv."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


@dataclass
class SampleCase:
    """Metadata for a pre-loaded diagnostic sample image."""
    id: str
    title: str
    desc: str
    image_path: str
    default_target_cls: int = 0
    tag: str = "sample"  # e.g., "fp", "tp", "clean"


class BaseModelAdapter(ABC):
    """
    Contract for integrating any PyTorch / timm / custom CV model into xcv.

    To add a new model:
      1. Subclass BaseModelAdapter in xcv/adapters/
      2. Implement load, preprocess, predict, and get_target_layers()
      3. Register it in xcv/adapters/__init__.py
    """

    id: str = "base"
    display_name: str = "Base Model"
    description: str = ""
    task_type: str = "classification"  # "classification" | "multilabel" | "detection" | "embedding"
    class_names: List[str] = []
    input_size: Tuple[int, int] = (256, 256)

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the loaded PyTorch nn.Module in eval() mode."""

    @abstractmethod
    def get_target_layers(self) -> List[nn.Module]:
        """Return the spatial layer(s) (usually last Conv2d stage) to extract Grad-CAM from."""

    @abstractmethod
    def preprocess(self, img_bgr: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess OpenCV BGR image into:
          - input_tensor: [1, C, H, W] on self.device
          - vis_rgb_float: [H, W, 3] float32 in [0.0, 1.0] RGB for heatmap overlay
        """

    @abstractmethod
    def predict(self, input_tensor: torch.Tensor) -> Dict[str, float]:
        """Run inference and return mapping {class_name: probability_or_score}."""

    def get_target_callable(self, target_cls_idx: int) -> Any:
        """
        Return target callable for pytorch-grad-cam.
        Override this for custom tasks (e.g. YOLO boxes, Cosine similarity, Segmentation masks).
        """
        return ClassifierOutputTarget(target_cls_idx)

    def get_reshape_transform(self) -> Optional[Callable]:
        """
        Return reshape transform for Vision Transformers (ViT/Swin) if needed.
        Default None for standard CNNs.
        """
        return None

    def get_embedding(self, input_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        Optional hook for future 2D Embedding Space Visualization (t-SNE / UMAP).
        Return 1D feature vector before the classification head.
        """
        return None

    def get_sample_cases(self) -> List[SampleCase]:
        """Return optional list of diagnostic demo images for this adapter."""
        return []
