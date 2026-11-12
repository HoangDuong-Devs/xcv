"""
Template Adapter: Copy this file to integrate a new CV model into xcv.

Steps:
  1. Copy this file to xcv/adapters/<your_model>.py
  2. Implement __init__, get_model, get_target_layers, preprocess, predict
  3. Register your adapter class in xcv/adapters/__init__.py
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn

from xcv.core.base_adapter import BaseModelAdapter, SampleCase


class TemplateCustomAdapter(BaseModelAdapter):
    id = "template_model"
    display_name = "Template Model (Example)"
    description = "Bộ khung mẫu để gắn bất kỳ mô hình PyTorch / ONNX / YOLO nào vào xcv."
    task_type = "classification"  # "classification" | "multilabel" | "detection"
    class_names = ["class_0", "class_1"]
    input_size = (224, 224)

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device)
        # TODO: Load your PyTorch nn.Module + checkpoint weights here
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, len(self.class_names)),
        ).to(self.device)
        self.model.eval()

    def get_model(self) -> nn.Module:
        return self.model

    def get_target_layers(self) -> List[nn.Module]:
        # Point to the last 2D spatial convolutional layer before global pooling
        return [self.model[0]]

    def preprocess(self, img_bgr: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        vis_rgb = cv2.resize(img_rgb, self.input_size).astype(np.float32) / 255.0
        # Add your model's mean/std normalization here if needed
        tensor = torch.from_numpy(vis_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        return tensor, vis_rgb

    def predict(self, input_tensor: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            logits = self.model(input_tensor)[0]
            probs = torch.softmax(logits, dim=0).cpu().numpy()
        return {name: float(probs[i]) for i, name in enumerate(self.class_names)}

    def get_sample_cases(self) -> List[SampleCase]:
        return []
