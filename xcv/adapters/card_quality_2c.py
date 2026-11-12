"""Adapter for eKYC Card Quality 2-Class Gate (Cut Corner & Occlusion)."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn

from xcv.core.base_adapter import BaseModelAdapter, SampleCase

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class CardQuality2CNet(nn.Module):
    """MobileNetV3-Large multi-label classifier for card corner cut & occlusion."""

    def __init__(self, backbone_name: str = "mobilenetv3_large_100", num_attribs: int = 2, dropout_p: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            num_features = self.backbone(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.Hardswish(),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_attribs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)


class CardQuality2CAdapter(BaseModelAdapter):
    id = "card_quality_2c"
    display_name = "eKYC Card Quality (2-Class: Cut & Occluded)"
    description = "MobileNetV3-Large (256x256) kiểm tra mất góc & ngón tay che thông tin thẻ CCCD."
    task_type = "multilabel"
    class_names = ["is_cut_corner", "is_occluded"]
    input_size = (256, 256)

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, weights_path: Optional[Path] = None, device: Optional[torch.device] = None):
        super().__init__(device)
        self.weights_path = weights_path or (ROOT_DIR / "weights/cccd_quality_2class.pt")
        self.model = CardQuality2CNet().to(self.device)
        if self.weights_path.exists():
            ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=False)
            state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
            self.model.load_state_dict(state)
        else:
            print(f"[Warning] Weights not found at {self.weights_path}; running with random weights.")
        self.model.eval()

    def get_model(self) -> nn.Module:
        return self.model

    def get_target_layers(self) -> List[nn.Module]:
        # MobileNetV3 places global pooling before conv_head; blocks[-1] is the last 2D spatial stage (8x8).
        return [self.model.backbone.blocks[-1]]

    def preprocess(self, img_bgr: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        vis_rgb = cv2.resize(img_rgb, self.input_size).astype(np.float32) / 255.0
        norm = (vis_rgb - self.MEAN) / self.STD
        tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        return tensor, vis_rgb

    def predict(self, input_tensor: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            probs = torch.sigmoid(self.model(input_tensor))[0].cpu().numpy()
        return {name: float(probs[i]) for i, name in enumerate(self.class_names)}

    def get_embedding(self, input_tensor: torch.Tensor) -> Optional[np.ndarray]:
        with torch.no_grad():
            emb = self.model.backbone(input_tensor)[0].cpu().numpy()
        return emb

    def get_sample_cases(self) -> List[SampleCase]:
        samples_dir = ROOT_DIR / "demo_samples"
        candidates = [
            SampleCase("val_0675", "val_0675 · Ngón tay cầm viền (False Positive)", "Ngón tay ở mép nhựa ngoài, không che chữ — model đoán che 99.7%.", str(samples_dir / "val_0675.jpg"), default_target_cls=1, tag="fp"),
            SampleCase("val_0457", "val_0457 · Vệt lóa đèn chéo (False Positive)", "Vệt sáng phản chiếu trên thẻ, không có vật cản — model đoán che 89.6%.", str(samples_dir / "val_0457.jpg"), default_target_cls=1, tag="fp"),
            SampleCase("val_0220", "val_0220 · Che nửa thẻ thật (True Positive)", "Vật/thẻ khác đè lên nửa trên CCCD.", str(samples_dir / "val_0220.jpg"), default_target_cls=1, tag="tp"),
            SampleCase("val_0227", "val_0227 · Ngón tay đè chữ thật (True Positive)", "Ngón tay đè trực tiếp lên vùng thông tin mặt sau.", str(samples_dir / "val_0227.jpg"), default_target_cls=1, tag="tp"),
            SampleCase("val_0048", "val_0048 · Cắt góc thật (True Positive Cut)", "Góc trên bên trái bị cắt vát.", str(samples_dir / "val_0048.jpg"), default_target_cls=0, tag="tp"),
            SampleCase("val_0111", "val_0111 · Cắt góc sâu (True Positive Cut)", "Vết cắt khuyết sâu góc thẻ.", str(samples_dir / "val_0111.jpg"), default_target_cls=0, tag="tp"),
            SampleCase("val_0000", "val_0000 · Thẻ sạch đạt chuẩn (Clean)", "Thẻ nguyên vẹn, không che khuất, không cắt góc.", str(samples_dir / "val_0000.jpg"), default_target_cls=1, tag="clean"),
        ]
        return [c for c in candidates if Path(c.image_path).exists()]
