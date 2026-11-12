"""Generic CAM computation engine wrapping pytorch-grad-cam."""

import base64
from typing import Dict, Any
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from .base_adapter import BaseModelAdapter

SUPPORTED_CAM_METHODS = {
    "GradCAM": GradCAM,
    "GradCAM++": GradCAMPlusPlus,
    "EigenCAM": EigenCAM,
    "LayerCAM": LayerCAM,
}


def _to_b64(img_uint8_rgb: np.ndarray) -> str:
    """Encode an RGB uint8 numpy image to base64 JPEG string."""
    bgr = cv2.cvtColor(img_uint8_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode("utf-8")


class CamEngine:
    """Executes CAM attribution across any BaseModelAdapter."""

    def __init__(self):
        self._cam_cache: Dict[tuple, Any] = {}

    def _get_cam(self, adapter: BaseModelAdapter, method_name: str):
        key = (adapter.id, method_name)
        if key not in self._cam_cache:
            cam_cls = SUPPORTED_CAM_METHODS.get(method_name, GradCAM)
            self._cam_cache[key] = cam_cls(
                model=adapter.get_model(),
                target_layers=adapter.get_target_layers(),
                reshape_transform=adapter.get_reshape_transform(),
            )
        return self._cam_cache[key]

    def analyze(
        self,
        adapter: BaseModelAdapter,
        img_bgr: np.ndarray,
        target_cls_idx: int = 0,
        method_name: str = "GradCAM",
    ) -> Dict[str, Any]:
        """Run prediction and generate CAM heatmap + diagnostics."""
        input_tensor, vis_rgb_float = adapter.preprocess(img_bgr)
        scores = adapter.predict(input_tensor)

        cam = self._get_cam(adapter, method_name)
        targets = [adapter.get_target_callable(target_cls_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        h, w = vis_rgb_float.shape[:2]
        orig_uint8 = np.clip(vis_rgb_float * 255.0, 0, 255).astype(np.uint8)

        # Standalone JET colormap for frontend alpha slider blending
        heat_bgr = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

        # Blended preview image
        overlay_rgb = show_cam_on_image(vis_rgb_float, grayscale_cam, use_rgb=True)

        # Locate spatial peak
        peak_y, peak_x = np.unravel_index(np.argmax(grayscale_cam), grayscale_cam.shape)
        hot_loc = []
        hot_loc.append("Mép trên" if peak_y < h * 0.33 else ("Mép dưới" if peak_y > h * 0.67 else "Giữa dọc"))
        hot_loc.append("Góc trái" if peak_x < w * 0.33 else ("Góc phải" if peak_x > w * 0.67 else "Giữa ngang"))

        return {
            "scores": scores,
            "target_class": adapter.class_names[target_cls_idx] if target_cls_idx < len(adapter.class_names) else str(target_cls_idx),
            "method": method_name,
            "orig_b64": _to_b64(orig_uint8),
            "heat_b64": _to_b64(heat_rgb),
            "overlay_b64": _to_b64(overlay_rgb),
            "peak_x": int(peak_x),
            "peak_y": int(peak_y),
            "width": int(w),
            "height": int(h),
            "hotspot_desc": " · ".join(hot_loc),
        }
