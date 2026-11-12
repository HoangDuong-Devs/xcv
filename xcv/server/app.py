"""Interactive Web Server for xcv (Explainable Computer Vision)."""

from dataclasses import asdict
from pathlib import Path
from typing import Dict
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from xcv.adapters import ADAPTER_REGISTRY, create_adapter
from xcv.core.base_adapter import BaseModelAdapter
from xcv.core.cam_engine import SUPPORTED_CAM_METHODS, CamEngine

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

app = FastAPI(title="xcv — Explainable Computer Vision", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cam_engine = CamEngine()
_loaded_adapters: Dict[str, BaseModelAdapter] = {}


def get_adapter(model_id: str) -> BaseModelAdapter:
    if model_id not in _loaded_adapters:
        _loaded_adapters[model_id] = create_adapter(model_id)
    return _loaded_adapters[model_id]


@app.get("/api/meta")
def get_metadata():
    """Return registered models and supported CAM methods."""
    models = []
    for mid, cls in ADAPTER_REGISTRY.items():
        models.append({
            "id": mid,
            "display_name": cls.display_name,
            "description": cls.description,
            "task_type": cls.task_type,
            "class_names": cls.class_names,
        })
    return {
        "models": models,
        "cam_methods": list(SUPPORTED_CAM_METHODS.keys()),
        "default_model": "card_quality_2c",
    }


@app.get("/api/cases")
def get_cases(model_id: str = Query("card_quality_2c")):
    """Return built-in demo cases for the selected model adapter."""
    adapter = get_adapter(model_id)
    return [asdict(c) for c in adapter.get_sample_cases()]


@app.get("/api/analyze_case")
def analyze_case(
    model_id: str = Query("card_quality_2c"),
    case_id: str = Query(...),
    target_cls: int = Query(0),
    method: str = Query("GradCAM"),
):
    adapter = get_adapter(model_id)
    cases = adapter.get_sample_cases()
    case = next((c for c in cases if c.id == case_id), None)
    if not case:
        return JSONResponse({"error": "Case not found"}, status_code=404)

    img_bgr = cv2.imread(case.image_path)
    if img_bgr is None:
        return JSONResponse({"error": f"Failed to read {case.image_path}"}, status_code=404)

    res = cam_engine.analyze(adapter, img_bgr, target_cls_idx=target_cls, method_name=method)
    res["case_info"] = asdict(case)
    return res


@app.post("/api/analyze_upload")
async def analyze_upload(
    file: UploadFile = File(...),
    model_id: str = Form("card_quality_2c"),
    target_cls: int = Form(0),
    method: str = Form("GradCAM"),
):
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse({"error": "Invalid image file"}, status_code=400)

    adapter = get_adapter(model_id)
    res = cam_engine.analyze(adapter, img_bgr, target_cls_idx=target_cls, method_name=method)
    res["case_info"] = {
        "id": "upload",
        "title": f"Uploaded: {file.filename}",
        "desc": "Phân tích trực tiếp trên ảnh người dùng tải lên.",
        "tag": "upload",
    }
    return res


@app.get("/", response_class=HTMLResponse)
def index_page():
    html_path = TEMPLATES_DIR / "index.html"
    return html_path.read_text(encoding="utf-8")


def main():
    uvicorn.run("xcv.server.app:app", host="0.0.0.0", port=8087, reload=False)


if __name__ == "__main__":
    main()
