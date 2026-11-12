# 🔬 xcv — Explainable Computer Vision Studio

Bộ khung công cụ trực quan hóa và chẩn đoán mô hình Thị giác máy tính (**XAI & Model Diagnostics**), được thiết kế theo kiến trúc **Plugin / Adapter** để có thể cắm nhanh (swap) bất kỳ mô hình PyTorch / `timm` / YOLO / Face Recognition nào vào soi lỗi và phân tích dữ liệu.

---

## ✨ Tính năng hiện có (v0.1 Demo)

1. **Hot-Swap Model Adapters:** Chuyển đổi qua lại giữa các mô hình ngay trên Web UI dropdown mà không cần sửa code server.
2. **Đa thuật toán CAM:** Hỗ trợ trực tiếp `GradCAM`, `GradCAM++`, `EigenCAM` và `LayerCAM`.
3. **Interactive Heatmap Explorer:**
   - Thanh trượt Opacity (0%–100%) giúp nhìn thấu vùng ảnh gốc nằm dưới bản đồ nhiệt.
   - Tự động tính tọa độ đỉnh kích hoạt (`Hotspot Peak`) để phát hiện nhanh hiện tượng *học tắt (Shortcut Learning)*.
   - Hỗ trợ upload ảnh trực tiếp từ máy hoặc duyệt qua bộ mẫu chẩn đoán có sẵn (`demo_samples/`).
4. **Sẵn sàng cho Embedding Explorer (Hook `get_embedding`):** Interface `BaseModelAdapter` đã thiết kế sẵn cổng trích xuất vector đặc trưng để mở rộng tính năng chiếu cụm 2D (**UMAP / t-SNE**).

---

## 📂 Cấu trúc thư mục

```text
xcv/
├── xcv/
│   ├── core/
│   │   ├── base_adapter.py     # Interface chuẩn cho mọi mô hình CV
│   │   └── cam_engine.py       # Engine chạy Grad-CAM / GradCAM++ / EigenCAM
│   ├── adapters/
│   │   ├── __init__.py         # Registry đăng ký & nạp nóng các model
│   │   ├── card_quality_2c.py  # Demo: eKYC Card Quality 2-Class (Cut & Occluded)
│   │   └── template_adapter.py # Bộ khung mẫu để gắn model mới trong 5 phút
│   └── server/
│       ├── app.py              # FastAPI server
│       └── templates/
│           └── index.html      # Giao diện Web UI (Bootstrap 5)
├── demo_samples/               # Các ảnh mẫu đi kèm (val_0675, val_0457, ...)
├── weights/                    # Nơi chứa file trọng số (.pt/.pth, đã gitignore)
├── requirements.txt
└── README.md
```

---

## 🚀 Cách cài đặt & Chạy Demo

### 1. Cài đặt thư viện (trong môi trường Python có sẵn PyTorch):
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy Web UI:
```bash
python -m xcv.server.app
```
Mở trình duyệt truy cập: **`http://localhost:8087`**

---

## 🔌 Hướng dẫn Swap / Thêm mô hình mới (3 bước)

Hệ thống thiết kế theo mô hình Adapter để giải quyết từng bài toán riêng mà không đụng vào phần lõi:

1. **Tạo file Adapter mới** bằng cách copy từ `xcv/adapters/template_adapter.py` sang `xcv/adapters/<ten_bai_toan>.py`.
2. **Cài đặt 4 hàm cốt lõi** trong class kế thừa `BaseModelAdapter`:
   - `get_model()` $\rightarrow$ Trả về PyTorch `nn.Module` ở chế độ `.eval()`.
   - `get_target_layers()` $\rightarrow$ Chỉ định tầng Conv 2D cuối cùng trước lớp pooling (VD: `[model.blocks[-1]]` với MobileNetV3, hoặc `[model.layer4[-1]]` với ResNet).
   - `preprocess(img_bgr)` $\rightarrow$ Chuẩn hóa ảnh đầu vào đúng như lúc train (resize, mean/std).
   - `predict(input_tensor)` $\rightarrow$ Trả về `dict` xác suất cho từng class.
   - *(Nâng cao)* Nếu làm **YOLO / Detection / ArcFace Cosine**, override thêm hàm `get_target_callable()` để định nghĩa mục tiêu tính gradient riêng.
   - *(Nâng cao)* Nếu làm **Vision Transformer (ViT / Swin)**, override thêm `get_reshape_transform()`.
3. **Đăng ký vào `xcv/adapters/__init__.py`:**
   ```python
   ADAPTER_REGISTRY[MyNewAdapter.id] = MyNewAdapter
   ```
   Khởi động lại server là mô hình mới sẽ tự động xuất hiện trong dropdown trên Web UI!

---

## 🗺️ Roadmap phát triển tiếp

- [ ] **Tab 2D Embedding Space (UMAP / t-SNE):** Chiếu toàn bộ tập validation lên mặt phẳng 2D qua hook `get_embedding()`, bấm vào từng chấm bị lạc cụm để bật ngay Grad-CAM của ảnh đó.
- [ ] **YOLO / Detection Eigen-CAM Adapter:** Hỗ trợ soi trực tiếp các model `yolov8` / `yolo26` nhận diện object và segmentation.
- [ ] **ViT / Face Recognition Adapter:** Hỗ trợ `Attention Rollout` và `Cosine Similarity Attribution` cho bài toán nhận diện khuôn mặt / Liveness / Deepfake.
