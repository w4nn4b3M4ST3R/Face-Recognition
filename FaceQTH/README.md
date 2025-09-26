# Hệ thống Nhận diện Khuôn mặt FaceQTH bằng MTCNN và FaceNet  
**Tác giả**: Võ Phúc Thịnh  
**Ngày**: 31/08/2025  

---

## Giới thiệu:  
Hệ thống này kết hợp **MTCNN** (phát hiện & căn chỉnh khuôn mặt) và **FaceNet** (tạo vector embedding), cùng với bộ phân loại (**SVM**) để **nhận diện khuôn mặt chính xác cao**.  

Hỗ trợ cả:  
- **Giao diện Web** (_Streamlit/WebRTC_)  
- **Giao diện GUI** (_PyQt_)  
- **Camera trực tiếp**  
- **Video có sẵn**  

---

## Điểm nổi bật:  

### GUI (PyQt)  
- Nút dễ sử dụng: **bật/tắt camera**, **thêm người dùng**, **nhận diện**.  
- Hiển thị **bounding box, tên, FPS** và **độ tin cậy (confidence)** trên ảnh trực tiếp.  
- Có thể gắn **logo thương hiệu**.  

### Web (Streamlit + WebRTC)  
- Hỗ trợ **nhận diện trực tiếp qua webcam** trong trình duyệt.  
- Hiển thị **bounding box, tên, FPS** và **độ tin cậy (confidence)**.  
- Giao diện đẹp, trực quan, dễ dùng.  

---

## Tính năng chính:  
- **Phát hiện & căn chỉnh khuôn mặt** bằng _MTCNN_ với **5 điểm mốc** (mắt, mũi, miệng).  
- **Sinh vector embedding** bằng _FaceNet_ (128 chiều).  
- **Nhận diện danh tính** bằng **SVM**.  
- Quản lý dataset:  
  - Ảnh gốc → `Dataset/FaceData/raw/`  
  - Ảnh đã căn chỉnh → `Dataset/FaceData/processed/` (_tự động sinh bounding boxes_).  

---

## Khởi động nhanh:  

### 1) Cài môi trường:  
```bash
python -m venv venv  
source venv/bin/activate  # Linux/Mac  
venv\Scripts\activate     # Windows  

pip install -r requirements.txt
```
*(Tải weight và lưu vào thư mục `Models` tại file `Models/weight_installing`)*
## Yêu cầu hệ thống:
- Python: 3.8 (đã thử và gợi ý).  
- OS: Windows / Linux / MacOS.  
- RAM: ≥ 4GB (khuyên dùng ≥ 8GB).  
- GPU: NVIDIA CUDA/cuDNN (tùy chọn, để tăng tốc).  
### 2) Chạy chương trình:
```bash
python GUI.py #giao diện local
streamlit run app.py #giao diện web
```
## Cách sử dụng:
*1. Thêm dữ liệu người dùng:*  
-Thêm ảnh trực tiếp vào file `Dataset/FaceData/raw`.  
-Chụp ảnh trực tiếp từ **webcam**  
*2. Căn chỉnh và tiền xử lý:*  
```bash
python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25 #cắt riêng khuôn mặt
```
(sinh ảnh căn chỉnh và bounding boxes trong `Dataset/FaceData/processed`)  
*3. Huấn luyện bộ phân loại:*  
```bash
python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000 #model training
```
*4. Nhận diện trực tiếp hoặc qua video:*  
Chạy webcam, video bằng local hoặc web, hiển thị **tên**, **độ tự tin** và **FPS**.  
*5. Đánh giá hiệu năng:*  
```bash
python validate_on_ifw.py
```
## Kiến trúc hệ thống:
```mermaid
flowchart LR
A[Ảnh / Video đầu vào] --> B[MTCNN: Phát hiện & Căn chỉnh]
B --> C[Ảnh 160x160]

subgraph Trích đặc trưng
    C --> D[FaceNet: Embedding 128D]
end

subgraph Phân loại
    D --> E{SVM}
end

E --> F[Kết quả: Tên, Confidence, FPS]
```
## Cấu trúc dự án:  
```text
FaceQTH/
├── Dataset/
│   └── FaceData/
│       ├── raw/          # Ảnh gốc
│       └── processed/    # Ảnh căn chỉnh + bounding boxes
├── Models/
│   ├── 20180402-114759.pb  # FaceNet pretrained
│   ├── facemodel.pkl       # Bộ phân loại
│   └── *.ckpt              # Checkpoints
├── src/
│   ├── align/                # MTCNN detector
│   ├── facenet.py            # Tiện ích FaceNet
│   ├── classifier.py         # Huấn luyện classifier
│   ├── face_rec_cam.py       # Nhận diện camera
│   ├── face_rec_cam_web.py   # Phiên bản Streamlit
│   └── align_dataset_mtcnn.py # Tiền xử lý dataset
├── test/        # Unit test
├── video/       # Video test
├── app.py       # Entry cho Streamlit
├── GUI.py       # Entry cho PyQt GUI
└── requirements.txt  # Thư viện
```
## Cấu hình và tinh chỉnh:  
- Ngưỡng phát hiện: điều chỉnh trong dectect_face.py.  
- Kích thước ảnh: 160x160 (mặc định).  
- Bộ phân loại: SVM (mặc định).
- FPS: tính bằng time.time() trong vòng lặp.  
## Hiệu năng và đánh giá:  
![Với tệp data hiện tại, model nhận diện rất tốt](matrix.jpg)
## Ảnh minh họa:  
- Giao diện local: `docs/GUI.png`.  
- Giao diện webcam local: `docs/GUI_cam.png`.  
- Giao diện video local: `docs/GUI_video.png`.  
- Giao diện web: `docs/web.png`.  
- Giao diện webcam web: `docs/web_cam.png`.  
- Giao diện webcam video: `docs/web_video.png`.  
## Phát triển và đóng góp:  
*1. Quy trình đóng góp:*  
- Fork repository và tạo nhánh mới (`feature/*`).  
- Commit & push thay đổi của bạn.  
- Mở Pull Request.  
*2. Báo cáo lỗi:* Vui lòng cung cấp các thông tin sau:  
- Hệ điều hành (OS).
- Phiên bản Python.
- Log đầy đủ.
- Các bước tái hiện lỗi.
- Screenshot (nếu có).
## Lời cảm ơn và liên hệ:  
- **MTCNN** – nhóm tác giả gốc đã phát triển mô hình Multi-task Cascaded Convolutional Networks cho phát hiện khuôn mặt.  
- **FaceNet** – nhóm nghiên cứu của Google đã xây dựng nền tảng nhận diện khuôn mặt mạnh mẽ, mở đường cho các hệ thống embedding hiện đại.  
- **Scikit-learn** – cộng đồng phát triển đã cung cấp công cụ SVM và nhiều thuật toán machine learning tiện lợi.  
- **Streamlit & Streamlit-webrtc** – đội ngũ Streamlit đã giúp việc xây dựng giao diện và chạy real-time camera trở nên dễ dàng.  
- **TensorFlow – framework** hỗ trợ huấn luyện và triển khai mô hình deep learning.  
**Tác giả**: *Võ Phúc Thịnh*  
Nếu có thắc mắc, đề xuất hay muốn đóng góp, bạn có thể mở **Issue** hoặc **Pull Request** trực tiếp trên GitHub repository.
