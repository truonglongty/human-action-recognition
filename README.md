Đây là kế hoạch tổng thể để thiết kế và triển khai ứng dụng Flask với đầy đủ các yêu cầu mà bạn đã liệt kê. Chúng ta sẽ chia thành các thành phần chính:

Yêu cầu cụ thể:
Chức năng đầu vào:
Cho phép người dùng tải lên video (hỗ trợ các định dạng phổ biến như MP4, AVI) thông qua giao diện web.
Hỗ trợ mở webcam trực tiếp để ghi lại và xử lý video thời gian thực.
Có nút hoặc tùy chọn để chuyển đổi giữa hai chế độ (tải video hoặc webcam).
Chức năng đầu ra:
Hiển thị video đầu ra trên giao diện web với các khung bao quanh (bounding box) các đối tượng được xác định là con người trong video
Gắn nhãn hành động dự đoán (ví dụ: "WalkingWithDog", "Punch", "JumpRope", "PushUps", "Typing") trên khung bao quanh hoặc ở góc video.
Hiển thị xác suất dự đoán của hành động (nếu có) để tăng tính minh bạch.
Lựa chọn mô hình dự đoán:
Cho phép người dùng chọn một trong các mô hình đã được huấn luyện trước đó:
SVM: Sử dụng mô hình SVM với đặc trưng trích xuất từ ResNet50.
Random Forest: Sử dụng mô hình Random Forest với đặc trưng từ ResNet50.
CNN-LSTM: Mô hình học sâu kết hợp CNN và LSTM.
Giao diện cần có menu thả xuống (dropdown) hoặc nút radio để chọn mô hình.
Các mô hình được tải từ file đã lưu (ví dụ: .pkl cho SVM/Random Forest, .h5 cho CNN-LSTM) và được tích hợp sẵn trong ứng dụng.
Giao diện người dùng (UI):
Thiết kế giao diện đẹp, hiện đại, dễ sử dụng sử dụng HTML, CSS (có thể dùng framework như Bootstrap), và JavaScript.
Giao diện bao gồm:
Khu vực để tải video hoặc bật webcam.
Nút chọn mô hình dự đoán.
Khu vực hiển thị video đầu ra với khung bao quanh và nhãn hành động.
Thông báo trạng thái (ví dụ: "Đang xử lý video...", "Dự đoán hoàn tất").
Đảm bảo giao diện responsive, hoạt động tốt trên cả máy tính và thiết bị di động.
Sử dụng màu sắc hài hòa, font chữ dễ đọc, và bố cục rõ ràng.
Xử lý và dự đoán:
Sử dụng OpenCV để xử lý video (đọc khung hình, phát hiện đối tượng, vẽ khung bao quanh).
Đối với video tải lên:
Xử lý từng khung hình, trích xuất đặc trưng (nếu dùng SVM/Random Forest) hoặc đưa trực tiếp vào mô hình CNN-LSTM.
Lưu video đầu ra với khung bao quanh và nhãn hành động vào thư mục tạm thời và cung cấp link tải về.
Đối với webcam:
Xử lý video thời gian thực, hiển thị kết quả trực tiếp trên giao diện web (sử dụng WebRTC hoặc stream video qua Flask).
Tối ưu hóa hiệu suất để đảm bảo xử lý mượt mà, đặc biệt với webcam.


---

### 🔧 1. Cấu trúc thư mục dự án

```
action_app/
│
├── app.py                         # Flask app chính
├── static/
│   ├── css/                       # CSS custom
│   └── js/                        # JavaScript cho xử lý UI/webcam
├── templates/
│   └── index.html                 # Giao diện chính
├── uploads/                      # Lưu video người dùng tải lên
├── outputs/                      # Lưu video kết quả
├── models/
│   ├── svm_model.pkl
│   ├── rf_model.pkl
│   └── cnn_lstm_model.h5
├── utils/
│   ├── feature_extractor.py      # Trích xuất đặc trưng ResNet50
│   ├── predictor.py              # Dự đoán hành động
│   └── detector.py               # Phát hiện người
```


```
action_app/
│
├── app.py                         # Flask app chính
├── static/
│   ├── css/                       # CSS custom
│   └── js/                        # JavaScript cho xử lý UI/webcam
├── templates/
│   └── index.html                 # Giao diện chính
├── uploads/                      # Lưu video người dùng tải lên
├── outputs/                      # Lưu video kết quả
├── models/
│   ├── svm_model.pkl
│   ├── rf_model.pkl
│   └── cnn_lstm_model.h5
├── utils/
│   ├── object_detection.py      
│   ├── predict_utils.py              
```

---

### 🧠 2. Mô hình dự đoán

* **SVM/Random Forest**:

  * Dùng `ResNet50` trích đặc trưng từ từng frame: `(2048,)`
* **CNN-LSTM**:

  * Nhận tensor có dạng `(20, 64, 64, 1)` hoặc tương đương chuỗi frame video đã resize.

---

### 📸 3. Phát hiện người

* Dùng OpenCV Haar Cascade hoặc YOLOv5/YOLOv8 (nếu muốn chính xác hơn).
* Trả về bounding box + cắt frame người để dự đoán hành động.

---

### 🌐 4. Flask App – `app.py`

#### Các endpoint:

* `/`: Trang chính.
* `/upload`: Xử lý video tải lên, lưu vào `uploads/`.
* `/predict`: Dự đoán hành động trên video hoặc webcam.
* `/video_feed`: Stream webcam video thời gian thực.
* `/download/<filename>`: Tải video đầu ra đã xử lý.

---

### 🎨 5. Giao diện `index.html`

Dùng Bootstrap:

* **Dropdown chọn mô hình** (SVM / RF / CNN-LSTM).
* **Tabs**:

  * Tải video từ máy.
  * Xử lý webcam thời gian thực.
* **Hiển thị video đầu ra** (với box + nhãn).
* Thông báo trạng thái và nút tải kết quả (nếu có).

---

### ⚙️ 6. Luồng xử lý video

#### Với video upload:

1. Đọc video → Lấy từng frame.
2. Phát hiện người trong frame.
3. Dự đoán hành động trên từng người.
4. Vẽ bounding box + nhãn + xác suất.
5. Ghi lại video kết quả → trả link tải.

#### Với webcam:

1. Dùng OpenCV mở webcam.
2. Phát hiện người + dự đoán hành động liên tục.
3. Trả luồng video dưới dạng MJPEG tới HTML `<img>`.

---

### 🧪 7. Tối ưu hóa

* Resize và crop người đúng tỷ lệ đầu vào model.
* Cache mô hình sau khi load.
* Hạn chế số lượng frame xử lý mỗi giây nếu cần (webcam).

---
