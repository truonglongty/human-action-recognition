# Human Action Recognition (HAR) - Nhận dạng Hành động Người từ Camera Giám sát

## Mô tả dự án

Hệ thống nhận dạng hành động người từ video camera giám sát sử dụng nhiều phương pháp học máy:
- Trích xuất đặc trưng thủ công + mô hình truyền thống (SVM tự viết, SVM scikit-learn, Random Forest)
- Học sâu end-to-end (CNN + LSTM với Keras/TensorFlow)
- Ứng dụng minh họa giao diện web (Flask)

## Các hành động nhận dạng
- WalkingWithDog, Punch, "umpRope, PushUps, Typing

## Cấu trúc thư mục

```
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── cnn_lstm_model.h5
│   ├── rf_model.pkl
│   ├── svc_model.pkl
│   └── svm_model.pkl
├── uploads/
├── outputs/
├── utils/
│   ├── object_detection.py
│   └── predict_utils.py
├── static/
│   ├── css/
│   └── js/
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   └── webcam.html
```

## Hướng dẫn cài đặt

1. **Clone repo:**
   ```sh
   git clone https://github.com/truonglongty/human-action-recognition.git
   cd human-action-recognition
   ```

2. **Cài đặt thư viện:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Chuẩn bị mô hình:**
   - Đảm bảo các file mô hình `.h5`, `.pkl` đã có trong thư mục `models/`
   - Nếu chưa có, hãy huấn luyện theo hướng dẫn trong tài liệu hoặc liên hệ tác giả.

4. **Chạy ứng dụng web:**
   ```sh
   python app.py
   ```
   Truy cập [http://localhost:5000](http://localhost:5000)

## Hướng dẫn sử dụng

- Tải lên video hoặc sử dụng webcam để nhận dạng hành động.
- Xem kết quả dự đoán và video đã xử lý.
- Có thể chọn mô hình dự đoán (SVM, RF, CNN+LSTM).

## Các phương pháp đã triển khai

1. **Trích xuất đặc trưng thủ công + mô hình truyền thống**
   - Optical Flow, HOG, HOF
   - SVM tự viết, SVC (scikit-learn), Random Forest

2. **Học sâu (Deep Learning)**
   - CNN (trích xuất đặc trưng không gian) + LSTM (mô hình hóa chuỗi thời gian)
   - Keras/TensorFlow

3. **Phát hiện người (YOLOv8)**
   - Tập trung vào vùng chứa người để giảm nhiễu nền

## Đánh giá & trực quan hóa

- Tính Accuracy, F1-score, Confusion Matrix cho từng mô hình
- So sánh hiệu năng các phương pháp
- Biểu đồ trực quan hóa kết quả

## Demo Video Output

<div align="center">

<h4>Video demo nhận dạng hành động "Punch"</h4>
<video width="480" controls>
  <source src="outputs/output_punch1.mp4" type="video/mp4">
  Trình duyệt của bạn không hỗ trợ video.
</video>

<h4>Video demo nhận dạng hành động "PushUps"</h4>
<video width="480" controls>
  <source src="outputs/output_pushups1.mp4" type="video/mp4">
  Trình duyệt của bạn không hỗ trợ video.
</video>

</div>

Nếu không xem được video trực tiếp trên GitHub, bạn có thể tải về từ thư mục [outputs/](outputs/).
