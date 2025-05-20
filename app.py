from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
import os
import cv2
from werkzeug.utils import secure_filename
from utils.predict_utils import predict_video, gen_webcam_frames
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
SELECTED_CLASSES = ["WalkingWithDog", "Punch", "JumpRope", "PushUps", "Typing"]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=2000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.models = {}
        self.classes = None
        self.feature_means = None
        self.feature_stds = None
    
    def _normalize_features(self, X, fit=False):
        """Chuẩn hóa đặc trưng để tăng tốc độ hội tụ"""
        if fit:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0) + 1e-8  # Tránh chia cho 0
            return (X - self.feature_means) / self.feature_stds
        else:
            return (X - self.feature_means) / self.feature_stds
    
    def fit(self, X, y):
        """Huấn luyện mô hình SVM với Batch Gradient Descent và tốc độ học được điều chỉnh"""
        # Chuẩn hóa đặc trưng
        X = self._normalize_features(X, fit=True)
        
        # Lấy các lớp duy nhất
        self.classes = np.unique(y)
        
        # Huấn luyện từng mô hình cho mỗi lớp (One-vs-Rest)
        for cls in self.classes:
            print(f"Huấn luyện SVM cho lớp {cls}...")
            y_binary = np.where(y == cls, 1, -1)
            n_samples, n_features = X.shape
            
            # Khởi tạo tham số
            w = np.zeros(n_features)
            b = 0
            
            # Tốc độ học ban đầu
            current_lr = self.lr
            
            # Huấn luyện với cập nhật theo batch
            for iteration in range(self.n_iters):
                # Tính toán margin phân loại cho tất cả mẫu
                margins = y_binary * (np.dot(X, w) + b)
                
                # Xác định mẫu nằm trong biên hoặc bị phân loại sai
                within_margin = margins < 1
                
                # Tính gradient sử dụng toàn bộ batch
                dw = self.lambda_param * w - np.sum(X[within_margin] * y_binary[within_margin].reshape(-1, 1), axis=0) / n_samples
                db = -np.sum(y_binary[within_margin]) / n_samples
                
                # Cập nhật tham số
                w = w - current_lr * dw
                b = b - current_lr * db
                
                # Giảm learning rate sau mỗi giai đoạn
                if (iteration + 1) % 200 == 0:
                    current_lr *= 0.75
                
                # Điều kiện dừng sớm
                if iteration > 500 and np.sum(within_margin) == 0:
                    print(f"Dừng sớm tại vòng lặp {iteration} - đạt phân tách hoàn hảo")
                    break
                    
            # Lưu tham số mô hình
            self.models[cls] = {'w': w, 'b': b}
        
        return self
    
    def predict(self, X):
        """Dự đoán nhãn lớp cho các mẫu trong X"""
        # Chuẩn hóa đặc trưng
        X = self._normalize_features(X)
        
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, len(self.classes)))
        
        # Tính điểm quyết định cho mỗi lớp
        for i, cls in enumerate(self.classes):
            model = self.models[cls]
            scores[:, i] = np.dot(X, model['w']) + model['b']
        
        # Trả về lớp có điểm cao nhất (margin lớn nhất)
        return self.classes[np.argmax(scores, axis=1)]

# Load models
svm_model_instance = None
try:
    svm_model_instance = pickle.load(open('models/svm_model.pkl', 'rb'))
    logger.info("SVM model (svm_model.pkl) loaded successfully")
except Exception as e:
    logger.error(f"Error loading SVM model (svm_model.pkl): {e}")


svc_model_instance = None
try:
    svc_model_instance = pickle.load(open('models/svc_model.pkl', 'rb'))
    logger.info("SVC model (svc_model.pkl) loaded successfully")
except Exception as e:
    logger.error(f"Error loading SVC model (svc_model.pkl): {e}")

rf_model = None
try:
    rf_model = pickle.load(open('models/rf_model.pkl', 'rb'))
    logger.info("Random Forest model loaded successfully")
except Exception as e:
    logger.error(f"Error loading Random Forest model: {e}")

cnn_lstm_model = None
try:
    cnn_lstm_model = load_model('models/cnn_lstm_model.h5')
    logger.info("CNN-LSTM model loaded successfully")
    # Compile CNN-LSTM model
    if cnn_lstm_model:
        cnn_lstm_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
except Exception as e:
    logger.error(f"Error loading CNN-LSTM model: {e}")


MODELS = {
    'SVM': svm_model_instance, # Will use the model from svc_model.pkl if loaded, else from svm_model.pkl, else None
    'SVC': svm_model_instance, # Same as above
    'Random Forest': rf_model,
    'CNN-LSTM': cnn_lstm_model
}

@app.route('/')
def index():
    return render_template('index.html', models=list(MODELS.keys()))

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return render_template('index.html', models=list(MODELS.keys()), error="No video file selected")
    file = request.files['video']
    model_name = request.form['model']
    if file.filename == '':
        return render_template('index.html', models=list(MODELS.keys()), error="No video file selected")
    if file and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if MODELS.get(model_name) is None: # Use .get() for safer access
            logger.error(f"Model {model_name} not found or not loaded.")
            return render_template('index.html', models=list(MODELS.keys()), error=f"Model {model_name} not loaded or not available")

        output_path = predict_video(filepath, MODELS[model_name], model_name, SELECTED_CLASSES)
        if output_path is None:
            logger.error(f"Error processing video {filepath} with model {model_name}")
            return render_template('index.html', models=list(MODELS.keys()), error="Error processing video")
        
        rel_output_path = os.path.basename(output_path).replace('\\', '/')
        logger.info(f"Output video path for template: {rel_output_path}")
        return render_template('result.html', video_path=rel_output_path, model_name=model_name)
    
    logger.warning(f"File format not supported for file: {file.filename}")
    return render_template('index.html', models=list(MODELS.keys()), error="File format not supported")

@app.route('/webcam', methods=['GET'])
def webcam():
    return render_template('webcam.html', models=list(MODELS.keys()))

@app.route('/video_feed/<model_name>')
def video_feed(model_name):
    if MODELS.get(model_name) is None: # Use .get()
        logger.error(f"Model {model_name} not loaded for webcam feed.")
        return Response("Model not loaded", status=500, mimetype='text/plain')
    return Response(gen_webcam_frames(MODELS[model_name], model_name, SELECTED_CLASSES),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download/<path:filename>')
def download_file(filename):
    logger.info(f"Serving file: {filename} from {app.config['OUTPUT_FOLDER']}")
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, mimetype='video/mp4')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=True)
