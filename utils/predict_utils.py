import cv2
import numpy as np
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from .object_detection import ObjectDetector
from collections import deque
import logging
import ffmpeg

logger = logging.getLogger(__name__)

# Load feature extractor
try:
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    logger.info("ResNet50 model loaded successfully")
except Exception as e:
    logger.error(f"Error loading ResNet50 model: {e}")
    resnet_model = None

detector = ObjectDetector()

def preprocess_frame(frame):
    """Preprocess frame with background subtraction and histogram equalization.

    Args:
        frame: Input frame (numpy array).

    Returns:
        Processed grayscale frame.
    """
    subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
    fg_mask = subtractor.apply(frame)
    frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray

def extract_features(video_frames, num_frames=20, img_size=(64, 64)):
    """Extract features from video frames using ResNet50.

    Args:
        video_frames: List of video frames.
        num_frames: Number of frames to sample (default: 20).
        img_size: Target frame size (default: (64, 64)).

    Returns:
        Mean feature vector or None if extraction fails.
    """
    if resnet_model is None:
        logger.error("ResNet50 model not loaded")
        raise ValueError("ResNet50 model is not available")
    total_frames = len(video_frames)
    if total_frames == 0:
        logger.error("No frames provided")
        raise ValueError("No frames provided for feature extraction")
    if total_frames < num_frames:
        logger.warning(f"Video has only {total_frames} frames, padding with last frame")
        video_frames = video_frames + [video_frames[-1]] * (num_frames - total_frames)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    features = []
    for idx in frame_indices:
        frame = video_frames[idx]
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            frame = np.repeat(gray[:, :, np.newaxis], 3, axis=-1)
        frame = cv2.resize(frame, img_size)
        frame = frame.astype('float32') / 255.0
        x = image.img_to_array(frame)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = resnet_model.predict(x, verbose=0)
        features.append(feat.flatten())
    return np.mean(features, axis=0)

def draw_action_text(frame, label, prob, show_text=True):
    """Draw action label and probability on the frame.

    Args:
        frame: Input frame to annotate.
        label: Predicted action label.
        prob: Prediction probability (optional).
        show_text: Whether to display the action text (default: True).

    Returns:
        Annotated frame.
    """
    if show_text:
        text = f"Action: {label} ({prob:.2f})" if prob else f"Action: {label}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def predict_action(frames, model, model_name, classes, num_frames=20, img_size=(64, 64), is_cnn_lstm=False):
    """Predict action from video frames.

    Args:
        frames: List of video frames.
        model: Trained model for prediction.
        model_name: Name of the model ('CNN-LSTM', 'SVM', 'Random Forest', etc.).
        classes: List of class labels.
        num_frames: Number of frames to sample (default: 20).
        img_size: Target frame size (default: (64, 64)).
        is_cnn_lstm: Whether the model is CNN-LSTM (default: False).

    Returns:
        Tuple of (predicted label, probability) or (None, None) if prediction fails.
    """
    label_map = {i: cls for i, cls in enumerate(classes)}
    try:
        if is_cnn_lstm and len(frames) >= num_frames:
            frame_indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            frames_buffer = []
            for idx in frame_indices:
                gray = preprocess_frame(frames[idx])
                resized = cv2.resize(gray, img_size)
                resized = resized.astype('float32') / 255.0
                frames_buffer.append(resized)
            input_seq = np.array(frames_buffer)[..., np.newaxis]
            input_seq = np.expand_dims(input_seq, axis=0)
            pred_probs = model.predict(input_seq, verbose=0)[0]
            pred_index = np.argmax(pred_probs)
            return label_map.get(pred_index, "Unknown"), float(pred_probs[pred_index])
        else:
            feature = extract_features(frames, num_frames=num_frames, img_size=img_size)
            pred_input = np.array([feature])
            if hasattr(model, 'predict_proba'):
                try:
                    pred_probs = model.predict_proba(pred_input)[0]
                    pred = np.argmax(pred_probs)
                    prob = float(pred_probs[pred])
                except Exception:
                    pred = model.predict(pred_input)[0]
                    prob = None
            else:
                pred = model.predict(pred_input)[0]
                prob = None
            actual_pred_index = pred if isinstance(pred, (int, np.integer)) else classes.index(pred) if pred in classes else 0
            return label_map.get(actual_pred_index, "Unknown"), prob
    except Exception as e:
        logger.error(f"Error predicting action with model {model_name}: {e}")
        return None, None

def predict_video(video_path, model, model_name, classes, num_frames=20, img_size=(64, 64), batch_size=100, show_action_text=False):
    """Predict actions in a video and save the output with annotations.

    Args:
        video_path: Path to input video file.
        model: Trained model for prediction.
        model_name: Name of the model.
        classes: List of class labels.
        num_frames: Number of frames to sample (default: 20).
        img_size: Target frame size (default: (64, 64)).
        batch_size: Number of frames to process per batch (default: 100).
        show_action_text: Whether to display action text on frames (default: False).

    Returns:
        Path to output video or None if processing fails.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file {video_path}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 1:
        logger.error("Video has no frames")
        cap.release()
        return None

    # Prepare output paths
    base_name = os.path.basename(video_path)
    name_without_ext = os.path.splitext(base_name)[0]
    temp_output_filename = f"temp_output_{name_without_ext}.mp4"
    temp_out_path = os.path.join('outputs', temp_output_filename)
    output_filename = f"output_{name_without_ext}.mp4"
    out_path = os.path.join('outputs', output_filename)

    # Initialize OpenCV VideoWriter with mp4v codec (temporary)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    if not out.isOpened():
        logger.error(f"Cannot create temporary output video at {temp_out_path}")
        cap.release()
        return None

    # Read and process frames in batches
    frames = []
    label_queue = deque(maxlen=3)  # For label smoothing
    frame_count_total = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count_total += 1

        if len(frames) >= batch_size or frame_count_total == frame_count:
            # Predict action for the batch
            label, prob = predict_action(
                frames, model, model_name, classes,
                num_frames=num_frames, img_size=img_size,
                is_cnn_lstm=(model_name == 'CNN-LSTM')
            )
            if label is None:
                logger.error("Prediction failed for batch")
                out.release()
                cap.release()
                return None

            # Process each frame in the batch
            for frame in frames:
                current_frame = frame.copy()
                bboxes = detector.detect_humans(current_frame)
                label_queue.append(label)
                smoothed_label = max(set(label_queue), key=label_queue.count)
                current_frame = detector.draw_bboxes(current_frame, bboxes, smoothed_label, prob)
                current_frame = draw_action_text(current_frame, smoothed_label, prob, show_action_text)
                out.write(current_frame)
                logger.debug(f"Frame {frame_count_total}: Wrote frame with label {smoothed_label}")
            frames.clear()

    cap.release()
    out.release()
    logger.info(f"Temporary video saved at {temp_out_path}")

    # Convert to H.264
    try:
        stream = ffmpeg.input(temp_out_path)
        stream = ffmpeg.output(stream, out_path, vcodec='libx264', acodec='aac', format='mp4', pix_fmt='yuv420p')
        ffmpeg.run(stream)
        logger.info(f"Converted video to H.264 and saved at {out_path}")
        os.remove(temp_out_path)
        logger.info(f"Removed temporary video at {temp_out_path}")
    except ffmpeg.Error as e:
        logger.error(f"Error converting video to H.264: {e}")
        return None

    return out_path

def gen_webcam_frames(model, model_name, classes, num_frames=20, img_size=(64, 64), buffer_size=20):
    """Generate frames from webcam with action predictions (real-time, minimal lag)."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return

    frame_buffer = deque(maxlen=buffer_size)
    label_queue = deque(maxlen=3)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame from webcam")
                break

            current_frame = frame.copy()
            bboxes = detector.detect_humans(current_frame)

            # Chỉ lấy frame hiện tại cho SVM/RF, buffer cho CNN-LSTM
            if model_name == 'CNN-LSTM':
                frame_buffer.append(frame)
                frames_for_pred = list(frame_buffer)
            else:
                frames_for_pred = [frame]

            label, prob = predict_action(
                frames_for_pred, model, model_name, classes,
                num_frames=num_frames, img_size=img_size,
                is_cnn_lstm=(model_name == 'CNN-LSTM')
            )
            if label is None:
                label = "Unknown"
                prob = None

            label_queue.append(label)
            smoothed_label = max(set(label_queue), key=label_queue.count)
            current_frame = detector.draw_bboxes(current_frame, bboxes, smoothed_label, prob)
            current_frame = draw_action_text(current_frame, smoothed_label, prob, show_text=True)

            ret_jpeg, jpeg = cv2.imencode('.jpg', current_frame)
            if not ret_jpeg:
                logger.warning("Failed to encode frame to JPEG for webcam stream")
                continue
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        cap.release()
        logger.info("Webcam released")