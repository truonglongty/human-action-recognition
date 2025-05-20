from ultralytics import YOLO
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self):
        try:
            self.model = YOLO('yolov8n.pt')  # Sử dụng YOLOv8 nano
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {e}")
            self.model = None

    def detect_humans(self, frame):
        """Phát hiện con người trong khung hình và trả về danh sách bounding box"""
        if self.model is None:
            logger.error("YOLO model not loaded")
            return []
        results = self.model(frame, classes=[0])  # Class 0 là 'person' trong COCO
        bboxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            if conf > 0.5:  # Ngưỡng tin cậy
                bboxes.append((x1, y1, x2, y2, conf))
        logger.debug(f"Detected {len(bboxes)} humans in frame")
        return bboxes

    def draw_bboxes(self, frame, bboxes, label, prob=None):
        """Vẽ khung bao quanh và nhãn hành động"""
        for (x1, y1, x2, y2, conf) in bboxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # display_label = f"{label} ({prob:.2f})" if prob is not None else label
            display_label = label
            cv2.putText(frame, display_label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        logger.debug(f"Drew {len(bboxes)} bboxes with label {label}")
        return frame