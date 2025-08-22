import sys
import cv2
import numpy as np
from datetime import datetime
from threading import Thread
import queue
import time
from typing import Optional, Tuple, List
import base64
import requests
from flask import Flask, jsonify

from module.camera_module import Camera
from module.detection_module import ObjectDetector
from utils.audio import play_event_sound, play_score_sound 
from utils.processing import check_object_center, warp_crop_to_original, calculate_score

# Thay thế bằng địa chỉ IP chính xác của máy Mac của bạn
SERVER_MAC_URL = "http://192.168.1.134:5000"

ORIGINAL_IMAGE_PATH = "images/original/bia_so_4.jpg"
DEFAULT_MASK_PATH = "images/mask/mask_bia_so_4.jpg"

app = Flask(__name__)

class ProcessingWorker(Thread):
    def __init__(self, process_queue, detector):
        super().__init__()
        self.process_queue = process_queue
        self.detector = detector
        self.original_img = cv2.imread(ORIGINAL_IMAGE_PATH)
        self.mask = cv2.imread(DEFAULT_MASK_PATH, cv2.IMREAD_GRAYSCALE)
        self.daemon = True
        self.running = True
        print("💡 ProcessingWorker đã khởi động.")
    
    def run(self):
        while self.running:
            try:
                frame, capture_time, center_coords = self.process_queue.get(timeout=0.1)
                self._process_frame(frame, capture_time, center_coords)
                self.process_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Lỗi trong luồng xử lý: {e}")
    
    def _process_frame(self, frame, capture_time, center_coords):
        print(f"✅ Bắt đầu xử lý ảnh chụp lúc {capture_time}...")
        
        results = self.detector.detect(frame, conf=0.6)
        status, obj_crop, shot_point = check_object_center(results, frame, center_coords, conf_threshold=0.6)
        
        result_data = {
            'time': capture_time,
            'target': 'Bia số 4',
            'score': '--',
            'image_data': ''
        }
        
        processed_image = frame.copy()
        score = 0
        
        # <<< KHỐI LỆNH ĐÃ ĐƯỢC SỬA ĐỔI HOÀN TOÀN >>>
        if status == "TRÚNG" and obj_crop is not None:
            # Luôn bắt đầu với ảnh bia gốc sạch sẽ
            processed_image = self.original_img.copy()
            
            # Cố gắng warp để tìm tọa độ chính xác trên bia gốc
            warped_img, transformed_point = warp_crop_to_original(self.original_img, obj_crop, shot_point)
            
            if warped_img is not None and transformed_point is not None:
                print("Đã warp thành công. Đang tính điểm.")
                score = calculate_score(transformed_point, self.original_img, self.mask)
                # Vẽ điểm chạm đã warp lên bia gốc
                cv2.drawMarker(processed_image, (int(transformed_point[0]), int(transformed_point[1])), 
                               (0, 0, 255), cv2.MARKER_CROSS, markerSize=40, thickness=3)
            else:
                print("❌ Warp thất bại. Đang tính điểm trên ảnh crop.")
                # Nếu warp thất bại, tính điểm dựa trên việc co giãn ảnh crop
                h_orig, w_orig = self.original_img.shape[:2]
                h_crop, w_crop = obj_crop.shape[:2]
                
                scaled_shot_point_x = int(shot_point[0] * w_orig / w_crop)
                scaled_shot_point_y = int(shot_point[1] * h_orig / h_crop)
                scaled_shot_point = (scaled_shot_point_x, scaled_shot_point_y)
                
                score = calculate_score(scaled_shot_point, self.original_img, self.mask)
                # Vẽ điểm chạm đã co giãn lên bia gốc
                cv2.drawMarker(processed_image, scaled_shot_point, 
                               (0, 0, 255), cv2.MARKER_CROSS, markerSize=40, thickness=3)
            
            result_data.update({"score": score})
            play_score_sound(score)
            
        elif status == "TRƯỢT":
            print("❌ Bắn không trúng mục tiêu.")
            result_data.update({"score": 0, "target": "Không trúng mục tiêu"})
            processed_image = frame.copy() 
            cv2.drawMarker(processed_image, shot_point, 
                           (0, 0, 255), cv2.MARKER_CROSS, markerSize=30, thickness=2)
            play_score_sound(0)
            
        else: # Bao gồm cả trường hợp "KHÔNG_PHÁT_HIỆN"
            print("⚠ Không xử lý được kết quả.")
            result_data.update({"score": 0, "target": "Không xử lý được"})
            processed_image = frame.copy()
            cv2.drawMarker(processed_image, shot_point, 
                           (0, 0, 255), cv2.MARKER_CROSS, markerSize=30, thickness=2)
            play_score_sound(0)
        # <<< KẾT THÚC SỬA ĐỔI >>>
        
        _, img_buffer = cv2.imencode('.jpg', processed_image)
        result_data['image_data'] = base64.b64encode(img_buffer).decode('utf-8')
        
        try:
            requests.post(f"{SERVER_MAC_URL}/processed_data_upload", json=result_data, timeout=5)
            print("🚀 Đã gửi dữ liệu xử lý lên server thành công.")
        except requests.exceptions.RequestException as e:
            print(f"❌ Lỗi khi gửi dữ liệu xử lý: {e}")

    def stop(self):
        self.running = False
        print("🛑 ProcessingWorker đã dừng.")