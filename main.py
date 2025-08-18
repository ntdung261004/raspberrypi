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
from collections import deque
import RPi.GPIO as GPIO

from module.camera_module import Camera
from module.detection_module import ObjectDetector
from app import ProcessingWorker 
from utils.audio import play_event_sound

# Thay thế bằng địa chỉ IP chính xác của máy Mac của bạn
SERVER_MAC_URL = "http://192.168.1.95:5000"

# --- Cấu hình GPIO cho nút bấm ---
GPIO.setmode(GPIO.BCM)
TRIGGER_PIN = 17
GPIO.setup(TRIGGER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) 

# Bộ đệm vòng để lưu trữ các khung hình gần nhất
# Kích thước 2: lưu 2 khung hình gần nhất. Có thể điều chỉnh nếu cần.
RING_BUFFER = deque(maxlen=2)

class SenderWorker(Thread):
    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.daemon = True
        self.running = True

    def run(self):
        while self.running:
            try:
                jpg_buffer = self.frame_queue.get(timeout=0.1)
                try:
                    requests.post(
                        f"{SERVER_MAC_URL}/video_upload",
                        data=jpg_buffer,
                        headers={'Content-Type': 'image/jpeg'},
                        timeout=0.5
                    )
                except requests.exceptions.RequestException:
                    pass
                self.frame_queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self.running = False

def run_main():
    cam = Camera(width=640, height=480)
    processing_queue = queue.Queue(maxsize=5)
    frame_queue = queue.Queue(maxsize=10)
    detector = ObjectDetector(model_path="my_model.pt")
    
    cam.start()
    
    print("🔥 Đang khởi động và làm nóng hệ thống... Vui lòng chờ.")
    dummy_frame = cam.capture_frame()
    if dummy_frame is not None:
        detector.detect(dummy_frame)
    print("✅ Hệ thống đã sẵn sàng!")
    
    processing_worker = ProcessingWorker(process_queue=processing_queue, detector=detector)
    processing_worker.start()
    sender_worker = SenderWorker(frame_queue)
    sender_worker.start()
    
    time.sleep(2)
    play_event_sound(-1) # Phát âm thanh khởi động
    print("🎥 Camera đã khởi động thành công. Bắt đầu livestream...")
    
    last_trigger_time = 0
    debounce_time = 0.5
    
    try:
        while True:
            # Chụp khung hình mới nhất liên tục và đưa vào bộ đệm vòng
            frame = cam.capture_frame()
            if frame is None:
                continue
            RING_BUFFER.append(frame)

            h, w, _ = frame.shape
            center_x, center_y = w // 2, h // 2
            cv2.drawMarker(frame, (center_x, center_y), (0, 0, 255), 
                           markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2, line_type=cv2.LINE_AA)

            _, jpg_buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60]) # Giảm chất lượng xuống 60%
            
            if frame_queue.qsize() < frame_queue.maxsize:
                frame_queue.put(jpg_buffer.tobytes())
            
            print("Nhấn trigger để chụp ảnh:", end='\r')
            
            current_time = time.time()
            if GPIO.input(TRIGGER_PIN) == GPIO.HIGH and (current_time - last_trigger_time > debounce_time):
                last_trigger_time = current_time
                
                capture_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"📸 Chụp ảnh lúc {capture_time}...")
                
                play_event_sound(-3) # Phát âm thanh "Đã bắn"

                # Lấy khung hình mới nhất từ bộ đệm vòng để xử lý
                if len(RING_BUFFER) > 0:
                    frame_to_process = RING_BUFFER[-1]
                    if processing_queue.qsize() < processing_queue.maxsize:
                        processing_queue.put((frame_to_process.copy(), capture_time))
                    else:
                        print("⚠️ Hàng đợi xử lý đang đầy, bỏ qua frame.")
                else:
                    print("⚠️ Bộ đệm vòng rỗng, không có khung hình để xử lý.")
                
    except KeyboardInterrupt:
        print("\n🛑 Lỗi hệ thống, thoát...")
    finally:
        processing_worker.stop()
        sender_worker.stop()
        cam.stop()
        cv2.destroyAllWindows()
        GPIO.cleanup()


if __name__ == '__main__':
    run_main()