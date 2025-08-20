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
SERVER_MAC_URL = "http://192.168.1.196:5000"

# --- Cấu hình GPIO cho nút bấm ---
GPIO.setmode(GPIO.BCM)
TRIGGER_PIN = 17
GPIO.setup(TRIGGER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) 

# Bộ đệm vòng để lưu trữ các khung hình gần nhất
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

def set_zoom(picam2, zoom_factor, stream_size):
    """
    Thiết lập zoom kỹ thuật số, đảm bảo giữ đúng tỷ lệ khung hình của stream.
    zoom_factor: 1.0 = không zoom, 2.0 = zoom 2x.
    stream_size: Tuple (width, height) của khung hình cuối cùng (ví dụ: (480, 640)).
    """
    if zoom_factor < 1.0:
        zoom_factor = 1.0

    full_width, full_height = picam2.camera_properties['PixelArraySize']
    stream_width, stream_height = stream_size

    # Tính tỷ lệ khung hình của stream (ví dụ: 480/640 = 0.75)
    target_aspect_ratio = stream_width / stream_height

    # Tính toán kích thước vùng crop ban đầu
    crop_width = full_width / zoom_factor
    crop_height = full_height / zoom_factor

    # Điều chỉnh vùng crop để có tỷ lệ khung hình chính xác
    # Bằng cách giữ chiều cao và tính lại chiều rộng
    new_crop_width = crop_height * target_aspect_ratio
    
    # Nếu chiều rộng mới nhỏ hơn chiều rộng crop ban đầu, ta dùng nó.
    # Ngược lại, ta phải giữ chiều rộng và tính lại chiều cao.
    if new_crop_width <= crop_width:
        crop_width = new_crop_width
    else:
        crop_height = crop_width / target_aspect_ratio

    # Tính toán điểm bắt đầu để crop từ chính giữa cảm biến
    crop_x = (full_width - crop_width) / 2
    crop_y = (full_height - crop_height) / 2
    
    # Tạo và áp dụng vùng crop
    crop_region = (int(crop_x), int(crop_y), int(crop_width), int(crop_height))
    picam2.set_controls({"ScalerCrop": crop_region})
    print(f"🔎 Đã thiết lập zoom kỹ thuật số: {zoom_factor}x, giữ tỷ lệ {stream_width}:{stream_height}")

def run_main():
    stream_width, stream_height = 480, 640
    cam = Camera(width=stream_width, height=stream_height)
    
    processing_queue = queue.Queue(maxsize=5)
    frame_queue = queue.Queue(maxsize=10)
    detector = ObjectDetector(model_path="my_model.pt")
    
    cam.start()
    
    # Áp dụng zoom 3x, bạn có thể thay đổi số này
    set_zoom(cam.picam2, 5, (stream_width, stream_height))
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
    
    previous_button_state = GPIO.LOW
    
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

            _, jpg_buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            
            if frame_queue.qsize() < frame_queue.maxsize:
                frame_queue.put(jpg_buffer.tobytes())
            
            current_button_state = GPIO.input(TRIGGER_PIN)
            
            if current_button_state == GPIO.HIGH and previous_button_state == GPIO.LOW:
                
                play_event_sound(-3) # Phát âm thanh "Đã bắn"

                if len(RING_BUFFER) > 0:
                    frame_to_process = RING_BUFFER[-1]
                    if processing_queue.qsize() < processing_queue.maxsize:
                        processing_queue.put((frame_to_process.copy(), capture_time))
                    else:
                        print("⚠️ Hàng đợi xử lý đang đầy, bỏ qua frame.")
                else:
                    print("⚠️ Bộ đệm vòng rỗng, không có khung hình để xử lý.")

            previous_button_state = current_button_state
            time.sleep(0.01)
                
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