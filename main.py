import sys
import cv2
import numpy as np
from datetime import datetime
from threading import Thread
import queue
import time
import requests
from collections import deque
import RPi.GPIO as GPIO
import json
import os

from module.camera_module import Camera
from module.detection_module import ObjectDetector
from app import ProcessingWorker 
from utils.audio import play_event_sound

SERVER_MAC_URL = "http://192.168.1.196:5000"
CONFIG_FILE = "config.json"

GPIO.cleanup()
GPIO.setmode(GPIO.BCM)
TRIGGER_PIN = 17
GPIO.setup(TRIGGER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) 

RING_BUFFER = deque(maxlen=2)
CALIBRATED_CENTER = None
CURRENT_ZOOM = 1.0

def save_config():
    config_data = { 'zoom': CURRENT_ZOOM, 'center': CALIBRATED_CENTER }
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"💾 Đã lưu cấu hình: {config_data}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu file cấu hình: {e}")

def load_config():
    global CURRENT_ZOOM, CALIBRATED_CENTER
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config_data = json.load(f)
                CURRENT_ZOOM = config_data.get('zoom', 1.0)
                CALIBRATED_CENTER = config_data.get('center', None)
                print(f"✅ Đã tải cấu hình từ phiên trước: Zoom={CURRENT_ZOOM}, Tâm={CALIBRATED_CENTER}")
    except Exception as e:
        print(f"❌ Lỗi khi tải file cấu hình, sử dụng giá trị mặc định: {e}")

def report_initial_config():
    """Gửi cấu hình đã tải từ file lên server."""
    config_data = { 'zoom': CURRENT_ZOOM, 'center': CALIBRATED_CENTER }
    try:
        requests.post(f"{SERVER_MAC_URL}/report_config", json=config_data, timeout=10)
        print(f"📢 Đã báo cáo cấu hình ban đầu lên server: {config_data}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Không thể báo cáo cấu hình ban đầu: {e}")

class SenderWorker(Thread):
    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.daemon = True
        self.running = True
    def run(self):
        while self.running:
            try:
                jpg_buffer = self.frame_queue.get_nowait()
                try:
                    requests.post(f"{SERVER_MAC_URL}/video_upload", data=jpg_buffer, headers={'Content-Type': 'image/jpeg'}, timeout=0.5)
                except requests.exceptions.RequestException as e:
                    print(f"LỖI SENDER: {e}")
                self.frame_queue.task_done()
            except queue.Empty:
                time.sleep(0.01)
                continue
    def stop(self):
        self.running = False

class CommandPoller(Thread):
    def __init__(self, command_queue):
        super().__init__()
        self.command_queue = command_queue
        self.daemon = True
        self.running = True
    def run(self):
        while self.running:
            try:
                response = requests.get(f"{SERVER_MAC_URL}/get_command", timeout=1.0)
                if response.status_code == 200:
                    command = response.json()
                    if command:
                        self.command_queue.put(command)
            except requests.exceptions.RequestException as e:
                pass
            time.sleep(1)
    def stop(self):
        self.running = False

def set_zoom(picam2, zoom_factor, stream_size):
    if zoom_factor < 1.0: zoom_factor = 1.0
    full_width, full_height = picam2.camera_properties['PixelArraySize']
    stream_width, stream_height = stream_size
    target_aspect_ratio = stream_width / stream_height
    crop_width = full_width / zoom_factor
    crop_height = full_height / zoom_factor
    new_crop_width = crop_height * target_aspect_ratio
    if new_crop_width <= crop_width:
        crop_width = new_crop_width
    else:
        crop_height = crop_width / target_aspect_ratio
    crop_x = (full_width - crop_width) / 2
    crop_y = (full_height - crop_height) / 2
    crop_region = (int(crop_x), int(crop_y), int(crop_width), int(crop_height))
    picam2.set_controls({"ScalerCrop": crop_region})
    print(f"🔎 Đã thiết lập zoom kỹ thuật số: {zoom_factor}x")

def main():
    global CALIBRATED_CENTER, CURRENT_ZOOM
    
    load_config()
    Thread(target=report_initial_config, daemon=True).start()
    
    stream_width, stream_height = 480, 640
    cam = Camera(width=stream_width, height=stream_height)
    
    processing_queue = queue.Queue(maxsize=5)
    frame_queue = queue.Queue(maxsize=10)
    command_queue = queue.Queue(maxsize=5)

    detector = ObjectDetector(model_path="my_model.pt")
    
    processing_worker = ProcessingWorker(process_queue=processing_queue, detector=detector)
    sender_worker = SenderWorker(frame_queue)
    command_poller = CommandPoller(command_queue)
    
    workers = [processing_worker, sender_worker, command_poller]
    for worker in workers:
        worker.start()

    cam.start()
    set_zoom(cam.picam2, CURRENT_ZOOM, (stream_width, stream_height))
    
    print("✅ Hệ thống đã sẵn sàng!")
    play_event_sound(-1)
    print("🎥 Bắt đầu livestream...")
    
    previous_button_state = GPIO.LOW
    last_status_print_time = 0
    
    try:
        while True:
            try:
                command = command_queue.get_nowait()
                if command.get('type') == 'center':
                    new_center = command.get('value')
                    if new_center:
                        CALIBRATED_CENTER = { 'x': int(new_center.get('x')), 'y': int(new_center.get('y')) }
                        print(f"🎯 Tâm ngắm đã được cập nhật thành: {CALIBRATED_CENTER}")
                        save_config()
                elif command.get('type') == 'zoom':
                    zoom_value = command.get('value')
                    if zoom_value:
                        CURRENT_ZOOM = float(zoom_value)
                        set_zoom(cam.picam2, CURRENT_ZOOM, (stream_width, stream_height))
                        save_config()
            except queue.Empty:
                pass
            
            frame = cam.capture_frame()
            if frame is None:
                continue

            if CALIBRATED_CENTER:
                center_to_draw = (CALIBRATED_CENTER['x'], CALIBRATED_CENTER['y'])
            else:
                h, w, _ = frame.shape
                center_to_draw = (w // 2, h // 2)
            
            cv2.drawMarker(frame, center_to_draw, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2)
            RING_BUFFER.append(frame)

            _, jpg_buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not frame_queue.full():
                frame_queue.put(jpg_buffer.tobytes())
            
            current_time = time.time()
            if current_time - last_status_print_time > 3:
                print("Hệ thống đang hoạt động, chờ trigger...")
                last_status_print_time = current_time
            
            current_button_state = GPIO.input(TRIGGER_PIN)
            if current_button_state == GPIO.HIGH and previous_button_state == GPIO.LOW:
                capture_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"📸 Chụp ảnh lúc {capture_time}...")
                play_event_sound(-3)
                if len(RING_BUFFER) > 0 and not processing_queue.full():
                    frame_to_process = RING_BUFFER[0]
                    processing_queue.put((frame_to_process.copy(), capture_time, CALIBRATED_CENTER))

            previous_button_state = current_button_state
            time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\n🛑 Thoát...")
    finally:
        print("Đang dừng các luồng phụ...")
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join()
        
        cam.stop()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Đã dọn dẹp và thoát.")

if __name__ == '__main__':
    main()