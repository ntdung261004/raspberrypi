import sys
import cv2
import queue
import time
import json
import os
import socket
from threading import Thread
from collections import deque
import requests

from module.camera_module import Camera
from module.detection_module import ObjectDetector
from app import ProcessingWorker 
from utils.audio import play_event_sound
# <<< THÊM MỚI: Import các worker từ file mới >>>
from threads.workers import SenderWorker, CommandPoller, TriggerListener

# --- Cấu hình ---
SERVER_HOSTNAME = "Minh-Luan.local" # Chỉ cần định nghĩa tên máy chủ ở đây
CONFIG_FILE = "config.json"
SERVER_IS_CONNECTED = True

# --- Biến toàn cục ---
RING_BUFFER = deque(maxlen=2)
CALIBRATED_CENTER = None
CURRENT_ZOOM = 1.0

# --- Các hàm ---
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

def report_initial_config(server_url):
    config_data = { 'zoom': CURRENT_ZOOM, 'center': CALIBRATED_CENTER }
    try:
        requests.post(f"{server_url}/report_config", json=config_data, timeout=10)
        print(f"📢 Đã báo cáo cấu hình ban đầu lên server: {config_data}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Không thể báo cáo cấu hình ban đầu: {e}")

def resolve_hostname(hostname):
    print(f"🔄 Đang phân giải hostname '{hostname}'...")
    while True:
        try:
            ip_address = socket.gethostbyname(hostname)
            print(f"✅ Phân giải thành công: {hostname} -> {ip_address}")
            return ip_address
        except socket.gaierror:
            print(f"⚠️ Không thể phân giải hostname. Thử lại sau 5 giây...")
            time.sleep(5)

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
    
    server_ip = resolve_hostname(SERVER_HOSTNAME)
    server_mac_url = f"http://{server_ip}:5000"
    
    load_config()
    Thread(target=report_initial_config, args=(server_mac_url,), daemon=True).start()
    
    stream_width, stream_height = 480, 640
    cam = Camera(width=stream_width, height=stream_height)
    
    processing_queue = queue.Queue(maxsize=5)
    frame_queue = queue.Queue(maxsize=10)
    command_queue = queue.Queue(maxsize=5)

    detector = ObjectDetector(model_path="my_model.pt")
    
    # Khởi tạo các luồng
    # <<< SỬA LỖI 3: Khởi tạo các luồng theo đúng thiết kế module >>>
    processing_worker = ProcessingWorker(process_queue=processing_queue, detector=detector, server_url=server_mac_url)
    sender_worker = SenderWorker(frame_queue=frame_queue, server_url=server_mac_url)
    command_poller = CommandPoller(command_queue=command_queue, server_url=server_mac_url)
    # TriggerListener cần được cung cấp queue và buffer
    trigger_listener = TriggerListener(processing_queue=processing_queue, ring_buffer=RING_BUFFER)
    
    workers = [processing_worker, sender_worker, command_poller, trigger_listener]
    for worker in workers:
        worker.start()

    cam.start()
    # <<< THÊM MỚI TẠI ĐÂY: Giai đoạn "Làm nóng" >>>
    print("🔥 Đang làm nóng model AI... Vui lòng chờ.")
    # Chụp một frame thật từ camera để có kích thước đúng
    dummy_frame = cam.capture_frame()
    if dummy_frame is not None:
        # Thực hiện một lần nhận diện giả để tải model vào bộ nhớ
        detector.detect(dummy_frame)
    print("✅ Model đã được làm nóng!")
    # <<< KẾT THÚC PHẦN THÊM MỚI >>>
    set_zoom(cam.picam2, CURRENT_ZOOM, (stream_width, stream_height))
    
    print("✅ Hệ thống đã sẵn sàng!")
    play_event_sound(-1)
    print("🎥 Bắt đầu livestream...")
    
    last_status_print_time = 0
    
    try:
        while True:
            # Vòng lặp chính giờ đây rất gọn gàng
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
            
            RING_BUFFER.append(frame.copy())

            center_to_draw = (CALIBRATED_CENTER['x'], CALIBRATED_CENTER['y']) if CALIBRATED_CENTER else (stream_width // 2, stream_height // 2)
            cv2.drawMarker(frame, center_to_draw, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2)

            _, jpg_buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            if not frame_queue.full():
                frame_queue.put(jpg_buffer.tobytes())
            
            current_time = time.time()
            if current_time - last_status_print_time > 3:
                print("Hệ thống đang hoạt động, chờ trigger Bluetooth...")
                last_status_print_time = current_time
            
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
        print("Đã dọn dẹp và thoát.")

if __name__ == '__main__':
    main()