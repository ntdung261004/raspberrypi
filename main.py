import sys
import cv2
import queue
import time
import json
import os
import socket
import logging
from threading import Thread
from collections import deque
import requests

# <<< TỐI ƯU: Import module thay vì class lẻ >>>
from module import camera_module, detection_module
# [THAY ĐỔI 1] Import thêm app và set_camera_instance
from app import ProcessingWorker, app, set_camera_instance
from threads.workers import SenderWorker, CommandPoller, TriggerListener, ConfigReporter
from utils.audio import audio_manager

CONFIG_FILE = "config.json"

# <<< TỐI ƯU: Tạo một lớp để quản lý trạng thái chia sẻ >>>
class SharedState:
    def __init__(self):
        self.server_is_connected = True
        self.calibrated_center = None
        self.current_zoom = 1.0

def setup_logging():
    """Thiết lập hệ thống logging tập trung."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("shooting_range.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Hệ thống logging đã được khởi tạo.")

def load_config():
    """Tải toàn bộ cấu hình từ file JSON."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        logging.info("Tải file cấu hình thành công.")
        return config
    except FileNotFoundError:
        logging.error(f"Lỗi nghiêm trọng: Không tìm thấy file {CONFIG_FILE}! Vui lòng tạo file.")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"Lỗi nghiêm trọng: File {CONFIG_FILE} không đúng định dạng JSON.")
        sys.exit(1)

def save_runtime_settings(config, shared_state):
    """Lưu các cài đặt thay đổi trong lúc chạy (zoom, center)."""
    config['saved_settings']['zoom'] = shared_state.current_zoom
    config['saved_settings']['center'] = shared_state.calibrated_center
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        logging.info(f"Đã lưu cài đặt mới: {config['saved_settings']}")
    except Exception as e:
        logging.error(f"Lỗi khi lưu file cấu hình: {e}")

def resolve_hostname(hostname):
    """Phân giải hostname thành địa chỉ IP, thử lại nếu thất bại."""
    logging.info(f"Đang phân giải hostname '{hostname}'...")
    while True:
        try:
            ip_address = socket.gethostbyname(hostname)
            logging.info(f"Phân giải thành công: {hostname} -> {ip_address}")
            return ip_address
        except socket.gaierror:
            logging.warning(f"Không thể phân giải hostname. Thử lại sau 5 giây...")
            time.sleep(5)

def set_zoom(picam2, zoom_factor, stream_size):
    """Thiết lập zoom kỹ thuật số cho camera."""
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
    logging.info(f"Đã thiết lập zoom kỹ thuật số: {zoom_factor}x")

# <<< SỬA LỖI: Thêm hàm is_ip_address() đã bị thiếu >>>
def is_ip_address(hostname):
    """Kiểm tra xem một chuỗi có phải là định dạng IP hợp lệ không."""
    parts = hostname.split('.')
    if len(parts) != 4:
        return False
    for item in parts:
        if not item.isdigit() or not 0 <= int(item) <= 255:
            return False
    return True

def main():
    setup_logging()
    config = load_config()
    shared_state = SharedState()
    
    # Tải cài đặt từ phiên trước
    shared_state.current_zoom = config['saved_settings'].get('zoom', 1.0)
    shared_state.calibrated_center = config['saved_settings'].get('center', None)

    # <<< Logic kiểm tra IP/hostname đã được tinh chỉnh >>>
    hostname = config['server']['hostname']
    if not is_ip_address(hostname):
        logging.info(f"Giá trị '{hostname}' không phải IP, tiến hành phân giải tên miền...")
        server_ip = resolve_hostname(hostname)
    else:
        logging.info(f"Sử dụng địa chỉ IP tĩnh đã cấu hình: {hostname}")
        server_ip = hostname
    
    server_url = f"http://{server_ip}:{config['server']['port']}"
    
    stream_cfg = config['camera']
    # [THAY ĐỔI 2] Truyền shared_state vào Camera để luồng stream có thể truy cập
    cam = camera_module.Camera(
        width=stream_cfg['stream_width'], 
        height=stream_cfg['stream_height'],
        shared_state=shared_state 
    )
    
    # [THAY ĐỔI 3] Gửi đối tượng camera cho máy chủ streaming
    set_camera_instance(cam)
    logging.info("Đối tượng camera đã được đăng ký với máy chủ streaming.")
    
    # --- Phần khởi tạo worker và vòng lặp chính (giữ nguyên) ---
    processing_queue = queue.Queue(maxsize=5)
    frame_queue = queue.Queue(maxsize=10) # Hàng đợi này sẽ không được sử dụng nữa
    command_queue = queue.Queue(maxsize=5)
    ring_buffer = deque(maxlen=2)

    detector = detection_module.ObjectDetector(model_path=config['model']['path'])
    
    # [THAY ĐỔI 4] XÓA SenderWorker khỏi danh sách
    workers = [
        ProcessingWorker(processing_queue, detector, server_url, config, shared_state),
        # SenderWorker(frame_queue, server_url, shared_state), # <<< ĐÃ XÓA/VÔ HIỆU HÓA
        CommandPoller(command_queue, server_url, shared_state),
        TriggerListener(processing_queue, ring_buffer, config, shared_state),
        ConfigReporter(server_url, config, shared_state)
    ]
    
    for worker in workers:
        worker.start()

    cam.start()
    
    logging.info("🔥 Đang làm nóng model AI... Vui lòng chờ.")
    dummy_frame = cam.capture_frame()
    if dummy_frame is not None:
        detector.detect(dummy_frame)
    logging.info("✅ Model đã được làm nóng!")
    
    set_zoom(cam.picam2, shared_state.current_zoom, (stream_cfg['stream_width'], stream_cfg['stream_height']))
    
    logging.info("✅ Hệ thống đã sẵn sàng!")
    audio_manager.play_connected()
    
    try:
        # [THAY ĐỔI 5] Khởi chạy máy chủ streaming trong một luồng riêng
        stream_server_thread = Thread(
            target=lambda: app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False),
            daemon=True
        )
        stream_server_thread.start()
        logging.info(f"Máy chủ streaming MJPEG đã bắt đầu tại http://0.0.0.0:8000/streamSTV.mjpg")

        while True:
            try:
                command = command_queue.get_nowait()
                if command.get('type') == 'center':
                    new_center = command.get('value')
                    if new_center:
                        shared_state.calibrated_center = { 'x': int(new_center['x']), 'y': int(new_center['y']) }
                        logging.info(f"🎯 Tâm ngắm đã được cập nhật: {shared_state.calibrated_center}")
                        save_runtime_settings(config, shared_state)
                elif command.get('type') == 'zoom':
                    zoom_value = command.get('value')
                    if zoom_value:
                        shared_state.current_zoom = float(zoom_value)
                        set_zoom(cam.picam2, shared_state.current_zoom, (stream_cfg['stream_width'], stream_cfg['stream_height']))
                        save_runtime_settings(config, shared_state)
            except queue.Empty:
                pass
            
            # Luồng chính chỉ cần lấy frame và đưa vào ring_buffer cho TriggerListener
            frame = cam.capture_frame()
            if frame is None:
                continue
            
            ring_buffer.append(frame.copy())

            # [THAY ĐỔI 6] ĐÃ XÓA CÁC DÒNG MÃ HÓA VÀ PUSH VÀO frame_queue
            # if shared_state.calibrated_center:
            #     ... (đã chuyển logic vẽ hồng tâm vào generate_frames)
            # _, jpg_buffer = cv2.imencode(...)
            # if not frame_queue.full():
            #     frame_queue.put(jpg_buffer.tobytes())
            
            # Vòng lặp chính giờ đây rất nhẹ
            time.sleep(0.01) # Giữ cho vòng lặp không chạy quá nhanh
                
    except KeyboardInterrupt:
        logging.info("\n🛑 Nhận tín hiệu thoát...")
    finally:
        logging.info("Đang dừng các luồng phụ...")
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join()
        
        cam.stop()
        cv2.destroyAllWindows()
        logging.info("Đã dọn dẹp và thoát chương trình.")

if __name__ == '__main__':
    main()