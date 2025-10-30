# main.py

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

from module import camera_module, detection_module
from app import ProcessingWorker
from threads.workers import SenderWorker, CommandPoller, TriggerListener, ConfigReporter
from utils.audio import audio_manager

CONFIG_FILE = "config.json"

class SharedState:
    def __init__(self):
        self.server_is_connected = True
        self.calibrated_center = None
        self.current_zoom = 1.0

def setup_logging():
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
    config['saved_settings']['zoom'] = shared_state.current_zoom
    config['saved_settings']['center'] = shared_state.calibrated_center
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        logging.info(f"Đã lưu cài đặt mới: {config['saved_settings']}")
    except Exception as e:
        logging.error(f"Lỗi khi lưu file cấu hình: {e}")

def resolve_hostname(hostname):
    logging.info(f"Đang phân giải hostname '{hostname}'...")
    while True:
        try:
            ip_address = socket.gethostbyname(hostname)
            logging.info(f"Phân giải thành công: {hostname} -> {ip_address}")
            return ip_address
        except socket.gaierror:
            logging.warning(f"Không thể phân giải hostname. Thử lại sau 5 giây...")
            time.sleep(5)

def set_zoom(cam_obj, zoom_factor):
    if hasattr(cam_obj, 'set_zoom'):
        cam_obj.set_zoom(zoom_factor)
        logging.info(f"Đã gửi lệnh zoom {zoom_factor}x đến camera module.")
    else:
        logging.warning("Đối tượng camera hiện tại không hỗ trợ hàm set_zoom.")

def is_ip_address(hostname):
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

    shared_state.current_zoom = config['saved_settings'].get('zoom', 1.0)
    shared_state.calibrated_center = config['saved_settings'].get('center', None)

    hostname = config['server']['hostname']
    server_ip = resolve_hostname(hostname) if not is_ip_address(hostname) else hostname
    server_url = f"http://{server_ip}:{config['server']['port']}"

    stream_cfg = config['camera']
    cam = camera_module.Camera(
        stream_width=stream_cfg['stream_width'],
        stream_height=stream_cfg['stream_height']
    )

    processing_queue = queue.Queue(maxsize=5)
    frame_queue = queue.Queue(maxsize=10)
    command_queue = queue.Queue(maxsize=5)
    ring_buffer = deque(maxlen=2)

    detector = detection_module.ObjectDetector(model_path=config['model']['path'])

    workers = [
        ProcessingWorker(processing_queue, detector, server_url, config, shared_state),
        SenderWorker(frame_queue, server_url, shared_state),
        CommandPoller(command_queue, server_url, shared_state),
        TriggerListener(processing_queue, ring_buffer, config, shared_state),
        ConfigReporter(server_url, config, shared_state)
    ]

    for worker in workers:
        worker.start()

    cam.start()

    logging.info("🔥 Đang chờ khung hình đầu tiên từ webcam để làm nóng model...")
    dummy_frame = None
    while dummy_frame is None:
        dummy_frame = cam.capture_frame()
        time.sleep(0.5)

    detector.detect(dummy_frame)
    logging.info("✅ Model đã được làm nóng!")
    set_zoom(cam, shared_state.current_zoom)
    logging.info("✅ Hệ thống đã sẵn sàng!")
    audio_manager.play_connected()

    # <<< BẮT ĐẦU TỐI ƯU FPS VÀ CHẤT LƯỢNG ẢNH >>>
    TARGET_FPS = 25  # Mục tiêu 25 khung hình/giây
    FRAME_TIME = 1.0 / TARGET_FPS
    JPEG_QUALITY = 45 # Giảm nhẹ chất lượng để tiết kiệm băng thông

    logging.info(f"🚀 Tối ưu luồng video: Mục tiêu {TARGET_FPS} FPS, Chất lượng JPEG {JPEG_QUALITY}")
    # <<< KẾT THÚC TỐI ƯU FPS VÀ CHẤT LƯỢNG ẢNH >>>

    try:
        while True:
            start_time = time.time() # Ghi lại thời điểm bắt đầu vòng lặp

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
                        set_zoom(cam, shared_state.current_zoom)
                        save_runtime_settings(config, shared_state)
            except queue.Empty:
                pass

            frame = cam.capture_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            ring_buffer.append(frame.copy())

            if shared_state.calibrated_center:
                center_to_draw = (shared_state.calibrated_center['x'], shared_state.calibrated_center['y'])
                cv2.circle(frame, center_to_draw, 16, (255, 255, 255), 1)
                cv2.drawMarker(frame, center_to_draw, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

            # Sử dụng chất lượng JPEG đã được tối ưu
            _, jpg_buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not frame_queue.full():
                frame_queue.put(jpg_buffer.tobytes())

            # Điều tiết vòng lặp để đạt được TARGET_FPS
            elapsed_time = time.time() - start_time
            sleep_time = FRAME_TIME - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

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