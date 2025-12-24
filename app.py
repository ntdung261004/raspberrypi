import cv2
import queue
import requests
import logging
import os
import time
from threading import Thread
from flask import Flask, Response  # <<< THÊM MỚI

# Import các module chức năng
from utils.processing import check_object_center
from utils.handles import handle_hit_bia_so_4, handle_hit_bia_so_7, handle_hit_bia_so_8, handle_miss

class ProcessingWorker(Thread):
    # <<< TỐI ƯU: Nhận config và shared_state từ main >>>
    def __init__(self, process_queue, detector, server_url, config, shared_state):
        super().__init__()
        self.setName('ProcessingWorker') # Đặt tên cho luồng để log rõ ràng hơn
        self.process_queue = process_queue
        self.detector = detector
        self.server_url = server_url
        self.config = config
        self.shared_state = shared_state # Đối tượng trạng thái chia sẻ
        
        self._load_resources()

        self.daemon = True
        self.running = True
        logging.info("ProcessingWorker đã khởi động.")

    def _load_resources(self):
        """Tải các tài nguyên ảnh và mask từ đường dẫn trong config."""
        logging.info("Đang tải tài nguyên hình ảnh cho các loại bia...")
        paths = self.config['resource_paths']
        
        # <<< TỐI ƯU: Thêm try-except để kiểm tra file tồn tại >>>
        try:
            self.original_img_bia4 = cv2.imread(paths['original_bia4'])
            self.mask_bia4 = cv2.imread(paths['mask_bia4'], cv2.IMREAD_GRAYSCALE)
            self.original_img_bia4_alt = cv2.imread(paths['original_bia4_alt'])
            # ... Tương tự cho các bia khác
            self.original_img_bia7 = cv2.imread(paths['original_bia7'])
            self.mask_bia7 = cv2.imread(paths['mask_bia7'], cv2.IMREAD_GRAYSCALE)
            self.original_img_bia7_alt = cv2.imread(paths['original_bia7_alt'])
            self.original_img_bia8 = cv2.imread(paths['original_bia8'])
            self.mask_bia8 = cv2.imread(paths['mask_bia8'], cv2.IMREAD_GRAYSCALE)
            self.original_img_bia8_alt = cv2.imread(paths['original_bia8_alt'])
            
            # Kiểm tra nếu bất kỳ file nào không tải được
            if self.original_img_bia4 is None or self.mask_bia4 is None:
                raise FileNotFoundError("Một hoặc nhiều file tài nguyên cho bia 4 không tồn tại.")
        except Exception as e:
            logging.error(f"Lỗi nghiêm trọng khi tải tài nguyên: {e}. Vui lòng kiểm tra đường dẫn trong config.json.")
            # Có thể thoát chương trình ở đây nếu tài nguyên là bắt buộc
            os._exit(1)
    
    def run(self):
        while self.running:
            try:
                frame, capture_time = self.process_queue.get(timeout=1)
                # <<< TỐI ƯU: Lấy center_coords từ shared_state >>>
                center_coords = self.shared_state.calibrated_center
                self._process_frame(frame, capture_time, center_coords)
                self.process_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Lỗi không xác định trong luồng xử lý: {e}", exc_info=True)
    
    def _send_results(self, result_data):
        """Hàm riêng để gửi kết quả về server."""
        try:
            requests.post(f"{self.server_url}/processed_data_upload", json=result_data, timeout=5)
            logging.info("🚀 Gửi dữ liệu xử lý lên server thành công.")
        except requests.exceptions.RequestException as e:
            # <<< TỐI ƯU: Sử dụng shared_state để quản lý kết nối >>>
            if self.shared_state.server_is_connected:
                logging.warning(f"Mất kết nối khi gửi dữ liệu xử lý: {e}")
                self.shared_state.server_is_connected = False

    def _process_frame(self, frame, capture_time, center_coords):
        """Hàm điều phối, gọi đến các handler tương ứng."""
        detections = self.detector.detect(frame, conf=self.config['model']['confidence'])
        status, hit_info = check_object_center(detections, frame, center_coords)
        
        result_data = None
        
        if status == "TRÚNG":
            target_name = hit_info.get('name')
            if target_name == 'bia_so_4':
                result_data = handle_hit_bia_so_4(hit_info, capture_time, frame, self.original_img_bia4, self.original_img_bia4_alt, self.mask_bia4)
            elif target_name == 'bia_so_7_8':
                result_data = handle_hit_bia_so_7(hit_info, capture_time, frame, self.original_img_bia7, self.original_img_bia7_alt, self.mask_bia7)
            elif target_name == 'bia_so_8':
                result_data = handle_hit_bia_so_8(hit_info, capture_time, frame, self.original_img_bia8, self.original_img_bia8_alt, self.mask_bia8)
            else:
                print(f"Phát hiện trúng mục tiêu không xác định: {target_name}")
                result_data = handle_miss(hit_info, capture_time, frame)
        
        else: # TRƯỢT
            result_data = handle_miss(hit_info, capture_time, frame)

        if result_data:
            self._send_results(result_data)

    def stop(self):
        self.running = False
        print("🛑 ProcessingWorker đã dừng.")

# =================================================================
# === PHẦN THÊM MỚI: MÁY CHỦ STREAMING VIDEO CHO PICAMERA2 ===
# =================================================================

app = Flask(__name__)

camera_instance = None

def set_camera_instance(cam):
    """
    Hàm này được gọi từ main.py để đưa đối tượng camera vào app.
    """
    global camera_instance
    camera_instance = cam

def generate_frames():
    """
    Generator tạo luồng MJPEG từ đối tượng camera (Picamera2).
    """
    global camera_instance
    
    # Chờ cho đến khi camera được khởi tạo bởi main.py
    while camera_instance is None or not camera_instance.is_running():
        logging.warning("Streamer: Đang chờ camera khởi tạo...")
        time.sleep(1)
        
    logging.info("Streamer: Camera đã sẵn sàng, bắt đầu luồng video.")
    
    last_frame_time = 0
    while True:
        try:
            # Giới hạn tốc độ khung hình của luồng stream
            current_time = time.time()
            if (current_time - last_frame_time) < 0.03: # ~30 FPS
                time.sleep(0.01)
                continue
            last_frame_time = current_time

            # Lấy khung hình trực tiếp từ camera_module
            frame = camera_instance.capture_frame() 
            
            if frame is None:
                logging.warning("Streamer: Không nhận được khung hình từ camera.")
                time.sleep(0.5)
                continue

            # Vẽ hồng tâm lên luồng video
            center_coords = camera_instance.shared_state.calibrated_center
            if center_coords:
                center_to_draw = (center_coords['x'], center_coords['y'])
                cv2.drawMarker(frame, center_to_draw, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

            # Mã hóa khung hình thành JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            
            if not flag:
                continue
                
            # Trả về khung hình dưới dạng byte
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            
        except Exception as e:
            logging.error(f"Lỗi trong luồng generate_frames: {e}")
            break

@app.route('/streamSTV.mjpg')
def video_feed():
    """
    Route chính để cung cấp luồng video cho máy chủ.
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')