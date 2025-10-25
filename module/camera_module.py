#
# ----- BẮT ĐẦU NỘI DUNG FILE: camera_module.py -----
#
import cv2
import logging
import time
from threading import Thread, Lock
import numpy as np

class Camera:
    def __init__(self, stream_width=480, stream_height=640, device_index=0):
        # Yêu cầu webcam chụp ở độ phân giải ngang cao hơn
        self.capture_width = 1280
        self.capture_height = 720
        
        self.stream_width = stream_width
        self.stream_height = stream_height
        self.device_index = device_index
        
        self.cap = None
        self.thread = None
        self.frame = None
        self.running = False
        self.lock = Lock()
        
        # [TÍNH NĂNG MỚI] Thêm biến để lưu trữ mức zoom
        self.zoom_factor = 1.0
        
        logging.info(f"✅ Module Camera (chế độ Webcam USB với crop 3:4 và zoom) đã được cấu hình.")

    # [TÍNH NĂNG MỚI] Thêm phương thức để cập nhật mức zoom từ main.py
    def set_zoom(self, zoom_factor):
        with self.lock:
            self.zoom_factor = max(1.0, float(zoom_factor)) # Đảm bảo zoom không nhỏ hơn 1.0

    def _initialize_capture(self):
        self.cap = cv2.VideoCapture(self.device_index)
        
        if not self.cap.isOpened():
            logging.error(f"❌ Không thể mở webcam ở vị trí {self.device_index}.")
            self.cap = None
            return False
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        
        ret, frame = self.cap.read()
        if not ret:
            logging.error(f"❌ Không thể đọc khung hình đầu tiên từ webcam {self.device_index}.")
            self.cap.release()
            self.cap = None
            return False

        self.frame = self._process_frame(frame)
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logging.info(f"✅ Webcam USB đã khởi tạo. Chụp ở {actual_width}x{actual_height}, xuất ra {self.stream_width}x{self.stream_height}.")
        return True

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        [THAY ĐỔI] Cắt và resize khung hình, có tính đến yếu tố zoom.
        """
        source_height, source_width, _ = frame.shape
        
        # [THAY ĐỔI] Tính toán kích thước vùng nhìn sau khi zoom
        # Lấy zoom factor an toàn từ self.zoom_factor
        zoom = self.zoom_factor
        zoomed_width = source_width / zoom
        zoomed_height = source_height / zoom
        
        # Tính toán để cắt ra vùng 3:4 từ vùng đã zoom
        target_aspect_ratio = self.stream_width / self.stream_height

        crop_width = int(zoomed_height * target_aspect_ratio)
        crop_height = int(zoomed_height)

        if crop_width > zoomed_width:
            crop_width = int(zoomed_width)
            crop_height = int(zoomed_width / target_aspect_ratio)

        # Tọa độ để cắt từ ảnh gốc
        crop_x_start = int((source_width - crop_width) / 2)
        crop_y_start = int((source_height - crop_height) / 2)
        
        cropped_frame = frame[
            crop_y_start : crop_y_start + crop_height,
            crop_x_start : crop_x_start + crop_width
        ]

        final_frame = cv2.resize(cropped_frame, (self.stream_width, self.stream_height))
        return final_frame

    def _update(self):
        while self.running:
            if self.cap is None:
                if not self._initialize_capture():
                    time.sleep(5)
                    continue

            ret, frame = self.cap.read()
            with self.lock:
                if ret:
                    self.frame = self._process_frame(frame)
                else:
                    logging.warning("⚠️ Mất kết nối hoặc không thể đọc khung hình từ webcam.")
                    if self.cap:
                        self.cap.release()
                    self.cap = None
                    time.sleep(2)

    def start(self):
        if not self.running:
            self.running = True
            if not self._initialize_capture():
                 logging.warning("Khởi tạo webcam ban đầu thất bại, luồng sẽ tự động thử lại.")
            
            self.thread = Thread(target=self._update, name="CameraThread", daemon=True)
            self.thread.start()
            logging.info("Luồng đọc và xử lý webcam đã bắt đầu.")

    def capture_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        return None

    def stop(self):
        logging.info("Đang dừng luồng camera...")
        if self.running:
            self.running = False
            if self.thread is not None:
                self.thread.join()
            
            if self.cap is not None:
                self.cap.release()
            logging.info("Webcam đã được giải phóng.")
#
# ----- KẾT THÚC NỘI DUNG FILE: camera_module.py -----
#